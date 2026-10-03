"""Stage 4: dynamic + static compression to GGUF.

1. Convert the healed model to a bf16 GGUF.
2. Compute a task importance matrix (imatrix) from the task data.
3. Dynamic pass: choose per-layer expert bit widths from the layer importance
   measured in the REAP stage, under the TaskSpec size budget (stages/bits.py).
4. Static pass: fixed types for attention, embeddings and output head.
   llama-quantize applies both passes in one run.
The dense student, if present, gets a plain imatrix Q4_K_M. Writes:
  work/candidates/<name>.gguf
  work/allocation.json   bit widths, size and speed estimates per candidate
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from common.progress import emit
from stages import bits
from stages._util import DRY_RUN, Job, run

STAGE = "quantize"


def to_gguf(job: Job, hf_dir: Path, name: str) -> tuple[Path, Path]:
    lc = Path(job.config.llama_cpp)
    bf16 = job.path("work", f"{name}-bf16.gguf")
    imatrix = job.path("work", f"{name}-imatrix.gguf")
    run(["python", str(lc / "convert_hf_to_gguf.py"), str(hf_dir), "--outtype", "bf16", "--outfile", str(bf16)], STAGE)
    run([str(lc / "build/bin/llama-imatrix"), "-m", str(bf16), "-f", str(job.path("data", "calib.txt")),
         "-o", str(imatrix), "-ngl", "99", "--chunks", "200"], STAGE)
    return bf16, imatrix


def quantize(job: Job, bf16: Path, imatrix: Path, out: Path, extra: list[str], base: str = "Q4_K_M") -> None:
    lc = Path(job.config.llama_cpp)
    run([str(lc / "build/bin/llama-quantize"), "--imatrix", str(imatrix), *extra, str(bf16), str(out), base], STAGE)


def run_stage(job: Job) -> None:
    cfg, spec = job.config, job.spec
    cand_dir = job.path("work", "candidates")
    cand_dir.mkdir(exist_ok=True)

    shape = bits.MoEShape.from_hf_config(json.loads(job.path("work", "healed", "config.json").read_text()))
    importance = json.loads(job.path("work", "layer_importance.json").read_text())
    budget = spec.target.max_size_gb - cfg.size_margin_gb
    report: dict = {"candidates": {}}

    emit(STAGE, pct=2, msg=f"{shape.total_params / 1e9:.1f}B params, budget {budget:.1f} GB")
    if not DRY_RUN:
        bf16, imatrix = to_gguf(job, job.path("work", "healed"), "healed")
    emit(STAGE, pct=40, msg="allocating bits by saliency")

    out = cand_dir / "lobbot-moe.gguf"
    for attempt in range(3):
        layers = bits.allocate(shape, importance, budget, cfg.bit_floor, cfg.bit_ceiling)
        if DRY_RUN:
            out.write_bytes(b"GGUF dry run")
            actual_gb = bits.estimate_size_gb(shape, layers)
            break
        quantize(job, bf16, imatrix, out, bits.quantize_args(layers))
        actual_gb = out.stat().st_size / 1e9
        if actual_gb <= spec.target.max_size_gb:
            break
        budget -= actual_gb - spec.target.max_size_gb + 0.2
        emit(STAGE, pct=60 + 5 * attempt, msg=f"{actual_gb:.2f} GB is over target, retrying at {budget:.2f} GB")
    else:
        raise RuntimeError(f"could not fit under {spec.target.max_size_gb} GB")

    report["candidates"]["lobbot-moe"] = {
        "path": str(out),
        "size_gb": round(actual_gb, 2),
        "params_b": round(shape.total_params / 1e9, 1),
        "tok_s_est": round(bits.estimate_tok_s(shape, layers, cfg.laptop_bandwidth_gb_s), 1),
        "bytes_per_token_gb": round(bits.bytes_per_token_gb(shape, layers), 2),
        "layers": [asdict(l) for l in layers],
    }
    report["bit_widths"] = bits.heatmap(layers)
    emit(STAGE, pct=75, msg=f"lobbot-moe {actual_gb:.2f} GB")

    dense_dir = job.path("work", "dense")
    if cfg.dense_fallback and dense_dir.exists():
        dout = cand_dir / "dense.gguf"
        if DRY_RUN:
            dout.write_bytes(b"GGUF dry run")
            size = 2.5
        else:
            d_bf16, d_imatrix = to_gguf(job, dense_dir, "dense")
            quantize(job, d_bf16, d_imatrix, dout, [])
            size = dout.stat().st_size / 1e9
        # Dense models read every weight per token.
        report["candidates"]["dense"] = {
            "path": str(dout), "size_gb": round(size, 2),
            "tok_s_est": round(cfg.laptop_bandwidth_gb_s * 0.65 / size, 1),
        }

    job.path("work", "allocation.json").write_text(json.dumps(report, indent=2))
    job.mark_done(STAGE, {k: v["size_gb"] for k, v in report["candidates"].items()})
    emit(STAGE, "done", 100, ", ".join(f"{k} {v['size_gb']} GB" for k, v in report["candidates"].items()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    run_stage(Job(ap.parse_args().job))
