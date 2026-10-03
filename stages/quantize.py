"""Stage 4: dynamic + static compression to GGUF.

1. Convert the healed model to a bf16 GGUF.
2. Compute a task importance matrix (imatrix) on chat-formatted task data.
3. Dynamic pass: choose per-layer expert bit widths under the TaskSpec size
   budget (stages/bits.py). Sensitivity per layer and projection combines the
   REAP-stage layer importance with the imatrix output energy of each tensor.
4. Static pass: fixed types for attention, embeddings and output head.
   llama-quantize applies both passes in one run.
The dense student, if present, gets a plain imatrix Q4_K_M. Writes:
  work/candidates/<name>.gguf
  work/allocation.json   bit widths, size and speed per candidate
Every intermediate (bf16 GGUF, imatrix) is reused if present, so a rerun after
a crash only redoes the step that failed.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from common.progress import emit
from stages import bits
from stages._util import DRY_RUN, Job, read_jsonl, run

STAGE = "quantize"
IMATRIX_CHUNKS = 200  # x 512 tokens of task text
EXPERT_PROJ = {"ffn_gate_exps": "gate", "ffn_up_exps": "up", "ffn_down_exps": "down"}


def gguf_lib(job: Job):
    """gguf-py from the same llama.cpp checkout that wrote the files."""
    p = str(Path(job.config.llama_cpp) / "gguf-py")
    if p not in sys.path:
        sys.path.insert(0, p)
    import gguf
    return gguf


def bin_path(job: Job, name: str) -> str:
    p = Path(job.config.llama_cpp) / "build" / "bin" / name
    if not p.exists():
        raise FileNotFoundError(f"{p} missing; build llama.cpp (scripts/setup_vm.sh)")
    return str(p)


def calibration_text(job: Job, hf_dir: Path) -> Path:
    """Task conversations rendered with the model's own chat template, so the
    imatrix sees the same tokens (including <|im_start|> etc.) as inference."""
    out = job.path("work", "calib-chat.txt")
    if out.exists():
        return out
    train = job.path("data", "train.jsonl")
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(str(hf_dir))
        rows = read_jsonl(train)[:1000]
        texts = [tok.apply_chat_template(r["messages"], tokenize=False) for r in rows]
        if not texts:
            raise ValueError("no training rows")
        out.write_text("\n".join(texts))
    except Exception as e:  # fall back to the data stage's plain text
        emit(STAGE, msg=f"chat-formatted calibration unavailable ({e}); using data/calib.txt")
        out.write_text(job.path("data", "calib.txt").read_text())
    return out


def to_gguf(job: Job, hf_dir: Path, name: str) -> tuple[Path, Path]:
    lc = Path(job.config.llama_cpp)
    bf16 = job.path("work", f"{name}-bf16.gguf")
    imatrix = job.path("work", f"{name}-imatrix.gguf")
    if not bf16.exists():
        tmp = bf16.with_name(bf16.name + ".part")
        run([sys.executable, str(lc / "convert_hf_to_gguf.py"), str(hf_dir), "--outtype", "bf16", "--outfile", str(tmp)], STAGE)
        tmp.rename(bf16)
    if not imatrix.exists():
        calib = calibration_text(job, hf_dir)
        tmp = imatrix.with_name("tmp-" + imatrix.name)  # imatrix wants a .gguf suffix
        run([bin_path(job, "llama-imatrix"), "-m", str(bf16), "-f", str(calib), "-o", str(tmp),
             "-ngl", "999", "-c", "512", "--chunks", str(IMATRIX_CHUNKS), "--parse-special", "--no-ppl"], STAGE)
        tmp.rename(imatrix)
    return bf16, imatrix


def _f32(gguf, ggml_type, data):
    """F32/F16/BF16 tensor data as a flat float32 array (gguf-py exposes BF16 as raw bytes)."""
    import numpy as np

    if ggml_type == gguf.GGMLQuantizationType.BF16:
        return (np.asarray(data).reshape(-1).view(np.uint16).astype(np.uint32) << 16).view(np.float32)
    return np.asarray(data, dtype=np.float32).reshape(-1)


def imatrix_energy(job: Job, bf16: Path, imatrix: Path, n_layers: int) -> dict[str, list[float]]:
    """Expected output energy of each expert tensor on task tokens:
        E_l = sum_experts sum_j in_sum2[e, j] * ||W_e[:, j]||^2
    in_sum2 already sums x_j^2 only over tokens routed to expert e, so rarely
    used experts count less. Quantization noise in a tensor scales with this.
    """
    import numpy as np

    gguf = gguf_lib(job)
    im = {t.name: t for t in gguf.GGUFReader(str(imatrix)).tensors}
    energy = {k: [float("nan")] * n_layers for k in ("gate", "up", "down")}
    for t in gguf.GGUFReader(str(bf16)).tensors:
        parts = t.name.split(".")
        if len(parts) != 4 or parts[0] != "blk" or parts[2] not in EXPERT_PROJ:
            continue
        layer, kind = int(parts[1]), EXPERT_PROJ[parts[2]]
        s2 = im.get(t.name + ".in_sum2")
        if s2 is None or layer >= n_layers:
            continue
        n_exp, n_out, n_in = (int(x) for x in reversed(t.shape.tolist()))
        sum2 = np.asarray(s2.data, dtype=np.float32).reshape(-1, n_in)
        w = t.data.reshape(n_exp, -1)
        total = 0.0
        for e in range(min(n_exp, sum2.shape[0])):  # one expert at a time keeps RAM low
            we = _f32(gguf, t.tensor_type, w[e])
            col2 = (we.reshape(n_out, n_in) ** 2).sum(axis=0)
            total += float(sum2[e] @ col2)
        energy[kind][layer] = total
    if all(x != x for v in energy.values() for x in v):
        raise RuntimeError("imatrix has no expert tensors")
    return energy


def gguf_report(job: Job, path: Path) -> dict:
    """Exact size, per-token bytes and per-tensor types of a written GGUF."""
    gguf = gguf_lib(job)
    r = gguf.GGUFReader(str(path))
    arch = str(bytes(r.fields["general.architecture"].parts[-1]), "utf-8")

    def kv(key, default):
        f = r.fields.get(f"{arch}.{key}")
        return int(f.parts[-1][0]) if f else default

    n_exp, n_used = kv("expert_count", 0), kv("expert_used_count", 0)
    frac = n_used / n_exp if n_exp else 1.0
    per_token, widths = 0, []
    for t in r.tensors:
        if t.name.startswith("token_embd"):
            continue  # one row per token, negligible
        per_token += t.n_bytes * (frac if "_exps" in t.name else 1.0)
        parts = t.name.split(".")
        if len(parts) == 4 and parts[2] in EXPERT_PROJ:
            block, type_size = gguf.GGML_QUANT_SIZES[t.tensor_type]
            widths.append({"layer": int(parts[1]), "tensor": parts[2], "type": t.tensor_type.name.lower(),
                           "bits": round(type_size * 8 / block, 3)})
    widths.sort(key=lambda w: (w["layer"], w["tensor"]))
    return {"arch": arch, "size_gb": path.stat().st_size / 1e9, "bytes_per_token_gb": per_token / 1e9,
            "bit_widths": widths}


def quantize(job: Job, bf16: Path, imatrix: Path, out: Path, extra: list[str], base: str = "Q4_K_M") -> None:
    tmp = out.with_name("tmp-" + out.name)
    run([bin_path(job, "llama-quantize"), "--imatrix", str(imatrix), *extra, str(bf16), str(tmp), base], STAGE)
    tmp.replace(out)


def tok_s(cfg, gb_per_token: float) -> float:
    return round(cfg.laptop_bandwidth_gb_s * 0.65 / max(gb_per_token, 1e-9), 1)


def healed_dir(job: Job) -> Path:
    """Scaffold layout first; the REAP thread's early layout as a fallback."""
    for p in (job.path("work", "healed"), job.path("heal", "model"), job.path("reap", "model")):
        if (p / "config.json").exists():
            return p
    return job.path("work", "healed")


def layer_importance(job: Job, n_layers: int) -> list[float] | None:
    """work/layer_importance.json, else per-layer REAP saliency summed over kept experts."""
    p = job.path("work", "layer_importance.json")
    if p.exists():
        imp = json.loads(p.read_text())
    else:
        sal = next((q for q in (job.path("work", "reap_saliency.json"), job.path("reap", "saliency.json")) if q.exists()), None)
        if sal is None:
            return None
        layers = sorted(json.loads(sal.read_text())["layers"], key=lambda l: l["layer"])
        imp = [l.get("layer_saliency_kept") or sum(l["saliency"][e] for e in l["kept"]) for l in layers]
    if len(imp) != n_layers:
        emit(STAGE, msg=f"layer importance has {len(imp)} entries for {n_layers} layers; ignoring")
        return None
    return imp


def run_stage(job: Job) -> None:
    cfg, spec = job.config, job.spec
    cand_dir = job.path("work", "candidates")
    cand_dir.mkdir(exist_ok=True)
    healed = healed_dir(job)

    shape = bits.MoEShape.from_hf_config(json.loads((healed / "config.json").read_text()))
    importance = layer_importance(job, shape.n_layers)
    budget = spec.target.max_size_gb - cfg.size_margin_gb
    # Laptop decode speed is bandwidth-bound, so the tok/s floor is a cap on bytes read per token.
    max_per_token = cfg.laptop_bandwidth_gb_s * 0.65 / spec.target.min_tok_s
    report: dict = {"candidates": {}}

    emit(STAGE, pct=2, msg=f"{shape.total_params / 1e9:.1f}B params, {shape.n_experts} experts, budget {budget:.1f} GB")
    energy = None
    if not DRY_RUN:
        emit(STAGE, pct=5, msg="converting healed model to GGUF")
        bf16, imatrix = to_gguf(job, healed, "healed")
        emit(STAGE, pct=35, msg="measuring per-tensor sensitivity from the task imatrix")
        try:
            energy = imatrix_energy(job, bf16, imatrix, shape.n_layers)
        except Exception as e:
            emit(STAGE, msg=f"imatrix sensitivity unavailable ({e}); using REAP importance only")
    emit(STAGE, pct=40, msg="allocating bits by saliency")

    out = cand_dir / "lobbot-moe.gguf"
    for attempt in range(3):
        layers = bits.allocate(shape, importance, budget, cfg.bit_floor, cfg.bit_ceiling,
                               proj_energy=energy, max_gb_per_token=max_per_token)
        est = bits.estimate_size_gb(shape, layers)
        emit(STAGE, pct=45 + 5 * attempt, msg=f"allocation {est:.2f} GB est.; quantizing")
        if DRY_RUN:
            out.write_bytes(b"GGUF dry run")
            actual = {"size_gb": est, "bytes_per_token_gb": bits.bytes_per_token_gb(shape, layers),
                      "bit_widths": bits.heatmap(layers)}
            break
        quantize(job, bf16, imatrix, out, bits.quantize_args(layers))
        actual = gguf_report(job, out)
        if actual["size_gb"] <= spec.target.max_size_gb:
            break
        budget -= actual["size_gb"] - spec.target.max_size_gb + 0.2
        emit(STAGE, pct=60 + 5 * attempt, msg=f"{actual['size_gb']:.2f} GB is over target, retrying at {budget:.2f} GB")
    else:
        raise RuntimeError(f"could not fit under {spec.target.max_size_gb} GB")

    size_gb = round(actual["size_gb"], 2)
    report["candidates"]["lobbot-moe"] = {
        "path": str(out),
        "size_gb": size_gb,
        "params_b": round(shape.total_params / 1e9, 1),
        "tok_s_est": tok_s(cfg, actual["bytes_per_token_gb"]),
        "bytes_per_token_gb": round(actual["bytes_per_token_gb"], 3),
        "layers": [asdict(l) for l in layers],
        "sensitivity": bits.sensitivities(shape.n_layers, importance, energy),
    }
    report["bit_widths"] = actual["bit_widths"]
    emit(STAGE, pct=75, msg=f"lobbot-moe {size_gb:.2f} GB, ~{report['candidates']['lobbot-moe']['tok_s_est']} tok/s on the laptop")

    dense_dir = job.path("work", "dense")
    has_dense = dense_dir.exists() if DRY_RUN else (dense_dir / "config.json").exists()
    if cfg.dense_fallback and has_dense:
        dout = cand_dir / "dense.gguf"
        emit(STAGE, pct=80, msg="quantizing dense student")
        if DRY_RUN:
            dout.write_bytes(b"GGUF dry run")
            d = {"size_gb": 2.5, "bytes_per_token_gb": 2.5}
        else:
            d_bf16, d_imatrix = to_gguf(job, dense_dir, "dense")
            quantize(job, d_bf16, d_imatrix, dout, [])
            d = gguf_report(job, dout)
        report["candidates"]["dense"] = {
            "path": str(dout), "size_gb": round(d["size_gb"], 2),
            "tok_s_est": tok_s(cfg, d["bytes_per_token_gb"]),
            "bytes_per_token_gb": round(d["bytes_per_token_gb"], 3),
        }

    job.path("work", "allocation.json").write_text(json.dumps(report, indent=2))
    job.mark_done(STAGE, {k: v["size_gb"] for k, v in report["candidates"].items()})
    emit(STAGE, "done", 100, ", ".join(f"{k} {v['size_gb']} GB" for k, v in report["candidates"].items()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    run_stage(Job(ap.parse_args().job))
