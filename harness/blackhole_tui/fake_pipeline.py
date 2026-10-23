"""A pretend pipeline for --demo: reads the job's config.json and prints the
same events and log lines a real run would, at a watchable speed. It runs
detached like the real one, so stop, reattach and logs can all be tried.

    python -m blackhole_tui.fake_pipeline --job ~/.blackhole/jobs/<job>

BLACKHOLE_DEMO_SPEED=4 runs it four times faster.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import signal
import sys
import time
from pathlib import Path

from blackhole_tui.catalog import find_model, find_schema, model_from_config, out_gb


def emit(stage: str, status: str = "running", pct: float | None = None, msg: str = "", **extra) -> None:
    ev = {"stage": stage, "status": status, "ts": round(time.time(), 3)}
    if pct is not None:
        ev["pct"] = round(pct, 1)
    if msg:
        ev["msg"] = msg
    ev.update(extra)
    print(json.dumps(ev), flush=True)


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def stage_lines(stage: str, i: int, n: int, ctx: dict) -> str:
    kind = stage.rstrip("0123456789")
    m = ctx["model"]
    return {
        "detect": ["reading config.json", "reading weight index", "matching router layout"][min(i * 3 // n, 2)],
        "plan": ["reading config.json", "sizing teacher and student", "picking kernels"][min(i * 3 // n, 2)],
        "weights": f"{m.id} shard {i}/{n}",
        "calibrate": f"teacher answers {i * 64}/{n * 64} for “{ctx['purpose'] or 'general'}”",
        "synth": f"teacher wrote {i * 1500}/{n * 1500} samples",
        "reap": f"scoring layer {i * 4}/{n * 4}" if i < n else f"{m.experts} → {ctx['keep']} experts / layer",
        "heal": f"step {i * 120}/{n * 120} · kl {0.62 - 0.4 * i / n:.3f}",
        "sft": f"epoch 1 · step {i * 80}/{n * 80} · loss {1.9 - 0.9 * i / n:.2f}",
        "dpo": f"pairs {i * 300}/{n * 300} · margin {0.1 + 0.6 * i / n:.2f}",
        "merge": f"merging adapters {i}/{n}",
        "eval": f"held-out {i * 32}/{n * 32} vs teacher",
        "gguf": f"{ctx['quant']}: per-layer expert bits, block {i}/{n}",
        "quant": f"routed experts to {ctx['quant']}: shard {i}/{n}",
        "qat": f"QAT round {i}/{n} · kld {0.05 - 0.04 * i / n:.3f}",
        "gptq": f"GPTQ int4 layer {i * 3}/{n * 3}",
        "reaq": f"codebooks 16/8 by REAP rank, layer {i * 3}/{n * 3}",
        "score": f"calibrating {i * 16}/{n * 16}",
        "prune": f"shard {i}/{n}",
        "package": ["writing Modelfile", "model.gguf checksum", "ollama create (dry run)"][min(i * 3 // n, 2)],
    }.get(kind, f"{stage} {i}/{n}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    job = Path(ap.parse_args().job)
    cfg = json.loads((job / "config.json").read_text())
    model = find_model(cfg["model"]) or model_from_config(cfg)
    schema = find_schema(cfg["schema"])
    speed = float(os.environ.get("BLACKHOLE_DEMO_SPEED", "1"))
    keep = round(model.experts * (1 - schema.prune)) if model.experts else 0
    ctx = {"model": model, "purpose": cfg.get("purpose", ""), "keep": keep, "quant": schema.quant}

    stopped = False

    def on_term(*_):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGTERM, on_term)

    emit("pipeline", "running", 0, f"started {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"job {job.name}: {model.id} · schema {schema.id} · target {cfg.get('target')}")
    random.seed(job.name)
    for stage in schema.stages():
        if stage.startswith("reap") and not model.experts:
            emit(stage, "skipped", 100, "dense model: no routed experts to prune")
            log(f"── {stage}: skipped, dense model")
            continue
        emit(stage, "running", 0)
        log(f"── {stage}")
        n = random.randint(10, 18)
        for i in range(1, n + 1):
            time.sleep(0.18 / speed)
            if stopped:
                emit(stage, "error", 100 * i / n, "stopped by user")
                emit("pipeline", "error", msg="stopped", outcome="stopped")
                return 1
            line = stage_lines(stage, i, n, ctx)
            emit(stage, "running", 100 * i / n, line)
            if i % 4 == 0:
                log(f"   {line}")
        extra = {}
        if stage == "detect":
            shape = f"moe {model.total_b:g}B, {model.experts} experts top-{model.top_k}" if model.experts else \
                    f"dense {model.total_b:g}B"
            extra = {"msg": shape}
        if stage == "eval":
            extra = {"top1": 0.86, "kl": 0.21, "ppl_teacher": 1.31, "ppl_student": 1.58, "gate": "passed"}
            log("   top-1 agreement 86% · ppl 1.31 → 1.58 · kl 0.210")
        if stage == "gptq":
            extra = {"kl": 0.034, "top1": 0.91, "expert_gb": 10.4, "msg": "demo numbers"}
        if stage == "reaq":
            extra = {"kl": 0.033, "top1": 0.91, "expert_gb": 9.5, "msg": "demo numbers"}
        emit(stage, "done", 100, **extra)
    gb = out_gb(model, schema)
    name = f"{model.name.replace(' ', '-')}-{schema.id}-{schema.quant.upper()}"
    out = job / "out" / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "README.md").write_text(f"# {name}\n\nDemo output: no weights were written.\n")
    emit("pipeline", "done", 100, model=str(out), size_gb=round(gb, 1), outcome="done", gates_passed=True)
    log(f"done: {out} (~{gb:.1f} GB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
