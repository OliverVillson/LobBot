"""Stage 3: heal by distillation.

LoRA SFT of the pruned MoE on the unpruned teacher's answers (router,
attention and experts), then merge. If dense_fallback is on, the dense
student is SFT'd on the same data at the same time in a child process, so a
crash there cannot take down the main path. Writes:
  work/healed/   pruned + healed MoE (merged HF model)
  work/dense/    dense student (merged HF model), if enabled

Knobs: heal_* and student_epochs in <job>/config.json (Config). The
wall-clock caps are env vars, LOBBOT_HEAL_MAX_MINUTES and
LOBBOT_STUDENT_MAX_MINUTES (default 60 each: training stops early at the cap
and still merges and saves; 0 means no cap). LOBBOT_HEAL_MAX_LEN (default
2048) is the per-example token limit for both models; longer examples are cut
short, so raise it for long answers such as code.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

from common.progress import emit, parse
from stages._util import DRY_RUN, Job

STAGE = "heal"
_lock = threading.Lock()


def _knob(cfg, name: str, env: str, default):
    v = getattr(cfg, name, None)
    if v is not None:
        return v
    raw = os.environ.get(env)
    if raw is None:
        return default
    if isinstance(default, bool):
        return raw.lower() not in ("0", "false", "no")
    return type(default)(raw)


def heal_moe(job: Job, progress) -> None:
    from stages.sft import train_sft
    from stages.taskdata import load_examples

    cfg = job.config
    kd = float(_knob(cfg, "heal_kd_weight", "LOBBOT_KD_WEIGHT", 0.0))
    train_sft(
        str(job.path("work", "reaped")), load_examples(job.path("data", "train.jsonl")), job.path("work", "healed"),
        epochs=cfg.heal_epochs, lr=cfg.heal_lr, r=cfg.heal_lora_r, alpha=2 * cfg.heal_lora_r,
        targets=cfg.heal_targets, train_experts=bool(_knob(cfg, "heal_train_experts", "LOBBOT_HEAL_EXPERTS", True)),
        train_router=True, kd_teacher=job.model_path(cfg.teacher) if kd > 0 else None, kd_weight=kd,
        max_len=int(_knob(cfg, "heal_max_len", "LOBBOT_HEAL_MAX_LEN", 2048)),
        max_tokens=16384, max_minutes=float(_knob(cfg, "heal_max_minutes", "LOBBOT_HEAL_MAX_MINUTES", 60.0)),
        progress=progress)


def train_dense(job: Job, progress) -> None:
    from stages.sft import train_sft
    from stages.taskdata import load_examples

    cfg = job.config
    train_sft(
        job.model_path(cfg.student), load_examples(job.path("data", "train.jsonl")), job.path("work", "dense"),
        epochs=float(_knob(cfg, "student_epochs", "LOBBOT_STUDENT_EPOCHS", 2.0)), lr=2e-4, r=32, alpha=64,
        max_len=int(_knob(cfg, "heal_max_len", "LOBBOT_HEAL_MAX_LEN", 2048)),
        max_tokens=32768, max_minutes=float(_knob(cfg, "student_max_minutes", "LOBBOT_STUDENT_MAX_MINUTES", 60.0)),
        progress=progress)


def run_stage(job: Job) -> None:
    cfg = job.config
    if DRY_RUN:
        import shutil
        shutil.copytree(job.path("work", "reaped"), job.path("work", "healed"), dirs_exist_ok=True)
        if cfg.dense_fallback:
            job.path("work", "dense").mkdir(exist_ok=True)
        emit(STAGE, pct=90, msg="dry run: copied pruned model")
        job.mark_done(STAGE, {"dense": cfg.dense_fallback})
        emit(STAGE, "done", 100, "healed" + (" + dense student" if cfg.dense_fallback else ""))
        return

    pct = {"moe": 0.0, "dense": 0.0 if cfg.dense_fallback else 100.0}

    def report(part: str, p: float, m: str) -> None:
        pct[part] = p
        total = (pct["moe"] + pct["dense"]) / 2 if cfg.dense_fallback else pct["moe"]
        with _lock:
            emit(STAGE, pct=total, msg=("MoE: " if part == "moe" else "dense: ") + m)

    child = reader = None
    if cfg.dense_fallback:
        child = subprocess.Popen(
            [sys.executable, "-u", "-m", "stages.heal", "--job", str(job.root), "--part", "dense"],
            cwd=Path(__file__).resolve().parents[1], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, errors="replace")

        def pump():
            for line in child.stdout:
                ev = parse(line)
                if ev is None:
                    with _lock:
                        sys.stdout.write("[dense] " + line)
                        sys.stdout.flush()
                elif ev["status"] == "error":
                    with _lock:
                        print(f"[dense] failed: {ev.get('msg')}", flush=True)
                elif "pct" in ev:
                    report("dense", ev["pct"], ev.get("msg", ""))
        reader = threading.Thread(target=pump, daemon=True)
        reader.start()

    try:
        heal_moe(job, lambda p, m: report("moe", p, m))
    except BaseException:
        if child:
            child.terminate()
        raise

    dense_ok = False
    if child:
        dense_ok = child.wait() == 0 and job.path("work", "dense", "config.json").exists()
        reader.join(timeout=5)
        if not dense_ok:
            print("heal: dense student failed; continuing with the MoE only (see [dense] lines above)", flush=True)
    job.mark_done(STAGE, {"dense": dense_ok})
    emit(STAGE, "done", 100, "healed" + (" + dense student" if dense_ok else ""))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--part", choices=["all", "moe", "dense"], default="all")
    a = ap.parse_args()
    job = Job(a.job)
    if a.part == "all":
        run_stage(job)
    else:
        (heal_moe if a.part == "moe" else train_dense)(job, lambda p, m: emit(STAGE, pct=p, msg=m))
        emit(STAGE, "done", 100, f"{a.part} trained")
