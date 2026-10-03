"""LobBot compression orchestrator.

    python pipeline.py --job <job_dir> [--from <stage>] [--only <stage>]

Runs each stage as its own subprocess (so GPU memory is fully released
between vLLM, training and llama.cpp), relays their progress lines, and
skips stages already completed in this job dir. <job_dir>/taskspec.json must
exist. Exit code 0 means the whole pipeline finished.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from common.progress import emit
from stages._util import Job

STAGES = ["data", "reap", "heal", "quantize", "eval", "package"]
ROOT = Path(__file__).resolve().parent

# vLLM and llm-compressor/TRL pin different torch and transformers versions,
# so setup_vm.sh installs them in separate venvs. The data stage runs in the
# vLLM venv; everything else in the training venv.
STAGE_PYTHON = {"data": os.environ.get("LOBBOT_VLLM_PY")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--job", required=True)
    ap.add_argument("--from", dest="start", choices=STAGES, help="rerun from this stage onward")
    ap.add_argument("--only", choices=STAGES, help="run a single stage")
    args = ap.parse_args()

    job = Job(args.job)
    if not job.path("taskspec.json").exists():
        emit("pipeline", "error", msg=f"missing {job.path('taskspec.json')}")
        return 2
    job.save_config()  # record the effective config for this run
    if args.start:
        job.clear_from(STAGES, args.start)

    todo = [args.only] if args.only else STAGES
    for stage in todo:
        if job.is_done(stage) and not args.only:
            emit(stage, "skipped", 100, "cached")
            continue
        emit(stage, "running", 0, "starting")
        proc = subprocess.run([STAGE_PYTHON.get(stage) or sys.executable, "-m", f"stages.{stage}", "--job", str(job.root)], cwd=ROOT)
        if proc.returncode != 0 or not job.is_done(stage):
            emit(stage, "error", msg=f"stage exited with {proc.returncode}")
            return 1
    emit("pipeline", "done", 100, str(job.path("out", "model.gguf")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
