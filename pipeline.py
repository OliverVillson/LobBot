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
import signal
import subprocess
import sys
import time
from pathlib import Path

from common.progress import emit, parse
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
        rc, tail = run_stage(stage, job)
        if rc != 0 or not job.is_done(stage):
            emit(stage, "error", msg=error_message(rc, tail))
            return 1
    emit("pipeline", "done", 100, str(job.path("out", "model.gguf")))
    return 0


_child: subprocess.Popen | None = None


def run_stage(stage: str, job: Job) -> tuple[int, list[str]]:
    """Run one stage, relaying its output and teeing it to <job>/logs/<stage>.log.

    Returns the exit code and the last non-protocol lines (for the error message).
    """
    global _child
    cmd = [STAGE_PYTHON.get(stage) or sys.executable, "-u", "-m", f"stages.{stage}", "--job", str(job.root)]
    log_dir = job.path("logs")
    log_dir.mkdir(exist_ok=True)
    tail: list[str] = []
    with open(log_dir / f"{stage}.log", "a") as log:
        log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(cmd)}\n")
        _child = subprocess.Popen(cmd, cwd=ROOT, env={**os.environ, "PYTHONUNBUFFERED": "1"},
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                                  errors="replace")
        assert _child.stdout
        for line in _child.stdout:
            log.write(line)
            sys.stdout.write(line)
            sys.stdout.flush()
            if parse(line) is None and line.strip():
                tail = (tail + [line.rstrip()])[-30:]
        rc = _child.wait()
        _child = None
    return rc, tail


def error_message(rc: int, tail: list[str]) -> str:
    """Best one-line explanation: the exception line of a traceback, else the last log line."""
    for line in reversed(tail):
        if line and not line.startswith(" ") and ("Error" in line or "Exception" in line):
            return line[:400]
    if rc < 0:
        return f"stage killed by signal {-rc}"
    return (tail[-1][:400] if tail else f"stage exited with {rc}")


def _stop(signum, _frame):
    """Forward SIGTERM/SIGINT to the running stage so the GPU is freed, then exit."""
    if _child and _child.poll() is None:
        _child.terminate()
        try:
            _child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            _child.kill()
    sys.exit(128 + signum)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    sys.exit(main())
