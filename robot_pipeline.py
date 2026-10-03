"""LobBot robot pipeline: map pick -> farm sim -> GR00T fine-tune -> compressed candidates.

    python robot_pipeline.py --job <job_dir> [--from <stage>] [--only <stage>]

Same mechanics as pipeline.py: each stage (robot/stages.py) runs as its own
subprocess, its progress lines are relayed, its output is teed to
<job>/logs/<stage>.log, and stages already completed in this job dir are
skipped. <job_dir>/robotspec.json must exist. See docs/robot-mvp.md.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time

from common.progress import emit, parse
from pipeline import ROOT, error_message
from robot.stages import STAGES, RobotJob

# Isaac-GR00T pins its own torch; setup puts it in a venv of its own.
STAGE_PYTHON = {s: os.environ.get("LOBBOT_GR00T_PY") for s in ("finetune", "compress", "simeval")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--job", required=True)
    ap.add_argument("--from", dest="start", choices=STAGES, help="rerun from this stage onward")
    ap.add_argument("--only", choices=STAGES, help="run a single stage")
    args = ap.parse_args()

    job = RobotJob(args.job)
    if not job.path("robotspec.json").exists():
        emit("pipeline", "error", msg=f"missing {job.path('robotspec.json')}")
        return 2
    job.save_config()
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
    video, graph = job.path("out", "training.mp4"), job.path("out", "footprint_vs_performance.png")
    if job.is_done("report") and video.exists():
        emit("pipeline", "done", 100, str(video), video=str(video), graph=str(graph))
    else:
        emit("pipeline", "done", 100, f"ran {', '.join(todo)}; no report yet", video=None, graph=None)
    return 0


_child: subprocess.Popen | None = None


def run_stage(stage: str, job: RobotJob) -> tuple[int, list[str]]:
    """Run one stage, relaying its output and teeing it to <job>/logs/<stage>.log."""
    global _child
    cmd = [STAGE_PYTHON.get(stage) or sys.executable, "-u", "-m", "robot.stages", stage, "--job", str(job.root)]
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
