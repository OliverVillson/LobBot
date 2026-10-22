"""Run the REAPer pipeline.

    python -m reaper_pipeline --job DIR [--from STAGE] [--only STAGE]
    python -m reaper_pipeline --model ID_OR_DIR [--purpose TEXT] [--prune 0.5] [--job DIR]

Each stage group runs as its own process (so the GPU is fully released
between the vLLM teacher, the REAP pass and eval), its output is relayed to
stdout and teed to DIR/logs/<stage>.log, and finished stages are skipped on a
rerun. The last line is {"stage": "pipeline", "status": "done", "model": "<out dir>"}.
Exit code 0 means the whole pipeline finished. SIGTERM stops the running stage.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from dataclasses import fields

from reaper_pipeline import PROCESSES, STAGES
from reaper_pipeline.events import emit, parse
from reaper_pipeline.job import Config, Job

_child: subprocess.Popen | None = None


def run_process(mod: str, job: Job, env: dict) -> tuple[int, list[str], dict | None]:
    """Run one stage process. Returns exit code, last log lines, last event it reported."""
    global _child
    py = os.environ.get("REAPER_VLLM_PY") if mod == "calibrate" else None
    cmd = [py or sys.executable, "-u", "-m", f"reaper_pipeline.stages.{mod}", "--job", str(job.root)]
    tail: list[str] = []
    last = None
    with open(job.path("logs", f"{mod}.log"), "a") as log:
        log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(cmd)}\n")
        _child = subprocess.Popen(cmd, env={**os.environ, **env, "PYTHONUNBUFFERED": "1"},
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                                  errors="replace")
        assert _child.stdout
        for line in _child.stdout:
            log.write(line)
            log.flush()
            sys.stdout.write(line)
            sys.stdout.flush()
            ev = parse(line)
            if ev is None:
                if line.strip():
                    tail = (tail + [line.rstrip()])[-30:]
            else:
                last = ev
        rc = _child.wait()
        _child = None
    return rc, tail, last


def error_message(rc: int, tail: list[str]) -> str:
    for line in reversed(tail):
        if line and not line.startswith(" ") and ("Error" in line or "Exception" in line):
            return line[:400]
    if rc < 0:
        return f"stage killed by signal {-rc}"
    return tail[-1][:400] if tail else f"stage exited with {rc}"


def run(job: Job, start: str | None = None, only: str | None = None) -> int:
    if start:
        job.clear_from(STAGES, start)
    if only:
        job.clear_from(STAGES, only)
    for mod, stages in PROCESSES:
        if only and only not in stages:
            continue
        if all(job.is_done(s) for s in stages):
            for s in stages:
                emit(s, "skipped", 100, "cached")
            continue
        emit(next(s for s in stages if not job.is_done(s)), "running", 0, "starting")
        env = {k: str(v) for k, v in ((job.read("plan.json") or {}).get("env") or {}).items()}
        rc, tail, last = run_process(mod, job, env)
        failed = next((s for s in stages if not job.is_done(s)), None)
        if rc != 0 or failed:
            stage = failed or (last or {}).get("stage") or stages[-1]
            if not (last and last["status"] == "error"):  # else the stage already said why
                emit(stage, "error", msg=error_message(rc, tail))
            emit("pipeline", "error", msg=f"{stage} failed")
            return 1
    if job.is_done("prune"):
        emit("pipeline", "done", 100, str(job.out_dir), model=str(job.out_dir))
    else:
        emit("pipeline", "done", 100, "partial run; no pruned model yet", model=None)
    return 0


def _stop(signum, _frame):
    """Forward SIGTERM/SIGINT to the running stage so the GPU is freed, then exit."""
    if _child and _child.poll() is None:
        _child.terminate()
        try:
            _child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            _child.kill()
    emit("pipeline", "error", msg="stopped")
    sys.exit(128 + signum)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--job", help="job directory (holds config.json)")
    ap.add_argument("--model", help="create/overwrite the job's config for this model")
    ap.add_argument("--purpose", default=None)
    ap.add_argument("--prune", type=float, default=None)
    ap.add_argument("--calib", default=None)
    ap.add_argument("--calib-samples", type=int, default=None)
    ap.add_argument("--from", dest="start", choices=STAGES, help="rerun from this stage onward")
    ap.add_argument("--only", choices=STAGES, help="run the process holding this stage")
    a = ap.parse_args(argv)
    if not a.job and not a.model:
        ap.error("give --job DIR or --model ID")
    if a.model:
        cfg = Config(model=a.model)
        if a.job and (Job(a.job).path("config.json")).exists():
            cfg = Job(a.job).config
            cfg.model = a.model
        for name in ("purpose", "prune", "calib", "calib_samples"):
            if getattr(a, name) is not None:
                setattr(cfg, name, getattr(a, name))
        job = Job.create(cfg, a.job)
    else:
        job = Job(a.job)
        if not job.path("config.json").exists():
            emit("pipeline", "error", msg=f"missing {job.path('config.json')}")
            return 2
    unknown = set(job.read("config.json") or {}) - {f.name for f in fields(Config)}
    if unknown:
        print(f"config.json: ignoring unknown keys {sorted(unknown)}", flush=True)
    job.path("pid").write_text(str(os.getpid()))
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    return run(job, a.start, a.only)


if __name__ == "__main__":
    sys.exit(main())
