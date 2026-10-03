"""LobBot backend HTTP API: the contract the desktop app talks to.

    LOBBOT_TOKEN=... uvicorn agent.server:app --host 127.0.0.1 --port 8700

Binds to localhost; reach it from a laptop with
    ssh -L 8700:127.0.0.1:8700 <vm>
Each job is a directory under LOBBOT_JOBS running pipeline.py as a
subprocess. Progress lines are appended to <job>/events.jsonl, which is the
source of truth: jobs keep running if the client disconnects, and state is
rebuilt from disk if the server restarts.
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse

from common.progress import parse
from common.taskspec import TaskSpec

ROOT = Path(__file__).resolve().parents[1]
JOBS = Path(os.environ.get("LOBBOT_JOBS", "/mnt/nvme/jobs"))
TOKEN = os.environ.get("LOBBOT_TOKEN", "")
DRY_RUN = os.environ.get("LOBBOT_DRY_RUN") == "1"
STAGES = ["data", "reap", "heal", "quantize", "eval", "package"]

app = FastAPI(title="LobBot")
_running: dict[str, subprocess.Popen] = {}
_lock = threading.Lock()


def auth(authorization: str = Header(default="")) -> None:
    if not TOKEN:
        raise HTTPException(500, "LOBBOT_TOKEN is not set on the server")
    if not secrets.compare_digest(authorization, f"Bearer {TOKEN}"):
        raise HTTPException(401, "bad token")


def job_dir(job_id: str) -> Path:
    d = JOBS / job_id
    if not job_id.isalnum() or not (d / "taskspec.json").exists():
        raise HTTPException(404, "no such job")
    return d


def read_events(d: Path) -> list[dict]:
    """Events of the latest run only (a resume starts a new run)."""
    f = d / "events.jsonl"
    if not f.exists():
        return []
    events = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    starts = [i for i, e in enumerate(events) if e["stage"] == "pipeline" and e["status"] == "running"]
    return events[starts[-1]:] if starts else events


def _mtime(p: Path) -> float:
    return p.stat().st_mtime if p.exists() else 0.0


def disk_state(d: Path) -> tuple[str, dict, str | None]:
    """State of a job run with pipeline.py directly (no events.jsonl from this
    server): .done markers, plus the last progress line of each stage's log.
    Besides the usual states this can be "partial": some stages ran cleanly
    (--only/--from by hand) and there is no packaged model."""
    stages = {s: {"status": "pending", "pct": 0, "msg": ""} for s in STAGES}
    error = None
    for s in STAGES:
        if (d / ".done" / s).exists():
            stages[s] = {"status": "done", "pct": 100, "msg": ""}
            continue
        log = d / "logs" / f"{s}.log"
        if not log.exists():
            continue
        last = None
        for line in log.read_text(errors="replace").split("\n===== ")[-1].splitlines():
            e = parse(line)
            if e and e.get("stage") == s:
                last = e
        if last:
            stages[s] = {"status": last["status"], "pct": last.get("pct") or 0, "msg": last.get("msg", "")}
            if last["status"] == "error":
                error = last.get("msg") or f"{s} failed"
    statuses = [v["status"] for v in stages.values()]
    if all(st == "done" for st in statuses) and (d / "out" / "model.gguf").exists():
        return "done", stages, None
    if error:
        return "error", stages, error
    if not any(st != "pending" for st in statuses):
        return "queued", stages, None
    if _pipeline_alive(d):
        return "running", stages, None
    if all(st in ("done", "pending") for st in statuses):
        # e.g. `pipeline.py --only reap`: what ran finished cleanly, the rest never started
        return "partial", stages, None
    return "error", stages, "pipeline stopped before the end; resume to continue"


def _pipeline_alive(d: Path) -> bool:
    """Is a pipeline.py started by hand still running on this job dir?"""
    try:
        r = subprocess.run(["pgrep", "-f", f"pipeline.py --job [^ ]*{d.name}( |$)"], capture_output=True)
        return r.returncode == 0
    except FileNotFoundError:  # no pgrep: a log written recently means it is still going
        newest = max((_mtime(p) for p in (d / "logs").glob("*.log")), default=0.0)
        return time.time() - newest < 300


def job_state(job_id: str, d: Path) -> dict:
    stages = {s: {"status": "pending", "pct": 0, "msg": ""} for s in STAGES}
    state, error = "queued", None
    newest_log = max((_mtime(p) for p in (d / "logs").glob("*.log")), default=0.0)
    events = d / "events.jsonl"
    if job_id not in _running and (not events.exists() or newest_log > _mtime(events)):
        # Last run was pipeline.py started by hand (CLI over SSH), not by this server.
        state, stages, error = disk_state(d)
        return _state_dict(job_id, d, state, stages, error)
    for e in read_events(d):
        if e["stage"] == "pipeline":
            if e["status"] == "running":
                state = "running"
            else:
                state = "done" if e["status"] == "done" else "error"
                error = e.get("msg") if e["status"] == "error" else None
            continue
        if e["stage"] in stages:
            stages[e["stage"]] = {"status": e["status"], "pct": e.get("pct", 0), "msg": e.get("msg", "")}
            state = "running"
            if e["status"] == "error":
                error = e.get("msg") or f"{e['stage']} failed"
    if state == "running" and job_id not in _running:
        # The pipeline process is gone without a final event (server restart or crash).
        state, error = "error", error or "pipeline stopped unexpectedly; resume to continue"
    return _state_dict(job_id, d, state, stages, error)


def _state_dict(job_id: str, d: Path, state: str, stages: dict, error: str | None) -> dict:
    spec = json.loads((d / "taskspec.json").read_text())
    return {
        "job_id": job_id,
        "task_name": spec.get("task_name"),
        "state": state,
        "created": (d / "taskspec.json").stat().st_mtime,
        "stages": stages,
        "error": error,
    }


def launch(job_id: str, d: Path, start: str | None = None) -> None:
    with _lock:
        proc = _running.get(job_id)
        if proc and proc.poll() is None:
            raise HTTPException(409, "job is already running")
        cmd = [sys.executable, str(ROOT / "pipeline.py"), "--job", str(d)]
        if start:
            cmd += ["--from", start]
        # Marks the start of a run: state and the event stream begin here.
        with (d / "events.jsonl").open("a") as ev:
            ev.write(json.dumps({"stage": "pipeline", "status": "running", "msg": "started", "ts": time.time()}) + "\n")
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        _running[job_id] = proc

    def pump() -> None:
        assert proc.stdout
        with (d / "events.jsonl").open("a") as ev, (d / "pipeline.log").open("a") as log:
            for line in proc.stdout:
                log.write(line)
                log.flush()
                event = parse(line)
                if event:
                    ev.write(json.dumps(event) + "\n")
                    ev.flush()
            code = proc.wait()
            # Make sure every run ends with a pipeline event, even on a crash.
            last = read_events(d)[-1:] or [{}]
            if last[0].get("stage") != "pipeline":
                ev.write(json.dumps({"stage": "pipeline", "status": "error", "msg": f"exited with {code}", "ts": time.time()}) + "\n")
        with _lock:
            _running.pop(job_id, None)

    threading.Thread(target=pump, daemon=True).start()


@app.get("/health")
def health() -> dict:
    gpu = None
    if shutil.which("nvidia-smi"):
        try:
            gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                 capture_output=True, text=True, timeout=5).stdout.strip().splitlines()[0]
        except (subprocess.SubprocessError, IndexError):
            pass
    return {"ok": True, "dry_run": DRY_RUN, "gpu": gpu}


@app.post("/jobs", dependencies=[Depends(auth)])
def create_job(body: dict) -> dict:
    try:
        spec = TaskSpec.from_dict(body)
    except (KeyError, TypeError, ValueError) as e:
        raise HTTPException(422, f"invalid TaskSpec: {e}")
    job_id = secrets.token_hex(5)
    d = JOBS / job_id
    d.mkdir(parents=True)
    spec.save(d / "taskspec.json")
    launch(job_id, d)
    return {"job_id": job_id}


@app.get("/jobs", dependencies=[Depends(auth)])
def list_jobs() -> list[dict]:
    if not JOBS.exists():
        return []
    rows = []
    for d in JOBS.iterdir():
        if (d / "taskspec.json").exists():
            s = job_state(d.name, d)
            rows.append({k: s[k] for k in ("job_id", "task_name", "state", "created")})
    return sorted(rows, key=lambda r: r["created"], reverse=True)


@app.get("/jobs/{job_id}", dependencies=[Depends(auth)])
def get_job(job_id: str) -> dict:
    return job_state(job_id, job_dir(job_id))


@app.post("/jobs/{job_id}/resume", dependencies=[Depends(auth)])
def resume_job(job_id: str, body: dict | None = None) -> dict:
    d = job_dir(job_id)
    start = (body or {}).get("from")
    if start is not None and start not in STAGES:
        raise HTTPException(422, f"from must be one of {STAGES}")
    launch(job_id, d, start)
    return {"job_id": job_id}


@app.get("/jobs/{job_id}/events", dependencies=[Depends(auth)])
async def job_events(job_id: str) -> StreamingResponse:
    d = job_dir(job_id)

    async def stream():
        sent = 0
        idle = 0.0
        while True:
            events = read_events(d)
            for e in events[sent:]:
                yield f"data: {json.dumps(e)}\n\n"
                if e["stage"] == "pipeline" and e["status"] != "running":
                    return
            sent = len(events)
            await asyncio.sleep(0.5)
            idle += 0.5
            if idle >= 15:
                idle = 0.0
                yield ": keepalive\n\n"
            if job_id not in _running and sent == len(read_events(d)):
                # Nothing running and nothing new: report the stored state and stop.
                state = job_state(job_id, d)
                if state["state"] in ("error", "queued"):
                    yield f"data: {json.dumps({'stage': 'pipeline', 'status': 'error', 'msg': state['error'] or 'not running', 'ts': time.time()})}\n\n"
                    return

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/jobs/{job_id}/eval", dependencies=[Depends(auth)])
def job_eval(job_id: str) -> dict:
    f = job_dir(job_id) / "out" / "eval.json"
    if not f.exists():
        raise HTTPException(404, "eval not ready")
    return json.loads(f.read_text())


@app.get("/jobs/{job_id}/model", dependencies=[Depends(auth)])
def job_model(job_id: str) -> FileResponse:
    d = job_dir(job_id)
    f = d / "out" / "model.gguf"
    if not f.exists() or not (d / ".done" / "package").exists():
        raise HTTPException(404, "model not ready")
    task = json.loads((d / "taskspec.json").read_text()).get("task_name", job_id)
    return FileResponse(f, media_type="application/octet-stream", filename=f"lobbot-{task}.gguf")


@app.get("/jobs/{job_id}/modelfile", dependencies=[Depends(auth)])
def job_modelfile(job_id: str) -> PlainTextResponse:
    f = job_dir(job_id) / "out" / "Modelfile"
    if not f.exists():
        raise HTTPException(404, "model not ready")
    return PlainTextResponse(f.read_text())
