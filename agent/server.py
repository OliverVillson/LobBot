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
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse

from common.progress import parse
from common.taskspec import TaskSpec

ROOT = Path(__file__).resolve().parents[1]
JOBS = Path(os.environ.get("LOBBOT_JOBS", "/mnt/nvme/jobs"))
TOKEN = os.environ.get("LOBBOT_TOKEN", "")
DRY_RUN = os.environ.get("LOBBOT_DRY_RUN") == "1"
STAGES = ["data", "reap", "heal", "quantize", "eval", "package"]
# Robot jobs (robot_pipeline.py, docs/robot-mvp.md) share JOBS and the one-job lock;
# robotspec.json marks them, so the /jobs routes (taskspec.json) never see them.
SITES = Path(os.environ.get("LOBBOT_SITES", "/mnt/nvme/sites"))
ROBOT_STAGES = ["site", "demos", "finetune", "compress", "simeval", "report"]
# kind -> (stages, spec file, final output, pipeline script)
KINDS = {"llm": (STAGES, "taskspec.json", "out/model.gguf", "pipeline.py"),
         "robot": (ROBOT_STAGES, "robotspec.json", "out/training.mp4", "robot_pipeline.py")}

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


def disk_state(d: Path, kind: str = "llm") -> tuple[str, dict, str | None]:
    """State of a job run with pipeline.py directly (no events.jsonl from this
    server): .done markers, plus the last progress line of each stage's log.
    Besides the usual states this can be "partial": some stages ran cleanly
    (--only/--from by hand) and there is no packaged model."""
    names, _, final, _ = KINDS[kind]
    stages = {s: {"status": "pending", "pct": 0, "msg": ""} for s in names}
    error = None
    for s in names:
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
    newest_log = max((_mtime(p) for p in (d / "logs").glob("*.log")), default=0.0)
    if (d / "stopped").exists() and _mtime(d / "stopped") > newest_log - 60 and not _pipeline_alive(d):
        # POST /stop, and nothing ran since (a stage may still log its exit just after)
        return "stopped", stages, None
    if all(st == "done" for st in statuses) and (d / final).exists():
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


def job_state(job_id: str, d: Path, kind: str = "llm") -> dict:
    stages = {s: {"status": "pending", "pct": 0, "msg": ""} for s in KINDS[kind][0]}
    state, error = "queued", None
    newest_log = max((_mtime(p) for p in (d / "logs").glob("*.log")), default=0.0)
    events = d / "events.jsonl"
    if job_id not in _running and (not events.exists() or newest_log > _mtime(events)):
        # Last run was pipeline.py started by hand (CLI over SSH), not by this server.
        state, stages, error = disk_state(d, kind)
        return _state_dict(job_id, d, state, stages, error, kind)
    for e in read_events(d):
        if e["stage"] == "pipeline":
            if e["status"] == "running":
                state = "running"
            elif e["status"] in ("done", "stopped"):
                state, error = e["status"], None
            else:
                state, error = "error", e.get("msg")
            continue
        if e["stage"] in stages:
            stages[e["stage"]] = {"status": e["status"], "pct": e.get("pct", 0), "msg": e.get("msg", "")}
            state = "running"
            if e["status"] == "error":
                error = e.get("msg") or f"{e['stage']} failed"
    if state == "running" and job_id not in _running:
        # The pipeline process is gone without a final event (server restart or crash).
        state, error = "error", error or "pipeline stopped unexpectedly; resume to continue"
    return _state_dict(job_id, d, state, stages, error, kind)


def _state_dict(job_id: str, d: Path, state: str, stages: dict, error: str | None, kind: str = "llm") -> dict:
    spec_file = d / KINDS[kind][1]
    spec = json.loads(spec_file.read_text())
    return {
        "job_id": job_id,
        "task_name": spec.get("task_name"),
        "state": state,
        "created": spec_file.stat().st_mtime,
        "stages": stages,
        "error": error,
    }


class Busy(Exception):
    """Another pipeline holds the GPU: one job runs at a time."""

    def __init__(self, job_id: str):
        self.job_id = job_id


@app.exception_handler(Busy)
def _busy(_request, e: Busy) -> JSONResponse:
    # running_job lets a client offer "watch it" or "stop it" without parsing the message.
    return JSONResponse(status_code=409, content={
        "detail": f"job {e.job_id} is running; wait for it or stop it (POST /jobs/{e.job_id}/stop)",
        "running_job": e.job_id})


def _hand_run_pipelines() -> dict[str, list[int]]:
    """pipeline.py processes not started by this server, by job id (dir name)."""
    try:
        out = subprocess.run(["pgrep", "-af", r"pipeline\.py --job "], capture_output=True, text=True).stdout
    except FileNotFoundError:
        return {}
    ours = {p.pid for p in _running.values()}
    found: dict[str, list[int]] = {}
    for line in out.splitlines():
        pid, _, cmd = line.partition(" ")
        args = cmd.split()
        if int(pid) in ours or "--job" not in args or args.index("--job") + 1 >= len(args):
            continue
        if not any(a.endswith("pipeline.py") for a in args):
            continue  # e.g. a shell whose command line mentions pipeline.py
        found.setdefault(Path(args[args.index("--job") + 1]).name, []).append(int(pid))
    return found


def running_job() -> str | None:
    """The job holding the GPU, started by this server or by hand, if any."""
    for job_id, proc in _running.items():
        if proc.poll() is None:
            return job_id
    return next(iter(_hand_run_pipelines()), None)


def launch(job_id: str, d: Path, start: str | None = None, kind: str = "llm") -> None:
    with _lock:
        busy = running_job()
        if busy:
            raise Busy(busy)
        (d / "stopped").unlink(missing_ok=True)
        cmd = [sys.executable, str(ROOT / KINDS[kind][3]), "--job", str(d)]
        if start:
            cmd += ["--from", start]
        # Marks the start of a run: state and the event stream begin here.
        with (d / "events.jsonl").open("a") as ev:
            ev.write(json.dumps({"stage": "pipeline", "status": "running", "msg": "started", "ts": time.time()}) + "\n")
        env = {**os.environ, "LOBBOT_SITES": str(SITES)} if kind == "robot" else None
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                                env=env)
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
            if (d / "stopped").exists():
                ev.write(json.dumps({"stage": "pipeline", "status": "stopped", "msg": "stopped; resume to continue",
                                     "ts": time.time()}) + "\n")
            elif last[0].get("stage") != "pipeline":
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
    busy = running_job()
    if busy:
        raise Busy(busy)  # before creating a job dir that would never run
    job_id = secrets.token_hex(5)
    d = JOBS / job_id
    d.mkdir(parents=True)
    spec.save(d / "taskspec.json")
    try:
        launch(job_id, d)
    except Busy:
        shutil.rmtree(d, ignore_errors=True)  # lost a race with another start
        raise
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


@app.post("/jobs/{job_id}/stop", dependencies=[Depends(auth)])
def stop_job(job_id: str) -> dict:
    """SIGTERM the job's pipeline.py, which forwards it to the running stage and
    frees the GPU. Finished stages stay cached; resume continues from there."""
    return _stop_job(job_id, job_dir(job_id))


def _stop_job(job_id: str, d: Path, kind: str = "llm") -> dict:
    proc = _running.get(job_id)
    pids = [proc.pid] if proc and proc.poll() is None else _hand_run_pipelines().get(job_id, [])
    if not pids:
        raise HTTPException(409, "job is not running")
    (d / "stopped").write_text(str(time.time()))
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    # pipeline.py gives its stage 30s to exit before killing it.
    end = time.time() + 40
    while time.time() < end and any(_alive(p) for p in pids):
        time.sleep(0.2)
    for pid in pids:
        if _alive(pid):
            os.kill(pid, signal.SIGKILL)
    if proc:
        for _ in range(50):  # let pump write the final "stopped" event
            if job_id not in _running:
                break
            time.sleep(0.1)
    return {"job_id": job_id, "state": job_state(job_id, d, kind)["state"]}


def _alive(pid: int) -> bool:
    proc = next((p for p in _running.values() if p.pid == pid), None)
    if proc:
        return proc.poll() is None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    try:  # an exited child its parent has not reaped yet is a zombie, not running
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except (OSError, IndexError):
        return True


@app.get("/jobs/{job_id}/events", dependencies=[Depends(auth)])
async def job_events(job_id: str) -> StreamingResponse:
    return _event_stream(job_id, job_dir(job_id))


def _event_stream(job_id: str, d: Path, kind: str = "llm") -> StreamingResponse:
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
                state = job_state(job_id, d, kind)
                if state["state"] in ("error", "queued"):
                    yield f"data: {json.dumps({'stage': 'pipeline', 'status': 'error', 'msg': state['error'] or 'not running', 'ts': time.time()})}\n\n"
                    return
                if state["state"] in ("done", "partial", "stopped"):  # a run started by hand, read from the job dir
                    status, msg = {"done": ("done", "done"), "partial": ("done", "partial run, no packaged model"),
                                   "stopped": ("stopped", "stopped; resume to continue")}[state["state"]]
                    yield f"data: {json.dumps({'stage': 'pipeline', 'status': status, 'msg': msg, 'ts': time.time()})}\n\n"
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


# ----------------------------------------------------------------- robot MVP (docs/robot-mvp.md)


@app.get("/map", include_in_schema=False)
def map_page() -> FileResponse:
    """The map picker. A static page: its API calls carry the bearer token."""
    f = ROOT / "farmsim" / "static" / "map.html"
    if not f.exists():
        raise HTTPException(404, "map page not installed")
    return FileResponse(f, media_type="text/html")


def _site_meta(site) -> dict:
    from farmsim.site import read_meta

    return read_meta(site)


def _load_site(site_id: str):
    from farmsim.site import load_site

    if not site_id.isalnum():
        raise HTTPException(404, "no such site")
    try:
        return load_site(site_id, sites_dir=SITES)
    except (FileNotFoundError, ValueError, TypeError):
        raise HTTPException(404, "no such site")


@app.post("/sites", dependencies=[Depends(auth)])
async def create_site(body: dict) -> dict:
    """Build a site from a map pick: {lat, lon, size_m, name, source}. Fetching tiles
    can take a while, so it runs in a worker thread."""
    from common.robotspec import SitePick
    from farmsim.site import build_site

    try:
        p = SitePick(**body)
        lat, lon, size = float(p.lat), float(p.lon), float(p.size_m)
    except (TypeError, ValueError) as e:
        raise HTTPException(422, f"invalid site: {e}")
    try:
        site = await asyncio.to_thread(build_site, lat, lon, size, name=p.name, source=p.source, sites_dir=SITES)
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception as e:  # every source failed (network, credentials)
        raise HTTPException(502, f"could not build the site: {e}")
    return _site_meta(site)


@app.get("/sites", dependencies=[Depends(auth)])
def list_sites_route() -> list[dict]:
    from farmsim.site import list_sites

    return [_site_meta(s) for s in list_sites(SITES)]


@app.get("/sites/{site_id}", dependencies=[Depends(auth)])
def get_site(site_id: str) -> dict:
    return _site_meta(_load_site(site_id))


@app.get("/sites/{site_id}/preview", dependencies=[Depends(auth)])
def site_preview(site_id: str) -> FileResponse:
    f = _load_site(site_id).preview_path
    if not f.exists():
        raise HTTPException(404, "no preview")
    return FileResponse(f, media_type="image/png")


def robot_job_dir(job_id: str) -> Path:
    d = JOBS / job_id
    if not job_id.isalnum() or not (d / "robotspec.json").exists():
        raise HTTPException(404, "no such robot job")
    return d


ROBOT_CONFIG_KEYS = {"img_size", "row_length_m", "clip_width", "clip_height", "fps", "demo_seed", "eval_seed", "candidates",
                     "video_seconds", "base_model"}


@app.post("/robot-jobs", dependencies=[Depends(auth)])
def create_robot_job(body: dict) -> dict:
    """RobotSpec body (site {lat, lon, ...} or site_id); an optional "config" key holds
    robot/stages.py:RobotConfig overrides."""
    from common.robotspec import RobotSpec

    body = dict(body)
    config = body.pop("config", None) or {}
    if not isinstance(config, dict) or set(config) - ROBOT_CONFIG_KEYS:
        raise HTTPException(422, f"config keys must be among {sorted(ROBOT_CONFIG_KEYS)}")
    try:
        spec = RobotSpec.from_dict(body)
    except (KeyError, TypeError, ValueError) as e:
        raise HTTPException(422, f"invalid RobotSpec: {e}")
    if spec.site_id:
        _load_site(spec.site_id)  # 404 before making a job that cannot run
    busy = running_job()
    if busy:
        raise Busy(busy)
    job_id = secrets.token_hex(5)
    d = JOBS / job_id
    d.mkdir(parents=True)
    spec.save(d / "robotspec.json")
    if config:
        (d / "config.json").write_text(json.dumps(config, indent=2))
    try:
        launch(job_id, d, kind="robot")
    except Busy:
        shutil.rmtree(d, ignore_errors=True)
        raise
    return {"job_id": job_id}


@app.get("/robot-jobs", dependencies=[Depends(auth)])
def list_robot_jobs() -> list[dict]:
    if not JOBS.exists():
        return []
    rows = []
    for d in JOBS.iterdir():
        if (d / "robotspec.json").exists():
            s = job_state(d.name, d, "robot")
            rows.append({k: s[k] for k in ("job_id", "task_name", "state", "created")})
    return sorted(rows, key=lambda r: r["created"], reverse=True)


@app.get("/robot-jobs/{job_id}", dependencies=[Depends(auth)])
def get_robot_job(job_id: str) -> dict:
    return job_state(job_id, robot_job_dir(job_id), "robot")


@app.post("/robot-jobs/{job_id}/resume", dependencies=[Depends(auth)])
def resume_robot_job(job_id: str, body: dict | None = None) -> dict:
    d = robot_job_dir(job_id)
    start = (body or {}).get("from")
    if start is not None and start not in ROBOT_STAGES:
        raise HTTPException(422, f"from must be one of {ROBOT_STAGES}")
    launch(job_id, d, start, kind="robot")
    return {"job_id": job_id}


@app.post("/robot-jobs/{job_id}/stop", dependencies=[Depends(auth)])
def stop_robot_job(job_id: str) -> dict:
    return _stop_job(job_id, robot_job_dir(job_id), "robot")


@app.get("/robot-jobs/{job_id}/events", dependencies=[Depends(auth)])
async def robot_job_events(job_id: str) -> StreamingResponse:
    return _event_stream(job_id, robot_job_dir(job_id), "robot")


@app.get("/robot-jobs/{job_id}/eval", dependencies=[Depends(auth)])
def robot_job_eval(job_id: str) -> dict:
    f = robot_job_dir(job_id) / "out" / "eval.json"
    if not f.exists():
        raise HTTPException(404, "eval not ready")
    return json.loads(f.read_text())


@app.get("/robot-jobs/{job_id}/graph", dependencies=[Depends(auth)])
def robot_job_graph(job_id: str) -> FileResponse:
    f = robot_job_dir(job_id) / "out" / "footprint_vs_performance.png"
    if not f.exists():
        raise HTTPException(404, "graph not ready")
    return FileResponse(f, media_type="image/png")


@app.get("/robot-jobs/{job_id}/video", dependencies=[Depends(auth)])
def robot_job_video(job_id: str) -> FileResponse:
    """out/training.mp4 (Range supported, so a browser can seek)."""
    d = robot_job_dir(job_id)
    f = d / "out" / "training.mp4"
    if not f.exists():
        raise HTTPException(404, "video not ready")
    return FileResponse(f, media_type="video/mp4", filename=f"lobbot-robot-{job_id}.mp4",
                        content_disposition_type="inline")
