import json
import os
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytest.importorskip("fastapi")


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("LOBBOT_DRY_RUN", "1")
    import agent.server as server
    monkeypatch.setattr(server, "JOBS", tmp_path)
    monkeypatch.setattr(server, "TOKEN", "t0k")
    from fastapi.testclient import TestClient
    return TestClient(server.app)


H = {"Authorization": "Bearer t0k"}
SPEC = json.loads((ROOT / "examples/support-tickets.taskspec.json").read_text())


def wait_done(client, job_id, timeout=60):
    end = time.time() + timeout
    while time.time() < end:
        s = client.get(f"/jobs/{job_id}", headers=H).json()
        if s["state"] in ("done", "error"):
            return s
        time.sleep(0.3)
    raise AssertionError("job did not finish")


def test_auth_required(client):
    assert client.get("/health").json()["ok"]
    assert client.get("/jobs").status_code == 401
    assert client.get("/jobs", headers={"Authorization": "Bearer nope"}).status_code == 401


def test_invalid_taskspec(client):
    r = client.post("/jobs", headers=H, json={"task_name": "x"})
    assert r.status_code == 422


def test_full_dry_run_job(client):
    job_id = client.post("/jobs", headers=H, json=SPEC).json()["job_id"]
    s = wait_done(client, job_id)
    assert s["state"] == "done", s
    assert all(v["status"] == "done" for v in s["stages"].values())

    with client.stream("GET", f"/jobs/{job_id}/events", headers=H) as r:
        events = [json.loads(l[6:]) for l in r.iter_lines() if l.startswith("data: ")]
    assert events[-1] == {**events[-1], "stage": "pipeline", "status": "done"}

    ev = client.get(f"/jobs/{job_id}/eval", headers=H).json()
    assert ev["winner"] in {c["name"] for c in ev["candidates"]}

    full = client.get(f"/jobs/{job_id}/model", headers=H)
    assert full.status_code == 200 and full.content
    part = client.get(f"/jobs/{job_id}/model", headers={**H, "Range": "bytes=2-5"})
    assert part.status_code == 206 and part.content == full.content[2:6]
    assert "FROM ./model.gguf" in client.get(f"/jobs/{job_id}/modelfile", headers=H).text

    assert client.get("/jobs", headers=H).json()[0]["job_id"] == job_id

    # Resume from a stage starts a new run that reruns the tail only.
    client.post(f"/jobs/{job_id}/resume", headers=H, json={"from": "eval"})
    s = wait_done(client, job_id)
    assert s["state"] == "done"
    assert [k for k, v in s["stages"].items() if v["status"] == "skipped"] == ["data", "reap", "heal", "quantize"]
    with client.stream("GET", f"/jobs/{job_id}/events", headers=H) as r:
        events = [json.loads(l[6:]) for l in r.iter_lines() if l.startswith("data: ")]
    assert events[0]["stage"] == "pipeline" and events[0]["status"] == "running"
    assert events[-1]["stage"] == "pipeline" and events[-1]["status"] == "done"


def test_unknown_job(client):
    assert client.get("/jobs/abc123/eval", headers=H).status_code == 404
    assert client.get("/jobs/..%2Fetc/eval", headers=H).status_code == 404


def test_job_run_by_hand_reports_disk_state(client, tmp_path):
    """A job run with pipeline.py directly (no events.jsonl) is not 'queued'."""
    import os
    import subprocess
    import sys

    job = tmp_path / "byhand"
    job.mkdir()
    (job / "taskspec.json").write_text(json.dumps(SPEC))
    env = {**os.environ, "LOBBOT_DRY_RUN": "1"}
    run = lambda *a: subprocess.run([sys.executable, "pipeline.py", "--job", str(job), *a],
                                    cwd=ROOT, env=env, capture_output=True, check=True)

    run("--only", "data")
    s = client.get("/jobs/byhand", headers=H).json()
    assert s["stages"]["data"]["status"] == "done" and s["stages"]["reap"]["status"] == "pending"
    assert s["state"] == "partial" and s["error"] is None

    # A stage that died mid-way (last line "running", no process) is an error.
    with (job / "logs/reap.log").open("a") as f:
        f.write('\n===== now\n{"stage": "reap", "status": "running", "pct": 40, "msg": "", "ts": 0}\n')
    s = client.get("/jobs/byhand", headers=H).json()
    assert s["state"] == "error" and "resume" in s["error"]

    run()
    s = client.get("/jobs/byhand", headers=H).json()
    assert s["state"] == "done", s
    with client.stream("GET", "/jobs/byhand/events", headers=H) as r:  # ends instead of idling forever
        events = [json.loads(l[6:]) for l in r.iter_lines() if l.startswith("data: ")]
    assert events[-1]["stage"] == "pipeline" and events[-1]["status"] == "done"
    assert all(v["status"] == "done" for v in s["stages"].values())


SLOW_PIPELINE = '''import json, sys, time
print(json.dumps({"stage": "data", "status": "running", "pct": 1, "msg": "", "ts": time.time()}), flush=True)
time.sleep(120)
'''


def test_one_job_at_a_time_and_stop(client, tmp_path, monkeypatch):
    import agent.server as server

    fake = tmp_path / "fakeroot"
    fake.mkdir()
    (fake / "pipeline.py").write_text(SLOW_PIPELINE)
    monkeypatch.setattr(server, "ROOT", fake)

    first = client.post("/jobs", headers=H, json=SPEC).json()["job_id"]
    r = client.post("/jobs", headers=H, json=SPEC)
    assert r.status_code == 409 and r.json()["running_job"] == first
    assert sorted(p.name for p in tmp_path.iterdir() if (p / "taskspec.json").exists()) == [first]  # no orphan dir
    assert client.post(f"/jobs/{first}/resume", headers=H, json={}).status_code == 409

    r = client.post(f"/jobs/{first}/stop", headers=H)
    assert r.status_code == 200 and r.json()["state"] == "stopped", r.text
    assert client.get(f"/jobs/{first}", headers=H).json()["state"] == "stopped"
    with client.stream("GET", f"/jobs/{first}/events", headers=H) as s:
        events = [json.loads(l[6:]) for l in s.iter_lines() if l.startswith("data: ")]
    assert events[-1]["stage"] == "pipeline" and events[-1]["status"] == "stopped"
    assert client.post(f"/jobs/{first}/stop", headers=H).status_code == 409  # nothing left to stop

    second = client.post("/jobs", headers=H, json=SPEC)
    assert second.status_code == 200
    client.post(f"/jobs/{second.json()['job_id']}/stop", headers=H)


def test_stop_a_job_run_by_hand(client, tmp_path):
    import subprocess
    import sys

    job = tmp_path / "byhand2"
    job.mkdir()
    (job / "taskspec.json").write_text(json.dumps(SPEC))
    fake = tmp_path / "pipeline.py"
    fake.write_text(SLOW_PIPELINE)
    proc = subprocess.Popen([sys.executable, str(fake), "--job", str(job)])
    try:
        time.sleep(0.5)
        r = client.post("/jobs", headers=H, json=SPEC)
        assert r.status_code == 409 and r.json()["running_job"] == "byhand2"
        assert client.post("/jobs/byhand2/stop", headers=H).status_code == 200
        assert proc.wait(timeout=10) != 0
        assert client.get("/jobs/byhand2", headers=H).json()["state"] == "stopped"
    finally:
        proc.kill()
