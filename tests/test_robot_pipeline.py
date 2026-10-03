import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from common.progress import parse
from common.robotspec import RobotSpec

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = json.loads((ROOT / "examples/farm-uppsala.robotspec.json").read_text())
# Tiny settings: a 3-episode eval on small images keeps the whole run well under a minute.
TINY = {**EXAMPLE, "site": {**EXAMPLE["site"], "source": "synthetic", "size_m": 150},
        "n_demos": 2, "eval_episodes": 3, "max_steps": 300}
TINY_CONFIG = {"img_size": 48, "row_length_m": 12, "clip_width": 128, "clip_height": 72, "video_seconds": 12,
               "candidates": ["teacher", "mixed"]}


def test_robotspec_roundtrip_and_validation(tmp_path):
    spec = RobotSpec.from_dict(EXAMPLE)
    assert spec.site.lat == pytest.approx(59.81) and spec.n_demos == 50 and spec.target.min_hz == 10
    spec.save(tmp_path / "r.json")
    assert RobotSpec.load(tmp_path / "r.json") == spec
    assert RobotSpec.from_dict({"task_name": "t", "description": "d", "site_id": "abc1"}).site is None
    for bad in ({"site": None}, {"site_id": "s1"}, {"site": {**EXAMPLE["site"], "source": "bing"}},
                {"site": {**EXAMPLE["site"], "size_m": 20}}, {"eval_episodes": 0}, {"version": 2}):
        with pytest.raises(ValueError):
            RobotSpec.from_dict({**EXAMPLE, **bad})


def test_degraded_policy_is_paired_and_scales_with_compression():
    from robot.simeval import DegradedPolicy, severity

    class Straight:
        def reset(self):
            pass

        def act(self, obs):
            return np.array([0.0, 2.0], np.float32)

    teacher = {"name": "teacher", "denoise_steps": 4, "footprint": {"size_gb": 6.0}}
    small = {"name": "nvfp4", "denoise_steps": 2, "footprint": {"size_gb": 2.0}}
    assert severity(teacher, teacher) == 0 and 0.5 < severity(small, teacher) <= 1

    def steer(sev, episode):
        p = DegradedPolicy(Straight(), sev, "x", seed=7)
        p.episode = episode
        p.reset()
        return np.array([p.act({})[0] for _ in range(50)])

    assert np.array_equal(steer(0.5, 3), steer(0.5, 3))  # same seed, same noise
    assert not np.array_equal(steer(0.5, 3), steer(0.5, 4))
    assert np.abs(steer(0.9, 3)).mean() > np.abs(steer(0.0, 3)).mean()
    lagged = steer(1.0, 3)
    assert np.all(lagged[:4] == 0)  # control lag: the first steps keep the wheels straight


def test_report_plot_from_an_eval(tmp_path):
    from robot.report import plot

    ev = {"dry_run": True, "episodes": 3, "target": {"max_size_gb": 4.0},
          "expert": {"success_rate": 1.0, "cte_mean_m": 0.05},
          "candidates": [{"name": "teacher", "size_gb": 6.1, "bits_per_weight": 16, "success_rate": 0.9, "cte_mean_m": 0.2},
                         {"name": "fp8", "size_gb": 3.1, "bits_per_weight": 8, "success_rate": 0.8, "cte_mean_m": 0.3}]}
    p = plot(ev, tmp_path / "g.png", "Ultuna")
    from PIL import Image

    assert Image.open(p).size == (1600, 800)


def run(job, *extra):
    env = {**os.environ, "LOBBOT_DRY_RUN": "1", "MUJOCO_GL": os.environ.get("MUJOCO_GL", "osmesa"),
           "LOBBOT_SITES": str(job.parent / "sites")}
    p = subprocess.run([sys.executable, "robot_pipeline.py", "--job", str(job), *extra],
                       cwd=ROOT, env=env, capture_output=True, text=True)
    return p, [e for e in map(parse, p.stdout.splitlines()) if e]


def test_robot_dry_run_end_to_end(tmp_path):
    job = tmp_path / "job"
    job.mkdir()
    (job / "robotspec.json").write_text(json.dumps(TINY))
    (job / "config.json").write_text(json.dumps(TINY_CONFIG))

    p, events = run(job)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    done = [e["stage"] for e in events if e["status"] == "done"]
    assert done == ["site", "demos", "finetune", "compress", "simeval", "report", "pipeline"]
    assert events[-1]["video"] == str(job / "out/training.mp4")

    ev = json.loads((job / "out/eval.json").read_text())
    assert ev["kind"] == "robot" and ev["dry_run"] is True and ev["episodes"] == 3
    names = [c["name"] for c in ev["candidates"]]
    assert names == ["teacher", "mixed"]
    for c in ev["candidates"]:
        assert {"size_gb", "params", "bits_per_weight", "latency_ms", "success_rate", "cte_mean_m",
                "progress", "video"} <= set(c)
        assert (job / c["video"]).stat().st_size > 0
        assert [e["seed"] for e in c["per_episode"]] == ev["seeds"]  # paired seeds
    assert {"success_rate", "cte_mean_m"} <= set(ev["expert"])
    assert (job / "data/lerobot").is_dir() and (job / "out/clips/expert.mp4").exists()
    assert (job / "out/footprint_vs_performance.png").stat().st_size > 10_000
    assert (job / "out/training.mp4").stat().st_size > 10_000
    assert json.loads((job / "out/report.json").read_text())["dry_run"] is True

    _, events = run(job)  # fully cached; --from reruns the tail
    assert {e["status"] for e in events if e["stage"] != "pipeline"} == {"skipped"}
    _, events = run(job, "--from", "report")
    assert [e["stage"] for e in events if e["status"] == "skipped"] == ["site", "demos", "finetune", "compress", "simeval"]


def test_missing_robotspec_fails(tmp_path):
    p, events = run(tmp_path / "job")
    assert p.returncode == 2 and events[0]["status"] == "error"


# ----------------------------------------------------------------- API


@pytest.fixture()
def client(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    monkeypatch.setenv("LOBBOT_DRY_RUN", "1")
    monkeypatch.setenv("MUJOCO_GL", os.environ.get("MUJOCO_GL", "osmesa"))
    import agent.server as server

    monkeypatch.setattr(server, "JOBS", tmp_path / "jobs")
    monkeypatch.setattr(server, "SITES", tmp_path / "sites")
    monkeypatch.setattr(server, "TOKEN", "t0k")
    from fastapi.testclient import TestClient

    return TestClient(server.app)


H = {"Authorization": "Bearer t0k"}


def test_map_page_needs_no_token(client):
    r = client.get("/map")
    assert r.status_code == 200 and "<html" in r.text.lower()
    assert client.get("/sites").status_code == 401


def test_sites_and_robot_job_api(client):
    pick = {"lat": 59.81, "lon": 17.66, "size_m": 150, "name": "Ultuna", "source": "synthetic"}
    site = client.post("/sites", headers=H, json=pick).json()
    assert site["source"] == "synthetic" and site["id"].isalnum() and site["name"] == "Ultuna"
    assert [s["id"] for s in client.get("/sites", headers=H).json()] == [site["id"]]
    assert client.get(f"/sites/{site['id']}", headers=H).json()["lat"] == pytest.approx(59.81)
    png = client.get(f"/sites/{site['id']}/preview", headers=H)
    assert png.status_code == 200 and png.content[:4] == b"\x89PNG"
    assert client.get("/sites/nope1/preview", headers=H).status_code == 404
    assert client.post("/sites", headers=H, json={"lat": 59.8}).status_code == 422

    assert client.post("/robot-jobs", headers=H, json={"task_name": "x"}).status_code == 422
    assert client.post("/robot-jobs", headers=H, json={**TINY, "site": None, "site_id": "nosuch1"}).status_code == 404
    # Smallest possible run: the end-to-end test above covers the stages themselves.
    body = {**TINY, "site": None, "site_id": site["id"], "n_demos": 1, "eval_episodes": 1,
            "config": {**TINY_CONFIG, "candidates": ["teacher"]}}
    job_id = client.post("/robot-jobs", headers=H, json=body).json()["job_id"]
    assert client.get("/jobs", headers=H).json() == []  # the LLM job list ignores robot jobs
    assert client.get(f"/jobs/{job_id}", headers=H).status_code == 404

    end = time.time() + 120
    while time.time() < end:
        s = client.get(f"/robot-jobs/{job_id}", headers=H).json()
        if s["state"] in ("done", "error"):
            break
        time.sleep(0.5)
    assert s["state"] == "done", s
    assert list(s["stages"]) == ["site", "demos", "finetune", "compress", "simeval", "report"]
    assert client.get("/robot-jobs", headers=H).json()[0]["job_id"] == job_id

    with client.stream("GET", f"/robot-jobs/{job_id}/events", headers=H) as r:
        events = [json.loads(l[6:]) for l in r.iter_lines() if l.startswith("data: ")]
    assert events[-1]["stage"] == "pipeline" and events[-1]["status"] == "done"
    ev = client.get(f"/robot-jobs/{job_id}/eval", headers=H).json()
    assert ev["dry_run"] is True and ev["site"] == site["id"]
    assert client.get(f"/robot-jobs/{job_id}/graph", headers=H).content[:4] == b"\x89PNG"
    full = client.get(f"/robot-jobs/{job_id}/video", headers=H)
    assert full.status_code == 200 and full.headers["content-type"] == "video/mp4"
    part = client.get(f"/robot-jobs/{job_id}/video", headers={**H, "Range": "bytes=4-11"})
    assert part.status_code == 206 and part.content == full.content[4:12]
