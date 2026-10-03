import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from common.progress import parse

ROOT = Path(__file__).resolve().parents[1]


def run(job, *extra):
    env = {**os.environ, "LOBBOT_DRY_RUN": "1"}
    p = subprocess.run([sys.executable, "pipeline.py", "--job", str(job), *extra],
                       cwd=ROOT, env=env, capture_output=True, text=True)
    events = [e for e in map(parse, p.stdout.splitlines()) if e]
    return p, events


def test_dry_run_end_to_end(tmp_path):
    job = tmp_path / "job"
    job.mkdir()
    shutil.copy(ROOT / "examples/support-tickets.taskspec.json", job / "taskspec.json")

    p, events = run(job)
    assert p.returncode == 0, p.stdout + p.stderr
    done = [e["stage"] for e in events if e["status"] == "done"]
    assert done == ["data", "reap", "heal", "quantize", "eval", "package", "pipeline"]

    report = json.loads((job / "out/eval.json").read_text())
    assert report["winner"] in {c["name"] for c in report["candidates"]}
    assert report["bit_widths"]
    assert (job / "out/model.gguf").exists()
    assert (job / "out/Modelfile").exists()
    assert events[-1]["model"] == str((job / "out/model.gguf").resolve())

    # Second run is fully cached; --from reruns the tail.
    _, events = run(job)
    assert {e["status"] for e in events if e["stage"] != "pipeline"} == {"skipped"}
    _, events = run(job, "--from", "quantize")
    assert [e["stage"] for e in events if e["status"] == "skipped"] == ["data", "reap", "heal"]


def test_only_does_not_claim_a_model(tmp_path):
    job = tmp_path / "job"
    job.mkdir()
    shutil.copy(ROOT / "examples/support-tickets.taskspec.json", job / "taskspec.json")

    p, events = run(job, "--only", "data")
    assert p.returncode == 0, p.stdout + p.stderr
    final = events[-1]
    assert final["stage"] == "pipeline" and final["status"] == "done"
    assert final["model"] is None
    assert "model.gguf" not in final["msg"]


def test_missing_taskspec_fails(tmp_path):
    p, events = run(tmp_path)
    assert p.returncode == 2
    assert events[0]["status"] == "error"


def test_config_accepts_time_caps(tmp_path):
    from stages._util import Job

    (tmp_path / "config.json").write_text(json.dumps({"heal_max_minutes": 20, "student_max_minutes": 15}))
    cfg = Job(tmp_path).config
    assert (cfg.heal_max_minutes, cfg.student_max_minutes) == (20, 15)
