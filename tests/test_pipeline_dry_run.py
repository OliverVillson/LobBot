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

    # Second run is fully cached; --from reruns the tail.
    _, events = run(job)
    assert {e["status"] for e in events if e["stage"] != "pipeline"} == {"skipped"}
    _, events = run(job, "--from", "quantize")
    assert [e["stage"] for e in events if e["status"] == "skipped"] == ["data", "reap", "heal"]


def test_missing_taskspec_fails(tmp_path):
    p, events = run(tmp_path)
    assert p.returncode == 2
    assert events[0]["status"] == "error"
