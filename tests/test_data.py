import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from stages.data import check_answer, parse_array

ROOT = Path(__file__).resolve().parents[1]


def run_data(job):
    env = {**os.environ, "LOBBOT_DRY_RUN": "1"}
    return subprocess.run([sys.executable, "pipeline.py", "--job", str(job), "--only", "data"],
                          cwd=ROOT, env=env, capture_output=True, text=True)


@pytest.fixture
def job(tmp_path):
    j = tmp_path / "job"
    j.mkdir()
    shutil.copy(ROOT / "examples/support-tickets.taskspec.json", j / "taskspec.json")
    (j / "config.json").write_text(json.dumps({"n_generate": 200, "n_heldout": 20}))
    return j


def read(p):
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def test_data_outputs(job):
    p = run_data(job)
    assert p.returncode == 0, p.stdout + p.stderr
    train, held = read(job / "data/train.jsonl"), read(job / "data/heldout.jsonl")
    spec = json.loads((job / "taskspec.json").read_text())
    assert len(held) == 20
    assert 200 <= len(train) <= 200 + 2 * len(spec["seed_examples"])
    for r in train:
        roles = [m["role"] for m in r["messages"]]
        assert roles == ["system", "user", "assistant"]
        json.loads(r["messages"][2]["content"])  # broken teacher answers were dropped
    for r in held:
        assert set(r) == {"input", "reference"}
        json.loads(r["reference"])
    # held-out inputs never appear in train
    assert not {r["input"] for r in held} & {r["messages"][1]["content"] for r in train}
    stats = json.loads((job / "data/stats.json").read_text())
    assert stats["dropped"] and stats["train"] == len(train)
    assert (job / "data/calib.txt").read_text().strip()


def test_data_resumes_from_cached_inputs(job):
    assert run_data(job).returncode == 0
    n_inputs = len(read(job / "work/data_inputs.jsonl"))
    (job / ".done/data").unlink()
    p = run_data(job)
    assert p.returncode == 0
    assert "round 1" not in p.stdout  # inputs came from the resume cache
    assert len(read(job / "work/data_inputs.jsonl")) == n_inputs


@pytest.mark.parametrize("text,finished,ok", [
    ('{"a": 1}', True, True),
    ('```json\n{"a": 1}\n```', True, True),
    ('<think>hmm</think>\n{"a": 1}', True, True),
    ('Sure! Here it is: {"a": 1}', True, True),
    ('{"a": 1', True, False),
    ('{"a": 1}', False, False),
    ("", True, False),
])
def test_check_answer(text, finished, ok):
    ans, why = check_answer(text, finished, want_json=True)
    assert (ans is not None) == ok, why
    if ok:
        json.loads(ans)


def test_parse_array():
    assert parse_array('Here:\n```json\n["first input here", "second input here"]\n```') == [
        "first input here", "second input here"]
    assert parse_array('[{"input": "an input as an object"}, "a plain string input"]') == [
        "an input as an object", "a plain string input"]
    assert parse_array("no list at all") == []
    assert parse_array("[broken") == []


def test_failing_stage_reports_cause_and_logs(job):
    (job / "taskspec.json").write_text((job / "taskspec.json").read_text())
    (job / "config.json").write_text(json.dumps({"n_generate": 200, "n_heldout": 5000}))
    p = run_data(job)
    assert p.returncode == 1
    from common.progress import parse
    err = [e for e in map(parse, p.stdout.splitlines()) if e and e["status"] == "error"][0]
    assert "RuntimeError" in err["msg"] and "usable" in err["msg"]
    assert "Traceback" in (job / "logs/data.log").read_text()


def test_gemini_written_heldout(job, monkeypatch):
    """With testgen on, the held-out inputs are Gemini's, answered by the teacher, and never trained on."""
    import random

    from stages import data, testgen
    from stages._util import Job

    j = Job(job)
    fake = [f"gemini test input number {i} with enough words" for i in range(25)]
    monkeypatch.setenv("GEMINI_API_KEY", "g-key")
    monkeypatch.setattr(data, "DRY_RUN", False)
    monkeypatch.setattr(data, "Teacher", lambda path: data.DryRunTeacher(j.spec, random.Random(0)))
    monkeypatch.setattr(testgen, "held_out_inputs", lambda spec, cfg, n, seen, *a, **k: fake[:n])
    data.run_stage(j)
    held, train = read(job / "data/heldout.jsonl"), read(job / "data/train.jsonl")
    assert {r["input"] for r in held} <= set(fake) and len(held) >= 15
    assert not set(fake) & {r["messages"][1]["content"] for r in train}
    stats = json.loads((job / "data/stats.json").read_text())
    assert stats["heldout_source"] == "gemini-3.8-flash" and "heldout_dropped" in stats

    # a rerun reuses the cached Gemini inputs instead of asking again
    (job / ".done/data").unlink()
    monkeypatch.setattr(testgen, "held_out_inputs", lambda *a, **k: pytest.fail("Gemini called again"))
    data.run_stage(j)
    assert [r["input"] for r in read(job / "work/data_testgen.jsonl")] == fake[:20]
    assert {r["input"] for r in read(job / "data/heldout.jsonl")} <= set(fake[:20])


def test_answer_max_tokens_env(monkeypatch):
    import importlib

    from stages import data

    monkeypatch.setenv("LOBBOT_DATA_ANSWER_MAX_TOKENS", "4096")
    assert importlib.reload(data).ANSWER_MAX_TOKENS == 4096
    monkeypatch.delenv("LOBBOT_DATA_ANSWER_MAX_TOKENS")
    assert importlib.reload(data).ANSWER_MAX_TOKENS == 1536
