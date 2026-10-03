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
