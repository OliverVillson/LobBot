"""reaper_pipeline: purpose calibration, config, and a full run on a tiny MoE (CPU)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from reaper_pipeline import STAGES, calib
from reaper_pipeline.events import parse

ROOT = Path(__file__).resolve().parents[1]


class FakeTeacher:
    """Answers the sub-task request with a list, and each request batch with debugging prompts."""

    def __init__(self):
        self.calls = []

    def chat(self, convs, max_tokens, temperature=0.7):
        out = []
        for c in convs:
            text = c[-1]["content"]
            self.calls.append(text)
            if "sub-tasks" in text:
                out.append('Sure:\n["stack traces", "flaky tests", "memory leaks", "race conditions"]')
            else:
                focus = text.split("Focus: ")[1].split(".\n")[0]
                n = len(self.calls)
                out.append(json.dumps([f"My {focus} case {n}-{i}: why does this code fail? x = foo()" for i in range(6)]))
        return out


def test_purpose_drives_the_calibration_set():
    t = FakeTeacher()
    rows, info = calib.build_prompts(t, "Debugging only", "mix", 40, mix=0.25)
    assert len(rows) == 40
    assert info["preset"] == "code" and info["topics"][:2] == ["stack traces", "flaky tests"]
    teacher_rows = [r for r in rows if r["source"] == "teacher"]
    assert len(teacher_rows) == 30 and len({r["prompt"] for r in teacher_rows}) == 30
    assert any("Debugging only" in c for c in t.calls)


def test_no_purpose_uses_the_preset(monkeypatch):
    rows, info = calib.build_prompts(None, "", "builtin", 10)
    assert len(rows) == 10 and all(r["source"] == "builtin" for r in rows)
    assert calib.preset_for("Debugging only") == "code"
    assert calib.preset_for("grade school maths tutor") == "math"
    assert calib.preset_for("poems about cheese") is None


def test_parse_list_handles_json_and_bullets():
    assert calib.parse_list('<think>[no]</think> here: ["a b", "c"]') == ["a b", "c"]
    assert calib.parse_list("1. first one\n2) second\n- third") == ["first one", "second", "third"]


@pytest.mark.parametrize("arch", ["qwen3_moe", "flint"])
def test_full_run_on_a_tiny_moe(tmp_path, arch):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    sys.path.insert(0, str(Path(__file__).parent))
    from tiny_moe import build
    model = build(arch, tmp_path / f"{arch}-tiny")
    job = tmp_path / "job"
    job.mkdir()
    (job / "config.json").write_text(json.dumps({
        "model": str(model), "purpose": "Debugging only", "prune": 0.5, "calib": "mix", "calib_samples": 8,
        "heldout": 4, "max_seq": 96, "answer_max_tokens": 12, "writer_max_tokens": 24, "teacher_backend": "hf"}))
    env = {**os.environ, "PYTHONPATH": str(ROOT), "REAPER_HW": "builtin", "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
           "REAPER_HOME": str(tmp_path / "home")}
    p = subprocess.run([sys.executable, "-m", "reaper_pipeline", "--job", str(job)], env=env, cwd=tmp_path,
                       capture_output=True, text=True, timeout=900)
    events = [e for e in map(parse, p.stdout.splitlines()) if e]
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    done = [e["stage"] for e in events if e["status"] == "done"]
    assert done == STAGES + ["pipeline"]
    out = Path(events[-1]["model"])
    assert (out / "config.json").exists() and (out / "README.md").exists()
    cfg = json.loads((out / "config.json").read_text())
    assert 4 in (cfg.get("num_experts"), cfg.get("num_local_experts"))
    assert out.name.endswith("-FP8") and cfg["quantization_config"]["quant_method"] == "fp8"
    ev = json.loads((job / "work" / "eval.json").read_text())
    assert 0 <= ev["top1_agreement"] <= 1 and ev["loads"]

    # a rerun skips everything that is done
    p2 = subprocess.run([sys.executable, "-m", "reaper_pipeline", "--job", str(job)], env=env, cwd=tmp_path,
                        capture_output=True, text=True, timeout=300)
    ev2 = [e for e in map(parse, p2.stdout.splitlines()) if e]
    assert p2.returncode == 0 and [e["status"] for e in ev2[:-1]] == ["skipped"] * len(STAGES)


def test_not_a_moe_fails_in_plan(tmp_path):
    m = tmp_path / "dense"
    m.mkdir()
    (m / "config.json").write_text(json.dumps({"model_type": "llama", "num_hidden_layers": 2}))
    job = tmp_path / "job"
    job.mkdir()
    (job / "config.json").write_text(json.dumps({"model": str(m)}))
    p = subprocess.run([sys.executable, "-m", "reaper_pipeline", "--job", str(job)],
                       env={**os.environ, "PYTHONPATH": str(ROOT)}, capture_output=True, text=True, timeout=300)
    events = [e for e in map(parse, p.stdout.splitlines()) if e]
    assert p.returncode == 1
    assert any(e["stage"] == "plan" and e["status"] == "error" and "not a mixture-of-experts" in e["msg"]
               for e in events)
    assert events[-1]["stage"] == "pipeline" and events[-1]["status"] == "error"
