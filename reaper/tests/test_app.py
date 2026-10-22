import asyncio
import json
import os
import sys
import textwrap
import time

import pytest

from reaper_tui import backend
from reaper_tui.app import Reaper

FAKE_PIPELINE = textwrap.dedent('''
    """Stand-in for the real reaper_pipeline: emits the progress protocol."""
    import argparse, json, pathlib, sys, time
    STAGES = ["plan", "weights", "calibrate", "score", "prune", "eval", "package"]
    if __name__ == "__main__":
        ap = argparse.ArgumentParser(); ap.add_argument("--job"); job = pathlib.Path(ap.parse_args().job)
        cfg = json.loads((job / "config.json").read_text())
        for st in STAGES:
            for pct in (0, 50):
                print(json.dumps({"stage": st, "status": "running", "pct": pct, "msg": cfg["purpose"]}), flush=True)
                time.sleep(0.05)
            print("some log line", flush=True)
            extra = {"top1_agreement": 0.9, "kl": 0.05, "ppl_teacher": 4.0, "ppl_pruned": 4.4} if st == "eval" else {}
            print(json.dumps({"stage": st, "status": "done", "pct": 100, **extra}), flush=True)
        print(json.dumps({"stage": "pipeline", "status": "done", "pct": 100, "model": str(job / "out")}), flush=True)
''')


async def _type(pilot, text):
    await pilot.click("#prompt")
    for ch in text:
        await pilot.press(ch if ch != " " else "space")
    await pilot.press("enter")
    await pilot.pause(0.3)


def test_demo_cli_launch():
    """`reaper --demo "qwen3-30b-a3b" "Debugging only"` finds the model and runs."""
    async def run():
        app = Reaper(demo=True, model="qwen3-30b-a3b", purpose="Debugging only")
        async with app.run_test(size=(150, 52)) as pilot:
            s = app.state
            for _ in range(200):
                await pilot.pause(0.1)
                if s.result:
                    break
            assert s.model.id == "Qwen/Qwen3-30B-A3B-Instruct-2507"
            assert s.purpose == "Debugging only"
            assert all(s.stages[n]["status"] == "done" for n in s.stage_names)
            assert backend.eval_line(s.stages["eval"]).startswith("top-1 agreement 91%")

    asyncio.run(run())


def test_settings_commands():
    async def run():
        app = Reaper(demo=True)
        async with app.run_test(size=(150, 52)) as pilot:
            for cmd in ["/model gpt-oss 20b", "/prune 25", "/purpose SQL only", "/samples 256"]:
                await _type(pilot, cmd)
            s = app.state
            assert s.model.id == "openai/gpt-oss-20b" and s.ratio == 0.25
            assert s.purpose == "SQL only" and s.samples == 256
            await _type(pilot, "/model someorg/Some-New-MoE")
            assert s.model.id == "someorg/Some-New-MoE"

    asyncio.run(run())


def test_real_backend_run_and_reattach(tmp_path, monkeypatch):
    """Starts a (fake) pipeline module detached, tails its events, reattaches."""
    (tmp_path / "pkg" / "reaper_pipeline").mkdir(parents=True)
    (tmp_path / "pkg" / "reaper_pipeline" / "__init__.py").write_text("STAGES = ['plan', 'weights', 'calibrate', 'score', 'prune', 'eval', 'package']\n")
    (tmp_path / "pkg" / "reaper_pipeline" / "__main__.py").write_text(FAKE_PIPELINE)
    monkeypatch.syspath_prepend(str(tmp_path / "pkg"))
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "pkg") + os.pathsep + os.environ.get("PYTHONPATH", ""))
    monkeypatch.setenv("REAPER_HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)  # `python -m` puts the cwd first; keep the real reaper_pipeline out of the way
    monkeypatch.setattr(backend, "_nvidia_smi", lambda: [backend.Gpu("NVIDIA B200", 179.1, 10.0)])
    real_module = backend._module
    monkeypatch.setattr(backend, "_module", lambda *n: None if "reaper_hw" in n else real_module(*n))

    async def run():
        app = Reaper(model="Qwen/Qwen3-30B-A3B-Instruct-2507", purpose="Debugging only")
        async with app.run_test(size=(150, 52)) as pilot:
            s = app.state
            for _ in range(100):
                await pilot.pause(0.1)
                if s.result:
                    break
            assert s.result.endswith("/out"), s.result
            assert all(s.stages[n]["status"] == "done" for n in s.stage_names)
            assert backend.eval_line(s.stages["eval"]) == "top-1 agreement 90% · KL 0.050 · ppl 4.00 → 4.40"
            cfg = json.loads((s.run.root / "config.json").read_text())
            assert cfg["purpose"] == "Debugging only" and cfg["model"] == "Qwen/Qwen3-30B-A3B-Instruct-2507"
        # a fresh TUI picks the finished job back up
        app2 = Reaper()
        async with app2.run_test(size=(150, 52)) as pilot:
            await pilot.pause(0.3)
            assert app2.state.result and app2.state.purpose == "Debugging only"

    asyncio.run(run())


def test_resolve_ranking():
    class M:
        def __init__(self, i, d): self.id, self.downloads = i, d

    class Api:
        def model_info(self, q): raise KeyError(q)
        def list_models(self, search, sort, limit):
            pool = [("unsloth/Qwen3.8-27B-A3B-Instruct-GGUF", 90000), ("Qwen/Qwen3.8-27B-A3B", 50000),
                    ("Qwen/Qwen3.8-27B-A3B-Instruct", 40000), ("Qwen/Qwen3.8-27B-A3B-Instruct-FP8", 30000)]
            return [M(i, d) for i, d in pool if search.lower() in i.lower()]

    rid, others = backend.resolve_model("qwen3.8-27B-A3B", Api())
    assert rid == "Qwen/Qwen3.8-27B-A3B-Instruct"
    assert backend.resolve_model("glm 4.5 air", Api())[0] == "zai-org/GLM-4.5-Air"


def test_machine_panel_with_reaper_hw(monkeypatch, tmp_path):
    """With the hardware package installed: a 1x B200 is detected and planned."""
    det = pytest.importorskip("reaper_hw.hardware")
    smi = "0, NVIDIA B200, 183359, 182900, 10.0, GPU-b200-0, 570.124.06"
    head = "| NVIDIA-SMI 570.124.06   Driver Version: 570.124.06   CUDA Version: 12.8 |"
    monkeypatch.setattr(det, "_run", lambda cmd, timeout=20: head if cmd == ["nvidia-smi"] else
                        ("" if cmd[:2] == ["nvidia-smi", "topo"] else smi))
    monkeypatch.setattr(det, "_torch", lambda hw, probe: (setattr(hw, "torch", "2.8.0"), setattr(hw, "torch_cuda", "12.8"),
                                                       setattr(hw, "torch_arches", ["sm_100"]), [])[-1])
    monkeypatch.setattr(det, "_meminfo", lambda: (256.0, 240.0))
    monkeypatch.setattr(det, "_disk_free", lambda path: 1800.0)
    monkeypatch.setenv("REAPER_HOME", str(tmp_path))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")  # config lookups fail fast; marketplace sizes are used

    async def run():
        app = Reaper()
        async with app.run_test(size=(150, 52)) as pilot:
            await _type(pilot, "/model qwen3-30b-a3b")
            for _ in range(50):
                await pilot.pause(0.1)
                if app.state.plan:
                    break
            s = app.state
            assert s.hw.gpus[0].cc == 10.0 and "B200" in s.hw.gpu_line()
            assert s.plan is not None and s.plan.ok, s.plan
            assert any("marketplace" in n for n in s.plan.notes)
            assert backend.fits(s.hw, 30.5, "Qwen/Qwen3-30B-A3B-Instruct-2507").startswith("✓")

    asyncio.run(run())
