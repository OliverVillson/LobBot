import asyncio
import time

import pytest

from blackhole_tui import art, architect
from blackhole_tui.catalog import SCHEMAS, find_model, find_schema, find_target, fit_mark, out_gb
from blackhole_tui.jobs import Run


@pytest.fixture(autouse=True)
def bh_home(tmp_path, monkeypatch):
    monkeypatch.setenv("BLACKHOLE_HOME", str(tmp_path))
    monkeypatch.setenv("BLACKHOLE_DEMO_SPEED", "40")


def test_logo_sizes():
    small, large = art.HOLE.plain.splitlines(), art.HOLE_LARGE.plain.splitlines()
    assert max(map(len, small)) <= 50 and max(map(len, large)) <= 64
    assert len(large) > len(small)
    # the centre is the event horizon: unlit
    mid = small[len(small) // 2 - 2]
    assert mid[len(mid) // 2] == " "


def test_schema_stages_and_sizes():
    pc = find_schema("pocket-coder")
    assert pc.stages() == ["detect", "plan", "weights", "calibrate", "reap", "heal", "eval", "gguf", "package"]
    assert find_schema("reap-classic").stages()[-2] == "quant"
    q = find_model("qwen3 coder")
    assert 8 < out_gb(q, pc) < 11
    assert fit_mark(q, pc, find_target("rtx4090")) == "✓"
    assert fit_mark(find_model("kimi k2"), pc, find_target("mac16")) == "✗"
    assert len({s.id for s in SCHEMAS}) == len(SCHEMAS)


def test_describe_draft():
    s = architect.draft_offline("a tiny rust tutor from my docs for my laptop")
    kinds = [st.kind for st in s.steps]
    assert kinds[0] == "synth" and "reap" in kinds and s.quant == "Q4_K"


def wait(run, timeout=30):
    end = time.time() + timeout
    while run.running() and time.time() < end:
        time.sleep(0.1)


def test_fake_pipeline_detached_run():
    settings = {"model": "Qwen/Qwen3-Coder-30B-A3B-Instruct", "schema": "pocket-coder", "target": "rtx4090",
                "purpose": "debugging"}
    run = Run.start(settings, demo=True)
    wait(run)
    events = [e for e, _ in Run(run.root).replay() if e]
    assert events[-1]["stage"] == "pipeline" and events[-1]["outcome"] == "done"
    done = [e["stage"] for e in events if e["status"] == "done" and e["stage"] != "pipeline"]
    assert done == find_schema("pocket-coder").stages()
    assert Run.latest().root == run.root


def test_stop(monkeypatch):
    monkeypatch.setenv("BLACKHOLE_DEMO_SPEED", "1")
    run = Run.start({"model": "openai/gpt-oss-20b", "schema": "laptop-chat", "target": "mac16"}, demo=True)
    time.sleep(1)
    assert run.stop()
    wait(run)
    last = [e for e, _ in run.replay() if e][-1]
    assert last["outcome"] == "stopped"


def test_tui_run_and_reattach():
    from blackhole_tui.app import BlackHole

    async def go():
        app = BlackHole(demo=True, model="gpt-oss 20b", schema="laptop-chat", target="mac16")
        async with app.run_test(size=(150, 50)) as pilot:
            await pilot.pause(0.5)
            assert app.state.running
        app2 = BlackHole(demo=True)  # as after an SSH drop
        async with app2.run_test(size=(150, 50)) as pilot:
            assert app2.state.run is not None and app2.state.schema.id == "laptop-chat"
            for _ in range(200):
                if not app2.state.running:
                    break
                await pilot.pause(0.1)
            await pilot.pause(0.5)
            assert app2.state.result
            await pilot.press("f6")
            await pilot.pause(0.2)
            await pilot.press("escape")
    asyncio.run(go())


QWEN3_30B = {"model_type": "qwen3_moe", "architectures": ["Qwen3MoeForCausalLM"], "hidden_size": 2048,
             "num_hidden_layers": 48, "intermediate_size": 6144, "moe_intermediate_size": 768, "num_experts": 128,
             "num_experts_per_tok": 8, "num_attention_heads": 32, "num_key_value_heads": 4, "head_dim": 128,
             "vocab_size": 151936, "tie_word_embeddings": False, "torch_dtype": "bfloat16"}


def test_detect_estimate_matches_card():
    from blackhole_tui.detect import from_config
    d = from_config(QWEN3_30B, "Qwen/Qwen3-30B-A3B", "hub")
    assert d.moe and d.experts == 128 and d.top_k == 8 and not d.size_exact
    assert 29.5 < d.total_b < 31.5 and 2.8 < d.active_b < 3.8  # card: 30.5B total, 3.3B active


def test_detect_local_folder(tmp_path):
    import json
    from blackhole_tui.detect import DetectError, detect
    (tmp_path / "config.json").write_text(json.dumps(QWEN3_30B))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": 61_000_000_000}}))
    d = detect(str(tmp_path))
    assert d.where == "local" and d.size_exact and abs(d.total_b - 30.5) < 0.1
    with pytest.raises(DetectError):
        detect(str(tmp_path / "nope"))


def test_tui_any_model_detect_and_run():
    from textual.widgets import Input
    from blackhole_tui.app import BlackHole

    async def go():
        app = BlackHole(demo=True)
        async with app.run_test(size=(150, 52)) as pilot:
            await pilot.press("f3")
            await pilot.pause(0.2)
            app.screen.query_one("#any_model", Input).value = "mistralai/Mixtral-8x7B-Instruct-v0.1"
            await pilot.press("enter")
            await pilot.pause(0.5)
            assert app.screen.found is not None
            await pilot.press("enter")
            await pilot.pause(0.2)
            assert app.state.model.experts == 8 and app.state.detected.family == "mixtral"
        # straight from the command line, a dense model: reap is skipped
        app = BlackHole(demo=True, model="meta-llama/Llama-3.1-70B-Instruct", schema="pocket-coder")
        async with app.run_test(size=(150, 52)) as pilot:
            for _ in range(100):
                await pilot.pause(0.1)
                if app.state.result:
                    break
            assert app.state.stages["reap"]["status"] == "skipped"
            assert app.state.result
    asyncio.run(go())
