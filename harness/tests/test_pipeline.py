"""blackhole_pipeline: bit packing, and a full run on a tiny random Flint (CPU) that reuses an
earlier REAPer job's calibration and scores."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("reaper_pipeline")

from blackhole_pipeline import STAGES  # noqa: E402
from blackhole_pipeline import pack  # noqa: E402
from blackhole_tui.catalog import find_schema  # noqa: E402
from blackhole_tui.jobs import parse_event  # noqa: E402

REAPER = Path(__import__("reaper_pipeline").__file__).resolve().parents[1]
HARNESS = Path(__file__).resolve().parents[1]


def test_int4_pack_roundtrip_and_layout():
    codes = torch.randint(-8, 8, (5, 40), dtype=torch.int8)
    p = pack.pack_int4(codes)
    assert p.dtype == torch.int32 and p.shape == (5, 5)
    assert torch.equal(pack.unpack_int4(p, 40), codes)
    one = torch.full((1, 8), -8, dtype=torch.int8)
    one[0, 1] = 7  # element 1 sits in bits 4..7, stored offset by 8
    assert pack.pack_int4(one).item() == 15 << 4


def test_int_codes_from_gptq_scales():
    sys.path.insert(0, str(REAPER / "gpu_perf"))
    from gptq import gptq_batched
    w = torch.randn(2, 16, 256)
    x = torch.randn(512, 256)
    H = (x.T @ x)[None].repeat(2, 1, 1)
    Q, s = gptq_batched(w, H, "int4_g128", return_scales=True)
    assert s.shape == (2, 16, 2)
    c = pack.int_codes(Q, s, 128)
    assert torch.allclose(c.float() * s.repeat_interleave(128, -1), Q, atol=1e-5)
    assert torch.equal(gptq_batched(w, H, "int4_g128"), Q)


def test_codebook_roundtrip():
    sys.path.insert(0, str(REAPER / "gpu_perf"))
    from headers import header_parts, round_to_table
    W = torch.randn(3, 8, 64)
    for L in (16, 8):
        S, Lv = header_parts(W, "expert", L, 32)
        Q = round_to_table(W, S, Lv)
        es = pack.expert_scale(W)
        back = pack.codebook_dequant(pack.pack_nibbles(pack.table_indices(Q, S, Lv)), 64,
                                     pack.block_codes(S, es, 32), es, Lv, 32)
        assert torch.allclose(back, Q, rtol=1e-5, atol=1e-7)


def test_lab_schema_stages_match_the_pipeline():
    for name in ("reaq-codebook", "mac-16gb", "mac-16gb-375"):
        sc = find_schema(name)
        assert sc.stages() == [s for s in STAGES if s in sc.stages()] and "detect" in sc.stages()
    assert "lab" in find_schema("reaq-codebook").tags


def test_mlx_packing_matches_mlx():
    mx = pytest.importorskip("mlx.core")
    import numpy as np
    for bits in (3, 4, 6, 8):
        w = torch.randn(3, 5, 128)
        u, s, b = pack.mlx_rtn(w, bits, 64, torch.float32)
        words = pack.mlx_pack(u, bits)
        assert torch.equal(pack.mlx_unpack(words, bits), u)
        got = mx.dequantize(mx.array(words.numpy()), mx.array(s.numpy()), mx.array(b.numpy()), group_size=64, bits=bits)
        assert np.allclose(np.array(got), pack.mlx_dequant(u, s, b, 64).numpy(), atol=1e-5)
    # GPTQ's symmetric grid maps exactly onto MLX's affine one
    codes = torch.randint(-4, 4, (2, 64))
    sc = torch.rand(2, 1) + 0.1
    u, s, b = pack.mlx_from_symmetric(codes, sc, 3, torch.float32)
    assert torch.allclose(pack.mlx_dequant(u, s, b, 64), codes * sc)


@pytest.mark.parametrize("donor", ["scores", "data-only"])
def test_full_run_on_a_tiny_flint(tmp_path, donor):
    pytest.importorskip("transformers")
    sys.path.insert(0, str(REAPER / "tests"))
    from tiny_moe import build
    model = build("flint", tmp_path / "flint-tiny")
    home = tmp_path / "home"
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(REAPER), str(HARNESS)]), "REAPER_HW": "builtin",
           "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "REAPER_HOME": str(home),
           "BLACKHOLE_HOME": str(tmp_path / "bhhome")}
    small = {"calib": "mix", "heldout": 4, "max_seq": 96, "answer_max_tokens": 12, "writer_max_tokens": 24,
             "teacher_backend": "hf"}

    # an earlier REAPer run at 50% on the same model and purpose (like ~/.reaper/jobs/flint-prune50)
    old = home / "jobs" / "flint-tiny-prune50"
    old.mkdir(parents=True)
    (old / "config.json").write_text(json.dumps({"model": str(model), "purpose": "", "prune": 0.5,
                                                 "calib_samples": 8, **small}))
    p = subprocess.run([sys.executable, "-m", "reaper_pipeline", "--job", str(old)], env=env, cwd=tmp_path,
                       capture_output=True, text=True, timeout=900)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]

    if donor == "data-only":  # e.g. a job whose scoring never finished: reuse the teacher answers only
        (old / ".done" / "score").unlink()
        (old / "work" / "teacher_ref.pt").unlink()

    probs = tmp_path / "lcb.jsonl"
    probs.write_text("".join(json.dumps(r) + "\n" for r in LCB_ROWS))
    small_bench = {"bench_file": str(probs), "bench_max_tokens": 16}

    sc = find_schema("reaq-codebook")
    job = tmp_path / "bh" / "flint-tiny-reaq"
    job.mkdir(parents=True)
    (job / "config.json").write_text(json.dumps({
        "model": str(model), "schema": sc.id, "target": "rtx4090", "purpose": "",
        "steps": [s.__dict__ for s in sc.steps], "quant": sc.quant, "calib_samples": 8, **small,
        **small_bench}))
    p = subprocess.run([sys.executable, "-m", "blackhole_pipeline", "--job", str(job)], env=env, cwd=tmp_path,
                       capture_output=True, text=True, timeout=900)
    events = [e for e in map(parse_event, p.stdout.splitlines()) if e]
    assert p.returncode == 0, p.stdout[-4000:] + p.stderr[-3000:]
    by = {}
    for e in events:
        by.setdefault(e["stage"], []).append(e)
    assert by["calibrate"][-1]["status"] == "skipped" and "reused" in by["calibrate"][-1]["msg"]
    if donor == "scores":
        assert all(e["status"] == "skipped" for e in by["score"])  # scores reused, only prune ran
    else:
        assert by["score"][-1]["status"] == "done"
        assert "calibration usable, REAP scores not" in p.stdout
    for s in ("detect", "plan", "prune", "eval", "gptq", "reaq", "bench", "quant", "package"):
        assert by[s][-1]["status"] == "done", (s, by[s][-1])
    ev = by["eval"][-1]
    assert {"top1", "kl", "ppl_teacher", "ppl_student"} <= set(ev)
    assert "kl" in by["gptq"][-1] and "kl" in by["reaq"][-1]
    final = by["pipeline"][-1]
    assert final["status"] == "done" and final["outcome"] == "done"
    gdir = Path(final["model"])
    assert gdir.name.endswith("-REAP-6e-GPTQ-W4A16")  # 8 experts, 25% pruned

    # the int4 checkpoint: compressed-tensors config, packed experts that dequantize near the bf16 ones
    from safetensors import safe_open
    cfg = json.loads((gdir / "config.json").read_text())
    assert cfg["quantization_config"]["format"] == "pack-quantized" and cfg["num_experts"] == 6
    idx = json.loads((gdir / "model.safetensors.index.json").read_text())["weight_map"]
    bf16 = gdir.with_name(gdir.name[:-len("-GPTQ-W4A16")])
    bidx = json.loads((bf16 / "model.safetensors.index.json").read_text())["weight_map"] \
        if (bf16 / "model.safetensors.index.json").exists() else None
    key = next(k for k in idx if k.endswith("experts.0.gate_proj.weight_packed"))
    base = key[:-len(".weight_packed")]
    assert f"{base}.weight" not in idx
    with safe_open(gdir / idx[key], "pt") as f:
        packed, scale, shape = f.get_tensor(key), f.get_tensor(base + ".weight_scale"), f.get_tensor(base + ".weight_shape")
    out, n = shape.tolist()
    deq = pack.unpack_int4(packed, n).float() * scale.float().repeat_interleave(128, -1)[:, :n]
    bfile = bf16 / (bidx[base + ".weight"] if bidx else "model.safetensors")
    with safe_open(bfile, "pt") as f:
        w = f.get_tensor(base + ".weight").float()
    assert deq.shape == w.shape and (deq - w).norm() / w.norm() < 0.25

    # the REAQ codebook folder: tiers by REAP rank, dequantizes near the bf16 experts
    qdir = gdir.with_name(bf16.name + "-REAQ-codebook")
    assert (qdir / "REAQ_LAYOUT.txt").exists()
    qidx = json.loads((qdir / "model.safetensors.index.json").read_text())["weight_map"]
    t16 = [k for k in qidx if ".tier16.experts" in k]
    t8 = [k for k in qidx if ".tier8.experts" in k]
    assert t16 and len(t16) == len(t8)
    rec = json.loads((job / "receipt.json").read_text())
    assert rec["results"]["reaq_codebook"]["bpw"] == pytest.approx(3.75)
    assert rec["results"]["gptq_int4"]["bpw"] == pytest.approx(4.125)
    assert (job / "out" / gdir.name).is_symlink() and (job / "RESULTS.md").exists()
    assert rec["results"]["bench"]["problems"] == 2 and "LiveCodeBench" in (job / "RESULTS.md").read_text()
    if donor == "scores":
        check_serve(gdir, qdir)


def check_serve(gdir: Path, qdir: Path):
    """blackhole-serve loads the int4 and codebook folders (unpacked to bf16 = the measured weights)
    and answers OpenAI chat requests, streamed and not."""
    import threading
    import urllib.request
    from http.server import ThreadingHTTPServer

    from blackhole_pipeline import serve
    qm, _ = serve.load(qdir, "cpu")
    assert hasattr(qm, "fast_generate")
    model, tok = serve.load(gdir, "cpu")
    engine = serve.Engine(model, tok, "tiny")
    srv = ThreadingHTTPServer(("127.0.0.1", 0), serve.make_handler(engine))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/v1"

    def post(body):
        r = urllib.request.Request(url + "/chat/completions", json.dumps(body).encode(),
                                   {"Content-Type": "application/json"})
        return urllib.request.urlopen(r, timeout=120).read().decode()

    assert json.loads(urllib.request.urlopen(url + "/models").read())["data"][0]["id"] == "tiny"
    msgs = [{"role": "user", "content": [{"type": "text", "text": "fix this bug"}]}]
    r = json.loads(post({"model": "tiny", "messages": msgs, "max_tokens": 8, "temperature": 0}))
    assert r["choices"][0]["message"]["role"] == "assistant" and r["usage"]["completion_tokens"] <= 8
    lines = [l for l in post({"model": "tiny", "messages": msgs, "max_tokens": 8, "stream": True}).splitlines()
             if l.startswith("data: ")]
    assert lines[-1] == "data: [DONE]" and json.loads(lines[-2][6:])["choices"][0]["finish_reason"]
    srv.shutdown()


def _private(tests):
    import base64
    import pickle
    import zlib
    return base64.b64encode(zlib.compress(pickle.dumps(json.dumps(tests)))).decode()


LCB_ROWS = [
    {"question_title": "Sum", "question_content": "Read two integers and print their sum.", "platform": "atcoder",
     "question_id": "abc1", "contest_date": "2025-02-01", "starter_code": "", "difficulty": "easy",
     "public_test_cases": json.dumps([{"input": "1 2\n", "output": "3\n", "testtype": "stdin"}]),
     "private_test_cases": _private([{"input": "5 7\n", "output": "12\n", "testtype": "stdin"}]), "metadata": "{}"},
    {"question_title": "Twice", "question_content": "Return twice the list.", "platform": "leetcode",
     "question_id": "lc1", "contest_date": "2025-03-01", "difficulty": "medium",
     "starter_code": "class Solution:\n    def twice(self, nums: List[int], k: int) -> List[int]:\n        ",
     "public_test_cases": json.dumps([{"input": "[1, 2]\n2", "output": "[2, 4]", "testtype": "functional"}]),
     "private_test_cases": _private([{"input": "[3]\n2", "output": "[6]", "testtype": "functional"}]),
     "metadata": json.dumps({"func_name": "twice"})},
]


def test_lcb_problems_and_runner(tmp_path):
    from blackhole_pipeline import bench
    f = tmp_path / "p.jsonl"
    f.write_text("".join(json.dumps(r) + "\n" for r in LCB_ROWS))
    a, b = bench.load_problems(file=str(f))
    assert len(a["tests"]) == 2 and b["func"] == "twice" and "starter code" in bench.prompt(b)[1]["content"]
    good_a = "```python\na, b = map(int, input().split())\nprint(a + b)\n```"
    good_b = "<think>hm</think>```python\nclass Solution:\n    def twice(self, nums, k):\n        return [x * k for x in nums]\n```"
    assert bench.run_problem(a, bench.extract_code(good_a))["passed"]
    assert bench.run_problem(b, bench.extract_code(good_b))["passed"]
    bad = bench.run_problem(a, "print(3)")
    assert not bad["passed"] and "test 2" in bad["why"]
    assert not bench.run_problem(a, "while True: pass")["passed"]
    assert not bench.run_problem(a, "")["passed"]


def test_parse_reply_tool_calls_and_reasoning():
    from blackhole_pipeline.serve import parse_reply, to_template_messages
    c, r, calls = parse_reply('<think>look first</think>Sure.<tool_call>{"name": "read", "arguments": {"path": "a.py"}}'
                              '</tool_call>')
    assert (c, r) == ("Sure.", "look first") and calls[0]["function"]["name"] == "read"
    assert json.loads(calls[0]["function"]["arguments"]) == {"path": "a.py"}
    assert parse_reply("plain answer") == ("plain answer", "", [])
    m = to_template_messages([{"role": "assistant", "content": None, "tool_calls": [
        {"id": "x", "type": "function", "function": {"name": "read", "arguments": '{"path": "a"}'}}]}])
    assert m[0]["content"] == "" and m[0]["tool_calls"][0]["function"]["arguments"] == {"path": "a"}


def test_mac_schema_heal_and_mlx_on_a_tiny_flint(tmp_path):
    """mac-16gb: REAP 50% on code, distillation heal, MLX export (GPTQ experts with a REAQ hot tier),
    bench of the MLX model; the MLX folder loads in blackhole-serve (dequantized) and, when mlx is
    installed, in mlx-lm with the same logits."""
    pytest.importorskip("transformers")
    sys.path.insert(0, str(REAPER / "tests"))
    from tiny_moe import build
    model = build("flint", tmp_path / "flint-tiny")
    home = tmp_path / "home"
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(REAPER), str(HARNESS)]), "REAPER_HW": "builtin",
           "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "REAPER_HOME": str(home),
           "BLACKHOLE_HOME": str(tmp_path / "bhhome")}
    small = {"heldout": 4, "max_seq": 96, "answer_max_tokens": 12, "writer_max_tokens": 24, "teacher_backend": "hf"}
    probs = tmp_path / "lcb.jsonl"
    probs.write_text("".join(json.dumps(r) + "\n" for r in LCB_ROWS))
    sc = find_schema("mac-16gb")
    job = tmp_path / "bh" / "flint-tiny-mac"
    job.mkdir(parents=True)
    (job / "config.json").write_text(json.dumps({
        "model": str(model), "schema": sc.id, "target": "mac", "purpose": "",
        "steps": [s.__dict__ for s in sc.steps], "quant": sc.quant, "calib_samples": 8, **small,
        "mlx_group": 32, "mlx_hot": 0.5, "mlx_qo_bits": 3, "mlx_head_vocab": 190, "heal_prompts": 6, "heal_max_tokens": 10, "heal_accum": 2,
        "heal_lr": 1e-3, "heal_expert_rank": 2, "bench_file": str(probs), "bench_max_tokens": 16}))
    p = subprocess.run([sys.executable, "-m", "blackhole_pipeline", "--job", str(job)], env=env, cwd=tmp_path,
                       capture_output=True, text=True, timeout=900)
    events = [e for e in map(parse_event, p.stdout.splitlines()) if e]
    assert p.returncode == 0, p.stdout[-4000:] + p.stderr[-3000:]
    by = {}
    for e in events:
        by.setdefault(e["stage"], []).append(e)
    for s in ("prune", "eval", "heal", "mlx", "qheal", "bench", "package"):
        assert by[s][-1]["status"] == "done", (s, by[s][-1])
    qh = json.loads((job / "reaper" / "work" / "qheal.json").read_text())
    mj = json.loads((job / "reaper" / "work" / "mlx.json").read_text())
    assert qh["steps"] >= 1
    start = mj.get("unhealed_kl", mj["kl"])
    assert abs(qh["kl_before"] - start) <= 1e-3 + 1e-2 * start, (qh["kl_before"], start)  # qheal starts AS the MLX model
    if qh["kept"]:
        assert mj["out"].endswith("-healed") and Path(mj["out"], "config.json").exists() and mj["kl"] < start
    assert "gptq" not in by and "reaq" not in by and "quant" not in by  # not in this schema

    # qheal's writer: the pruned model made into the MLX model, changed, written back as an MLX
    # folder that loads with the same logits
    from blackhole_pipeline import serve
    from blackhole_pipeline.stages.heal import quantize_like, write_mlx
    mdir = Path(mj.get("unhealed_out", mj["out"]))
    st, _ = serve.load(mdir.parent / mdir.name.split("-MLX-")[0], "cpu")
    fq = quantize_like(st, mdir)
    assert any("self_attn.q_proj" in n for n in fq) and not any(".experts." in n for n in fq)
    ids = torch.randint(5, 300, (1, 12))
    with torch.no_grad():
        same = st(input_ids=ids).logits[0]
        ref_mlx, _ = serve.load(mdir, "cpu")
        assert torch.allclose(same, ref_mlx(input_ids=ids).logits[0], atol=1e-2)   # it IS the MLX model
        g = torch.Generator().manual_seed(3)
        for n, prm in st.named_parameters():
            if "parametrizations" in n or n.endswith("mlp.gate.weight") or "layernorm" in n:
                prm.add_(0.05 * prm.abs().mean() * torch.randn(prm.shape, generator=g))
        want = st(input_ids=ids).logits[0]
    assert (want - same).abs().max() > 1e-2
    write_mlx(st, fq, mdir, tmp_path / "healed-MLX")
    got_m, _ = serve.load(tmp_path / "healed-MLX", "cpu")
    with torch.no_grad():
        got = got_m(input_ids=ids).logits[0]
    assert torch.allclose(got, want, atol=2e-2 * want.abs().max()), (got - want).abs().max()
    rcfg = json.loads((job / "reaper" / "config.json").read_text())
    assert rcfg["calib"] == "code" and rcfg["prune"] == 0.5
    heal = json.loads((job / "reaper" / "work" / "heal.json").read_text())
    assert heal["steps"] >= 1 and heal["sequences"] == 6
    rec = json.loads((job / "receipt.json").read_text())
    xdir = Path(rec["model"])
    assert "-MLX-" in xdir.name and (xdir / "flint_mlx.py").exists()
    cfg = json.loads((xdir / "config.json").read_text())
    assert cfg["model_file"] == "flint_mlx.py" and cfg["num_experts"] == 4
    assert all(v == 2 for v in cfg["expert_tiers"].values())  # half of the 4 kept experts on the hot tier
    q = cfg["quantization"]
    hot = next(k for k in q if k.endswith("experts_hot.gate_proj"))
    assert q[hot]["bits"] == 4 and q[hot.replace("experts_hot", "experts")]["bits"] == 3
    assert q["lm_head"]["bits"] == 6 and q["model.embed_tokens"]["bits"] == 4
    assert cfg["head_vocab"] == 192                       # 190 rounded up to a multiple of 32
    assert q["model.layers.0.self_attn.q_proj"]["bits"] == 3 and q["model.layers.0.self_attn.k_proj"]["bits"] == 4
    assert rec["results"]["mlx"]["kl"] >= 0 and "mlx" in rec["results"]["bench"]
    assert "mlx_lm.server" in (job / "RESULTS.md").read_text()

    from blackhole_pipeline import serve
    tm, tok = serve.load(xdir, "cpu")
    ids = torch.randint(5, 300, (1, 12))
    with torch.no_grad():
        ref = tm(input_ids=ids).logits[0].float().numpy()
    try:
        import mlx.core as mx
        from mlx_lm.utils import load_model
    except ImportError:
        return
    import numpy as np
    mm, _ = load_model(xdir, trust_remote_code=True)
    mm.set_dtype(mx.float32)  # bf16 activations would hide a real mismatch on a tiny random model
    got = np.array(mm(mx.array(ids.numpy()))[0].astype(mx.float32))
    live = np.isfinite(got)                                # output-trimmed head: dropped tokens are -inf
    assert np.array_equal(live, ref > serve.HEAD_FILL) and live[0].sum() <= 192
    assert np.abs(got[live] - ref[live]).max() < 1e-3 * np.abs(ref[live]).max() + 1e-4
    assert np.array_equal(got.argmax(-1), ref.argmax(-1))


def test_shortest_passing_keeps_the_shortest_correct_answer():
    from blackhole_pipeline.bench import extract_code, run_problem
    from blackhole_pipeline.traces import shortest_passing
    add = {"id": "add", "question": "a+b", "starter": "", "func": None,
           "tests": [{"input": "1 2\n", "output": "3\n"}, {"input": "5 5\n", "output": "10\n"}]}
    neg = {"id": "neg", "question": "-a", "starter": "", "func": None, "tests": [{"input": "4\n", "output": "-4\n"}]}
    good = "```python\na, b = map(int, input().split())\nprint(a + b)\n```"
    replies = {("add", 0): ("long thinking " * 30, good), ("add", 1): ("short", good),
               ("add", 2): ("x", "```python\nprint(0)\n```"), ("neg", 0): ("y", "no code"),
               ("neg", 1): ("z", "```python\nprint(4)\n```"), ("neg", 2): ("w", "")}

    def gen(jobs):
        out = []
        for p, s in jobs:
            think, ans = replies[(p["id"], s)]
            text = f"<think>{think}</think>{ans}"
            out.append(([1, 2, 3], list(range(len(text.split()))), text))
        return out
    rows, st = shortest_passing([add, neg], 3, gen, lambda p, t: run_problem(p, extract_code(t))["passed"])
    assert st["solved"] == 1 and st["passed"] == 2 and st["answers"] == 6
    assert len(rows) == 1 and rows[0]["start"] == 3
    assert st["tokens_mean_kept"] < st["tokens_mean_passing"] and st["shorter_than_passing"] > 0
