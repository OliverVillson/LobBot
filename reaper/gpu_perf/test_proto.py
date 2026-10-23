"""CPU tests for the prototypes against REAPer's reference code, on its tiny real architectures.

    PYTHONPATH=<LobBot>/reaper:<LobBot>/reaper/tests:. pytest -q test_proto.py
"""

from __future__ import annotations

import json
import time

import pytest
import torch
from tiny_moe import ARCHS, build

from reaper_pipeline import ckpt, moe

import fast_prune
from fast_saliency import FastSaliency

ARCH_LIST = sorted(ARCHS)


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    root = tmp_path_factory.mktemp("tiny")
    return {a: build(a, root / a) for a in ARCH_LIST}


def load(path):
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).eval()


def batch(seed=1):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(5, 300, (3, 16), generator=g)
    mask = torch.ones_like(ids)
    mask[1, 11:] = 0  # right padding, as pad_batch makes it
    mask[2, 6:] = 0
    return ids, mask


def fwd(model, ids, mask):
    # the tiny deepseek_v3 config trips transformers' own sdpa path with a padding mask;
    # padding is still left out of the statistics through col.mask
    am = None if model.config.model_type == "deepseek_v3" else mask
    return model(input_ids=ids, attention_mask=am, use_cache=False).logits


def run(model, col, ids, mask):
    col.mask = mask.reshape(-1).bool()
    with torch.no_grad():
        out = fwd(model, ids, mask)
    col.mask = None
    return out


@pytest.mark.parametrize("reservoir", [4, 0])  # 4: per-expert loop path, 0: grouped-GEMM path
@pytest.mark.parametrize("arch", ARCH_LIST)
def test_fast_saliency_matches_reference_and_keeps_outputs(models, arch, reservoir):
    model = load(models[arch])
    ids, mask = batch()
    with torch.no_grad():
        plain = fwd(model, ids, mask)

    ref = moe.Saliency(model)
    for s in (1, 2):
        run(model, ref, *batch(s))
    ref.remove()
    want = ref.results()

    fast = FastSaliency(model, act_stats=True, reservoir=reservoir)
    out = run(model, fast, ids, mask)
    run(model, fast, *batch(2))
    fast.remove()
    got = fast.results()

    torch.testing.assert_close(out, plain, rtol=1e-4, atol=1e-4)  # scoring does not change the model
    with torch.no_grad():  # and the model is restored afterwards
        torch.testing.assert_close(fwd(model, ids, mask), plain)
    assert set(got) == set(want)
    for li in want:
        assert got[li]["freq"] == want[li]["freq"], (arch, li, fast.mode)
        torch.testing.assert_close(torch.tensor(got[li]["saliency"]), torch.tensor(want[li]["saliency"]),
                                   rtol=1e-4, atol=1e-6)
        if fast.mode[li] != "recompute":
            assert got[li]["in_sq"].shape[0] == moe.find_blocks(model)[0].num_experts
            routed = sum(1 for f in got[li]["freq"] if f > 0)
            if reservoir:
                assert sum(1 for b in fast.samples[li] if b) == routed
    print(arch, reservoir, fast.mode, fast._grouped_ok)


def _tensors(d):
    from safetensors import safe_open
    out = {}
    for f in sorted(d.glob("*.safetensors")):
        with safe_open(f, "pt") as h:
            out.update({k: h.get_tensor(k) for k in h.keys()})
    return out


@pytest.mark.parametrize("arch", ARCH_LIST)
def test_fast_prune_is_identical_to_reference(models, arch, tmp_path):
    src = models[arch]
    cfg = json.loads((src / "config.json").read_text())
    model = load(src)
    blocks = moe.find_blocks(model)
    E = blocks[0].num_experts
    g = torch.Generator().manual_seed(0)
    keep = {b.layer: sorted(torch.randperm(E, generator=g)[:5].tolist()) for b in blocks}
    L = cfg.get("num_hidden_layers")
    a = ckpt.prune_checkpoint(src, tmp_path / "ref", keep, E, L)
    t0 = time.time()
    b = fast_prune.prune_checkpoint(src, tmp_path / "fast", keep, E, L, workers=4)
    assert a["bytes_after"] == b["bytes_after"] and a["bytes_before"] == b["bytes_before"]
    ta, tb = _tensors(tmp_path / "ref"), _tensors(tmp_path / "fast")
    assert ta.keys() == tb.keys()
    for k in ta:
        assert ta[k].dtype == tb[k].dtype and torch.equal(ta[k], tb[k]), k
    assert json.loads((tmp_path / "ref" / "config.json").read_text()) == \
        json.loads((tmp_path / "fast" / "config.json").read_text())
    pruned = load(tmp_path / "fast")  # and it loads
    assert moe.find_blocks(pruned)[0].num_experts == 5
    print(arch, f"{time.time() - t0:.3f}s")


def test_tiered_experts_equal_per_expert_formats():
    from tiered_moe import TieredExperts, reference
    torch.manual_seed(0)
    E, H, I, N, k = 16, 64, 48, 40, 4
    gate_up, down = torch.randn(E, 2 * I, H) * 0.05, torch.randn(E, H, I) * 0.05
    tiers = {"fp8_block": [0, 3, 5], "nvfp4": [1, 2, 7, 8, 9], "int4_g128": [4, 6, 10, 11],
             "int2_g64": [12, 13], "drop": [14, 15]}
    fmt_of = {j: f for f, ids in tiers.items() for j in ids}
    x = torch.randn(N, H)
    logits = torch.randn(N, E)
    w, idx = logits.softmax(-1).topk(k, -1)
    mod = TieredExperts(gate_up, down, tiers)
    assert mod.formats == ["fp8_block", "nvfp4", "int4_g128", "int2_g64"]
    torch.testing.assert_close(mod(x, idx, w), reference(x, idx, w, gate_up, down, fmt_of), rtol=1e-4, atol=1e-5)


def test_fake_quant_error_ladder():
    from fakequant import BPW, fake_quant
    torch.manual_seed(0)
    w = torch.randn(256, 512) * 0.02
    rel = {f: float((w - fake_quant(w, f)).pow(2).sum() / w.pow(2).sum()) for f in BPW if f != "drop"}
    print({f: f"{v:.2e}" for f, v in rel.items()})
    assert rel["bf16"] < rel["int8_g128"] < rel["fp8_block"] < rel["nvfp4"] < rel["int3_g64"] < rel["int2_g64"]
    assert rel["nvfp4"] < rel["mxfp4"]  # finer group + fp8 scales beat power-of-two scales per 32


def test_probe_and_allocator_on_a_scored_model(models):
    import expert_bits as xb
    model = load(models["qwen3_moe"])
    fast = FastSaliency(model, reservoir=16)
    for s in (1, 2, 3):
        run(model, fast, *batch(s))
    fast.remove()
    res = fast.results()
    blocks = moe.find_blocks(model)
    ex = blocks[0].experts
    experts = []
    fmts = ["drop", "int2_g64", "int4_g128", "fp8_block"]
    H = model.config.hidden_size
    for b in blocks:
        r = res[b.layer]
        tot = sum(r["freq"])
        for j in range(b.num_experts):
            p = model.config.num_experts_per_tok * r["freq"][j] / tot
            e = xb.Expert(b.layer, j, p, r["saliency"][j], 3 * H * model.config.moe_intermediate_size)
            smp = fast.samples[b.layer][j]
            if smp:
                xj = torch.cat(smp).float()
                gu, dn = b.experts.gate_up_proj[j], b.experts.down_proj[j]
                if gu.shape[-1] != H:  # stored [H, 2I]: transpose to [2I, H]
                    gu = gu.T
                if dn.shape[0] != H:
                    dn = dn.T
                I = gu.shape[0] // 2
                gates = torch.cat(fast.sample_gates[b.layer][j])
                m = xb.probe({"gate": gu[:I], "up": gu[I:], "down": dn}, xj, gates, fmts[1:])
                e.measured = {f: p * v for f, v in m.items()} | {"drop": e.damage("drop")}
            experts.append(e)
    full = sum(e.storage("fp8_block") for e in experts)
    alloc = xb.allocate(experts, fmts, 0.4 * full, max_tiers=2)
    assert alloc.storage <= 0.4 * full + 1
    per_layer = {}
    for (L, j), f in alloc.choice.items():
        per_layer.setdefault(L, set()).add(f)
    assert all(len(v - {"drop"}) <= 2 for v in per_layer.values())
    print({f: sum(1 for v in alloc.choice.values() if v == f) for f in fmts})


def test_gptq_matches_rtn_on_identity_and_beats_it_on_correlated_inputs():
    from fakequant import fake_quant
    from gptq import gptq
    torch.manual_seed(0)
    W = torch.randn(32, 256)
    assert torch.allclose(gptq(W, torch.eye(256), "int4_g128", damp=0), fake_quant(W, "int4_g128"), atol=1e-6)
    X = torch.randn(2048, 256) @ torch.randn(256, 256) * 0.3
    W[:, 5:9] = 0
    Q = gptq(W, X.T @ X, "int3_g64")
    assert Q[:, 5:9].abs().max() == 0
    assert ((W - Q) @ X.T).norm() < 0.9 * ((W - fake_quant(W, "int3_g64")) @ X.T).norm()


def test_header_formats():
    from fakequant import fake_quant
    from headers import header_quant
    torch.manual_seed(0)
    W = torch.randn(3, 64, 256) * torch.tensor([1.0, 3.0, 0.2])[:, None, None]
    nv = torch.stack([fake_quant(w, "nvfp4") for w in W])
    assert torch.allclose(header_quant(W, "e2m1"), nv)              # per-expert header == NVFP4 per matrix
    err = lambda Q: float((W - Q).pow(2).sum())                     # noqa: E731
    assert err(header_quant(W, "expert", 8)) < err(header_quant(W, "int", 8))
    assert err(header_quant(W, "expert", 16)) <= err(header_quant(W, "e2m1")) * 1.01
