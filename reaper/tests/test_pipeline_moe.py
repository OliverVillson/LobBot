"""reaper_pipeline on tiny random checkpoints of real MoE architectures (CPU)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tiny_moe import ARCHS, build  # noqa: E402

from reaper_pipeline import ckpt, moe  # noqa: E402

ARCH_LIST = sorted(ARCHS)


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    root = tmp_path_factory.mktemp("tiny")
    return {a: build(a, root / a) for a in ARCH_LIST}


def load(path):
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).eval()


def ids(n=12, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(5, 300, (2, n), generator=g)


@pytest.mark.parametrize("arch", ARCH_LIST)
def test_routing_and_expert_outputs_rebuild_the_moe_output(models, arch):
    """sum_j g_j(x) f_j(x) over the routing we read equals what the experts module returns."""
    model = load(models[arch])
    blocks = moe.find_blocks(model)
    cfg = json.loads((models[arch] / "config.json").read_text())
    dense_first = cfg.get("first_k_dense_replace", 0) + len(cfg.get("mlp_only_layers") or [])
    assert len(blocks) == cfg["num_hidden_layers"] - dense_first
    assert all(b.num_experts == 8 for b in blocks)

    seen, checked = {}, []

    def on_router(blk):
        def hook(_m, args, kwargs, out):
            x = (args[0] if args else next(iter(kwargs.values())))
            x = x.reshape(-1, x.shape[-1])
            p = moe.parse_router_output(out, x.shape[0], blk.num_experts)
            if p is None:
                lg = out[0] if isinstance(out, tuple) else out
                p = moe.route_from_logits(blk, model.config, lg.reshape(-1, blk.num_experts))
            seen[blk.layer] = (x, *p)
        return hook

    busy = []

    def on_experts(blk):
        def hook(_m, _args, out):
            if busy:  # our own single-expert calls below
                return
            busy.append(1)
            x, idx, w = seen[blk.layer]
            ref = torch.zeros_like(x)
            for j in torch.unique(idx).tolist():
                tok, slot = torch.where(idx == j)
                ref[tok] += w[tok, slot, None].to(x.dtype) * moe.expert_forward(blk, model.config, j, x[tok])
            got = (out[0] if isinstance(out, tuple) else out).reshape(ref.shape)
            busy.clear()
            torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-5)
            checked.append((blk.layer, blk._strategy))
        return hook

    hs = []
    for b in blocks:
        hs.append(b.router.register_forward_hook(on_router(b), with_kwargs=True))
        if b.fused:
            hs.append(b.experts.register_forward_hook(on_experts(b)))
    with torch.no_grad():
        model(input_ids=ids(), use_cache=False)
    for h in hs:
        h.remove()
    if blocks[0].fused:
        assert sorted(li for li, _ in checked) == sorted(b.layer for b in blocks)
        print(arch, checked)


@pytest.mark.parametrize("arch", ARCH_LIST)
def test_saliency_scores_every_layer_without_changing_outputs(models, arch):
    model = load(models[arch])
    x = ids()
    with torch.no_grad():
        before = model(input_ids=x, use_cache=False).logits
        col = moe.Saliency(model)
        after = model(input_ids=x, use_cache=False).logits
        col.remove()
    torch.testing.assert_close(before, after)
    res = col.results()
    k = moe.top_k(col.blocks[0], model.config)
    for li, r in res.items():
        assert sum(r["freq"]) == x.numel() * k, (li, r["freq"])
        assert all(s >= 0 for s in r["saliency"]) and max(r["saliency"]) > 0


@pytest.mark.parametrize("arch", ARCH_LIST)
def test_pruned_checkpoint_is_exact_where_routing_is_unchanged(models, arch, tmp_path):
    """On one token, keep the experts it routes to (plus others): the pruned model
    must give the same logits, so router rows, biases, renumbering and config line up."""
    src = models[arch]
    model = load(src)
    blocks = moe.find_blocks(model)
    used: dict[int, set] = {}

    def on_router(blk):
        def hook(_m, args, kwargs, out):
            x = (args[0] if args else next(iter(kwargs.values())))
            x = x.reshape(-1, x.shape[-1])
            p = moe.parse_router_output(out, x.shape[0], blk.num_experts)
            if p is None:
                lg = out[0] if isinstance(out, tuple) else out
                p = moe.route_from_logits(blk, model.config, lg.reshape(-1, blk.num_experts))
            used[blk.layer] = set(p[0].flatten().tolist())
        return hook

    hs = [b.router.register_forward_hook(on_router(b), with_kwargs=True) for b in blocks]
    one = torch.tensor([[42]])
    with torch.no_grad():
        ref = model(input_ids=one, use_cache=False).logits
    for h in hs:
        h.remove()
    keep = {}
    for b in blocks:
        extra = [j for j in range(8) if j not in used[b.layer]]
        keep[b.layer] = sorted(used[b.layer] | set(extra[-(4 - len(used[b.layer])):]))
        assert len(keep[b.layer]) == 4
    n_layers = moe.text_config(model.config).num_hidden_layers
    info = ckpt.prune_checkpoint(src, tmp_path / "pruned", keep, 8, n_layers)
    assert info["bytes_after"] < info["bytes_before"]

    pruned = load(tmp_path / "pruned")
    pb = moe.find_blocks(pruned)
    assert [b.num_experts for b in pb] == [4] * len(blocks)
    with torch.no_grad():
        got = pruned(input_ids=one, use_cache=False).logits
    torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-4)
    cfg = json.loads((tmp_path / "pruned" / "config.json").read_text())
    assert 4 in [cfg.get(k) for k in moe.EXPERT_COUNT_KEYS]


def test_choose_keep_respects_expert_groups():
    s = [0.9, 0.8, 0.1, 0.2, 0.05, 0.04, 0.03, 0.02]
    f = [1] * 8
    assert moe.choose_keep(s, f, 4) == [0, 1, 2, 3]
    assert moe.choose_keep(s, f, 4, groups=2) == [0, 1, 4, 5]
    assert moe.experts_to_keep(256, 0.5, 8, groups=8, topk_group=4) == 128
    assert moe.experts_to_keep(8, 0.9, 2) == 2


def test_mtp_layers_are_dropped():
    keys = ["model.layers.0.mlp.experts.3.w.weight", "model.layers.0.mlp.gate.weight",
            "model.layers.0.mlp.gate.e_score_correction_bias", "model.layers.0.mlp.shared_experts.w.weight",
            "model.layers.1.mlp.experts.0.w.weight", "model.layers.1.eh_proj.weight"]
    actions, notes = ckpt.plan_keys(keys, {0: [1, 3]}, num_layers=1)
    assert actions[keys[0]] == ("rename", "model.layers.0.mlp.experts.1.w.weight")
    assert actions[keys[1]] == ("rows", [1, 3]) and actions[keys[2]] == ("rows", [1, 3])
    assert actions[keys[3]] == "copy"
    assert actions[keys[4]] == "drop" and actions[keys[5]] == "drop"
    assert notes


# --------------------------------------------------------------------------- remote-code style (Kimi K2, transformers 4.x)


class _Cfg:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _Mlp(torch.nn.Module):
    def __init__(self, h, i):
        super().__init__()
        self.gate_proj, self.up_proj = torch.nn.Linear(h, i, bias=False), torch.nn.Linear(h, i, bias=False)
        self.down_proj = torch.nn.Linear(i, h, bias=False)

    def forward(self, x):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class _KimiGate(torch.nn.Module):
    """DeepSeek-V3 remote-code MoEGate: returns (topk_idx, topk_weight, aux_loss)."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.weight = torch.nn.Parameter(torch.randn(cfg.n_routed_experts, cfg.hidden_size))
        self.e_score_correction_bias = torch.nn.Parameter(torch.randn(cfg.n_routed_experts) * 0.3)

    def forward(self, h):
        x = h.view(-1, h.shape[-1])
        s = torch.nn.functional.linear(x, self.weight).sigmoid()
        idx = (s + self.e_score_correction_bias).topk(self.cfg.num_experts_per_tok, dim=-1)[1]
        w = s.gather(1, idx)
        w = w / w.sum(-1, keepdim=True) * self.cfg.routed_scaling_factor
        return idx, w, None


class _LinearGate(torch.nn.Linear):
    pass


class _Block(torch.nn.Module):
    def __init__(self, cfg, kimi):
        super().__init__()
        self.cfg, self.kimi = cfg, kimi
        E, H = cfg.n_routed_experts, cfg.hidden_size
        self.gate = _KimiGate(cfg) if kimi else torch.nn.Linear(H, E, bias=False)
        self.experts = torch.nn.ModuleList(_Mlp(H, 16) for _ in range(E))

    def forward(self, h):
        x = h.view(-1, h.shape[-1])
        if self.kimi:
            idx, w, _ = self.gate(h)
        else:  # Mixtral: softmax, top-k, renormalise
            p = self.gate(x).softmax(-1)
            w, idx = p.topk(self.cfg.num_experts_per_tok, dim=-1)
            w = w / w.sum(-1, keepdim=True)
        y = torch.zeros_like(x)
        for t in range(x.shape[0]):
            for s in range(idx.shape[1]):
                y[t] += w[t, s] * self.experts[int(idx[t, s])](x[t])
        return y.view_as(h)


class _Toy(torch.nn.Module):
    def __init__(self, kimi):
        super().__init__()
        self.config = _Cfg(hidden_size=16, n_routed_experts=6, num_experts_per_tok=2, routed_scaling_factor=2.0,
                           num_hidden_layers=2, hidden_act="silu")
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList()
        for _ in range(2):
            layer = torch.nn.Module()
            layer.mlp = _Block(self.config, kimi)
            self.model.layers.append(layer)

    def forward(self, h):
        for layer in self.model.layers:
            h = h + layer.mlp(h)
        return h


@pytest.mark.parametrize("kimi", [True, False], ids=["kimi-remote-code", "linear-gate-modulelist"])
def test_modulelist_experts_and_tuple_or_logit_routers(kimi, tmp_path):
    from safetensors.torch import save_file
    torch.manual_seed(0)
    toy = _Toy(kimi).eval()
    blocks = moe.find_blocks(toy)
    assert [b.layer for b in blocks] == [0, 1] and not blocks[0].fused
    h = torch.randn(1, 1, 16)
    with torch.no_grad():
        ref = toy(h)
        col = moe.Saliency(toy)
        toy(torch.randn(3, 5, 16))
        col.remove()
    assert set(col.routed_from.values()) == {"router" if kimi else "logits"}
    assert all(sum(r["freq"]) == 15 * 2 for r in col.results().values())

    # prune on disk, keeping what the probe token used, then reload: same output
    used = {}
    hooks = []
    for b in blocks:
        def hook(_m, args, kwargs, out, b=b):
            x = args[0].reshape(-1, 16)
            p = moe.parse_router_output(out, x.shape[0], 6) or moe.route_from_logits(b, toy.config, out)
            used[b.layer] = set(p[0].flatten().tolist())
        hooks.append(b.router.register_forward_hook(hook, with_kwargs=True))
    with torch.no_grad():
        toy(h)
    for x in hooks:
        x.remove()
    keep = {li: sorted(u | set([j for j in range(6) if j not in u][: 3 - len(u)])) for li, u in used.items()}
    src = tmp_path / "src"
    src.mkdir()
    save_file({k: v.contiguous() for k, v in toy.state_dict().items()}, str(src / "model.safetensors"))
    (src / "config.json").write_text(json.dumps({"n_routed_experts": 6, "num_experts_per_tok": 2}))
    ckpt.prune_checkpoint(src, tmp_path / "out", keep, 6, 2)
    from safetensors.torch import load_file
    sd = load_file(str(tmp_path / "out" / "model.safetensors"))
    small = _Toy(kimi)
    small.config.n_routed_experts = 3
    for layer in small.model.layers:
        layer.mlp = _Block(small.config, kimi)
    small.load_state_dict(sd)
    with torch.no_grad():
        torch.testing.assert_close(small.eval()(h), ref)
    assert json.loads((tmp_path / "out" / "config.json").read_text())["n_routed_experts"] == 3


def test_flint_attention_padding_window_and_sinks(models):
    """REAPer's Flint: left padding changes nothing, the window cuts off old tokens,
    and a large sink logit soaks up the attention."""
    from reaper_pipeline.models.flint import FlintAttention
    model = load(models["flint"])
    x = ids(10)
    with torch.no_grad():
        ref = model(input_ids=x).logits
        pad = torch.cat([torch.zeros(2, 3, dtype=torch.long), x], 1)
        mask = torch.cat([torch.zeros(2, 3, dtype=torch.long), torch.ones_like(x)], 1)
        got = model(input_ids=pad, attention_mask=mask).logits[:, 3:]
    torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-4)

    attn = next(m for m in model.modules() if isinstance(m, FlintAttention) and m.window)
    h = torch.randn(1, 9, model.config.hidden_size)
    pos, keys = torch.arange(9)[None], torch.ones(1, 9, dtype=torch.bool)
    with torch.no_grad():
        full = attn(h, pos, keys)
        h2 = h.clone()
        h2[:, 0] += 5.0  # token 0 is outside the last query's window (4)
        assert torch.allclose(attn(h2, pos, keys)[:, -1], full[:, -1], atol=1e-5)
        assert not torch.allclose(attn(h2, pos, keys)[:, 1], full[:, 1], atol=1e-5)
        attn.softmax_offset.fill_(80.0)
        assert attn(h, pos, keys).abs().max() < 1e-6


def test_expert_fp8_quant(models, tmp_path):
    """Only the routed experts become block fp8, the scales dequantize them back closely,
    and the config lists every other linear layer as kept in bf16."""
    from safetensors.torch import load_file

    from reaper_pipeline import quant
    src = models["qwen3_moe"]
    info = quant.quantize_experts(src, tmp_path / "q")
    assert info["bytes_after"] < info["bytes_before"] and info["expert_weights"] == 2 * 8 * 3
    orig, got = {}, {}
    for f in src.glob("*.safetensors"):
        orig.update(load_file(f))
    for f in (tmp_path / "q").glob("*.safetensors"):
        got.update(load_file(f))
    k = "model.layers.1.mlp.experts.5.down_proj.weight"
    assert got[k].dtype == torch.float8_e4m3fn
    back = quant.dequantize_block_fp8(got[k], got[k[:-6] + "weight_scale_inv"])
    assert (back - orig[k].float()).abs().max() <= orig[k].float().abs().max() * 0.07
    assert got["model.layers.1.self_attn.q_proj.weight"].dtype == orig["model.layers.1.self_attn.q_proj.weight"].dtype
    q = json.loads((tmp_path / "q" / "config.json").read_text())["quantization_config"]
    assert q["quant_method"] == "fp8" and q["weight_block_size"] == [128, 128]
    assert "model.layers.1.mlp.gate" in q["ignored_layers"] and "lm_head" in q["ignored_layers"]
    assert not any(".experts." in m for m in q["ignored_layers"])


def test_flint_cached_generation_matches_full_recompute(models):
    """fast_generate (KV cache, grouped heads) gives the same greedy tokens as recomputing the
    whole left-padded sequence every step, through the sliding window and the no-RoPE layer."""
    model = load(models["flint"])
    x = ids(10)
    mask = torch.ones_like(x)
    x[1, :3], mask[1, :3] = 0, 0  # left padding on the second row
    new = model.fast_generate(x, mask, max_new_tokens=8)
    seq, m = x.clone(), mask.clone()
    for _ in range(8):
        with torch.no_grad():
            nxt = model(input_ids=seq, attention_mask=m).logits[:, -1].argmax(-1)
        seq, m = torch.cat([seq, nxt[:, None]], 1), torch.cat([m, torch.ones_like(m[:, :1])], 1)
    assert torch.equal(new, seq[:, 10:])
    # sampling runs and respects eos padding
    g = torch.Generator().manual_seed(0)
    s = model.fast_generate(x, mask, max_new_tokens=6, temperature=0.7, top_p=0.9, eos_token_id=[int(new[0, 0])],
                            generator=g)
    assert s.shape[0] == 2 and s.shape[1] <= 6
