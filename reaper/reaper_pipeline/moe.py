"""Find and read the sparse MoE blocks of any Hugging Face MoE model.

Nothing here is written per architecture. A MoE block is any module with an
`experts` child and a router child (`gate`, `router`, ...) whose weight is
[E, hidden], where the experts hold E experts either as a ModuleList
(transformers 4.x, remote code) or as one fused module with [E, ...] stacked
weights (transformers 5.x, gpt-oss).

Routing is read from what the router itself returns whenever it returns the
chosen experts and their weights (DeepSeek-V3, GLM-4.5, Kimi K2, gpt-oss and
every transformers 5.x router), so grouped top-k, correction biases and scaling
factors come out exactly as the model uses them. A plain linear gate returns
logits only; then the routing is recomputed from the config: sigmoid with the
correction bias when the block has one (MiniMax-M2), softmax top-k otherwise
(Qwen2/3-MoE, Qwen3-Next, Mixtral, OLMoE), renormalised as the config says.

Covered: Qwen3-MoE, Qwen3-Next, gpt-oss, GLM-4.5/4.6, DeepSeek-V3, Kimi K2,
Mixtral, MiniMax-M2, OLMoE and lookalikes. Llama 4 (router scales the expert
input) and Granite-MoE (no `experts` module) are not.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

ROUTER_NAMES = ("gate", "router", "gating", "gate_layer", "moe_gate")
EXPERT_COUNT_KEYS = ("num_experts", "num_local_experts", "n_routed_experts", "moe_num_experts")


def text_config(config):
    get = getattr(config, "get_text_config", None)
    return get() if callable(get) else getattr(config, "text_config", None) or config


def router_weight(router: nn.Module) -> torch.Tensor | None:
    w = getattr(router, "weight", None)
    return w if isinstance(w, torch.Tensor) and w.dim() == 2 else None


def count_experts(experts: nn.Module) -> int:
    if isinstance(experts, nn.ModuleList):
        return len(experts)
    for attr in ("num_experts", "num_local_experts"):
        if isinstance(getattr(experts, attr, None), int):
            return getattr(experts, attr)
    for p in experts.parameters(recurse=False):
        if p.dim() >= 2:
            return p.shape[0]
    return 0


@dataclass
class MoeBlock:
    name: str            # module path, e.g. model.layers.3.mlp
    layer: int           # decoder layer index (checkpoint keys use the same)
    module: nn.Module
    router_name: str
    num_experts: int
    _strategy: str | None = field(default=None, repr=False)

    @property
    def router(self) -> nn.Module:
        return getattr(self.module, self.router_name)

    @property
    def experts(self) -> nn.Module:
        return self.module.experts

    @property
    def fused(self) -> bool:
        return not isinstance(self.experts, nn.ModuleList)


def find_blocks(model: nn.Module) -> list[MoeBlock]:
    out: list[MoeBlock] = []
    for name, mod in model.named_modules():
        experts = getattr(mod, "experts", None)
        if not isinstance(experts, nn.Module):
            continue
        for rn in ROUTER_NAMES:
            r = getattr(mod, rn, None)
            w = router_weight(r) if isinstance(r, nn.Module) else None
            if w is None:
                continue
            E = w.shape[0]
            if E > 1 and count_experts(experts) == E:
                layers = re.findall(r"layers\.(\d+)", name)
                out.append(MoeBlock(name, int(layers[-1]) if layers else len(out), mod, rn, E))
                break
    return out


def top_k(block: MoeBlock, config) -> int:
    c = text_config(config)
    for obj, attr in ((block.module, "top_k"), (block.router, "top_k"), (c, "num_experts_per_tok"),
                      (c, "moe_k"), (c, "num_selected_experts")):
        v = getattr(obj, attr, None)
        if isinstance(v, int) and v > 0:
            return v
    raise AttributeError(f"{block.name}: cannot find top-k")


def n_groups(config) -> int:
    """Expert groups for DeepSeek-style group-limited routing (1 = none)."""
    g = getattr(text_config(config), "n_group", None)
    return g if isinstance(g, int) and g > 1 else 1


def _correction_bias(block: MoeBlock) -> torch.Tensor | None:
    for owner in (block.router, block.module):
        b = getattr(owner, "e_score_correction_bias", None)
        if isinstance(b, torch.Tensor) and b.numel() == block.num_experts:
            return b
    return None


# --------------------------------------------------------------------------- routing


def parse_router_output(out, n_tokens: int, E: int):
    """(indices [N, k], weights [N, k]) when the router returned its choice, else None."""
    ts = [t for t in (out if isinstance(out, (tuple, list)) else (out,)) if isinstance(t, torch.Tensor)]
    idx = next((t for t in ts if not t.is_floating_point() and t.dim() == 2 and t.shape[0] == n_tokens
                and t.shape[1] < E), None)
    if idx is None:
        return None
    w = next((t for t in reversed(ts) if t.is_floating_point() and t.shape == idx.shape), None)
    if w is None:  # dense scores with zeros off the chosen experts (gpt-oss 4.x)
        dense = next((t for t in ts if t.is_floating_point() and t.dim() == 2 and t.shape == (n_tokens, E)), None)
        if dense is None:
            return None
        w = dense.gather(1, idx.long())
    return idx.long(), w


def route_from_logits(block: MoeBlock, config, logits: torch.Tensor):
    """Recompute the routing of a plain linear gate from its logits [N, E]."""
    c = text_config(config)
    k = top_k(block, config)
    logits = logits.float()
    bias = _correction_bias(block)
    if bias is not None or getattr(c, "scoring_func", "softmax") == "sigmoid":
        scores = logits.sigmoid()
        choice = scores + (bias.float().to(scores.device) if bias is not None else 0)
        g = n_groups(config)
        if g > 1:
            N, E = choice.shape
            grp = choice.view(N, g, E // g).topk(2, dim=-1)[0].sum(-1)
            keep = grp.topk(getattr(c, "topk_group", g), dim=-1)[1]
            mask = torch.zeros_like(grp).scatter_(1, keep, 1).unsqueeze(-1).expand(N, g, E // g).reshape(N, E)
            choice = choice.masked_fill(mask == 0, float("-inf"))
        idx = choice.topk(k, dim=-1)[1]
        w = scores.gather(1, idx)
        if getattr(c, "norm_topk_prob", True):
            w = w / (w.sum(-1, keepdim=True) + 1e-20)
    else:
        probs = logits.softmax(-1)
        w, idx = probs.topk(k, dim=-1)
        # Mixtral-style configs carry no flag and always renormalise
        if getattr(c, "norm_topk_prob", True):
            w = w / w.sum(-1, keepdim=True)
    w = w * float(getattr(c, "routed_scaling_factor", 1.0) or 1.0)
    return idx, w


# --------------------------------------------------------------------------- expert outputs


def _call_fused(experts, x, idx, w):
    out = experts(x, idx, w)
    out = out[0] if isinstance(out, tuple) else out
    return out.reshape(-1, x.shape[-1])


def _act(experts, config):
    fn = getattr(experts, "act_fn", None)
    if fn is not None:
        return fn
    from transformers.activations import ACT2FN
    return ACT2FN[text_config(config).hidden_act]


def _manual(block: MoeBlock, config, j: int, x: torch.Tensor) -> torch.Tensor:
    """Stacked gate_up_proj/down_proj (or gate_proj/up_proj/down_proj) maths."""
    ex = block.experts
    H = x.shape[-1]

    def lin(t, W, b=None):
        y = F.linear(t, W) if W.shape[-1] == t.shape[-1] else t @ W
        return y + b if b is not None else y

    def bias(name):
        b = getattr(ex, name, None)
        return b[j] if isinstance(b, torch.Tensor) else None

    if hasattr(ex, "gate_up_proj"):
        h = lin(x, ex.gate_up_proj[j], bias("gate_up_proj_bias"))
        g, u = h.chunk(2, dim=-1)
    else:
        g = lin(x, ex.gate_proj[j], bias("gate_proj_bias"))
        u = lin(x, ex.up_proj[j], bias("up_proj_bias"))
    h = _act(ex, config)(g) * u
    dn = ex.down_proj[j]
    y = lin(h, dn) if (dn.shape[-1] == h.shape[-1] or dn.shape[0] != H) else h @ dn
    b = bias("down_proj_bias")
    return y + b if b is not None else y


def _choose_strategy(block: MoeBlock, config, x: torch.Tensor) -> str:
    """Pick how to get one expert's output from a fused experts module.

    Asking the module itself (forced routing to a single expert, weight 1) is
    preferred because it runs the model's own kernel and activation (gpt-oss
    clamps and interleaves). It is accepted only if two single-expert calls
    combine linearly into the module's own two-expert answer."""
    ex = block.experts
    E = block.num_experts
    x2 = x[:2]
    n = x2.shape[0]
    a, b = 0, min(1, E - 1)
    pair = torch.tensor([[a, b]] * n, device=x.device)
    pw = torch.tensor([[0.3, 0.7]] * n, device=x.device, dtype=x.dtype)

    def single(j, dense):
        idx = torch.full((n, 1), j, device=x.device, dtype=torch.long)
        w = (F.one_hot(idx[:, 0], E).to(x.dtype) if dense else torch.ones(n, 1, device=x.device, dtype=x.dtype))
        return _call_fused(ex, x2, idx, w)

    for dense in (False, True):
        try:
            with torch.no_grad():
                ya, yb = single(a, dense), single(b, dense)
                w2 = pw if not dense else torch.zeros(n, E, device=x.device, dtype=x.dtype).scatter_(1, pair, pw)
                yab = _call_fused(ex, x2, pair, w2)
            ref = 0.3 * ya.float() + 0.7 * yb.float()
            if ya.abs().sum() > 0 and torch.allclose(yab.float(), ref, rtol=2e-2,
                                                     atol=2e-2 * (ref.abs().max().item() + 1e-6)):
                return "call_dense" if dense else "call_topk"
        except Exception:
            continue
    try:
        with torch.no_grad():
            _manual(block, config, a, x2)
        return "manual"
    except Exception as e:
        raise NotImplementedError(f"{block.name}: cannot evaluate experts of {type(ex).__name__}: {e}") from e


def expert_forward(block: MoeBlock, config, j: int, x: torch.Tensor) -> torch.Tensor:
    """Output of expert j (no router weight) on tokens x [n, H] -> [n, H]."""
    if not block.fused:
        return block.experts[j](x)
    if block._strategy is None:
        block._strategy = _choose_strategy(block, config, x)
    s = block._strategy
    if s == "manual":
        return _manual(block, config, j, x)
    n = x.shape[0]
    idx = torch.full((n, 1), j, device=x.device, dtype=torch.long)
    if s == "call_dense":
        w = F.one_hot(idx[:, 0], block.num_experts).to(x.dtype)
    else:
        w = torch.ones(n, 1, device=x.device, dtype=x.dtype)
    return _call_fused(block.experts, x, idx, w)


# --------------------------------------------------------------------------- REAP saliency


class Saliency:
    """REAP scores collected with forward hooks on every router.

        S_j = mean over tokens routed to j of  g_j(x) * ||f_j(x)||_2

    g_j is the router weight the model actually applies and f_j the expert
    output. Set `mask` ([N] bool over the flattened batch) before a forward to
    leave padding out. Outputs are unchanged."""

    def __init__(self, model: nn.Module, blocks: list[MoeBlock] | None = None):
        self.config = model.config
        self.blocks = blocks if blocks is not None else find_blocks(model)
        if not self.blocks:
            raise RuntimeError("no MoE blocks found: no module has an `experts` child next to a "
                               "[num_experts, hidden] router; is this a mixture-of-experts model?")
        self.mask: torch.Tensor | None = None
        self.sum: dict[int, torch.Tensor] = {}
        self.freq: dict[int, torch.Tensor] = {}
        self.routed_from = {}
        self.handles = []
        for b in self.blocks:
            self.sum[b.layer] = torch.zeros(b.num_experts, dtype=torch.float64)
            self.freq[b.layer] = torch.zeros(b.num_experts, dtype=torch.float64)
            self.handles.append(b.router.register_forward_hook(self._hook(b), with_kwargs=True))

    def _hook(self, blk: MoeBlock):
        @torch.no_grad()
        def hook(_mod, args, kwargs, out):
            x = args[0] if args else next(iter(kwargs.values()))
            x = x.reshape(-1, x.shape[-1])
            parsed = parse_router_output(out, x.shape[0], blk.num_experts)
            if parsed is None:
                logits = out[0] if isinstance(out, (tuple, list)) else out
                idx, w = route_from_logits(blk, self.config, logits.reshape(-1, blk.num_experts))
                self.routed_from[blk.layer] = "logits"
            else:
                idx, w = parsed
                self.routed_from[blk.layer] = "router"
            if self.mask is not None and self.mask.numel() == x.shape[0]:
                m = self.mask.to(x.device)
                x, idx, w = x[m], idx[m], w[m]
            s = torch.zeros(blk.num_experts, dtype=torch.float64, device=x.device)
            f = torch.zeros(blk.num_experts, dtype=torch.float64, device=x.device)
            for j in torch.unique(idx).tolist():
                tok, slot = torch.where(idx == j)
                y = expert_forward(blk, self.config, j, x[tok])
                s[j] += (w[tok, slot].float() * y.float().norm(dim=-1)).sum().double()
                f[j] += tok.numel()
            self.sum[blk.layer] += s.cpu()
            self.freq[blk.layer] += f.cpu()
        return hook

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []

    def results(self) -> dict[int, dict]:
        return {li: {"saliency": (self.sum[li] / self.freq[li].clamp(min=1)).tolist(),
                     "freq": self.freq[li].tolist()} for li in self.sum}


def choose_keep(saliency: list[float], freq: list[float], n_keep: int, groups: int = 1) -> list[int]:
    """Highest-saliency experts (ties: more tokens, lower id). With expert
    groups (DeepSeek-V3 group-limited routing) the same number is kept in every
    group so the group structure still holds after pruning."""
    E = len(saliency)
    rank = lambda j: (saliency[j], freq[j], -j)  # noqa: E731
    if groups <= 1:
        return sorted(sorted(range(E), key=rank, reverse=True)[:n_keep])
    size, per = E // groups, n_keep // groups
    keep: list[int] = []
    for g in range(groups):
        ids = range(g * size, (g + 1) * size)
        keep += sorted(ids, key=rank, reverse=True)[:per]
    return sorted(keep)


def experts_to_keep(E: int, prune: float, k: int, groups: int = 1, topk_group: int | None = None) -> int:
    """Experts kept per layer for a prune ratio, never below what top-k routing needs."""
    n = E - int(round(E * prune))
    floor = k
    if groups > 1:
        per = -(-k // (topk_group or groups))  # chosen groups must still hold k experts
        floor = max(k, per * groups)
        n = max(floor, n // groups * groups)
    return max(floor, min(E, n))
