"""Version-tolerant access to the sparse MoE blocks of Qwen3-MoE style models
(Qwen3MoeSparseMoeBlock / Qwen2MoeSparseMoeBlock / Mixtral), covering both the
transformers 4.x layout (experts = ModuleList of MLPs, gate = nn.Linear) and the
fused 5.x layout (experts = one module holding [E, ...] stacked weights,
gate = a router module with a [E, H] weight)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def find_moe_blocks(model) -> list[tuple[int, nn.Module]]:
    """(decoder layer index, moe block) for every sparse layer."""
    layers = model.model.layers
    out = []
    for i, layer in enumerate(layers):
        mlp = getattr(layer, "mlp", None) or getattr(layer, "block_sparse_moe", None)
        if mlp is not None and hasattr(mlp, "experts") and hasattr(mlp, "gate"):
            out.append((i, mlp))
    return out


def gate_weight(block) -> torch.Tensor:
    return block.gate.weight  # [E, H] for both nn.Linear and router modules


def num_experts(block) -> int:
    return gate_weight(block).shape[0]


def top_k(block, config) -> int:
    for obj, name in ((block, "top_k"), (block.gate, "top_k"), (config, "num_experts_per_tok")):
        v = getattr(obj, name, None)
        if isinstance(v, int):
            return v
    raise AttributeError("cannot find top_k")


def route(block, config, x: torch.Tensor):
    """Replicates the HF router: softmax over all experts, top-k, optional
    renormalisation. x: [N, H]. Returns (weights [N, k], indices [N, k])."""
    logits = F.linear(x, gate_weight(block))
    probs = F.softmax(logits, dim=-1, dtype=torch.float)
    w, idx = torch.topk(probs, top_k(block, config), dim=-1)
    if getattr(config, "norm_topk_prob", False):
        w = w / w.sum(dim=-1, keepdim=True)
    return w, idx


def _act(experts, config):
    fn = getattr(experts, "act_fn", None)
    if fn is not None:
        return fn
    from transformers.activations import ACT2FN
    return ACT2FN[config.hidden_act]


def expert_forward(block, config, j: int, x: torch.Tensor) -> torch.Tensor:
    """Output of expert j on tokens x [n, H] -> [n, H] (without gate weight)."""
    experts = block.experts
    if isinstance(experts, nn.ModuleList):
        return experts[j](x)
    act = _act(experts, config)
    if hasattr(experts, "gate_up_proj"):
        gu = experts.gate_up_proj[j]  # [2I, H] (5.x) or [H, 2I] (some variants)
        dn = experts.down_proj[j]
        H = x.shape[-1]
        h = F.linear(x, gu) if gu.shape[-1] == H else x @ gu
        g, u = h.chunk(2, dim=-1)
        h = act(g) * u
        return F.linear(h, dn) if dn.shape[0] == H else h @ dn
    raise NotImplementedError(f"unknown experts layout: {type(experts).__name__}")


@torch.no_grad()
def prune_block(block, keep: list[int]) -> None:
    """Keep only the experts in `keep` (original ids, ascending), in place."""
    keep_t = torch.tensor(keep, dtype=torch.long, device=gate_weight(block).device)
    E = num_experts(block)
    n = len(keep)

    # router
    gate = block.gate
    if isinstance(gate, nn.Linear):
        new = nn.Linear(gate.in_features, n, bias=gate.bias is not None,
                        device=gate.weight.device, dtype=gate.weight.dtype)
        new.weight.copy_(gate.weight[keep_t])
        if gate.bias is not None:
            new.bias.copy_(gate.bias[keep_t])
        block.gate = new
    else:
        for name, p in list(gate.named_parameters(recurse=False)):
            if p.shape[0] == E:
                setattr(gate, name, nn.Parameter(p.data[keep_t].clone(), requires_grad=p.requires_grad))
        for name, b in list(gate.named_buffers(recurse=False)):
            if b.shape and b.shape[0] == E:
                setattr(gate, name, b[keep_t].clone())
    for obj in (gate, block):
        for attr in ("num_experts", "n_routed_experts", "num_local_experts"):
            if isinstance(getattr(obj, attr, None), int):
                setattr(obj, attr, n)

    # experts
    experts = block.experts
    if isinstance(experts, nn.ModuleList):
        block.experts = nn.ModuleList([experts[j] for j in keep])
    else:
        for name, p in list(experts.named_parameters(recurse=False)):
            if p.shape[0] == E:
                setattr(experts, name, nn.Parameter(p.data[keep_t.to(p.device)].clone(),
                                                    requires_grad=p.requires_grad))
        for attr in ("num_experts", "num_local_experts"):
            if isinstance(getattr(experts, attr, None), int):
                setattr(experts, attr, n)


def set_config_experts(config, n: int) -> None:
    for attr in ("num_experts", "num_local_experts", "n_routed_experts"):
        if getattr(config, attr, None) is not None:
            setattr(config, attr, n)


def write_expert_count_alias(model_dir) -> None:
    """transformers 5 saves the expert count as num_local_experts; the Qwen
    checkpoints and our quantize stage use num_experts. Write both."""
    import json
    from pathlib import Path
    p = Path(model_dir) / "config.json"
    cfg = json.loads(p.read_text())
    n = cfg.get("num_experts") or cfg.get("num_local_experts")
    if n is None:
        return
    cfg["num_experts"] = n
    cfg["num_local_experts"] = n
    p.write_text(json.dumps(cfg, indent=2) + "\n")
