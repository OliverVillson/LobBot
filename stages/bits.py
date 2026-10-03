"""Dynamic (saliency-weighted) bit allocation for MoE expert tensors.

Pure Python, no GPU: given per-layer importance scores and the model shape,
choose a llama.cpp quant type for each layer's stacked expert tensors so the
whole GGUF lands under a size budget. Salient layers get more bits, weak
layers fewer. Everything outside the experts gets a fixed ("static") type.

GGUF stores all experts of a layer in one tensor (blk.N.ffn_{gate,up,down}_exps),
so the finest granularity llama-quantize can vary is per layer and per projection.
"""

from __future__ import annotations

from dataclasses import dataclass

# Approximate bits per weight for llama.cpp quant types.
BPW = {
    "q2_k": 2.625,
    "q3_k": 3.4375,
    "q4_k": 4.5,
    "q5_k": 5.5,
    "q6_k": 6.5625,
    "q8_0": 8.5,
}
LADDER = ["q2_k", "q3_k", "q4_k", "q5_k", "q6_k"]

# Static types for everything outside the experts. Attention and the output
# head are read on every token, so their bits cost speed directly.
STATIC = {
    "attn": "q5_k",
    "output": "q6_k",
    "token_embd": "q4_k",
}


@dataclass
class MoEShape:
    n_layers: int
    hidden: int
    moe_intermediate: int
    n_experts: int
    n_experts_active: int
    vocab: int
    n_heads: int
    n_kv_heads: int
    head_dim: int

    @classmethod
    def from_hf_config(cls, cfg: dict) -> "MoEShape":
        n_heads = cfg["num_attention_heads"]
        return cls(
            n_layers=cfg["num_hidden_layers"],
            hidden=cfg["hidden_size"],
            moe_intermediate=cfg["moe_intermediate_size"],
            n_experts=cfg["num_experts"],
            n_experts_active=cfg["num_experts_per_tok"],
            vocab=cfg["vocab_size"],
            n_heads=n_heads,
            n_kv_heads=cfg.get("num_key_value_heads", n_heads),
            head_dim=cfg.get("head_dim") or cfg["hidden_size"] // n_heads,
        )

    # Parameter counts.
    @property
    def expert_params_per_proj(self) -> int:
        """One projection (gate, up or down) across all experts of one layer."""
        return self.n_experts * self.hidden * self.moe_intermediate

    @property
    def attn_params_per_layer(self) -> int:
        q = self.hidden * self.n_heads * self.head_dim
        kv = 2 * self.hidden * self.n_kv_heads * self.head_dim
        o = self.n_heads * self.head_dim * self.hidden
        return q + kv + o

    @property
    def embd_params(self) -> int:
        return self.vocab * self.hidden

    @property
    def total_params(self) -> int:
        experts = 3 * self.expert_params_per_proj * self.n_layers
        attn = self.attn_params_per_layer * self.n_layers
        return experts + attn + 2 * self.embd_params


def _gb(params: float, bpw: float) -> float:
    return params * bpw / 8 / 1e9


def static_size_gb(shape: MoEShape) -> float:
    return (
        _gb(shape.attn_params_per_layer * shape.n_layers, BPW[STATIC["attn"]])
        + _gb(shape.embd_params, BPW[STATIC["output"]])
        + _gb(shape.embd_params, BPW[STATIC["token_embd"]])
    )


@dataclass
class LayerBits:
    layer: int
    gate_up: str
    down: str


def allocate(
    shape: MoEShape,
    importance: list[float],
    budget_gb: float,
    floor: str = "q2_k",
    ceiling: str = "q6_k",
) -> list[LayerBits]:
    """Greedy allocation: start every layer at `floor`, then repeatedly buy the
    upgrade with the best (importance x error reduction) per extra byte until
    the next upgrade would exceed the budget.

    ffn_down is upgraded first and may sit up to two steps above gate/up,
    since it is usually the most quantization-sensitive projection.
    """
    if len(importance) != shape.n_layers:
        raise ValueError("importance must have one score per layer")
    lo, hi = LADDER.index(floor), LADDER.index(ceiling)
    total_w = sum(importance) or 1.0
    w = [x / total_w for x in importance]

    gate_up = [lo] * shape.n_layers
    down = [lo] * shape.n_layers
    proj = shape.expert_params_per_proj

    def size() -> float:
        s = static_size_gb(shape)
        for i in range(shape.n_layers):
            s += _gb(2 * proj, BPW[LADDER[gate_up[i]]]) + _gb(proj, BPW[LADDER[down[i]]])
        return s

    current = size()
    if current > budget_gb:
        raise ValueError(f"budget {budget_gb:.2f} GB is below the floor size {current:.2f} GB")

    def err(level: int) -> float:
        return 2.0 ** -BPW[LADDER[level]]

    while True:
        best = None
        for i in range(shape.n_layers):
            # Candidate 1: raise down by one (stays <= gate_up + 2).
            if down[i] < hi and down[i] < gate_up[i] + 2:
                extra = _gb(proj, BPW[LADDER[down[i] + 1]] - BPW[LADDER[down[i]]])
                gain = 1.5 * w[i] * (err(down[i]) - err(down[i] + 1))
                cand = (gain / extra, i, "down", extra)
                best = cand if best is None or cand[0] > best[0] else best
            # Candidate 2: raise gate+up by one (stays <= down).
            if gate_up[i] < hi and gate_up[i] < down[i]:
                extra = _gb(2 * proj, BPW[LADDER[gate_up[i] + 1]] - BPW[LADDER[gate_up[i]]])
                gain = 2 * w[i] * (err(gate_up[i]) - err(gate_up[i] + 1))
                cand = (gain / extra, i, "gate_up", extra)
                best = cand if best is None or cand[0] > best[0] else best
        if best is None or current + best[3] > budget_gb:
            break
        _, i, which, extra = best
        if which == "down":
            down[i] += 1
        else:
            gate_up[i] += 1
        current += extra

    return [LayerBits(i, LADDER[gate_up[i]], LADDER[down[i]]) for i in range(shape.n_layers)]


def estimate_size_gb(shape: MoEShape, layers: list[LayerBits]) -> float:
    proj = shape.expert_params_per_proj
    return static_size_gb(shape) + sum(
        _gb(2 * proj, BPW[l.gate_up]) + _gb(proj, BPW[l.down]) for l in layers
    )


def bytes_per_token_gb(shape: MoEShape, layers: list[LayerBits]) -> float:
    """Weights read per generated token: active experts, attention, output head."""
    frac = shape.n_experts_active / shape.n_experts
    proj = shape.expert_params_per_proj
    experts = sum(
        _gb(2 * proj * frac, BPW[l.gate_up]) + _gb(proj * frac, BPW[l.down]) for l in layers
    )
    attn = _gb(shape.attn_params_per_layer * shape.n_layers, BPW[STATIC["attn"]])
    head = _gb(shape.embd_params, BPW[STATIC["output"]])
    return experts + attn + head


def estimate_tok_s(shape: MoEShape, layers: list[LayerBits], bandwidth_gb_s: float = 120.0, efficiency: float = 0.65) -> float:
    """Bandwidth-bound decode estimate. Default: MacBook Air M4 (~120 GB/s)."""
    return bandwidth_gb_s * efficiency / bytes_per_token_gb(shape, layers)


def quantize_args(layers: list[LayerBits]) -> list[str]:
    """llama-quantize flags implementing an allocation."""
    args = [
        "--output-tensor-type", STATIC["output"],
        "--token-embedding-type", STATIC["token_embd"],
        "--tensor-type", f"attn_(q|k|v|output)\\.weight={STATIC['attn']}",
    ]
    for l in layers:
        args += ["--tensor-type", f"blk\\.{l.layer}\\.ffn_(gate|up)_exps={l.gate_up}"]
        args += ["--tensor-type", f"blk\\.{l.layer}\\.ffn_down_exps={l.down}"]
    return args


def heatmap(layers: list[LayerBits]) -> list[dict]:
    """bit_widths entries for eval.json (consumed by the scoreboard screen)."""
    out = []
    for l in layers:
        for tensor, t in (("ffn_gate_exps", l.gate_up), ("ffn_up_exps", l.gate_up), ("ffn_down_exps", l.down)):
            out.append({"layer": l.layer, "tensor": tensor, "type": t, "bits": BPW[t]})
    return out
