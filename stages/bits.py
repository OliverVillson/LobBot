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

# Bits per weight for llama.cpp quant types (block bytes * 8 / block size).
BPW = {
    "q2_k": 2.625,
    "q3_k": 3.4375,
    "q4_k": 4.5,
    "q5_k": 5.5,
    "q6_k": 6.5625,
    "q8_0": 8.5,
}
LADDER = ["q2_k", "q3_k", "q4_k", "q5_k", "q6_k"]

# Relative weight MSE, ||W - Q(W)||^2 / ||W||^2, measured by quantizing and
# dequantizing Qwen3-MoE expert tensors with llama.cpp (2026-10-03, no imatrix).
# Each step up the ladder cuts the error ~4x, i.e. ~2^(-2 * extra bits) as
# rate-distortion theory predicts. This is what an upgrade buys.
REL_MSE = {
    "q2_k": 0.0876,
    "q3_k": 0.0227,
    "q4_k": 0.00507,
    "q5_k": 0.00130,
    "q6_k": 0.000314,
    "q8_0": 0.0000196,
}

# ffn_down writes straight into the residual stream; llama.cpp's own mixes
# give it extra bits for that reason.
KIND_WEIGHT = {"gate": 1.0, "up": 1.0, "down": 1.5}

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
            # transformers 5 saves Qwen3-MoE configs with num_local_experts
            n_experts=cfg.get("num_experts") or cfg["num_local_experts"],
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


def _normalise(xs: list[float]) -> list[float]:
    """Scale to mean 1; non-finite or non-positive entries become the mean."""
    good = [x for x in xs if x == x and 0 < x < float("inf")]
    mean = sum(good) / len(good) if good else 1.0
    return [(x if x in good else mean) / mean for x in xs]


# The imatrix energy is an absolute output energy, and the residual stream grows
# with depth, so it climbs steeply with depth in every model (Qwen3-30B-A3B
# REAP-50, 2026-10-03 B200 run: ffn_down energy 0.01x the mean in early layers,
# 340x in the last). The REAP score (||moe_out|| / ||moe_in||) already measures
# a block's relative effect, so using the raw energy as well counts depth twice
# and put 111 of 144 expert tensors at q2_k to fund q5/q6 in the last layer.
# A fourth root keeps the energy's split between projections and its ordering
# while letting REAP lead; the clip stops any single layer from soaking up
# the budget.
ENERGY_POWER = 0.25
SENS_CLIP = (0.25, 4.0)


def sensitivities(
    n_layers: int,
    layer_importance: list[float] | None = None,
    proj_energy: dict[str, list[float]] | None = None,
    reap_weight: float = 1.0,
    energy_power: float = ENERGY_POWER,
) -> dict[str, list[float]]:
    """Per-layer, per-projection weight on quantization error.

    layer_importance: REAP-stage score per MoE block (how much the experts move
        the residual stream on task data). Shared by gate, up and down.
    proj_energy: task imatrix energy per projection, sum_j E[x_j^2] * ||W[:, j]||^2,
        the expected output energy of the tensor on task tokens. See
        quantize.imatrix_energy(). Damped by `energy_power`.
    Each signal is normalised to mean 1, they are multiplied, and the product
    is renormalised and clipped to SENS_CLIP; either signal may be missing.
    """
    li = _normalise(layer_importance) if layer_importance else [1.0] * n_layers
    if len(li) != n_layers:
        raise ValueError("importance must have one score per layer")
    out = {}
    for kind in ("gate", "up", "down"):
        e = _normalise(proj_energy[kind]) if proj_energy and kind in proj_energy else [1.0] * n_layers
        raw = _normalise([(li[i] ** reap_weight) * (e[i] ** energy_power) for i in range(n_layers)])
        lo, hi = SENS_CLIP
        out[kind] = [KIND_WEIGHT[kind] * min(hi, max(lo, w)) for w in raw]
    return out


def allocate(
    shape: MoEShape,
    importance: list[float] | None,
    budget_gb: float,
    floor: str = "q2_k",
    ceiling: str = "q6_k",
    proj_energy: dict[str, list[float]] | None = None,
    max_gb_per_token: float | None = None,
) -> list[LayerBits]:
    """Greedy rate-distortion allocation: start every layer at `floor`, then
    repeatedly buy the upgrade with the largest drop in weighted error
    (sensitivity x REL_MSE reduction) per extra byte, until the next upgrade
    would break the size budget (or the per-token read budget, which sets
    laptop tok/s).

    ffn_down may sit up to two steps above gate/up; gate/up never above down.
    """
    lo, hi = LADDER.index(floor), LADDER.index(ceiling)
    sens = sensitivities(shape.n_layers, importance, proj_energy)
    w_down = sens["down"]
    w_gu = [g + u for g, u in zip(sens["gate"], sens["up"])]
    frac = shape.n_experts_active / shape.n_experts

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
    per_token = bytes_per_token_gb(shape, [LayerBits(i, LADDER[lo], LADDER[lo]) for i in range(shape.n_layers)])

    def err(level: int) -> float:
        return REL_MSE[LADDER[level]]

    while True:
        cands = []
        for i in range(shape.n_layers):
            # Raise down by one (stays <= gate_up + 2).
            if down[i] < hi and down[i] < gate_up[i] + 2:
                extra = _gb(proj, BPW[LADDER[down[i] + 1]] - BPW[LADDER[down[i]]])
                gain = w_down[i] * (err(down[i]) - err(down[i] + 1))
                cands.append((gain / extra, i, "down", extra))
            # Raise gate+up by one (stays <= down).
            if gate_up[i] < hi and gate_up[i] < down[i]:
                extra = _gb(2 * proj, BPW[LADDER[gate_up[i] + 1]] - BPW[LADDER[gate_up[i]]])
                gain = w_gu[i] * (err(gate_up[i]) - err(gate_up[i] + 1))
                cands.append((gain / extra, i, "gate_up", extra))
        # Best value first; skip upgrades that do not fit, a cheaper one may.
        bought = False
        for _, i, which, extra in sorted(cands, reverse=True):
            if current + extra > budget_gb:
                continue
            if max_gb_per_token is not None and per_token + extra * frac > max_gb_per_token:
                continue
            if which == "down":
                down[i] += 1
            else:
                gate_up[i] += 1
            current += extra
            per_token += extra * frac
            bought = True
            break
        if not bought:
            break

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
    """bit_widths entries for eval.json (consumed by the scoreboard screen).
    quantize.py replaces this with the types actually written to the GGUF."""
    out = []
    for l in layers:
        for tensor, t in (("ffn_gate_exps", l.gate_up), ("ffn_up_exps", l.gate_up), ("ffn_down_exps", l.down)):
            out.append({"layer": l.layer, "tensor": tensor, "type": t, "bits": BPW[t]})
    return out
