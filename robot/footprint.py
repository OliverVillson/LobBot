"""Compression candidates and their weight footprint for GR00T N1.7.

Pure Python, no GPU: a candidate is a recipe (denoising steps, DiT blocks to
drop, a precision per layer group) and its footprint is the weight bytes that
recipe would ship. Works from a shape table (layer group -> param count) or,
when a torch model is given, from the real parameter counts.

Layer groups (shared by gr00t_policy.py and compress.py):

    vision          Qwen3-VL vision tower (backbone ...visual.*)
    llm.embed       token embedding
    llm.{i}.qkv     LLM layer i q/k/v projections   (i < select_layer = 12)
    llm.{i}.o       LLM layer i o_proj
    llm.{i}.gate_up LLM layer i gate/up projections
    llm.{i}.down    LLM layer i down_proj
    dit.{j}.attn    DiT block j attention (attn1)
    dit.{j}.ff      DiT block j everything else (ada norm, feed-forward)
    proj            action head outside the DiT blocks: state/action encoders,
                    action decoder, vlln, position embedding, DiT in/out layers

NVIDIA's Thor recipe keeps o_proj, down_proj and DiT attention at FP8 and puts
the rest of the LLM and DiT at NVFP4, so those groups never go below fp8 here
(see ALLOWED).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Effective bits per weight.
#   fp8:   e4m3 values plus one fp32 scale per output channel; 32 bits over a
#          ~2048-wide row is ~0.016 bits per weight.
#   nvfp4: e2m1 values in blocks of 16 along the input dim, one e4m3 scale per
#          block: 4 + 8/16 = 4.5 bits.
BPW = {"bf16": 16.0, "fp8": 8.0 + 32 / 2048, "nvfp4": 4.5}
LADDER = ["nvfp4", "fp8", "bf16"]  # cheapest first
NVFP4_BLOCK = 16


@dataclass
class Candidate:
    name: str  # "teacher", "steps2", "fp8", "mixed", "pruned"...
    denoise_steps: int = 4
    drop_dit_blocks: list[int] = field(default_factory=list)
    precision: dict[str, str] = field(default_factory=dict)  # group -> "bf16" | "fp8" | "nvfp4"
    note: str = ""
    footprint: dict = field(default_factory=dict)  # filled by compress.build_candidates


def allowed(group: str) -> list[str]:
    """Precisions a group may take, cheapest first (the Thor recipe floors)."""
    if re.fullmatch(r"llm\.\d+\.(o|down)", group) or re.fullmatch(r"dit\.\d+\.attn", group):
        return ["fp8", "bf16"]
    if group in ("llm.embed", "proj"):
        return ["fp8", "bf16"]
    return list(LADDER)


# ---------------------------------------------------------------------------
# Default shape table. APPROXIMATE: hand-computed from the Isaac-GR00T N1.7
# config defaults (gr00t/configs/model/gr00t_n1d7.py: select_layer=12, DiT
# num_layers=16, 32 heads x 48 = 1536 wide, cross-attention dim 2048,
# max_state/action_dim 132, 32 embodiments) and Cosmos-Reason2-2B (Qwen3-VL-2B:
# hidden 2048, intermediate 6144, 16 q heads / 8 kv heads x 128, vocab 151936).
# The backbone drops LLM layers >= select_layer at load
# (gr00t/model/modules/qwen3_backbone.py), so only 12 LLM layers ship.
# Overridden by real counts (shapes_from_model) whenever the model is loaded.
# ---------------------------------------------------------------------------
LLM_LAYERS = 12
DIT_BLOCKS = 16
_H, _I, _QD, _KVD, _VOCAB = 2048, 6144, 16 * 128, 8 * 128, 151936
_D, _XD = 32 * 48, 2048


def _default_shapes() -> dict[str, int]:
    s: dict[str, int] = {"vision": 400_000_000, "llm.embed": _VOCAB * _H}
    for i in range(LLM_LAYERS):
        s[f"llm.{i}.qkv"] = _H * _QD + 2 * _H * _KVD
        s[f"llm.{i}.o"] = _QD * _H
        s[f"llm.{i}.gate_up"] = 2 * _H * _I
        s[f"llm.{i}.down"] = _I * _H
    for j in range(DIT_BLOCKS):
        kv_in = _XD if j % 2 == 0 else _D  # even blocks cross-attend to the VL tokens
        s[f"dit.{j}.attn"] = 2 * _D * _D + 2 * kv_in * _D
        s[f"dit.{j}.ff"] = 2 * _D * 4 * _D + _D * 2 * _D  # gelu FF (x4) + AdaLayerNorm linear
    cats, sd, ad, hid, emb = 32, 132, 132, 1024, 1536
    s["proj"] = (
        cats * (sd * hid + hid * emb)  # state_encoder
        + cats * (ad * emb + 2 * emb * emb + emb * emb)  # action_encoder W1..W3
        + cats * (hid * hid + hid * ad)  # action_decoder
        + _D * 2 * _D + _D * hid + 256 * _D + _D * _D  # DiT proj_out_1/2, timestep encoder
    )
    return s


DEFAULT_SHAPES: dict[str, int] = _default_shapes()  # approximate, ~2.2B params


# ---------------------------------------------------------------------------
# Parameter name -> layer group (real model). Names follow the module tree of
# gr00t/model/gr00t_n1d7/gr00t_n1d7.py: Gr00tN1d7.backbone (Qwen3Backbone,
# .model = Qwen3VLForConditionalGeneration) and Gr00tN1d7.action_head
# (.model = AlternateVLDiT with .transformer_blocks).
# Real names (checked on a tiny random Gr00tN1d7 by tests/test_gr00t_compat.py):
#   backbone.model.model.language_model.layers.{i}.self_attn.{q,k,v,o}_proj.weight
#   backbone.model.model.language_model.layers.{i}.mlp.{gate,up,down}_proj.weight
#   backbone.model.model.language_model.embed_tokens.weight, backbone.model.model.visual.*
#   backbone.model.lm_head.weight (unused: GR00T reads hidden states, never logits)
#   action_head.model.transformer_blocks.{j}.attn1.*  /  .norm1.*, .ff.*
#   action_head.{state_encoder,action_encoder,action_decoder}.*.W (CategorySpecificLinear)
# ---------------------------------------------------------------------------
_LLM_RE = re.compile(r"language_model\.layers\.(\d+)\.(self_attn|mlp)\.(\w+)_proj\.")
_DIT_RE = re.compile(r"^action_head\.model\.transformer_blocks\.(\d+)\.(attn1\.)?")
_LLM_KIND = {"q": "qkv", "k": "qkv", "v": "qkv", "o": "o", "gate": "gate_up", "up": "gate_up", "down": "down"}


def group_of(param_name: str) -> str | None:
    """Layer group for a parameter name, or None (kept bf16, e.g. norms, lm_head)."""
    m = _LLM_RE.search(param_name)
    if m:
        return f"llm.{m.group(1)}.{_LLM_KIND[m.group(3)]}"
    if param_name.startswith("backbone."):
        if ".visual." in param_name:
            return "vision"
        if "embed_tokens" in param_name:
            return "llm.embed"
        return None
    m = _DIT_RE.match(param_name)
    if m:
        return f"dit.{m.group(1)}.{'attn' if m.group(2) else 'ff'}"
    if param_name.startswith("action_head."):
        return "proj"
    return None


def is_quantizable(param_name: str, ndim: int) -> bool:
    """Matrix weights only; biases and norm scales stay bf16."""
    leaf = param_name.rsplit(".", 1)[-1]
    return ndim >= 2 and leaf in ("weight", "W")


def shapes_from_model(model) -> dict[str, int]:
    """Real per-group counts of quantizable weights. Everything else is
    reported under "other" (bf16). Tied weights are counted once."""
    out: dict[str, int] = {}
    seen: set[int] = set()
    for name, p in model.named_parameters():  # deduplicates tied weights
        if id(p) in seen:
            continue
        seen.add(id(p))
        g = group_of(name) if is_quantizable(name, p.ndim) else None
        if g is None and "lm_head" in name:
            continue  # unused: GR00T reads layer-12 hidden states, never logits
        key = g or "other"
        out[key] = out.get(key, 0) + p.numel()
    return out


def _count(shapes: dict[str, int], precision: dict[str, str], drop: list[int]) -> tuple[int, float, dict]:
    dropped = {f"dit.{j}.{k}" for j in drop for k in ("attn", "ff")}
    params, bits, by_prec = 0, 0.0, {}
    for g, n in shapes.items():
        if g in dropped:
            continue
        prec = precision.get(g, "bf16")
        params += n
        bits += n * BPW[prec]
        by_prec[prec] = by_prec.get(prec, 0) + n
    return params, bits, by_prec


def footprint(model_or_shapes, cand: Candidate) -> dict:
    """Weight params, size in GB and mean bits per weight for a candidate.

    model_or_shapes: dict of layer group -> param count (DEFAULT_SHAPES or
    shapes_from_model output), or a torch model. Groups the candidate does not
    name are bf16; dropped DiT blocks remove their params.
    """
    shapes = model_or_shapes if isinstance(model_or_shapes, dict) else shapes_from_model(model_or_shapes)
    params, bits, by_prec = _count(shapes, cand.precision, cand.drop_dit_blocks)
    return {
        "params": int(params),
        "size_gb": round(bits / 8 / 1e9, 6),
        "bits_per_weight": round(bits / params, 3) if params else 0.0,
        "by_precision": by_prec,
    }


def size_gb(shapes: dict[str, int], precision: dict[str, str], drop: list[int] | None = None) -> float:
    """Unrounded weight size in GB (for budget arithmetic)."""
    return _count(shapes, precision, drop or [])[1] / 8 / 1e9
