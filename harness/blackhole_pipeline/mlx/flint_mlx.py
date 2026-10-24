"""Recursive's Flint for MLX / mlx-lm (Apple silicon), as exported by BlackHole-Harness.

mlx-lm loads it through the folder's config ("model_file": "flint_mlx.py"):

    mlx_lm.generate --model DIR --trust-remote-code --prompt "hi"
    mlx_lm.server --model DIR --trust-remote-code        # OpenAI API for opencode, pi, ...

Same semantics as REAPer's PyTorch Flint (reaper_pipeline/models/flint.py): sliding (window
counting the current token) and full attention layers, no RoPE on `no_rope_layer_indices`,
per-head q/k RMSNorm before RoPE, one learned sink logit per head, sigmoid top-k router
renormalised over the chosen experts, ungated shared expert, dense `mlp_only_layers`.

REAQ tiers: a MoE layer listed in `expert_tiers` ({layer: n_hot}) keeps its experts permuted so
the n_hot most salient (by REAP score) come first, in their own stacked tensors (`experts_hot`,
quantized with more bits) and the rest in `experts`; the router rows are permuted to match.

HEX experts (Header-Encoded eXperts, BlackHole-Harness): a MoE layer listed in `hex` stores its
routed experts as per-expert learned level tables instead of evenly spaced levels. Per tier
("experts" / "experts_hot") and projection, stacked over the tier's experts:
  idx           uint32 [n, out, in / 32 * bits]   level indices, `bits` per weight, a little-endian
                                                  bit stream per block of 32 inputs (bits words)
  block_scale   uint8  [n, out, in / 32]          fp8 e4m3 scale per block of 32 inputs
  expert_scale  float32 [n]                       one scale per expert matrix
  levels        float32 [n, 2**bits]              the expert's own level table (its header)
weight[e, o, i] = levels[e, idx] * fp8(block_scale[e, o, i // 32]) * expert_scale[e]
On Apple GPUs a Metal kernel reads the tables directly (one simdgroup per output row); elsewhere a
reference path dequantizes the chosen experts.

Output-only vocab trim: with `head_vocab` = K the LM head has K rows for the token ids in
`lm_head_ids`; logits of every other token are -inf (the model still reads all tokens).
"""

from dataclasses import dataclass, field
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from mlx_lm.models.cache import KVCache, RotatingKVCache
from mlx_lm.models.switch_layers import SwitchGLU


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "flint"
    vocab_size: int = 150144
    hidden_size: int = 2048
    intermediate_size: int = 6144
    num_hidden_layers: int = 49
    num_attention_heads: int = 32
    num_key_value_heads: int = 4
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    rope_parameters: Optional[dict] = None
    attention_bias: bool = False
    use_sliding_window: bool = True
    sliding_window: int = 513
    layer_types: Optional[list] = None
    no_rope_layer_indices: list = field(default_factory=list)
    moe_intermediate_size: int = 512
    shared_expert_intermediate_size: int = 1024
    num_experts_per_tok: int = 6
    num_experts: int = 256
    norm_topk_prob: bool = True
    mlp_only_layers: list = field(default_factory=list)
    tie_word_embeddings: bool = False
    expert_tiers: Optional[dict] = None   # {str(layer): n_hot}
    head_vocab: int = 0                   # output-only vocab trim: LM head rows (0 = vocab_size)
    hex: Optional[dict] = None            # {str(layer): {"experts": bits, "experts_hot": bits}}

    def __post_init__(self):
        if self.layer_types is None:
            self.layer_types = ["full_attention"] * self.num_hidden_layers
        if self.rope_parameters and "rope_theta" in self.rope_parameters:
            self.rope_theta = float(self.rope_parameters["rope_theta"])
        self.expert_tiers = {int(k): int(v) for k, v in (self.expert_tiers or {}).items()}
        self.hex = {int(k): dict(v) for k, v in (self.hex or {}).items()}


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer: int):
        super().__init__()
        H, KV, D = args.num_attention_heads, args.num_key_value_heads, args.head_dim
        self.nh, self.nkv, self.d = H, KV, D
        self.q_proj = nn.Linear(args.hidden_size, H * D, bias=args.attention_bias)
        self.k_proj = nn.Linear(args.hidden_size, KV * D, bias=args.attention_bias)
        self.v_proj = nn.Linear(args.hidden_size, KV * D, bias=args.attention_bias)
        self.o_proj = nn.Linear(H * D, args.hidden_size, bias=False)
        self.q_norm = nn.RMSNorm(D, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(D, eps=args.rms_norm_eps)
        self.softmax_offset = mx.zeros((H,))
        self.rope = None if layer in args.no_rope_layer_indices else nn.RoPE(D, traditional=False, base=args.rope_theta)
        self.scale = D ** -0.5

    def __call__(self, x, mask=None, cache=None):
        B, L, _ = x.shape
        q = self.q_norm(self.q_proj(x).reshape(B, L, self.nh, self.d)).transpose(0, 2, 1, 3)
        k = self.k_norm(self.k_proj(x).reshape(B, L, self.nkv, self.d)).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.nkv, self.d).transpose(0, 2, 1, 3)
        if self.rope is not None:
            off = cache.offset if cache is not None else 0
            q, k = self.rope(q, offset=off), self.rope(k, offset=off)
        if cache is not None:
            k, v = cache.update_and_fetch(k, v)
        o = scaled_dot_product_attention(q, k, v, cache, self.scale, mask=mask,
                                         sinks=self.softmax_offset.astype(q.dtype))
        return self.o_proj(o.transpose(0, 2, 1, 3).reshape(B, L, -1))


class MLP(nn.Module):
    def __init__(self, hidden: int, inter: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, inter, bias=False)
        self.up_proj = nn.Linear(hidden, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


HEX_BLOCK = 32
HEX_SKIP = 0xFFFFFFFF   # expert id of a (token, slot) another tier serves: output 0, no weights read
_HEX_KERNELS = {}

_HEX_HEADER = """
inline float hex_fp8(uint code) {          // e4m3 block scale
    uint ex = (code >> 3) & 15u, mt = code & 7u;
    float s = ex == 0 ? float(mt) * 0.001953125f : as_type<float>(((ex + 120u) << 23) | (mt << 20));
    return (code & 128u) ? -s : s;
}
"""

# One threadgroup = SG simdgroups x RPS output rows of one (token, slot). Each lane owns 16-weight
# chunks (half a block: two aligned index words), keeps its inputs in registers across the rows,
# and looks values up in the expert's level table held in threadgroup memory.
_HEX_DOT = """
        float acc{s} = 0.0f;
        for (uint t = 0; t < CPL; t++) {{
            uint c = lane + 32u * t;
            if (c < NCH) {{
                ulong b = base + (c >> 1);
                float sc = hex_fp8(block_scale{s}[b]);
                const device uint32_t* w = idx{s} + b * BITS;
                uint off = (c & 1u) * (16u * BITS);
                uint wi = off >> 5;
                ulong v = ((ulong)w[wi] | ((ulong)w[wi + 1] << 32)) >> (off & 31u);
                float part = 0.0f;
                for (uint j = 0; j < 16; j++) part += tlv{s}[(uint)(v >> (j * BITS)) & MASK] * xv[t][j];
                acc{s} += part * sc;
            }}
        }}
        acc{s} = simd_sum(acc{s}) * es{s};
"""


def _hex_source(glu: bool) -> str:
    sets = ("", "2") if glu else ("",)
    tl = "".join(f"""
    threadgroup float tlv{s}[1u << BITS];
    if (tid < (1u << BITS)) tlv{s}[tid] = levels{s}[e * (1u << BITS) + tid];
    float es{s} = expert_scale{s}[e];""" for s in sets)
    dots = "".join(_HEX_DOT.format(s=s) for s in sets)
    out = "acc / (1.0f + metal::exp(-acc)) * acc2" if glu else "acc"
    return """
    constexpr uint NCH = IN / 16;
    constexpr uint CPL = (NCH + 31) / 32;
    constexpr uint NBLK = IN / 32;
    constexpr uint MASK = (1u << BITS) - 1u;
    uint tid = thread_position_in_threadgroup.x;
    uint lane = tid & 31u;
    uint r = threadgroup_position_in_grid.y;
    uint row0 = threadgroup_position_in_grid.x * (SG * RPS) + (tid >> 5) * RPS;
    uint e = experts[r];
    if (e == 0xFFFFFFFFu) {            // this (token, slot) belongs to the other tier
        if (lane < RPS && row0 + lane < OUT) y[(ulong)r * OUT + row0 + lane] = static_cast<T>(0.0f);
        return;
    }""" + tl + """
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const device T* xrow = x + (ulong)(r / XDIV) * IN;
    float xv[CPL][16];
    for (uint t = 0; t < CPL; t++) {
        uint c = lane + 32u * t;
        for (uint j = 0; j < 16; j++) xv[t][j] = c < NCH ? float(xrow[c * 16 + j]) : 0.0f;
    }
    for (uint q = 0; q < RPS; q++) {
        uint row = row0 + q;
        if (row >= OUT) break;
        ulong base = ((ulong)e * OUT + row) * NBLK;""" + dots + f"""
        if (lane == 0) y[(ulong)r * OUT + row] = static_cast<T>({out});
    }}
"""


_HEX_DEQUANT = """
    constexpr uint NBLK = IN / 32;
    constexpr uint MASK = (1u << BITS) - 1u;
    uint c = thread_position_in_grid.x, row = thread_position_in_grid.y, e = thread_position_in_grid.z;
    if (c >= IN / 16) return;
    ulong b = ((ulong)e * OUT + row) * NBLK + (c >> 1);
    float sc = hex_fp8(block_scale[b]) * expert_scale[e];
    const device uint32_t* w = idx + b * BITS;
    uint off = (c & 1u) * (16u * BITS);
    uint wi = off >> 5;
    ulong v = ((ulong)w[wi] | ((ulong)w[wi + 1] << 32)) >> (off & 31u);
    const device float* lv = levels + e * (1u << BITS);
    device T* o = w_out + ((ulong)e * OUT + row) * IN + c * 16;
    for (uint j = 0; j < 16; j++) o[j] = static_cast<T>(lv[(uint)(v >> (j * BITS)) & MASK] * sc);
"""

HEX_SG, HEX_RPS = 4, 4


def _hex_kernel(kind: str):
    if kind not in _HEX_KERNELS:
        parts = ["idx", "block_scale", "expert_scale", "levels"]
        if kind == "dequant":
            names, out, src = parts, ["w_out"], _HEX_DEQUANT
        else:
            names = ["x"] + parts + ([p + "2" for p in parts] if kind == "glu" else []) + ["experts"]
            out, src = ["y"], _hex_source(kind == "glu")
        _HEX_KERNELS[kind] = mx.fast.metal_kernel(
            name=f"blackhole_hex_{kind}_v3", input_names=names, output_names=out, source=src, header=_HEX_HEADER)
    return _HEX_KERNELS[kind]


def _on_gpu() -> bool:
    return mx.metal.is_available() and mx.default_device() == mx.gpu


def hex_dequant(idx, block_scale, expert_scale, levels, bits: int):
    """Reference: weights [..., out, in] (float32) from HEX parts with matching leading dims."""
    *lead, out, nw = idx.shape
    nblk = nw // bits
    w = idx.reshape(*lead, out, nblk, bits).astype(mx.uint32)
    vals = []
    for j in range(HEX_BLOCK):
        off = j * bits
        wi, sh = off // 32, off % 32
        v = mx.right_shift(w[..., wi], sh)
        if sh + bits > 32:
            v = mx.bitwise_or(v, mx.left_shift(w[..., wi + 1], 32 - sh))
        vals.append(mx.bitwise_and(v, (1 << bits) - 1))
    u = mx.stack(vals, -1).reshape(*lead, out, nblk * HEX_BLOCK).astype(mx.int32)   # [..., out, in]
    L = levels.shape[-1]
    flat_lv = levels.reshape(-1, L)
    rows = mx.arange(flat_lv.shape[0]).reshape(*lead, 1, 1) * L if lead else mx.array(0)
    v = mx.take(flat_lv.reshape(-1), (u + rows).reshape(-1)).reshape(u.shape)
    c = block_scale.astype(mx.uint32)
    ex = mx.bitwise_and(mx.right_shift(c, 3), 15).astype(mx.float32)
    mt = mx.bitwise_and(c, 7).astype(mx.float32)
    s = mx.where(ex == 0, mt / 8.0 * 0.015625, (1.0 + mt / 8.0) * mx.power(2.0, ex - 7.0))
    s = mx.where(mx.bitwise_and(c, 128) > 0, -s, s) * expert_scale.reshape(*lead, 1, 1)
    return v * mx.repeat(s, HEX_BLOCK, axis=-1)


class HexSwitchLinear(nn.Module):
    """Stacked experts in HEX format: y = x @ W[expert].T for each (token, slot)."""

    def __init__(self, in_dims: int, out_dims: int, num_experts: int, bits: int):
        super().__init__()
        self.bits, self.in_dims, self.out_dims = bits, in_dims, out_dims
        nblk = in_dims // HEX_BLOCK
        self.idx = mx.zeros((num_experts, out_dims, nblk * bits), dtype=mx.uint32)
        self.block_scale = mx.zeros((num_experts, out_dims, nblk), dtype=mx.uint8)
        self.expert_scale = mx.ones((num_experts,), dtype=mx.float32)
        self.levels = mx.zeros((num_experts, 2 ** bits), dtype=mx.float32)

    def parts(self):
        return [self.idx, self.block_scale, self.expert_scale, self.levels]

    def _matvec(self, x, experts, other=None):
        """Metal: one output row per (token, slot); `other` (the up projection) fuses silu(self) * other."""
        R = experts.shape[0]
        ntg = -(-self.out_dims // (HEX_SG * HEX_RPS))
        return _hex_kernel("glu" if other is not None else "matvec")(
            inputs=[x, *self.parts(), *(other.parts() if other is not None else []), experts.astype(mx.uint32)],
            template=[("T", x.dtype), ("BITS", self.bits), ("IN", self.in_dims), ("OUT", self.out_dims),
                      ("XDIV", R // x.shape[0]), ("SG", HEX_SG), ("RPS", HEX_RPS)],
            grid=(32 * HEX_SG * ntg, R, 1), threadgroup=(32 * HEX_SG, 1, 1),
            output_shapes=[(R, self.out_dims)], output_dtypes=[x.dtype])[0]

    def dequant(self, dtype=mx.bfloat16):
        """All experts as dense [E, out, in] weights (Metal), for prompt processing."""
        E = self.idx.shape[0]
        return _hex_kernel("dequant")(
            inputs=self.parts(), template=[("T", dtype), ("BITS", self.bits), ("IN", self.in_dims),
                                           ("OUT", self.out_dims)],
            grid=(self.in_dims // 16, self.out_dims, E), threadgroup=(min(32, self.in_dims // 16), 1, 1),
            output_shapes=[(E, self.out_dims, self.in_dims)], output_dtypes=[dtype])[0]

    def __call__(self, x, experts, shared_x: bool = True):
        """x [R or N, in] (shared_x: one row per token for all its k slots), experts [N*k] -> [N*k, out]."""
        if _on_gpu():
            return self._matvec(x, experts)
        R = experts.shape[0]
        xdiv = R // x.shape[0]
        live = experts != HEX_SKIP
        e = mx.where(live, experts, 0)
        W = hex_dequant(self.idx[e], self.block_scale[e], self.expert_scale[e], self.levels[e], self.bits)
        xr = mx.repeat(x, xdiv, axis=0) if xdiv > 1 else x
        y = (W @ xr[..., None].astype(mx.float32))[..., 0]
        return mx.where(live[:, None], y, 0.0).astype(x.dtype)


class HexSwitchGLU(nn.Module):
    # (token, slot) pairs per expert from which prompt processing dequantizes the tier once and
    # runs sorted dense expert matmuls instead of one matrix-vector product per pair
    dequant_min_per_expert = 8

    def __init__(self, hidden: int, inter: int, num_experts: int, bits: int):
        super().__init__()
        self.gate_proj = HexSwitchLinear(hidden, inter, num_experts, bits)
        self.up_proj = HexSwitchLinear(hidden, inter, num_experts, bits)
        self.down_proj = HexSwitchLinear(inter, hidden, num_experts, bits)

    def _dense(self, xf, ids, k):
        E = self.gate_proj.idx.shape[0]
        live = ids != HEX_SKIP
        order = mx.argsort(mx.where(live, ids, E - 1))
        sid = mx.minimum(ids[order], E - 1)
        xs = xf[order // k][:, None].astype(mx.bfloat16)                                  # [R, 1, H]
        mm = lambda a, lin: mx.gather_mm(a, lin.dequant().swapaxes(-1, -2), rhs_indices=sid,  # noqa: E731
                                         sorted_indices=True)
        a = nn.silu(mm(xs, self.gate_proj)) * mm(xs, self.up_proj)
        y = mm(a, self.down_proj)[:, 0]
        y = y[mx.argsort(order)]
        return mx.where(live[:, None], y, 0).astype(xf.dtype)

    def __call__(self, x, indices):
        """x [..., H], indices [..., k] -> [..., k, H] (like SwitchGLU)."""
        shp, k = x.shape, indices.shape[-1]
        xf = x.reshape(-1, shp[-1])
        ids = indices.reshape(-1).astype(mx.uint32)   # HEX_SKIP marks slots another tier serves
        if _on_gpu():
            if ids.size >= self.dequant_min_per_expert * self.gate_proj.idx.shape[0]:
                y = self._dense(xf, ids, k)
            else:
                a = self.gate_proj._matvec(xf, ids, self.up_proj)                       # silu(gate) * up, fused
                y = self.down_proj._matvec(a, ids)
        else:
            a = nn.silu(self.gate_proj(xf, ids, True)) * self.up_proj(xf, ids, True)      # [N*k, I]
            y = self.down_proj(a, ids, False)                                              # [N*k, H]
        return y.reshape(*shp[:-1], k, shp[-1])


class MoE(nn.Module):
    split_min = 64   # (token, slot) pairs from which the tiers get only their own pairs (prompt processing)

    def _split_tiers(self, x, ids):
        """Each (token, slot) through its own tier only: one host sync per layer for the tier sizes,
        then every tier sees just its pairs (sorted by expert inside SwitchGLU). Decoding a single
        token keeps the sync-free path that runs both tiers on all slots."""
        shp, k = x.shape, ids.shape[-1]
        xf = x.reshape(-1, shp[-1])
        flat = ids.reshape(-1)
        hot = flat < self.n_hot
        n_h = int(hot.sum().item())
        order = mx.argsort(mx.logical_not(hot).astype(mx.int32))
        y = mx.zeros((flat.size, shp[-1]), dtype=x.dtype)
        for tier, pos, off in ((self.experts_hot, order[:n_h], 0), (self.experts, order[n_h:], self.n_hot)):
            if pos.size:
                yt = tier(xf[pos // k], (flat[pos] - off)[:, None])[:, 0]
                y = y.at[pos].add(yt)
        return y.reshape(*shp[:-1], k, shp[-1])

    def __init__(self, args: ModelArgs, n_hot: int = 0, hex_bits: Optional[dict] = None):
        super().__init__()
        H, I, E = args.hidden_size, args.moe_intermediate_size, args.num_experts
        self.k, self.norm, self.n_hot = args.num_experts_per_tok, args.norm_topk_prob, n_hot
        self.gate = nn.Linear(H, E, bias=False)
        hb = hex_bits or {}
        self.experts = HexSwitchGLU(H, I, E - n_hot, hb["experts"]) if "experts" in hb else SwitchGLU(H, I, E - n_hot)
        if n_hot:
            self.experts_hot = (HexSwitchGLU(H, I, n_hot, hb["experts_hot"]) if "experts_hot" in hb
                                else SwitchGLU(H, I, n_hot))
        self.shared_expert = MLP(H, args.shared_expert_intermediate_size)

    def __call__(self, x):
        scores = mx.sigmoid(self.gate(x).astype(mx.float32))
        ids = mx.stop_gradient(mx.argpartition(-scores, kth=self.k - 1, axis=-1)[..., :self.k])
        w = mx.take_along_axis(scores, ids, axis=-1)
        if self.norm:
            w = w / (w.sum(-1, keepdims=True) + 1e-20)
        w = w.astype(x.dtype)
        if self.n_hot and ids.size >= self.split_min:
            y = self._split_tiers(x, ids)
        elif self.n_hot and isinstance(self.experts, HexSwitchGLU) and isinstance(self.experts_hot, HexSwitchGLU):
            hot = ids < self.n_hot   # each (token, slot) is computed by its own tier only
            skip = mx.array(HEX_SKIP, dtype=mx.uint32)
            u = ids.astype(mx.uint32)
            y = (self.experts_hot(x, mx.where(hot, u, skip))
                 + self.experts(x, mx.where(hot, skip, u - self.n_hot)))
        elif self.n_hot:
            hot = ids < self.n_hot
            y = mx.where(hot[..., None], self.experts_hot(x, mx.where(hot, ids, 0)),
                         self.experts(x, mx.where(hot, 0, ids - self.n_hot)))
        else:
            y = self.experts(x, ids)
        return (y * w[..., None]).sum(-2) + self.shared_expert(x)


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer: int):
        super().__init__()
        self.self_attn = Attention(args, layer)
        self.mlp = (MLP(args.hidden_size, args.intermediate_size) if layer in args.mlp_only_layers
                    else MoE(args, args.expert_tiers.get(layer, 0), args.hex.get(layer)))
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, x, mask=None, cache=None):
        x = x + self.self_attn(self.input_layernorm(x), mask, cache)
        return x + self.mlp(self.post_attention_layernorm(x))


class FlintModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        types = args.layer_types
        self.window = args.sliding_window if args.use_sliding_window else None
        self.sliding = [t == "sliding_attention" and self.window is not None for t in types]
        self.swa_idx = self.sliding.index(True) if any(self.sliding) else None
        self.full_idx = self.sliding.index(False) if not all(self.sliding) else None

    def __call__(self, inputs, cache=None, input_embeddings=None):
        x = input_embeddings if input_embeddings is not None else self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)
        full = create_attention_mask(x, cache[self.full_idx]) if self.full_idx is not None else None
        swa = (create_attention_mask(x, cache[self.swa_idx], window_size=self.window)
               if self.swa_idx is not None else None)
        for layer, c, s in zip(self.layers, cache, self.sliding):
            x = layer(x, swa if s else full, c)
        return self.norm(x)


def pin_memory() -> None:
    """Let MLX wire (pin) up to the GPU working-set limit, so macOS cannot page the weights out:
    on a 16 GB Mac a 9 GB model otherwise gets partly swapped and a long prompt, which touches
    every expert, reloads them from disk (measured on an M5: 385 -> 777 tok/s prompt processing)."""
    try:
        info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
        if mx.metal.is_available() and info.get("max_recommended_working_set_size"):
            mx.set_wired_limit(int(info["max_recommended_working_set_size"]))
    except Exception:  # noqa: BLE001  (older MLX, or no GPU)
        pass


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        pin_memory()
        self.args = args
        self.model_type = args.model_type
        self.model = FlintModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.head_vocab or args.vocab_size, bias=False)
        if args.head_vocab:
            self.lm_head_ids = mx.zeros((args.head_vocab,), dtype=mx.int32)

    def __call__(self, inputs, cache=None, input_embeddings=None):
        h = self.model(inputs, cache, input_embeddings)
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(h)
        logits = self.lm_head(h)
        if not self.args.head_vocab:
            return logits
        full = mx.full((*logits.shape[:-1], self.args.vocab_size), -mx.inf, dtype=logits.dtype)
        idx = mx.broadcast_to(self.lm_head_ids.astype(mx.int32), logits.shape)
        return mx.put_along_axis(full, idx, logits, axis=-1)

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [RotatingKVCache(max_size=self.model.window) if s else KVCache() for s in self.model.sliding]

    def sanitize(self, weights):
        """Accept a plain Hugging Face Flint checkpoint too: per-expert tensors are stacked (no
        tiers) and the router weight moves to gate.weight (it already is in HF names)."""
        if not any(".experts.0." in k for k in weights):
            return weights
        out = {k: v for k, v in weights.items() if ".experts." not in k}
        for i in range(self.args.num_hidden_layers):
            pre = f"model.layers.{i}.mlp.experts"
            if f"{pre}.0.gate_proj.weight" not in weights:
                continue
            for name in ("gate_proj", "up_proj", "down_proj"):
                ks = sorted((k for k in weights if k.startswith(pre + ".") and k.endswith(f".{name}.weight")),
                            key=lambda k: int(k[len(pre) + 1:].split(".")[0]))
                out[f"{pre}.{name}.weight"] = mx.stack([weights[k] for k in ks])
        return out
