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

    def __post_init__(self):
        if self.layer_types is None:
            self.layer_types = ["full_attention"] * self.num_hidden_layers
        if self.rope_parameters and "rope_theta" in self.rope_parameters:
            self.rope_theta = float(self.rope_parameters["rope_theta"])
        self.expert_tiers = {int(k): int(v) for k, v in (self.expert_tiers or {}).items()}


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


class MoE(nn.Module):
    def __init__(self, args: ModelArgs, n_hot: int = 0):
        super().__init__()
        H, I, E = args.hidden_size, args.moe_intermediate_size, args.num_experts
        self.k, self.norm, self.n_hot = args.num_experts_per_tok, args.norm_topk_prob, n_hot
        self.gate = nn.Linear(H, E, bias=False)
        self.experts = SwitchGLU(H, I, E - n_hot)
        if n_hot:
            self.experts_hot = SwitchGLU(H, I, n_hot)
        self.shared_expert = MLP(H, args.shared_expert_intermediate_size)

    def __call__(self, x):
        scores = mx.sigmoid(self.gate(x).astype(mx.float32))
        ids = mx.stop_gradient(mx.argpartition(-scores, kth=self.k - 1, axis=-1)[..., :self.k])
        w = mx.take_along_axis(scores, ids, axis=-1)
        if self.norm:
            w = w / (w.sum(-1, keepdims=True) + 1e-20)
        w = w.astype(x.dtype)
        if self.n_hot:
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
                    else MoE(args, args.expert_tiers.get(layer, 0)))
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


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = FlintModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs, cache=None, input_embeddings=None):
        h = self.model(inputs, cache, input_embeddings)
        return self.model.embed_tokens.as_linear(h) if self.args.tie_word_embeddings else self.lm_head(h)

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
