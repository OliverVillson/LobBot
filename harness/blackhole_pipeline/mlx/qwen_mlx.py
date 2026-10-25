"""Qwen3.5 / 3.6 MoE (e.g. Qwen3.6-35B-A3B) for mlx-lm, as BlackHole exports it: mlx-lm's own
model (Gated DeltaNet linear attention, gated full attention every 4th layer, softmax top-k
router, sigmoid-gated shared expert) with BlackHole's routed experts:

  REAQ tiers   the most salient experts of each layer (REAP score) on their own stacked tensors
               (`experts_hot`, ids 0..n_hot-1 after the router's rows were reordered), the rest
               on `experts`; each (token, slot) runs through its own tier only
  HEX          per-expert learned value tables (`hex`: {layer: {tier: bits}}), Metal kernels
  vocab trim   with `head_vocab` = K the LM head has K rows for the ids in `lm_head_ids`; every
               other token's logit is -inf (the model still reads all tokens)

The tiers, HEX kernels and memory pinning come from flint_mlx.py next to this file. The vision
tower is not exported: this folder is the text model. Load with
`mlx_lm.server --model DIR --trust-remote-code`.
"""

import os
import sys
from dataclasses import dataclass, field

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models import qwen3_5_moe as Q

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flint_mlx import MoE as _TieredMoE  # noqa: E402
from flint_mlx import pin_memory  # noqa: E402


@dataclass
class ModelArgs(Q.ModelArgs):
    expert_tiers: dict = field(default_factory=dict)   # {layer: experts on the hot tier}
    hex: dict = field(default_factory=dict)            # {layer: {"experts_hot"/"experts": bits}}
    head_vocab: int = 0                                # output-only vocab trim: LM head rows (0 = all)


class QwenMoE(_TieredMoE):
    """Qwen's sparse MoE block on BlackHole's expert tiers."""

    def __init__(self, args, n_hot: int = 0, hex_bits=None):
        super().__init__(args, n_hot, hex_bits)
        self.shared_expert_gate = nn.Linear(args.hidden_size, 1, bias=False)

    def route(self, x):
        gates = mx.softmax(self.gate(x), axis=-1, precise=True)
        ids = mx.stop_gradient(mx.argpartition(gates, kth=-self.k, axis=-1)[..., -self.k:])
        w = mx.take_along_axis(gates, ids, axis=-1)
        if self.norm:
            w = w / w.sum(axis=-1, keepdims=True)
        return ids, w

    def shared(self, x):
        return mx.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)


class Model(Q.Model):
    def __init__(self, args: ModelArgs):
        super().__init__(args)
        pin_memory()
        targs = self.language_model.args
        tiers = {int(k): v for k, v in (args.expert_tiers or {}).items()}
        hexes = {int(k): v for k, v in (args.hex or {}).items()}
        for i, layer in enumerate(self.language_model.model.layers):
            if hasattr(layer.mlp, "switch_mlp"):
                layer.mlp = QwenMoE(targs, tiers.get(i, 0), hexes.get(i))
        self.head_vocab = args.head_vocab
        if args.head_vocab:
            self.language_model.lm_head = nn.Linear(targs.hidden_size, args.head_vocab, bias=False)
            self.lm_head_ids = mx.zeros((args.head_vocab,), dtype=mx.int32)

    def __call__(self, inputs, cache=None, input_embeddings=None):
        logits = super().__call__(inputs, cache=cache, input_embeddings=input_embeddings)
        if not self.head_vocab:
            return logits
        full = mx.full((*logits.shape[:-1], self.language_model.args.vocab_size), -mx.inf, dtype=logits.dtype)
        idx = mx.broadcast_to(self.lm_head_ids.astype(mx.int32), logits.shape)
        return mx.put_along_axis(full, idx, logits, axis=-1)

    def sanitize(self, weights):
        return weights   # BlackHole writes the folder in this model's own names and layouts
