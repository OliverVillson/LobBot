"""Qwen3.5 / 3.6 MoE (e.g. Qwen3.6-35B-A3B) for mlx-lm, as BlackHole exports it: mlx-lm's own
model (Gated DeltaNet linear attention, gated full attention every 4th layer, softmax top-k
router, sigmoid-gated shared expert) with BlackHole's routed experts:

  REAQ tiers   the most salient experts of each layer (REAP score) on their own stacked tensors
               (`experts_hot`, ids 0..n_hot-1 after the router's rows were reordered), the rest
               on `experts`; each (token, slot) runs through its own tier only
  HEX          per-expert learned value tables (`hex`: {layer: {tier: bits}}), Metal kernels
  vocab trim   with `head_vocab` = K the LM head has K rows for the ids in `lm_head_ids`; every
               other token's logit is -inf (the model still reads all tokens)
  think budget with `think_budget` = B (env BLACKHOLE_THINK_BUDGET overrides, 0 = off) a reply
               still thinking after B tokens is told to answer now: the model itself emits Qwen's
               "Considering the limited time ..." line and </think> (budget forcing), so it works
               in mlx_lm.server, opencode and pi unchanged. Ids in `think_ids` (think_budget.py)

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
    think_budget: int = 0                              # thinking tokens before the model must answer (0 = off)
    think_ids: dict = field(default_factory=dict)      # {"start": <think>, "end": </think>, "force": [ids]}


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


class ThinkBudget:
    """Budget forcing on the token stream the model is fed (prompt chunks, then one sampled token
    per step): counts thinking tokens since the last <think>; past the budget the next tokens are
    forced to the answer-now line ending in </think>. A new request re-syncs from its prompt."""

    def __init__(self, budget: int, start: int, end: int, force: list):
        self.budget, self.start, self.end, self.force = budget, start, end, list(force)
        self.open, self.count, self.k = False, 0, None   # k: next position in `force` while forcing

    def step(self, toks: list):
        """Feed the tokens of this call; returns the token the model must emit next, or None."""
        if self.k is not None:
            if len(toks) == 1 and toks[0] == self.force[self.k]:
                self.k += 1
            else:
                self.k = None   # a new request (or the stream diverged): stop forcing
        for t in toks:
            if t == self.start:
                self.open, self.count = True, 0
            elif t == self.end:
                self.open, self.k = False, None
            elif self.open:
                self.count += 1
        if self.k is not None and self.k < len(self.force):
            return self.force[self.k]
        self.k = None
        if self.open and self.count >= self.budget:
            self.k = 0
            return self.force[0]
        return None


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

        ids = args.think_ids or {}
        budget = int(os.environ.get("BLACKHOLE_THINK_BUDGET", args.think_budget or 0))
        self.think = ThinkBudget(budget, ids["start"], ids["end"], ids["force"]) if budget > 0 and ids else None

    def __call__(self, inputs, cache=None, input_embeddings=None):
        logits = super().__call__(inputs, cache=cache, input_embeddings=input_embeddings)
        if self.head_vocab:
            full = mx.full((*logits.shape[:-1], self.language_model.args.vocab_size), -mx.inf, dtype=logits.dtype)
            idx = mx.broadcast_to(self.lm_head_ids.astype(mx.int32), logits.shape)
            logits = mx.put_along_axis(full, idx, logits, axis=-1)
        if self.think is not None and inputs.ndim == 2 and inputs.shape[0] == 1:   # one sequence at a time
            tok = self.think.step(inputs[0].tolist())
            if tok is not None:
                last = mx.where(mx.arange(logits.shape[-1]) == tok, 0.0, -mx.inf).astype(logits.dtype)
                logits = mx.concatenate([logits[:, :-1], last[None, None]], axis=1)
        return logits

    def sanitize(self, weights):
        return weights   # BlackHole writes the folder in this model's own names and layouts
