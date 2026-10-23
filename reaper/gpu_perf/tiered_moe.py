"""Reference execution of a mixed-precision MoE layer: experts stored in tiers by format.

Every fast fused-MoE kernel (vLLM/SGLang fused_moe, Marlin MoE, DeepGEMM, CUTLASS/FlashInfer
FP4 MoE, llama.cpp's mul_mat_id) assumes one weight format for all experts of a layer.
Per-expert formats therefore need the layer split into tiers: experts permuted so each
format's experts are contiguous and stacked into their own [n_t, ...] tensors, and one
kernel launch per tier. The MoE output is linear in the experts,

    y(x) = sum_t  sum_{j in tier t}  g_j(x) f_j(x),

so each tier runs as an ordinary fused MoE over the full top-k routing with the other
tiers' slots masked out (expert id -1, the same mechanism vLLM uses for expert
parallelism's expert_map), and the tier outputs are summed. The router, top-k and the
total work are unchanged; the cost is (tiers - 1) extra launches per layer and slightly
emptier grouped GEMMs, which is why the allocator caps tiers per layer (2-3).

This module is the bit-exact reference the kernels have to match; the dequantized weights
come from fakequant.py. See the report for the vLLM and llama.cpp integration sketches.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from fakequant import fake_quant


class TieredExperts(nn.Module):
    def __init__(self, gate_up: torch.Tensor, down: torch.Tensor, tiers: dict[str, list[int]], act=F.silu):
        """gate_up [E, 2I, H] (gate rows first), down [E, H, I]; tiers {format: expert ids}.
        Experts in no tier, or in the "drop" tier, output zero."""
        super().__init__()
        E = gate_up.shape[0]
        self.act = act
        self.formats: list[str] = []
        tier_of = torch.full((E,), -1, dtype=torch.long)
        local_of = torch.full((E,), -1, dtype=torch.long)
        self.w13, self.w2 = nn.ParameterList(), nn.ParameterList()
        for fmt, ids in tiers.items():
            if fmt == "drop" or not ids:
                continue
            ids = sorted(ids)
            t = len(self.formats)
            self.formats.append(fmt)
            tier_of[ids] = t
            local_of[ids] = torch.arange(len(ids))
            # one stacked tensor per tier: what a fused kernel would get in that format
            self.w13.append(nn.Parameter(torch.stack([fake_quant(gate_up[j], fmt) for j in ids]), requires_grad=False))
            self.w2.append(nn.Parameter(torch.stack([fake_quant(down[j], fmt) for j in ids]), requires_grad=False))
        self.register_buffer("tier_of", tier_of)
        self.register_buffer("local_of", local_of)

    @staticmethod
    def fused_moe(x, ids, w, w13, w2, act):
        """Stand-in for one fused-MoE kernel launch: routing ids [N, k] into this tier's
        local experts (-1 = not here), weights [N, k]."""
        out = torch.zeros_like(x, dtype=torch.float32)
        flat = ids.reshape(-1)
        valid = flat >= 0
        if not valid.any():
            return out
        k = ids.shape[1]
        pos = torch.nonzero(valid)[:, 0]
        order = pos[torch.argsort(flat[pos], stable=True)]
        counts = torch.bincount(flat[order], minlength=w13.shape[0]).tolist()
        start = 0
        for j, c in enumerate(counts):
            if c == 0:
                continue
            sel = order[start:start + c]
            start += c
            tok = sel // k
            h = x[tok].float() @ w13[j].T
            g, u = h.chunk(2, dim=-1)
            y = (act(g) * u) @ w2[j].T
            out.index_add_(0, tok, w.reshape(-1)[sel, None].float() * y)
        return out

    def forward(self, x: torch.Tensor, idx: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(x, dtype=torch.float32)
        tiers = self.tier_of[idx]
        for t in range(len(self.formats)):
            local = torch.where(tiers == t, self.local_of[idx], torch.full_like(idx, -1))
            out += self.fused_moe(x, local, w, self.w13[t], self.w2[t], self.act)
        return out.to(x.dtype)


def reference(x, idx, w, gate_up, down, fmt_of: dict[int, str], act=F.silu):
    """Untiered maths with each expert's own dequantized weights, token by token."""
    out = torch.zeros_like(x, dtype=torch.float32)
    for n in range(x.shape[0]):
        for s in range(idx.shape[1]):
            j = int(idx[n, s])
            fmt = fmt_of.get(j, "drop")
            if fmt == "drop":
                continue
            g, u = (x[n].float() @ fake_quant(gate_up[j], fmt).T).chunk(2)
            out[n] += w[n, s].float() * ((act(g) * u) @ fake_quant(down[j], fmt).T)
    return out
