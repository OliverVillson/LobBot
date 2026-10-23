"""GPTQ (Frantar et al. 2022) for the fake-quant formats in fakequant.py: same symmetric
group grid as int_sym, but each column's rounding error is pushed onto the not-yet-quantized
columns using the inverse input Hessian H = sum x x^T, so the layer output error is minimised
instead of the weight error. Columns that are all zero in W or never active in H are treated as
dead (pruned) and stay zero."""

from __future__ import annotations

import torch

GRID = {"int8_g128": (8, 128), "int4_g128": (4, 128), "int3_g64": (3, 64), "int2_g64": (2, 64)}


def _scale(w: torch.Tensor, bits: int) -> torch.Tensor:
    qmax = 2 ** (bits - 1) - 1
    s = (w.abs().amax(-1, keepdim=True) / (1.5 if bits == 2 else max(qmax, 1))).half().float()
    return torch.where(s > 0, s, torch.ones_like(s))


def _round(w: torch.Tensor, s: torch.Tensor, bits: int) -> torch.Tensor:
    qmax = 2 ** (bits - 1) - 1
    if bits == 2:
        return ((w / s - 0.5).round().clamp(-2, 1) + 0.5) * s
    return (w / s).round().clamp(-qmax - 1, qmax) * s


@torch.no_grad()
def gptq(w: torch.Tensor, H: torch.Tensor, fmt: str, damp: float = 0.01, block: int = 128) -> torch.Tensor:
    """w [out, in], H [in, in] (fp32). Returns the dequantized weight (fp32)."""
    bits, g = GRID[fmt]
    assert block % g == 0
    W = w.float().clone()
    H = H.float().clone()
    n = W.shape[1]
    dead = (H.diagonal() == 0) | (W.abs().amax(0) == 0)
    H[dead, :] = 0
    H[:, dead] = 0
    H[dead, dead] = 1
    W[:, dead] = 0
    H.diagonal().add_(damp * H.diagonal()[~dead].mean() if (~dead).any() else damp)
    L = torch.linalg.cholesky(H)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(L), upper=True)
    Q = torch.zeros_like(W)
    s = None
    for b0 in range(0, n, block):
        b1 = min(b0 + block, n)
        Wb = W[:, b0:b1].clone()
        Eb = torch.zeros_like(Wb)
        Hb = Hinv[b0:b1, b0:b1]
        for j in range(b1 - b0):
            col = b0 + j
            if col % g == 0:
                s = _scale(Wb[:, j:j + g], bits)
            q = _round(Wb[:, j:j + 1], s, bits)
            Q[:, col:col + 1] = q
            e = (Wb[:, j:j + 1] - q) / Hb[j, j]
            Wb[:, j:] -= e @ Hb[j:j + 1, j:]
            Eb[:, j:j + 1] = e
        W[:, b1:] -= Eb @ Hinv[b0:b1, b1:]
    Q[:, dead] = 0
    return Q
