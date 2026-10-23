"""GPTQ (Frantar et al. 2022) for the fake-quant formats in fakequant.py: same symmetric
group grid as int_sym, but each column's rounding error is pushed onto the not-yet-quantized
columns using the inverse input Hessian H = sum x x^T, so the layer output error is minimised
instead of the weight error. Columns that are all zero in W or never active in H are treated as
dead (pruned) and stay zero."""

from __future__ import annotations

import torch

GRID = {"int8_g128": (8, 128), "int7_g128": (7, 128), "int6_g128": (6, 128), "int5_g128": (5, 128),
        "int4_g128": (4, 128), "int3_g64": (3, 64), "int2_g64": (2, 64)}


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
    return gptq_batched(w[None], H[None], fmt, damp, block)[0]


def _prep(w: torch.Tensor, H: torch.Tensor, damp: float):
    """Dead columns, damping and the upper Cholesky factor of H^-1 for a batch of problems."""
    W = w.float().clone()
    H = H.double().clone()
    B, _, n = W.shape
    eye = torch.eye(n, dtype=H.dtype, device=H.device)
    empty = (H.diagonal(dim1=1, dim2=2) == 0).all(1)          # no calibration tokens: plain RTN
    H[empty] = eye
    dead = (H.diagonal(dim1=1, dim2=2) == 0) | (W.abs().amax(1) == 0)             # [B, in]
    live = (~dead).to(H.dtype)
    H = H * live[:, :, None] * live[:, None, :] + eye * dead.to(H.dtype)[:, :, None]
    W = W * (~dead)[:, None, :]
    diag = H.diagonal(dim1=1, dim2=2)
    base = (diag * live).sum(1) / live.sum(1).clamp(min=1)
    base = torch.where(base > 0, base, torch.ones_like(base))
    # outlier features (massive activations) make H badly conditioned: factor in fp64 and raise
    # the damping of each problem until both Cholesky steps succeed
    d = torch.full((B,), damp, dtype=H.dtype, device=H.device)
    for _ in range(6):
        L, info = torch.linalg.cholesky_ex(H + (d * base)[:, None, None] * eye)
        U, info2 = torch.linalg.cholesky_ex(torch.cholesky_inverse(L), upper=True)
        bad = (info != 0) | (info2 != 0) | ~torch.isfinite(U).all(-1).all(-1)
        if not bad.any():
            break
        d = torch.where(bad, d * 3, d)
    else:
        raise RuntimeError("GPTQ: Hessian not positive definite even with heavy damping")
    return W, U.float(), dead


def _solve(W, Hinv, quant_col, block: int, on_col=None):
    """The GPTQ column sweep with lazy block updates; quant_col(Wb_col, col) -> q."""
    n = W.shape[-1]
    Q = torch.zeros_like(W)
    for b0 in range(0, n, block):
        b1 = min(b0 + block, n)
        Wb = W[..., b0:b1].clone()
        Eb = torch.zeros_like(Wb)
        Hb = Hinv[:, b0:b1, b0:b1]
        for j in range(b1 - b0):
            col = b0 + j
            if on_col is not None:
                on_col(Wb, j, col)
            q = quant_col(Wb[..., j:j + 1], col)
            Q[..., col:col + 1] = q
            e = (Wb[..., j:j + 1] - q) / Hb[:, j, j][:, None, None]
            Wb[..., j:] -= e @ Hb[:, j:j + 1, j:]
            Eb[..., j:j + 1] = e
        W[..., b1:] -= Eb @ Hinv[:, b0:b1, b1:]
    return Q


@torch.no_grad()
def gptq_batched(w: torch.Tensor, H: torch.Tensor, fmt: str, damp: float = 0.01, block: int = 128) -> torch.Tensor:
    """Independent GPTQ problems in one pass, e.g. all experts of a layer: w [B, out, in],
    H [B, in, in]. Returns the dequantized weights [B, out, in] (fp32)."""
    bits, g = GRID[fmt]
    assert block % g == 0
    W, Hinv, dead = _prep(w, H, damp)
    st = {}

    def on_col(Wb, j, col):
        if col % g == 0:
            st["s"] = _scale(Wb[..., j:j + g], bits)
    Q = _solve(W, Hinv, lambda c, col: _round(c, st["s"], bits), block, on_col)
    return Q * (~dead)[:, None, :]


@torch.no_grad()
def gptq_mixed(w: torch.Tensor, H: torch.Tensor, col_fmt: list[str], damp: float = 0.01,
               block: int = 128) -> torch.Tensor:
    """GPTQ with a format per input column (columns of one format contiguous, groups counted from
    the start of each run). Group scales are static, taken from the weights before the sweep.
    w [B, out, in], H [B, in, in]; the col_fmt pattern is shared by the whole batch."""
    W, Hinv, dead = _prep(w, H, damp)
    n = W.shape[-1]
    assert len(col_fmt) == n
    bits = [GRID[f][0] for f in col_fmt]
    S = torch.empty_like(W)
    c = 0
    while c < n:
        f = col_fmt[c]
        e = c
        while e < n and col_fmt[e] == f:
            e += 1
        g = GRID[f][1]
        for a in range(c, e, g):
            b = min(a + g, e)
            S[..., a:b] = _scale(W[..., a:b], GRID[f][0])
        c = e
    Q = _solve(W, Hinv, lambda x, col: _round(x, S[..., col:col + 1], bits[col]), block)
    return Q * (~dead)[:, None, :]
