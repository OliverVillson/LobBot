"""Per-expert header formats: NVFP4-style two-level scaling where every expert carries its own
header (an fp32 scale and, optionally, its own table of quantization levels), with low-bit
indices under it.

Layout per expert matrix [out, in]:
  header   fp32 expert scale (+ L fp16 levels for a learned table): <= 4 + 2*16 bytes, about
           1e-4 bits per weight on a 4.7M-weight expert
  blocks   one fp8 (e4m3) scale per `block` weights along the input dim
  indices  log2(L) bits per weight into the level table

Level tables:
  fixed    e2m1 (= NVFP4 values) or a symmetric int grid, the same for every expert
  layer    one learned table shared by all experts of a layer
  expert   a learned table per expert (the header idea)
Learned tables are a 1-D weighted k-means (Lloyd) on block-normalized weights, weighted by the
input channel's activation energy (diag of the GPTQ Hessian, the imatrix statistic).
Fake quantization: returns the dequantized weights, for quality measurements only.
"""

from __future__ import annotations

import math

import torch

FP8_MAX = 448.0
E2M1 = torch.tensor([-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]) / 6.0


def int_levels(bits: int) -> torch.Tensor:
    q = 2 ** (bits - 1)
    return torch.arange(-(q - 1), q).float() / (q - 1) if bits > 1 else torch.tensor([-1.0, 1.0])  # 2^b - 1 symmetric levels, like int_sym


def bpw(levels: int, block: int) -> float:
    return math.log2(levels) + 8 / block


def _blocks(W: torch.Tensor, block: int, shared_scale: bool):
    """W [E, out, in] -> block-normalized values in [-1, 1], dequant block scales, padding."""
    E, o, n = W.shape
    pad = -n % block
    x = torch.nn.functional.pad(W.float(), (0, pad)).view(E, o, (n + pad) // block, block)
    amax = x.abs().amax(-1, keepdim=True)
    top = amax.amax(dim=(1, 2, 3), keepdim=True)
    if shared_scale:
        top = top.amax(0, keepdim=True).expand_as(top)
    es = (top / FP8_MAX).clamp(min=1e-30)                       # the expert's (or layer's) fp32 scale
    blk = (amax / es).clamp(max=FP8_MAX).to(torch.float8_e4m3fn).float() * es
    blk = torch.where(blk > 0, blk, torch.ones_like(blk))
    return (x / blk).clamp(-1, 1), blk, pad


def _nearest(xn: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    """Round every value of xn [E, ...] to the nearest of its row's sorted levels [E, L]."""
    E = xn.shape[0]
    mids = (levels[:, 1:] + levels[:, :-1]) / 2
    flat = xn.reshape(E, -1)
    idx = torch.searchsorted(mids.contiguous(), flat.contiguous())
    return torch.gather(levels, 1, idx).view_as(xn)


def fit_levels(xn: torch.Tensor, wt: torch.Tensor | None, L: int, iters: int = 20,
               sample: int = 1 << 17, seed: int = 0) -> torch.Tensor:
    """Weighted 1-D k-means per row: xn [E, N] values, wt [E, N] weights. Returns sorted [E, L]."""
    E, N = xn.shape
    g = torch.Generator(device="cpu").manual_seed(seed)
    if N > sample:
        idx = torch.randint(0, N, (sample,), generator=g).to(xn.device)
        xn = xn[:, idx]
        wt = wt[:, idx] if wt is not None else None
    w = wt if wt is not None else torch.ones_like(xn)
    q = torch.linspace(0, 1, L + 2, device=xn.device)[1:-1]
    lev = torch.quantile(xn.float(), q, dim=1).T.contiguous()                # [E, L]
    for _ in range(iters):
        mids = (lev[:, 1:] + lev[:, :-1]) / 2
        a = torch.searchsorted(mids.contiguous(), xn.contiguous())          # [E, S]
        num = torch.zeros(E, L, device=xn.device).scatter_add_(1, a, xn * w)
        den = torch.zeros(E, L, device=xn.device).scatter_add_(1, a, w)
        lev = torch.where(den > 0, num / den.clamp(min=1e-30), lev)
        lev = lev.sort(1).values
    return lev


@torch.no_grad()
def header_quant(W: torch.Tensor, table: str, L: int = 16, block: int = 16, col_weight: torch.Tensor | None = None,
                 shared_scale: bool = False, levels: torch.Tensor | None = None) -> torch.Tensor:
    """W [E, out, in]. table: "e2m1" | "int" | "layer" | "expert". col_weight [E, in] (activation
    energy per input channel) weights the k-means. Returns dequantized [E, out, in] fp32."""
    E, o, n = W.shape
    xn, blk, pad = _blocks(W, block, shared_scale)
    dev = W.device
    if levels is None:
        if table == "e2m1":
            levels = E2M1.to(dev)[None].expand(E, -1)
        elif table == "int":
            levels = int_levels(int(round(math.log2(L)))).to(dev)[None].expand(E, -1)
        else:
            wt = None
            if col_weight is not None:
                cw = torch.nn.functional.pad(col_weight.float(), (0, pad)).view(E, 1, -1, block)
                wt = (cw.expand(E, o, -1, -1) * blk.pow(2)).reshape(E, -1)    # error in weight units
            flat = xn.reshape(E, -1)
            if table == "layer":
                lev = fit_levels(flat.reshape(1, -1), wt.reshape(1, -1) if wt is not None else None, L)
                levels = lev.expand(E, -1)
            else:
                levels = fit_levels(flat, wt, L)
    q = _nearest(xn, levels.contiguous().float()) * blk
    return q.view(E, o, -1)[..., :n]
