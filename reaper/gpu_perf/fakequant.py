"""Fake quantization (quantize then dequantize) of 2-D weights in the formats an expert tier
can be stored in, so the bit allocator can measure error per expert instead of assuming it.

Formats and their effective bits per weight (weights + scales):
  fp8_block   e4m3, one fp32 scale per 128x128 block (REAPer's quant stage)      ~8.00
  int8_g128   symmetric int8, fp16 scale per 128 inputs                           8.125
  int4_g128   symmetric int4, fp16 scale per 128 inputs (GPTQ/AWQ/Marlin layout)  4.125
  mxfp4       e2m1, e8m0 power-of-two scale per 32 (OCP MX, gpt-oss)              4.25
  nvfp4       e2m1, e4m3 scale per 16, plus one fp32 tensor scale (Blackwell)     4.50
  int3_g64    symmetric int3, fp16 scale per 64 (proxy for GGUF q3_k)             3.25
  int2_g64    symmetric int2, fp16 scale per 64 (proxy for GGUF q2_k)             2.25
The integer ones are round-to-nearest proxies; GPTQ/AWQ/imatrix rounding does better, so
treat measured errors as an upper bound that ranks experts correctly.
"""

from __future__ import annotations

import torch

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
FP8_MAX = 448.0

BPW = {"bf16": 16.0, "fp8_block": 8.0 + 32 / 16384, "int8_g128": 8.125, "int4_g128": 4.125,
       "mxfp4": 4.25, "nvfp4": 4.5, "int3_g64": 3.25, "int2_g64": 2.25, "drop": 0.0,
       "int7_g128": 7.125, "int6_g128": 6.125, "int5_g128": 5.125}


def _groups(w: torch.Tensor, g: int) -> tuple[torch.Tensor, int]:
    out, inp = w.shape
    pad = -inp % g
    x = torch.nn.functional.pad(w.float(), (0, pad))
    return x.view(out, -1, g), pad


def _ungroup(x: torch.Tensor, shape, pad: int) -> torch.Tensor:
    out, inp = shape
    return x.reshape(out, inp + pad)[:, :inp]


def int_sym(w: torch.Tensor, bits: int, g: int) -> torch.Tensor:
    x, pad = _groups(w, g)
    qmax = 2 ** (bits - 1) - 1
    s = (x.abs().amax(-1, keepdim=True) / max(qmax, 1)).half().float()
    if bits == 2:  # {-2,-1,0,1} with a midpoint grid works better than {-1,0,1}
        s = (x.abs().amax(-1, keepdim=True) / 1.5).half().float()
    s = torch.where(s > 0, s, torch.ones_like(s))  # all-zero group (e.g. pruned rows); fp16 scale underflow
    if bits == 2:
        q = (x / s - 0.5).round().clamp(-2, 1) + 0.5
    else:
        q = (x / s).round().clamp(-qmax - 1, qmax)
    return _ungroup(q * s, w.shape, pad)


def _e2m1(x: torch.Tensor) -> torch.Tensor:
    """Round |x| <= 6 to the nearest FP4 e2m1 value, keeping the sign."""
    grid = E2M1.to(x.device)
    a = x.abs().clamp(max=6.0)
    i = (a.unsqueeze(-1) - grid).abs().argmin(-1)
    return grid[i] * x.sign()


def mxfp4(w: torch.Tensor) -> torch.Tensor:
    x, pad = _groups(w, 32)
    amax = x.abs().amax(-1, keepdim=True).clamp(min=1e-30)
    s = torch.exp2(torch.floor(torch.log2(amax)) - 2)  # e8m0: largest power of two putting amax near 4..8
    return _ungroup(_e2m1(x / s) * s, w.shape, pad)


def nvfp4(w: torch.Tensor) -> torch.Tensor:
    x, pad = _groups(w, 16)
    tensor_scale = (x.abs().amax() / (6.0 * FP8_MAX)).clamp(min=1e-30)
    blk = (x.abs().amax(-1, keepdim=True) / 6.0 / tensor_scale).clamp(max=FP8_MAX)
    blk = blk.to(torch.float8_e4m3fn).float().clamp(min=1e-12) * tensor_scale
    return _ungroup(_e2m1(x / blk) * blk, w.shape, pad)


def fp8_block(w: torch.Tensor, block: int = 128) -> torch.Tensor:
    out, inp = w.shape
    po, pi = -out % block, -inp % block
    x = torch.nn.functional.pad(w.float(), (0, pi, 0, po)).view((out + po) // block, block, (inp + pi) // block, block)
    s = (x.abs().amax(dim=(1, 3), keepdim=True) / FP8_MAX).clamp(min=1e-12)
    q = (x / s).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).float() * s
    return q.view(out + po, inp + pi)[:out, :inp]


def fake_quant(w: torch.Tensor, fmt: str) -> torch.Tensor:
    if fmt == "bf16":
        return w.to(torch.bfloat16).float()
    if fmt == "drop":
        return torch.zeros_like(w, dtype=torch.float32)
    if fmt == "fp8_block":
        return fp8_block(w)
    if fmt == "nvfp4":
        return nvfp4(w)
    if fmt == "mxfp4":
        return mxfp4(w)
    bits, g = {"int8_g128": (8, 128), "int7_g128": (7, 128), "int6_g128": (6, 128), "int5_g128": (5, 128),
               "int4_g128": (4, 128), "int3_g64": (3, 64), "int2_g64": (2, 64)}[fmt]
    return int_sym(w, bits, g)
