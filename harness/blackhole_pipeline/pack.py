"""Bit packing for the two expert formats the pipeline writes.

int4 (GPTQ W4A16): compressed-tensors "pack-quantized", as llm-compressor writes it and vLLM
reads it: signed codes in [-8, 7] stored offset by 8, eight per int32 along the input dim,
element i of each group of eight at bits 4i..4i+3. weight = code * weight_scale (symmetric,
one scale per 128 inputs).

codebook (REAQ): per expert matrix a sorted table of L <= 16 levels, one fp8 scale per block of
inputs times an fp32 matrix scale, and a 4-bit index per weight into the table (two per byte,
low nibble first; 8-level tables leave the top bit unused in this container format, the 3-bit
size is what bpw() reports). weight = levels[idx] * fp8(block_scale) * expert_scale.
"""

from __future__ import annotations

import math

import torch

FP8_MAX = 448.0


def int_codes(Q: torch.Tensor, scales: torch.Tensor, group: int) -> torch.Tensor:
    """Integer codes of GPTQ output Q [..., out, in] with group scales [..., out, in / group]."""
    s = scales.repeat_interleave(group, -1)[..., :Q.shape[-1]]
    return (Q / s).round().clamp(-8, 7).to(torch.int8)


def int_codes_bits(Q: torch.Tensor, scales: torch.Tensor, group: int, bits: int) -> torch.Tensor:
    """int_codes for any symmetric GPTQ width: codes in [-2^(b-1), 2^(b-1) - 1]."""
    s = scales.repeat_interleave(group, -1)[..., :Q.shape[-1]]
    h = 2 ** (bits - 1)
    return (Q / s).round().clamp(-h, h - 1).to(torch.int8)


def pack_int4(codes: torch.Tensor) -> torch.Tensor:
    """[out, in] int8 codes in [-8, 7] -> [out, ceil(in / 8)] int32 (compressed-tensors layout)."""
    out, n = codes.shape
    pad = -n % 8
    v = torch.nn.functional.pad((codes.to(torch.int32) + 8), (0, pad)).view(out, -1, 8)
    shifts = torch.arange(8, dtype=torch.int64, device=codes.device) * 4
    x = (v.to(torch.int64) << shifts).sum(-1)
    return torch.where(x >= 2 ** 31, x - 2 ** 32, x).to(torch.int32)  # same bits, as int32


def unpack_int4(packed: torch.Tensor, n: int) -> torch.Tensor:
    """Inverse of pack_int4: [out, in / 8] int32 -> [out, n] int8 codes."""
    shifts = torch.arange(8, dtype=torch.int32, device=packed.device) * 4
    v = (packed[..., None] >> shifts) & 0xF
    return (v.reshape(packed.shape[0], -1)[:, :n] - 8).to(torch.int8)


def table_indices(Q: torch.Tensor, scales: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    """Index into each matrix's level table of a table-rounded Q [E, out, in] with per-element
    scales [E, out, in] and sorted levels [E, L]. Exact for output of headers.round_to_table."""
    E = Q.shape[0]
    mids = (levels[:, 1:] + levels[:, :-1]) / 2
    x = (Q / scales).clamp(-1, 1).reshape(E, -1)
    return torch.searchsorted(mids.contiguous(), x.contiguous()).view_as(Q).to(torch.uint8)


def pack_nibbles(idx: torch.Tensor) -> torch.Tensor:
    """[..., n] uint8 values < 16 -> [..., ceil(n / 2)] uint8, low nibble first."""
    if idx.shape[-1] % 2:
        idx = torch.nn.functional.pad(idx, (0, 1))
    return (idx[..., 0::2] | (idx[..., 1::2] << 4)).to(torch.uint8)


def unpack_nibbles(p: torch.Tensor, n: int) -> torch.Tensor:
    return torch.stack([p & 0xF, p >> 4], -1).reshape(*p.shape[:-1], -1)[..., :n]


def expert_scale(W: torch.Tensor) -> torch.Tensor:
    """The fp32 matrix scale headers._blocks uses (non-shared): max |w| / 448, per matrix [E]."""
    return (W.float().abs().amax(dim=(1, 2)) / FP8_MAX).clamp(min=1e-30)


def block_codes(scales: torch.Tensor, es: torch.Tensor, block: int) -> torch.Tensor:
    """Per-element dequant scales [E, out, in] (constant per block) -> fp8 block codes
    [E, out, ceil(in / block)] such that code * es reproduces them."""
    s = scales[..., ::block]
    return (s / es[:, None, None]).clamp(max=FP8_MAX).to(torch.float8_e4m3fn)


def codebook_dequant(packed: torch.Tensor, n: int, codes: torch.Tensor, es: torch.Tensor, levels: torch.Tensor,
                     block: int) -> torch.Tensor:
    """Weights [E, out, n] back from the saved codebook parts (for checks and a bf16 loader)."""
    idx = unpack_nibbles(packed, n).long()
    E = idx.shape[0]
    vals = torch.gather(levels, 1, idx.reshape(E, -1)).view(idx.shape)
    s = codes.float().repeat_interleave(block, -1)[..., :n] * es[:, None, None]
    return vals * s


def bpw(levels: int, block: int) -> float:
    return math.log2(levels) + 8 / block


# --------------------------------------------------------------------------- MLX (Apple silicon)
# MLX affine quantization: weight = u * scale + bias per group of `group` inputs, u in
# [0, 2^bits - 1], codes packed as one little-endian bit stream into uint32 words (code i at
# bits i*bits .. i*bits+bits-1 of the stream), scale and bias in the model dtype.

def mlx_pack(u: torch.Tensor, bits: int) -> torch.Tensor:
    """[..., n] integer codes (n * bits divisible by 32) -> [..., n * bits / 32] uint32."""
    shp = u.shape
    b = ((u.to(torch.int64)[..., None] >> torch.arange(bits, device=u.device)) & 1).to(torch.uint8)
    b = b.reshape(*shp[:-1], -1, 32).to(torch.int64)
    words = (b << torch.arange(32, device=u.device)).sum(-1)
    return torch.where(words >= 2 ** 31, words - 2 ** 32, words).to(torch.int32).view(torch.uint32)


def mlx_unpack(words: torch.Tensor, bits: int) -> torch.Tensor:
    """Inverse of mlx_pack: [..., w] uint32 -> [..., w * 32 / bits] int64 codes."""
    x = words.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    b = (x[..., None] >> torch.arange(32, device=x.device)) & 1
    b = b.reshape(*x.shape[:-1], -1, bits)
    return (b << torch.arange(bits, device=x.device)).sum(-1)


def mlx_from_symmetric(codes: torch.Tensor, scales: torch.Tensor, bits: int, dtype):
    """Signed GPTQ codes [..., out, in] in [-2^(b-1), 2^(b-1)-1] with group scales [..., out, in/g]
    -> MLX (u, scale, bias) with the same grid: u = code + 2^(b-1), bias = -2^(b-1) * scale."""
    half = 2 ** (bits - 1)
    s = scales.to(dtype)
    return (codes.to(torch.int64) + half), s, (-half * s.float()).to(dtype)


def mlx_rtn(w: torch.Tensor, bits: int, group: int, dtype):
    """Round-to-nearest MLX affine quantization of w [..., out, in] (min/max per group)."""
    x = w.float().reshape(*w.shape[:-1], -1, group)
    lo, hi = x.amin(-1, keepdim=True), x.amax(-1, keepdim=True)
    s = ((hi - lo) / (2 ** bits - 1)).clamp(min=1e-8).to(dtype).float()
    b = lo.to(dtype).float()
    u = ((x - b) / s).round().clamp(0, 2 ** bits - 1)
    return u.reshape(w.shape).to(torch.int64), s[..., 0].to(dtype), b[..., 0].to(dtype)


def mlx_dequant(u: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor, group: int) -> torch.Tensor:
    """u [..., out, in], scale/bias [..., out, in/g] -> fp32 weights."""
    x = u.float().reshape(*u.shape[:-1], -1, group)
    return (x * scale.float()[..., None] + bias.float()[..., None]).reshape(u.shape)


def mlx_bpw(bits: int, group: int, dtype_bits: int = 16) -> float:
    return bits + 2 * dtype_bits / group
