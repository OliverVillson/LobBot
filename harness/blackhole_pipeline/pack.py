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
