"""Expert-only FP8 quantization of a (pruned) checkpoint, shard by shard.

Routed experts hold almost all of a MoE's weights, so only they are quantized:
every 2-D `.experts.<j>.<proj>.weight` becomes float8_e4m3 with one float32
scale per 128x128 block (`<proj>.weight_scale_inv`, multiplied in to
dequantize). Attention, routers, shared experts, embeddings and the LM head
stay in bf16 and are listed in the config's ignore list. This is the
"fine-grained fp8" format of DeepSeek-V3 and Qwen3-*-FP8, which vLLM, SGLang
and transformers load natively (fp8 matmuls on Hopper and Blackwell).
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from reaper_pipeline.ckpt import SKIP_DIRS, WEIGHT_SUFFIXES, shard_map

BLOCK = 128
FP8_MAX = 448.0
_EXPERT_W = re.compile(r"\.experts\.\d+\.[^.]+\.weight$")


def already_quantized(cfg: dict) -> str | None:
    q = cfg.get("quantization_config") or (cfg.get("text_config") or {}).get("quantization_config")
    return (q or {}).get("quant_method") or ("quantized" if q else None)


def quantize_block_fp8(w):
    """[out, in] weight -> (fp8 weight, [ceil(out/128), ceil(in/128)] float32 scales)."""
    import torch
    out, inp = w.shape
    po, pi = -out % BLOCK, -inp % BLOCK
    x = torch.nn.functional.pad(w.float(), (0, pi, 0, po))
    x = x.view((out + po) // BLOCK, BLOCK, (inp + pi) // BLOCK, BLOCK)
    scale = (x.abs().amax(dim=(1, 3)) / FP8_MAX).clamp(min=1e-12)
    q = (x / scale[:, None, :, None]).clamp(-FP8_MAX, FP8_MAX)
    q = q.view(out + po, inp + pi)[:out, :inp].to(torch.float8_e4m3fn).contiguous()
    return q, scale.contiguous()


def dequantize_block_fp8(q, scale):
    import torch
    out, inp = q.shape
    s = scale.repeat_interleave(BLOCK, 0)[:out].repeat_interleave(BLOCK, 1)[:, :inp]
    return q.to(torch.float32) * s


def quantize_experts(src: str | Path, dst: str | Path, on_progress=None) -> dict:
    """Write src with expert weights in block fp8 to dst. Returns a summary."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    src, dst = Path(src), Path(dst)
    cfg = json.loads((src / "config.json").read_text())
    if (m := already_quantized(cfg)):
        raise ValueError(f"checkpoint is already quantized ({m})")
    shards = shard_map(src)
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    weight_map, ignored, before, after, n_q = {}, set(), 0, 0, 0
    for i, (fname, names) in enumerate(shards.items()):
        out = {}
        with safe_open(src / fname, "pt") as f:
            for k in names:
                t = f.get_tensor(k)
                before += t.numel() * t.element_size()
                if _EXPERT_W.search(k) and t.dim() == 2 and t.is_floating_point():
                    q, s = quantize_block_fp8(t)
                    out[k], out[k[:-len("weight")] + "weight_scale_inv"] = q, s
                    n_q += 1
                else:
                    out[k] = t
                    if k.endswith(".weight") and t.dim() == 2 and "embed" not in k:
                        ignored.add(k[:-len(".weight")])
        for k, t in out.items():
            weight_map[k] = fname
            after += t.numel() * t.element_size()
        save_file(out, str(tmp / fname), metadata={"format": "pt"})
        del out
        if on_progress:
            on_progress(i + 1, len(shards), fname)
    if not n_q:
        shutil.rmtree(tmp)
        raise ValueError("no per-expert 2-D weights found (fused expert tensors are not supported yet)")

    if (src / "model.safetensors.index.json").exists() or len(shards) > 1:
        (tmp / "model.safetensors.index.json").write_text(json.dumps(
            {"metadata": {"total_size": after}, "weight_map": dict(sorted(weight_map.items()))}, indent=2) + "\n")
    for p in src.iterdir():
        if p.name in SKIP_DIRS or p.is_dir() or p.name.endswith(WEIGHT_SUFFIXES) \
                or p.name in ("model.safetensors.index.json", "config.json"):
            continue
        shutil.copy2(p, tmp / p.name)
    skip = sorted(ignored)
    cfg["quantization_config"] = {
        "quant_method": "fp8", "fmt": "e4m3", "activation_scheme": "dynamic",
        "weight_block_size": [BLOCK, BLOCK], "modules_to_not_convert": skip, "ignored_layers": skip}
    (tmp / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    shutil.rmtree(dst, ignore_errors=True)
    tmp.rename(dst)
    return {"bytes_before": before, "bytes_after": after, "expert_weights": n_q, "kept_bf16": len(skip)}
