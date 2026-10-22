"""Prune a checkpoint on disk, shard by shard, in its own format.

Rewriting the safetensors files instead of calling save_pretrained keeps the
pruned model in exactly the layout of the original (per-expert or fused keys,
fp8 / mxfp4 blocks and scales, remote code), so whatever loads the original
(transformers, vLLM, SGLang, llama.cpp's converter) loads the pruned one, and
it needs one shard of memory instead of the whole model.

Per MoE layer L with kept experts keep[L] (original ids, ascending):
  <pre>.experts.<j>.<rest>      kept and renumbered to its position in keep[L], else dropped
  <pre>.experts.<rest>         fused [E, ...] tensor: rows keep[L]
  <pre>.<router>.<rest>        [E, ...] router weight / bias / correction bias: rows keep[L]
  <pre>.e_score_correction_bias  (MiniMax-M2 keeps it on the block) likewise
Layers past num_hidden_layers (DeepSeek/GLM multi-token-prediction heads) are
not scored by REAP; they are dropped and num_nextn_predict_layers set to 0.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from reaper_pipeline.moe import EXPERT_COUNT_KEYS, ROUTER_NAMES

SKIP_DIRS = {"original", "metal", ".cache", ".git"}
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".gguf", ".msgpack", ".h5")
_EXPERT = re.compile(r"^(?P<pre>.+?)\.experts\.(?P<rest>.+)$")
_LAYER = re.compile(r"layers\.(\d+)")


def _layer(key: str) -> int | None:
    m = _LAYER.findall(key)
    return int(m[-1]) if m else None


def shard_map(src: Path) -> dict[str, list[str]]:
    """shard file -> its tensor names."""
    idx = src / "model.safetensors.index.json"
    if idx.exists():
        out: dict[str, list[str]] = {}
        for k, f in json.loads(idx.read_text())["weight_map"].items():
            out.setdefault(f, []).append(k)
        return out
    single = src / "model.safetensors"
    if single.exists():
        from safetensors import safe_open
        with safe_open(single, "pt") as f:
            return {"model.safetensors": list(f.keys())}
    raise FileNotFoundError(f"no safetensors checkpoint in {src}")


def plan_keys(keys: list[str], keep: dict[int, list[int]], num_layers: int):
    """For every key: ("copy"|"drop"|("rename", new)|("rows", keep_ids)), plus notes."""
    prefixes: dict[str, int] = {}
    for k in keys:
        m = _EXPERT.match(k)
        if m and (L := _layer(m["pre"])) is not None:
            prefixes[m["pre"]] = L
    notes = []
    missing = sorted({L for L in prefixes.values() if L < num_layers and L not in keep})
    if missing:
        raise RuntimeError(f"checkpoint has experts in layers {missing} that REAP did not score")
    mtp = sorted({L for k in keys if (L := _layer(k)) is not None and L >= num_layers})
    if mtp:
        notes.append(f"dropped multi-token-prediction layers {mtp} (not scored by REAP)")

    actions = {}
    for k in keys:
        L = _layer(k)
        if L is not None and L >= num_layers:
            actions[k] = "drop"
            continue
        m = _EXPERT.match(k)
        if m and m["pre"] in prefixes:
            kept = keep[prefixes[m["pre"]]]
            head, _, tail = m["rest"].partition(".")
            if head.isdigit():
                j = int(head)
                actions[k] = ("rename", f"{m['pre']}.experts.{kept.index(j)}.{tail}") if j in kept else "drop"
            else:
                actions[k] = ("rows", kept)
            continue
        pre = next((p for p in prefixes if k.startswith(p + ".")), None)
        if pre is not None:
            rest = k[len(pre) + 1:]
            first = rest.split(".")[0]
            if first in ROUTER_NAMES or rest == "e_score_correction_bias":
                actions[k] = ("rows", keep[prefixes[pre]])
                continue
        actions[k] = "copy"
    return actions, notes


def prune_config(cfg: dict, n_keep: int, dropped_mtp: bool) -> dict:
    def fix(c: dict):
        for key in EXPERT_COUNT_KEYS:
            if isinstance(c.get(key), int) and c[key] > 1:
                c[key] = n_keep
        if isinstance(c.get("num_experts_per_tok"), int):
            c["num_experts_per_tok"] = min(c["num_experts_per_tok"], n_keep)
        if dropped_mtp and c.get("num_nextn_predict_layers"):
            c["num_nextn_predict_layers"] = 0
    fix(cfg)
    for sub in ("text_config", "language_config", "llm_config"):
        if isinstance(cfg.get(sub), dict):
            fix(cfg[sub])
    return cfg


def prune_checkpoint(src: str | Path, dst: str | Path, keep: dict[int, list[int]], num_experts: int,
                     num_layers: int, on_progress=None) -> dict:
    """Write the pruned copy of src to dst. Returns a summary."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    src, dst = Path(src), Path(dst)
    n_keep = len(next(iter(keep.values())))
    if any(len(v) != n_keep for v in keep.values()):
        raise ValueError("every layer must keep the same number of experts")
    shards = shard_map(src)
    actions, notes = plan_keys([k for ks in shards.values() for k in ks], keep, num_layers)

    tmp = dst.with_name(dst.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    weight_map, total, before = {}, 0, 0
    for i, (fname, names) in enumerate(shards.items()):
        out = {}
        with safe_open(src / fname, "pt") as f:
            for k in names:
                a = actions[k]
                t = f.get_tensor(k)
                before += t.numel() * t.element_size()
                if a == "drop":
                    continue
                if a == "copy":
                    out[k] = t
                elif a[0] == "rename":
                    out[a[1]] = t
                else:
                    if t.shape[0] != num_experts:
                        out[k] = t  # not per-expert after all (e.g. a [1, H] shared gate)
                    else:
                        out[k] = t.index_select(0, torch.tensor(a[1])).contiguous()
        for k, t in out.items():
            weight_map[k] = fname
            total += t.numel() * t.element_size()
        if out:
            save_file(out, str(tmp / fname), metadata={"format": "pt"})
        del out
        if on_progress:
            on_progress(i + 1, len(shards), fname)

    if (src / "model.safetensors.index.json").exists() or len(shards) > 1:
        idx_meta = {"total_size": total}
        (tmp / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": idx_meta, "weight_map": dict(sorted(weight_map.items()))}, indent=2) + "\n")

    for p in src.iterdir():
        if p.name in SKIP_DIRS or p.is_dir() or p.name.endswith(WEIGHT_SUFFIXES) \
                or p.name in ("model.safetensors.index.json", "config.json"):
            continue
        shutil.copy2(p, tmp / p.name)
    cfg = json.loads((src / "config.json").read_text())
    dropped_mtp = any("multi-token" in n for n in notes)
    (tmp / "config.json").write_text(json.dumps(prune_config(cfg, n_keep, dropped_mtp), indent=2) + "\n")

    shutil.rmtree(dst, ignore_errors=True)
    tmp.rename(dst)
    return {"bytes_before": before, "bytes_after": total, "notes": notes}
