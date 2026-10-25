"""Shared pieces of our stage processes: the REAP-pruned model loaded for quantization,
calibration batches, and the sequential (chunked) per-expert GPTQ loop both quant stages use."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import traceback
from pathlib import Path

from blackhole_pipeline import use_gpu_perf
from blackhole_pipeline.job import BHJob

use_gpu_perf()


def emit(stage: str, status: str = "running", pct: float | None = None, msg: str = "", **extra) -> None:
    from reaper_pipeline.events import emit as e
    e(stage, status, pct, msg, **extra)


def stage_main(run, stage: str) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    bh = BHJob(ap.parse_args().job)
    try:
        run(bh, bh.reaper())
    except Exception as e:
        traceback.print_exc()
        emit(stage, "error", msg=f"{type(e).__name__}: {e}"[:400])
        sys.exit(1)


class Pruned:
    """The REAP-pruned bf16 model, its MoE blocks, per-expert weight views and checkpoint names."""

    def __init__(self, rj, n_calib: int, src: Path | None = None):
        from bench_b200 import expert_views
        from reaper_pipeline import moe
        from reaper_pipeline.load import input_device, load_model, load_tokenizer
        from reaper_pipeline.stages._common import batches, encode, pad_batch, profile, read_jsonl

        self.rj = rj
        self.src = src or rj.out_dir
        if not (self.src / "config.json").exists():
            raise RuntimeError(f"no pruned model in {self.src}")
        prof = profile(rj)
        self.model = load_model(self.src, prof, rj.config.trust_remote_code)
        self.blocks = moe.find_blocks(self.model)
        if not self.blocks:
            raise RuntimeError("no MoE blocks in the pruned model")
        self.H = moe.text_config(self.model.config).hidden_size
        self.fused = self.blocks[0].fused   # [E, ...] expert tensors (transformers 5 Qwen3.5 / 3.6, ...)
        names = {id(p): n for n, p in self.model.named_parameters()}
        mods = {id(m): n for n, m in self.model.named_modules()}
        self.views, self.names = {}, {}
        for b in self.blocks:
            self.views[b.layer] = [expert_views(b, j, self.H) for j in range(b.num_experts)]
            ex = b.experts
            if self.fused:   # one module holds every expert: its name stands for all of them
                self.names[b.layer] = [{k: mods[id(ex)] for k in ("gate", "up", "down")}] * b.num_experts
                continue
            self.names[b.layer] = [{k: names[id(getattr(ex[j], f"{k}_proj").weight)][:-len(".weight")]
                                    for k in ("gate", "up", "down")} for j in range(b.num_experts)]
        tok = load_tokenizer(self.src, rj.config.trust_remote_code)
        rows = read_jsonl(rj.path("data", "calib.jsonl"))[:n_calib]
        if not rows:
            raise RuntimeError("no calibration data (data/calib.jsonl)")
        enc = [encode(tok, r["prompt"], r.get("answer"), rj.config.max_seq)[0] for r in rows]
        dev = input_device(self.model)
        bs = max(1, int((prof.get("reap") or {}).get("batch_seqs", 1)))
        self.batches = [pad_batch([enc[i] for i in g], tok.pad_token_id, dev)
                        for g in batches(enc, bs, bs * rj.config.max_seq)]
        self.n_calib = len(rows)
        self.params = sum(w.numel() for vs in self.views.values() for v in vs for w in v.values())

    def hessians(self, blocks, diag: bool = False) -> dict:
        from bench_b200 import expert_hessians
        return expert_hessians(self.model, blocks, self.batches, self.H, diag=diag)

    def chunks(self, size: int):
        for c0 in range(0, len(self.blocks), size):
            yield self.blocks[c0:c0 + size]

    def stacked(self, layer: int, js: list[int], device):
        """gate+up rows as one [n, 2I, H] matrix and down [n, H, I] for experts js (fp32 copies)."""
        import torch
        vs = self.views[layer]
        gu = torch.stack([torch.cat([vs[j]["gate"], vs[j]["up"]]) for j in js]).to(device).float()
        dn = torch.stack([vs[j]["down"] for j in js]).to(device).float()
        return gu, dn

    def put(self, layer: int, js: list[int], qgu, qdn) -> None:
        import torch
        with torch.no_grad():
            vs = self.views[layer]
            I = vs[js[0]]["gate"].shape[0]
            for i, j in enumerate(js):
                v = vs[j]
                v["gate"].copy_(qgu[i, :I].to(v["gate"].dtype))
                v["up"].copy_(qgu[i, I:].to(v["up"].dtype))
                v["down"].copy_(qdn[i].to(v["down"].dtype))

    def compare(self, on_progress=None) -> dict:
        import torch
        from reaper_pipeline.stages.eval import compare
        ref = self.rj.path("work", "teacher_ref.pt")
        if not ref.exists():
            return {}
        return compare(self.model, torch.load(ref, weights_only=False), on_progress)


def expert_keys(names: dict) -> set[str]:
    """Checkpoint tensor names of every routed expert weight ("<module>.weight")."""
    return {f"{n}.weight" for per in names.values() for e in per for n in e.values()}


def write_checkpoint(src: Path, dst: Path, replace: set[str], new_tensors, cfg_update: dict,
                     on_progress=None) -> int:
    """Copy src's safetensors to dst without the tensors in `replace`, then add the tensors yielded
    by new_tensors() as extra shards; copy tokenizer and config files; config.json gets
    cfg_update merged in. Returns bytes written."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    from reaper_pipeline.ckpt import SKIP_DIRS, WEIGHT_SUFFIXES, shard_map

    tmp = dst.with_name(dst.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    weight_map, total = {}, 0
    shards = shard_map(src)
    for i, (fname, keys) in enumerate(shards.items()):
        out = {}
        with safe_open(src / fname, "pt") as f:
            for k in keys:
                if k not in replace:
                    out[k] = f.get_tensor(k)
        if out:
            name = f"rest-{i:05d}.safetensors"
            save_file(out, str(tmp / name), metadata={"format": "pt"})
            for k, t in out.items():
                weight_map[k] = name
                total += t.numel() * t.element_size()
        if on_progress:
            on_progress(i + 1, len(shards), "copying non-expert tensors")
    for name, tensors in new_tensors():
        save_file(tensors, str(tmp / name), metadata={"format": "pt"})
        for k, t in tensors.items():
            weight_map[k] = name
            total += t.numel() * t.element_size()
    (tmp / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": total}, "weight_map": dict(sorted(weight_map.items()))}, indent=2) + "\n")
    for p in src.iterdir():
        if p.name in SKIP_DIRS or p.is_dir() or p.name.endswith(WEIGHT_SUFFIXES) \
                or p.name in ("model.safetensors.index.json", "config.json"):
            continue
        shutil.copy2(p, tmp / p.name)
    cfg = json.loads((src / "config.json").read_text())
    cfg.update(cfg_update)
    (tmp / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    shutil.rmtree(dst, ignore_errors=True)
    tmp.rename(dst)
    return total


def gib(n_bytes: float) -> float:
    return n_bytes / 1024 ** 3
