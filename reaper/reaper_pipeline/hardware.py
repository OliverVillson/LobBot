"""The run profile: how this machine runs each stage.

The hardware thread's `reaper_hw` package owns detection and planning
(`detect()` and `plan(hardware, model, prune)` returning an object with
`.ok/.notes/.problems` and per-stage settings). This module calls it when it is
installed and otherwise falls back to a small built-in planner, then flattens
either answer into one dict with every key the stages read:

    ok, problems, notes, summary, device ("cuda" | "cpu"), n_gpus, dtype,
    reap:    placement, max_memory ({gpu index | "cpu": "NGiB"}), batch_seqs
    teacher: tensor_parallel_size, gpu_memory_utilization, quantization,
             max_model_len, kv_cache_dtype, max_num_seqs
    env:     environment variables for the stage processes (vLLM backends)
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
from pathlib import Path

GB = 1024**3


def _as_dict(obj) -> dict:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    return {k: v for k, v in vars(obj).items() if not k.startswith("_")}


def _external(model: str, prune: float, max_seq: int):
    """(hardware dict, plan dict) from reaper_hw, or None when it is not installed
    (or REAPER_HW=builtin, which CPU tests use)."""
    if os.environ.get("REAPER_HW") == "builtin":
        return None
    mod = None
    for name in ("reaper_hw", "reaper_core.hw"):
        try:
            mod = __import__(name, fromlist=["detect"])
            break
        except ImportError:
            continue
    if mod is None:
        return None
    hw = mod.detect()
    planner = getattr(mod, "plan", None) or getattr(mod, "make_plan")
    try:
        p = planner(hw, model, prune, max_seq)
    except TypeError:
        p = planner(hw, model, prune)
    hw_d = _as_dict(hw)
    if hasattr(hw, "summary"):
        hw_d["summary"] = hw.summary() if callable(hw.summary) else hw.summary
    return hw_d, _as_dict(p)


# --------------------------------------------------------------------------- built-in fallback


def _gpus() -> list[dict]:
    try:
        import torch
        if not torch.cuda.is_available():
            return []
        out = []
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            free, total = torch.cuda.mem_get_info(i)
            out.append({"index": i, "name": p.name, "mem_gb": total / GB, "free_gb": free / GB,
                        "cc": p.major + p.minor / 10})
        return out
    except Exception:
        return []


def _ram_gb() -> float:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return float(line.split()[1]) / 1024**2
    except OSError:
        pass
    return 16.0


def model_bytes(model: str) -> float:
    """bf16-ish size of the checkpoint, from local files or the Hub index."""
    p = Path(model).expanduser()
    if p.is_dir():
        n = sum(f.stat().st_size for f in p.glob("*.safetensors"))
        if n:
            return n
    try:
        from huggingface_hub import hf_hub_download
        idx = json.loads(Path(hf_hub_download(model, "model.safetensors.index.json")).read_text())
        return float(idx["metadata"]["total_size"])
    except Exception:
        return 0.0


def builtin_plan(model: str, prune: float, max_seq: int) -> tuple[dict, dict]:
    gpus = _gpus()
    ram = _ram_gb()
    W = model_bytes(model) / GB
    hw = {"gpus": gpus, "cpus": os.cpu_count() or 1, "ram_free_gb": round(ram, 1),
          "disk_free_gb": round(shutil.disk_usage(Path.home()).free / GB, 1)}
    plan: dict = {"ok": True, "notes": [], "problems": [], "env": {}}
    if not gpus:
        hw["summary"] = f"no GPU · {hw['cpus']} CPUs · {ram:.0f} GB RAM free"
        plan.update(dtype="float32" if W * 2 < ram * 0.5 else "bfloat16",
                    reap={"placement": "cpu", "max_memory": None, "batch_seqs": 1},
                    teacher={"tensor_parallel_size": 0})
        plan["notes"].append("no GPU: running on CPU, fine for tiny test models only")
        return hw, plan
    g = gpus[0]
    cc = min(x["cc"] for x in gpus)
    hw["summary"] = f"{len(gpus)}x {g['name']} {g['mem_gb']:.0f} GB (sm_{int(cc * 10)}) · {ram:.0f} GB RAM free"
    dtype = "bfloat16" if cc >= 8.0 else "float16"
    per_gpu = min(x["mem_gb"] for x in gpus) - (6 + 0.02 * g["mem_gb"])
    total = per_gpu * len(gpus)
    if W <= total:
        placement, headroom = ("gpu" if W <= per_gpu else "multi-gpu"), total - W
    elif W <= total + ram * 0.8:
        placement, headroom = "offload", 2.0
        plan["notes"].append(f"REAP: {W - total:.0f} GB of weights spill to CPU RAM (slow)")
    else:
        placement, headroom = "none", 0.0
        plan["ok"] = False
        plan["problems"].append(f"the model needs ~{W:.0f} GB; GPUs + RAM hold {total + ram * 0.8:.0f} GB")
    mm = {str(x["index"]): f"{int(per_gpu)}GiB" for x in gpus}
    mm["cpu"] = f"{int(ram * 0.8)}GiB"
    batch = int(max(1, min(16, headroom / max(max_seq / 2048 * 2.5, 0.1))))
    tp = 1
    while tp < len(gpus) and tp * g["mem_gb"] * 0.9 < W * 1.1 + 8:
        tp *= 2
    fits = tp * g["mem_gb"] * 0.9 >= W * 1.1 + 8
    plan.update(dtype=dtype, reap={"placement": placement, "max_memory": mm, "batch_seqs": batch},
                teacher={"tensor_parallel_size": tp, "gpu_memory_utilization": 0.9,
                         "quantization": None if fits or cc < 8.9 else "fp8",
                         "max_model_len": max(8192, max_seq * 2), "kv_cache_dtype": "auto",
                         "max_num_seqs": 256})
    if cc >= 10.0:
        plan["env"] = {"VLLM_ATTENTION_BACKEND": "FLASH_ATTN", "VLLM_USE_FLASHINFER_SAMPLER": "0"}
        plan["notes"].append("Blackwell: prebuilt FLASH_ATTN backend for the vLLM teacher")
    return hw, plan


# --------------------------------------------------------------------------- profile


def profile(model: str, prune: float, max_seq: int = 2048) -> tuple[dict, dict]:
    """(hardware, run profile) for this machine and model."""
    hw, base = builtin_plan(model, prune, max_seq)
    source = "built-in"
    try:
        ext = _external(model, prune, max_seq)
    except Exception as e:  # never let a planner bug stop the run; say so instead
        ext = None
        base["notes"].append(f"reaper_hw failed ({e}); used the built-in planner")
    if ext is not None:
        hw, p = ext
        source = "reaper_hw"
        mine = base.get("reap", {}).get("max_memory")
        for k, v in p.items():
            if isinstance(v, dict) and isinstance(base.get(k), dict):
                base[k] = {**base[k], **v}
            elif v is not None:
                base[k] = v
        if p.get("layout") in ("shared", "split"):
            # The pipeline always runs the teacher first and frees the GPUs before REAP
            # loads, so each gets the whole machine rather than a share of it.
            base["teacher"]["gpu_memory_utilization"] = 0.9
            if mine:
                base["reap"]["max_memory"] = mine
            base["notes"].append("teacher and REAP run one after the other, each with all GPU memory")
    gpus = hw.get("gpus") or []
    base["device"] = "cuda" if gpus else "cpu"
    base["n_gpus"] = len(gpus)
    base["summary"] = hw.get("summary", "")
    base["source"] = source
    return hw, base


def torch_dtype(name: str):
    import torch
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}.get(name, torch.bfloat16)
