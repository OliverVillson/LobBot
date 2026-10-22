"""`reaper hw`: print what this machine is and how a model would run on it."""

from __future__ import annotations

import argparse
import json

from reaper_hw.hardware import detect
from reaper_hw.model import ModelInfo
from reaper_hw.plan import plan


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="reaper hw", description="Show detected hardware and the plan for a model.")
    ap.add_argument("model", nargs="?", help="Hugging Face id or local folder with config.json")
    ap.add_argument("--prune", type=float, default=0.5, help="share of routed experts to drop (default 0.5)")
    ap.add_argument("--seq", type=int, default=2048, help="tokens per calibration sequence")
    ap.add_argument("--sizes", metavar="TOTAL,ACTIVE,EXPERTS,TOPK",
                    help="plan from headline sizes (billions) instead of reading the config, e.g. 30.5,3.3,128,8")
    ap.add_argument("--work-dir", help="where models and jobs go (default $REAPER_HOME or ~/.reaper)")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    a = ap.parse_args(argv)

    hw = detect(a.work_dir)
    model = a.model
    if a.sizes:
        t, act, e, k = a.sizes.split(",")
        model = ModelInfo.from_sizes(a.model or "model", float(t), float(act), int(e), int(k))
    p = plan(hw, model, a.prune, a.seq) if model else None

    if a.json:
        print(json.dumps({"hardware": hw.to_dict(), "plan": p.to_dict() if p else None}, indent=2))
        return 0 if p is None or p.ok else 1

    print(hw.summary())
    for g in hw.gpus:
        print(f"  gpu{g.index}: {g.name}, {g.mem_gb:.0f} GB ({g.free_gb:.0f} free), {g.arch} {g.sm}")
    stack = [f"driver {hw.driver}" if hw.driver else "", f"CUDA {hw.cuda}" if hw.cuda else "",
             f"torch {hw.torch}" + (f"+cu{hw.torch_cuda}" if hw.torch_cuda else "") if hw.torch else "no torch"]
    print("  " + " · ".join(s for s in stack if s))
    if hw.gpus:
        print(f"  bf16 {'yes' if hw.bf16 else 'no'} · fp8 {'yes' if hw.fp8 else 'no'} · fp4 {'yes' if hw.fp4 else 'no'}")
    print(f"  cpu: {hw.cpu_model} ({hw.cpus} threads) · disk: {hw.disk_path}")
    for w in hw.warnings:
        print(f"  ! {w}")
    if p is None:
        return 0
    m = p.model
    if m:
        print(f"\n{m['id']}: {m['params_b']:g}B params ({m['expert_params_b']:g}B in experts), "
              f"{m['experts']} experts top-{m['top_k']}, {m['bf16_gb']:.0f} GB bf16")
    print(f"plan: {'ok' if p.ok else 'CANNOT RUN'} · layout {p.layout} · {p.dtype}")
    if p.teacher:
        t = p.teacher
        print(f"  teacher  TP {t['tensor_parallel_size']} on GPU {t['gpus']}, {t['gpu_memory_utilization']:.0%} memory, "
              f"weights {t['quantization'] or t['dtype']}, KV {t['kv_cache_dtype']}")
    if p.reap:
        r = p.reap
        print(f"  reap     {r['placement']} (device_map={r['device_map']}), {r['batch_seqs']} seqs/pass")
    if p.verify:
        print(f"  verify   TP {p.verify['tensor_parallel_size']}, pruned model ~{p.verify['pruned_gb']:.0f} GB")
    for n in p.notes:
        print(f"  · {n}")
    for x in p.problems:
        print(f"  ✗ {x}")
    return 0 if p.ok else 1
