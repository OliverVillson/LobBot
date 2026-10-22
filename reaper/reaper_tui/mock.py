"""--demo mode: a simulated B200 machine and pipeline run, for trying the TUI
on a laptop. Events use the same protocol as the real pipeline."""

from __future__ import annotations

import asyncio
import random
from typing import AsyncIterator

from reaper_tui.backend import DEFAULT_STAGES, Gpu, Hardware, Plan
from reaper_tui.catalog import MoeModel

DEMO_HW = Hardware([Gpu("NVIDIA B200", 179.1, 10.0)], 32, 256.0, 1800.0, source="demo")


def demo_plan(m: MoeModel | None) -> Plan:
    if m is None:
        return Plan(True)
    gb = m.total_b * 2
    if gb > 179 * 0.9:
        return Plan(True, [f"teacher: {gb:.0f} GB bf16 does not fit, serving with fp8 (demo)",
                           f"REAP: {gb - 165:.0f} GB of weights spill to CPU RAM (demo)"])
    return Plan(True, ["teacher: vLLM on 1 GPU, bf16 (demo)", "REAP: weights on GPU, 8 sequences per pass (demo)",
                       "Blackwell: prebuilt FLASH_ATTN kernels (demo)"])


async def run_pipeline(model: MoeModel, ratio: float) -> AsyncIterator[dict]:
    keep = round(model.experts * (1 - ratio))
    for stage in DEFAULT_STAGES:
        yield {"stage": stage, "status": "running", "pct": 0, "msg": ""}
        steps = random.randint(8, 14)
        for i in range(1, steps + 1):
            await asyncio.sleep(0.12)
            msg = {
                "plan": "reading model config",
                "weights": f"{model.id} shard {i}/{steps}",
                "calibrate": f"teacher: {i * 40} answers",
                "score": f"layer {i * 3}/{steps * 3}",
                "prune": f"{model.experts} → {keep} experts / layer" if i == steps else f"rewriting shard {i}/{steps}",
                "eval": f"held-out {i * 2}/{steps * 2}",
                "package": "writing model card",
            }[stage]
            yield {"stage": stage, "status": "running", "pct": round(100 * i / steps), "msg": msg}
        extra = ({"top1_agreement": 0.91, "kl": 0.084, "ppl_teacher": 4.12, "ppl_pruned": 4.57}
                 if stage == "eval" else {})
        yield {"stage": stage, "status": "done", "pct": 100, "msg": "", **extra}
    yield {"stage": "pipeline", "status": "done", "pct": 100,
           "model": f"~/.reaper/jobs/{model.id.split('/')[-1]}-prune{round(ratio * 100)}/out (demo)"}
