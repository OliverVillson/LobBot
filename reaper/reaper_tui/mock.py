"""Mocked provider calls and pipeline run. Nothing here touches the network.

The pipeline yields the same progress events as LobBot's pipeline.py
({"stage", "status", "pct", "msg"}) so the real VM pipeline can replace it later.
"""

from __future__ import annotations

import asyncio
import random
from typing import AsyncIterator

from reaper_tui.catalog import MoeModel

STAGES = [
    ("provision", "start the GPU VM"),
    ("weights", "download model weights"),
    ("calibrate", "generate calibration data"),
    ("score", "REAP saliency per expert"),
    ("prune", "drop low-saliency experts"),
    ("eval", "compare against the original"),
    ("package", "safetensors + GGUF"),
]


async def check_inference_key(provider_id: str, key: str) -> tuple[bool, str]:
    await asyncio.sleep(0.6)
    if len(key) < 12:
        return False, "key looks too short"
    return True, f"{provider_id}: key accepted (mock), 42 models visible"


async def check_compute(provider_id: str, key: str, gpu: str) -> tuple[bool, str]:
    await asyncio.sleep(0.6)
    if len(key) < 12:
        return False, "token looks too short"
    return True, f"{provider_id}: {gpu} available in eu-north (mock)"


async def run_pipeline(model: MoeModel, ratio: float) -> AsyncIterator[dict]:
    keep = round(model.experts * (1 - ratio))
    for stage, _ in STAGES:
        yield {"stage": stage, "status": "running", "pct": 0, "msg": ""}
        steps = random.randint(8, 14)
        for i in range(1, steps + 1):
            await asyncio.sleep(0.12)
            msg = {
                "provision": "booting vm",
                "weights": f"{model.id} shard {i}/{steps}",
                "calibrate": f"{i * 64} samples",
                "score": f"layer {i * 3}/{steps * 3}",
                "prune": f"{model.experts} → {keep} experts / layer",
                "eval": f"held-out {i * 10} answers",
                "package": "writing tensors",
            }[stage]
            yield {"stage": stage, "status": "running", "pct": round(100 * i / steps), "msg": msg}
        yield {"stage": stage, "status": "done", "pct": 100, "msg": ""}
