"""Stage quant: routed experts of the pruned model to block fp8 (quant.py).

Writes out/<name>-FP8/ next to the bf16 pruned model, which eval has already
checked. Skipped when config.quant is "none" or the checkpoint is already
quantized (fp8 / mxfp4 originals stay in their own format).
"""

from __future__ import annotations

import json

from reaper_pipeline import quant
from reaper_pipeline.events import emit, size
from reaper_pipeline.job import Job
from reaper_pipeline.stages._common import stage_main

STAGE = "quant"


def run(job: Job) -> None:
    src, dst = job.out_dir, job.quant_dir
    if not (src / "config.json").exists():
        raise RuntimeError(f"no pruned model in {src}")
    if job.config.quant == "none":
        job.mark_done(STAGE, {"skipped": "quant is none"})
        emit(STAGE, "skipped", 100, "quant: none")
        return
    if job.config.quant != "fp8-experts":
        raise ValueError(f"unknown quant {job.config.quant!r} (fp8-experts or none)")
    if (m := quant.already_quantized(json.loads((src / "config.json").read_text()))):
        job.mark_done(STAGE, {"skipped": f"already {m}"})
        emit(STAGE, "skipped", 100, f"checkpoint is already {m}")
        return
    r = quant.quantize_experts(src, dst, lambda i, n, f: emit(STAGE, pct=100 * i / n, msg=f"shard {i}/{n}"))
    job.write("work/quant.json", r)
    job.mark_done(STAGE, {"out": str(dst), **r})
    emit(STAGE, "done", 100, f"experts in fp8: {size(r['bytes_before'])} → {size(r['bytes_after'])}")


if __name__ == "__main__":
    stage_main(run, STAGE)
