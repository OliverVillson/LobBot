"""The BlackHole pipeline the TUI runs: `python -m blackhole_pipeline --job DIR`.

DIR is a TUI job (config.json from blackhole_tui: model, schema, target, purpose,
steps). The heavy lifting is REAPer's pipeline (reaper_pipeline) for detect →
calibrate → REAP → eval, plus two stages of our own on the pruned model:

  gptq   GPTQ int4 (group 128) on the routed experts, measured against the teacher and
         written as a compressed-tensors W4A16 checkpoint (the shippable artifact)
  reaq   REAQ per-expert codebooks: GPTQ rounding to a learned level table per expert
         matrix, the top half of each layer's experts by REAP score on 16 levels and the
         rest on 8 (block 32, fp8 block scales). No kernel serves this yet, so it is a
         measured stage: KL / top-1 / size against int4, and the packed indices, tables
         and scales are saved for a future kernel.
  bench  LiveCodeBench pass@1 of the original and the int4 model (one sample, 4k tokens)

Events are REAPer's JSON lines; stage names are STAGES.
"""

from __future__ import annotations

import sys
from pathlib import Path

STAGES = ["detect", "plan", "weights", "calibrate", "score", "prune", "eval", "gptq", "reaq", "bench", "quant", "package"]

# (process, stages it reports, who runs it): REAPer processes are reaper_pipeline.stages.<name>,
# ours blackhole_pipeline.stages.<name>. Each process frees the GPU when it exits.
PROCESSES = [
    ("detect", ["detect"], "bh"),
    ("plan", ["plan"], "reaper"),
    ("weights", ["weights"], "reaper"),
    ("calibrate", ["calibrate"], "reaper"),
    ("reap", ["score", "prune"], "reaper"),
    ("eval", ["eval"], "reaper"),
    ("gptq", ["gptq"], "bh"),
    ("reaq", ["reaq"], "bh"),
    ("bench", ["bench"], "bh"),
    ("quant", ["quant"], "reaper"),
    ("package", ["package"], "bh"),
]

DEFAULTS = {
    "prune": 0.25,           # REAP share of routed experts dropped (best of 2026-10-23: 25%)
    "calib_samples": 256,
    "gptq_calib": 64,        # calibration sequences for the GPTQ Hessians (64 was enough on B200)
    "layer_chunk": 12,       # MoE layers per sequential GPTQ chunk
    "reaq_top": 0.5,         # share of each layer's experts (by REAP score) on 16-level tables
    "reaq_block": 32,
    "bench": "lcb-v6",       # LiveCodeBench release (problems new in it) for original vs int4; "none" skips
    "bench_limit": 0,        # 0 = every problem of the release
    "bench_max_tokens": 4096,
    "bench_file": "",        # a local jsonl in LiveCodeBench's format instead of the download
}


def gpu_perf_dir() -> Path:
    """reaper/gpu_perf next to the installed reaper_pipeline (an editable install of LobBot)."""
    import reaper_pipeline
    return Path(reaper_pipeline.__file__).resolve().parents[1] / "gpu_perf"


def use_gpu_perf() -> None:
    d = str(gpu_perf_dir())
    if d not in sys.path:
        sys.path.insert(0, d)
