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
  heal   distillation of the pruned model from the teacher on fresh calibration prompts
  mlx    an MLX folder for Apple silicon: GPTQ 3-bit experts (optionally a REAQ 4-bit hot tier),
         4-bit attention and embeddings, flint_mlx.py for mlx-lm
  qheal  the heal again around the quantized MLX model (frozen 3-bit experts, fake-quant attention),
         so it repairs the quantization error too; the bench then measures the healed folder
  bench  LiveCodeBench pass@1 and tok/s of the original and the REAQ model (bench_models) (one sample, up to 32k tokens)

Events are REAPer's JSON lines; stage names are STAGES. A schema's stage_list (blackhole_tui.catalog)
picks which of them a job runs.
"""

from __future__ import annotations

import sys
from pathlib import Path

STAGES = ["detect", "plan", "weights", "calibrate", "score", "prune", "eval", "heal", "gptq", "reaq", "mlx", "qheal", "bench",
          "quant", "package"]

# (process, stages it reports, who runs it): REAPer processes are reaper_pipeline.stages.<name>,
# ours blackhole_pipeline.stages.<name>. Each process frees the GPU when it exits.
PROCESSES = [
    ("detect", ["detect"], "bh"),
    ("plan", ["plan"], "reaper"),
    ("weights", ["weights"], "reaper"),
    ("calibrate", ["calibrate"], "reaper"),
    ("reap", ["score", "prune"], "reaper"),
    ("eval", ["eval"], "reaper"),
    ("heal", ["heal"], "bh"),
    ("gptq", ["gptq"], "bh"),
    ("reaq", ["reaq"], "bh"),
    ("mlx", ["mlx"], "bh"),
    ("qheal", ["qheal"], "bh"),
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
    "heal_prompts": 1024,    # fresh calibration-set prompts the teacher answers for distillation
    "heal_max_tokens": 1536,
    "heal_epochs": 1.0,
    "heal_accum": 8,         # sequences per optimizer step
    "heal_lr": 3e-6,         # 1e-5 overfit Flint's attention (held-out KL 0.31 -> 0.40)
    "heal_eval_every": 8,    # optimizer steps between held-out checks (early stopping, best point kept)
    "heal_minutes": 45,      # wall-clock cap on training (0 = none)
    "mlx_bits": 3,           # MLX export: routed experts (GPTQ)
    "mlx_group": 64,
    "mlx_hot": 0.0,          # share of each layer's experts (by REAP score) on mlx_hot_bits
    "mlx_hot_bits": 4,
    "mlx_attn_bits": 4,      # attention, shared experts, dense MLPs (round-to-nearest)
    "mlx_embed_bits": 4,
    "mlx_head_bits": 6,
    "bench": "lcb-v6",       # LiveCodeBench release (problems new in it) for original vs int4; "none" skips
    "bench_models": "original,reaq",  # any of original, int4, reaq, mlx
    "bench_limit": 0,        # 0 = every problem of the release
    "bench_max_tokens": 32768,  # reasoning models think for tens of thousands of tokens; 4k cut most off
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
