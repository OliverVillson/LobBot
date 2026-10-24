"""Stage qheal: distillation around the quantized MLX model (see heal.py)."""

from __future__ import annotations

from blackhole_pipeline.common import stage_main
from blackhole_pipeline.stages.heal import run_quantized

if __name__ == "__main__":
    stage_main(run_quantized, "qheal")
