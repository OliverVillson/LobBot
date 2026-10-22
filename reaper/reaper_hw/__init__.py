"""Hardware detection and the run plan that tunes REAPer to the machine it runs on.

    from reaper_hw import detect, plan
    hw = detect()                       # GPUs, CPU, RAM, disk, CUDA / driver / torch
    p = plan(hw, "Qwen/Qwen3-30B-A3B-Instruct-2507", prune=0.5)
    p.ok, p.notes, p.problems           # what was adapted, and why it cannot run

    python -m reaper_hw [model] [--prune 0.5] [--json]
"""

from reaper_hw.hardware import Gpu, Hardware, detect, home
from reaper_hw.model import ModelInfo, model_info
from reaper_hw.plan import Plan, plan

__all__ = ["Gpu", "Hardware", "ModelInfo", "Plan", "detect", "home", "main", "model_info", "plan"]


def main(argv: list[str] | None = None) -> int:
    from reaper_hw.cli import main as cli_main
    return cli_main(argv)
