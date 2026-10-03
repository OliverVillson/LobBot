"""GR00T N1.7 NEW_EMBODIMENT fine-tune on the sim demos, via Isaac-GR00T.

Shells out to the Isaac-GR00T checkout (env LOBBOT_GR00T_REPO, default
/mnt/nvme/Isaac-GR00T, set up by scripts/setup_robot_vm.sh) and its own uv
venv:

    uv run python gr00t/experiment/launch_finetune.py --base-model-path ...
        --dataset-path <lerobot dir> --embodiment-tag NEW_EMBODIMENT
        --modality-config-path <dataset>/farm_tractor_config.py ...

The final model and its processor/ land in out_dir itself
(gr00t/experiment/experiment.py: trainer.save_model() into output_dir,
processor.save_pretrained(output_dir / "processor")), which is what
gr00t/policy/gr00t_policy.py:Gr00tPolicy loads.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path

from common.progress import emit
from stages._util import DRY_RUN, run as run_cmd

STAGE = "finetune"
GR00T_REPO = os.environ.get("LOBBOT_GR00T_REPO", "/mnt/nvme/Isaac-GR00T")
BASE_MODEL = os.environ.get("LOBBOT_GR00T_MODEL", "nvidia/GR00T-N1.7-3B")
MODALITY_CONFIG = "farm_tractor_config.py"  # written by farmsim/lerobot.py


def _base_model_path(base_model: str, models_dir: str | None = None) -> str:
    """Local snapshot from setup_robot_vm.sh if present, else the hub id."""
    root = models_dir or os.environ.get("LOBBOT_MODELS", "/mnt/nvme/models")
    local = Path(root) / base_model.split("/")[-1]
    return str(local) if local.exists() else base_model


def command(dataset_dir, out_dir, steps: int, base_model: str, batch: int = 32,
            num_gpus: int = 1, workers: int = 4, save_steps: int | None = None) -> list[str]:
    dataset_dir = Path(dataset_dir).resolve()
    return [
        "uv", "run", "python", "gr00t/experiment/launch_finetune.py",
        "--base-model-path", base_model,
        "--dataset-path", str(dataset_dir),
        "--embodiment-tag", "NEW_EMBODIMENT",
        "--modality-config-path", str(dataset_dir / MODALITY_CONFIG),
        "--num-gpus", str(num_gpus),
        "--output-dir", str(Path(out_dir).resolve()),
        "--max-steps", str(steps),
        "--save-steps", str(save_steps or max(500, steps // 4)),
        "--global-batch-size", str(batch),
        "--dataloader-num-workers", str(workers),
    ]


def _latest_step(out_dir: Path) -> int:
    steps = [int(m.group(1)) for p in out_dir.glob("checkpoint-*") if (m := re.fullmatch(r"checkpoint-(\d+)", p.name))]
    return max(steps, default=0)


def _watch(out_dir: Path, steps: int, stop: threading.Event, every_s: float = 20.0) -> None:
    """Progress from saved checkpoints (the trainer's own tqdm goes to the log)."""
    last = -1
    while not stop.wait(every_s):
        s = _latest_step(out_dir)
        if s != last:
            last = s
            emit(STAGE, pct=2 + 95 * s / max(steps, 1), msg=f"step {s}/{steps}")


def find_checkpoint(out_dir: Path) -> Path:
    """out_dir when the final save landed there, else the newest checkpoint-N."""
    if (out_dir / "config.json").exists():
        return out_dir
    s = _latest_step(out_dir)
    if s:
        return out_dir / f"checkpoint-{s}"
    raise FileNotFoundError(f"no GR00T checkpoint in {out_dir}")


def finetune(dataset_dir, out_dir, steps: int = 2000, base_model: str = BASE_MODEL,
             gr00t_repo: str | None = None, dry_run: bool | None = None, **kw) -> dict:
    """Run the fine-tune; returns {"checkpoint", "steps", "base_model", "seconds", ...}."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dry = DRY_RUN if dry_run is None else dry_run
    if dry:
        (out_dir / "config.json").write_text(json.dumps({"model_type": "Gr00tN1d7", "dry_run": True}))
        (out_dir / "DRY_RUN").write_text("placeholder checkpoint: no fine-tune ran\n")
        emit(STAGE, pct=100, msg="DRY RUN: placeholder checkpoint")
        return {"checkpoint": str(out_dir), "steps": 0, "base_model": base_model, "seconds": 0.0, "dry_run": True}

    repo = Path(gr00t_repo or GR00T_REPO)
    if not (repo / "gr00t" / "experiment" / "launch_finetune.py").exists():
        raise FileNotFoundError(f"Isaac-GR00T not found at {repo}; run scripts/setup_robot_vm.sh or set LOBBOT_GR00T_REPO")
    if not (Path(dataset_dir) / MODALITY_CONFIG).exists():
        raise FileNotFoundError(f"{MODALITY_CONFIG} missing in {dataset_dir} (farmsim/lerobot.py writes it)")
    base = _base_model_path(base_model)
    cmd = command(dataset_dir, out_dir, steps, base, **kw)
    emit(STAGE, pct=2, msg=f"fine-tuning {base} for {steps} steps")
    stop = threading.Event()
    watcher = threading.Thread(target=_watch, args=(out_dir, steps, stop), daemon=True)
    watcher.start()
    t0 = time.time()
    try:
        run_cmd(cmd, STAGE, cwd=repo)
    finally:
        stop.set()
    ckpt = find_checkpoint(out_dir)
    return {"checkpoint": str(ckpt), "steps": steps, "base_model": base, "seconds": round(time.time() - t0, 1),
            "dry_run": False, "command": cmd}


def run(job, dataset_dir, out_dir, steps: int = 2000, gr00t_repo: str | None = None,
        base_model: str = BASE_MODEL, dry_run: bool | None = None) -> dict:
    """Job wrapper: fine-tune and write work/finetune.json."""
    result = finetune(dataset_dir, out_dir, steps=steps, base_model=base_model,
                      gr00t_repo=gr00t_repo, dry_run=dry_run)
    job.path("work").mkdir(parents=True, exist_ok=True)
    job.path("work", "finetune.json").write_text(json.dumps(result, indent=2))
    return result
