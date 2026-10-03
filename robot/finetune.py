"""GR00T N1.7 NEW_EMBODIMENT fine-tune on the sim demos, via Isaac-GR00T.

Shells out to the Isaac-GR00T checkout (env LOBBOT_GR00T_REPO, default
/mnt/nvme/Isaac-GR00T, set up by scripts/setup_robot_vm.sh) with its own venv
python (<repo>/.venv/bin/python; `uv run python` if that is missing):

    python gr00t/experiment/launch_finetune.py --base-model-path ...
        --dataset-path <lerobot dir> --embodiment-tag NEW_EMBODIMENT
        --modality-config-path <dataset>/farm_tractor_config.py ...

Flags are checked against gr00t/configs/finetune_config.py:FinetuneConfig
(tyro CLI: --base-model-path, --dataset-path, --embodiment-tag,
--modality-config-path, --num-gpus, --output-dir, --max-steps, --save-steps,
--global-batch-size, --dataloader-num-workers, --color-jitter-params ...) and
were run end to end on CPU against a tiny random model (tests/test_gr00t_compat.py).
Launch follows examples/finetune.sh: one GPU runs plain python with
CUDA_VISIBLE_DEVICES pinned (HF Trainer would otherwise wrap the model in
DataParallel and crash), several GPUs run torchrun.

Output layout (gr00t/experiment/experiment.py, experiment_name unset):
out_dir/config.json + model.safetensors (trainer.save_model) and
out_dir/processor/ (processor.save_pretrained); every out_dir/checkpoint-N/ is
standalone (CheckpointFormatCallback copies the processor files into it).
gr00t/policy/gr00t_policy.py:Gr00tPolicy loads either.
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
# Same augmentation as Isaac-GR00T examples/finetune.sh (COLOR_JITTER_PARAMS default).
COLOR_JITTER = ["brightness", "0.3", "contrast", "0.4", "saturation", "0.5", "hue", "0.08"]


def _base_model_path(base_model: str, models_dir: str | None = None) -> str:
    """Local snapshot from setup_robot_vm.sh if present, else the hub id."""
    root = models_dir or os.environ.get("LOBBOT_MODELS", "/mnt/nvme/models")
    local = Path(root) / base_model.split("/")[-1]
    return str(local) if local.exists() else base_model


def _python(repo: Path) -> list[str]:
    venv = repo / ".venv" / "bin" / "python"
    return [str(venv)] if venv.exists() else ["uv", "run", "python"]


def command(dataset_dir, out_dir, steps: int, base_model: str, batch: int = 32,
            num_gpus: int = 1, workers: int = 4, save_steps: int | None = None,
            gr00t_repo: str | Path | None = None, master_port: int = 29500) -> list[str]:
    """launch_finetune.py argv, run with cwd = the Isaac-GR00T checkout."""
    if batch % num_gpus:
        raise ValueError(f"global batch {batch} must be divisible by num_gpus {num_gpus}")
    dataset_dir = Path(dataset_dir).resolve()
    py = _python(Path(gr00t_repo or GR00T_REPO))
    launcher = py if num_gpus == 1 else [*py, "-m", "torch.distributed.run",
                                         f"--nproc_per_node={num_gpus}", f"--master_port={master_port}"]
    return [
        *launcher, "gr00t/experiment/launch_finetune.py",
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
        "--color-jitter-params", *COLOR_JITTER,
    ]


def launch_env(num_gpus: int = 1) -> dict[str, str]:
    """Extra env for the fine-tune process (see examples/finetune.sh)."""
    env: dict[str, str] = {}
    if num_gpus == 1 and "CUDA_VISIBLE_DEVICES" not in os.environ:
        env["CUDA_VISIBLE_DEVICES"] = "0"
    nvme = os.environ.get("NVME", "/mnt/nvme")
    if "HF_HOME" not in os.environ and Path(nvme, "hf-cache").is_dir():
        env["HF_HOME"] = str(Path(nvme, "hf-cache"))  # where setup_robot_vm.sh put the gated backbone
    if "CUDA_HOME" not in os.environ and Path("/usr/local/cuda").is_dir():
        env["CUDA_HOME"] = "/usr/local/cuda"  # deepspeed (multi-GPU) needs it
    return env


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
    if (out_dir / "DRY_RUN").exists():  # a dry run's placeholder would pass for the final save
        for f in ("DRY_RUN", "config.json"):
            (out_dir / f).unlink(missing_ok=True)
    base = _base_model_path(base_model)
    cmd = command(dataset_dir, out_dir, steps, base, gr00t_repo=repo, **kw)
    emit(STAGE, pct=2, msg=f"fine-tuning {base} for {steps} steps")
    stop = threading.Event()
    watcher = threading.Thread(target=_watch, args=(out_dir, steps, stop), daemon=True)
    watcher.start()
    t0 = time.time()
    try:
        run_cmd(cmd, STAGE, cwd=repo, env=launch_env(kw.get("num_gpus", 1)))
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
