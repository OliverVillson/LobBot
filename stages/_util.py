"""Shared helpers for pipeline stages: job layout, config, subprocess runs."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from common.progress import emit
from common.taskspec import TaskSpec

# Set LOBBOT_DRY_RUN=1 to walk every stage without a GPU: stages skip the
# heavy work and write small placeholder outputs. Useful for testing the
# pipeline contract and the TUI end to end on a laptop.
DRY_RUN = os.environ.get("LOBBOT_DRY_RUN") == "1"


@dataclass
class Config:
    """Backend knobs. Override per job with <job>/config.json."""

    teacher: str = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    student: str = "Qwen/Qwen3-4B-Instruct-2507"
    models_dir: str = os.environ.get("LOBBOT_MODELS", "/mnt/nvme/models")
    llama_cpp: str = os.environ.get("LOBBOT_LLAMA_CPP", "/mnt/nvme/llama.cpp")
    # Data
    n_generate: int = 2000
    n_heldout: int = 100
    # REAP: fraction of experts removed per layer. 0.5 keeps 64 of 128.
    # Raising it frees bytes for more bits per remaining expert.
    reap_sparsity: float = 0.5
    reap_calib_samples: int = 512
    reap_max_seq: int = 2048
    # Heal (LoRA SFT on teacher answers)
    heal_epochs: float = 1.0
    heal_lr: float = 1e-4
    heal_lora_r: int = 16
    heal_targets: list[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    dense_fallback: bool = True
    # Quantize: aim below the TaskSpec max size by this margin.
    size_margin_gb: float = 0.5
    bit_floor: str = "q2_k"
    bit_ceiling: str = "q6_k"
    # Eval
    judge_model: str = "claude-sonnet-5-5"
    laptop_bandwidth_gb_s: float = 120.0  # MacBook Air M4; M5 is ~153


class Job:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        for sub in ("data", "work", "out", ".done"):
            (self.root / sub).mkdir(exist_ok=True)
        cfg_path = self.root / "config.json"
        overrides = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
        self.config = Config(**overrides)

    @property
    def spec(self) -> TaskSpec:
        return TaskSpec.load(self.root / "taskspec.json")

    def path(self, *parts: str) -> Path:
        return self.root.joinpath(*parts)

    def model_path(self, hf_id: str) -> str:
        """Local snapshot if setup_vm.sh downloaded it, else the hub id."""
        local = Path(self.config.models_dir) / hf_id.split("/")[-1]
        return str(local) if local.exists() else hf_id

    def is_done(self, stage: str) -> bool:
        return (self.root / ".done" / stage).exists()

    def mark_done(self, stage: str, info: dict | None = None) -> None:
        (self.root / ".done" / stage).write_text(json.dumps(info or {}))

    def clear_from(self, stages: list[str], start: str) -> None:
        for s in stages[stages.index(start):]:
            (self.root / ".done" / s).unlink(missing_ok=True)

    def save_config(self) -> None:
        (self.root / "config.json").write_text(json.dumps(asdict(self.config), indent=2))


def run(cmd: list[str], stage: str, cwd: str | Path | None = None, env: dict | None = None) -> None:
    """Run a command, forwarding its output as log lines (stderr is merged)."""
    emit(stage, msg="$ " + " ".join(cmd[:3]) + (" ..." if len(cmd) > 3 else ""))
    proc = subprocess.Popen(
        cmd, cwd=cwd, env={**os.environ, **(env or {})},
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    assert proc.stdout
    for line in proc.stdout:
        sys.stdout.write("[" + stage + "] " + line)
        sys.stdout.flush()
    if proc.wait() != 0:
        raise RuntimeError(f"{cmd[0]} exited with {proc.returncode}")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
