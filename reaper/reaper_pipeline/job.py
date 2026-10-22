"""Job directory and run settings.

    <job>/
        config.json        settings for this run (Config fields; unknown keys are ignored)
        events.log         progress events, one JSON object per line (the TUI tails this)
        pid                pid of the running pipeline
        hardware.json      what the plan stage detected
        plan.json          how each stage adapts to it
        logs/<stage>.log   full output of each stage process
        data/calib.jsonl   prompts (+ teacher answers) REAP is calibrated on
        data/heldout.jsonl prompts + teacher answers kept back for eval
        data/topics.json   sub-tasks the teacher derived from the purpose
        work/saliency.json per-layer, per-expert REAP scores and kept ids
        work/reap.json     experts before/after, sizes
        work/teacher_ref.pt  the unpruned model's top tokens on the held-out answers
        work/eval.json     pruned vs teacher
        out/<name>/        the pruned model (safetensors, same layout as the original) + README.md
        .done/<stage>      finished stages; a rerun skips them
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, fields
from pathlib import Path


def home() -> Path:
    """$REAPER_HOME, else ~/.reaper: models and jobs live here."""
    return Path(os.environ.get("REAPER_HOME") or Path.home() / ".reaper")


@dataclass
class Config:
    model: str = "Qwen/Qwen3-30B-A3B-Instruct-2507"  # HF repo id or local directory
    purpose: str = ""                # free text, e.g. "Debugging only"; empty = general (calib below)
    prune: float = 0.5               # share of routed experts dropped in every MoE layer
    calib: str = "mix"               # mix | chat | code | math | builtin | path to .jsonl/.txt prompts
    calib_samples: int = 512         # calibration sequences REAP is scored on
    heldout: int = 32                # teacher answers kept back for eval
    max_seq: int = 2048              # tokens per calibration sequence
    answer_max_tokens: int = 1024    # teacher answer length cap
    teacher_answers: bool = True     # calibrate on prompt + teacher answer (False: prompts only)
    teacher_backend: str = "auto"    # auto (vLLM when installed and on GPU, else transformers) | vllm | hf
    purpose_mix: float = 0.25        # share of a purpose's prompts taken from the closest public preset
    trust_remote_code: bool | None = None  # models whose code lives on the Hub (Kimi K2); None = when needed
    models_dir: str = ""             # where downloads go; default <REAPER_HOME>/models
    out_name: str = ""               # name of the output directory; default <model>-REAP-<n>e
    seed: int = 0

    @classmethod
    def load(cls, path: Path) -> "Config":
        raw = json.loads(path.read_text()) if path.exists() else {}
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in known})


def job_name(model: str, prune: float, purpose: str = "") -> str:
    base = re.sub(r"[^A-Za-z0-9._-]+", "-", model.rstrip("/").split("/")[-1])
    tag = re.sub(r"[^a-z0-9]+", "-", purpose.lower()).strip("-")[:24]
    return f"{base}-prune{round(prune * 100)}" + (f"-{tag}" if tag else "")


class Job:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        for sub in ("data", "work", "out", "logs", ".done"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)
        self.config = Config.load(self.root / "config.json")
        if not self.config.models_dir:
            self.config.models_dir = str(home() / "models")
        if self.config.trust_remote_code is None:
            self.config.trust_remote_code = self._needs_remote_code()

    def _needs_remote_code(self) -> bool:
        """True when the checkpoint ships its own modeling code for an architecture
        transformers does not know (Kimi K2). The user picked this model to run."""
        p = self.model_dir / "config.json"
        if not p.exists():
            return False
        cfg = json.loads(p.read_text())
        if not cfg.get("auto_map"):
            return False
        try:
            from transformers.models.auto.configuration_auto import CONFIG_MAPPING
            return cfg.get("model_type") not in CONFIG_MAPPING
        except ImportError:
            return True

    @classmethod
    def create(cls, cfg: Config, root: str | Path | None = None) -> "Job":
        root = Path(root) if root else home() / "jobs" / job_name(cfg.model, cfg.prune, cfg.purpose)
        root.mkdir(parents=True, exist_ok=True)
        (root / "config.json").write_text(json.dumps(asdict(cfg), indent=2) + "\n")
        return cls(root)

    def path(self, *parts: str) -> Path:
        return self.root.joinpath(*parts)

    @property
    def model_dir(self) -> Path:
        """Local copy of the model (the weights stage downloads it there)."""
        p = Path(self.config.model).expanduser()
        if p.is_dir():
            return p.resolve()
        return Path(self.config.models_dir) / self.config.model.replace("/", "--")

    @property
    def out_dir(self) -> Path:
        if self.config.out_name:
            return self.path("out", self.config.out_name)
        n = (self.read("work/reap.json") or {}).get("experts_kept")
        return self.path("out", Path(self.config.model).name + (f"-REAP-{n}e" if n else "-REAP"))

    def read(self, rel: str):
        p = self.path(rel)
        return json.loads(p.read_text()) if p.exists() else None

    def write(self, rel: str, obj) -> None:
        p = self.path(rel)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(obj, indent=2) + "\n")
        tmp.replace(p)

    def is_done(self, stage: str) -> bool:
        return self.path(".done", stage).exists()

    def mark_done(self, stage: str, info: dict | None = None) -> None:
        self.path(".done", stage).write_text(json.dumps(info or {}))

    def clear_from(self, stages: list[str], start: str) -> None:
        for s in stages[stages.index(start):]:
            self.path(".done", s).unlink(missing_ok=True)
