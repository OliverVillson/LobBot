"""A TUI job and the REAPer job inside it.

    <job>/config.json     written by the TUI (model, schema, target, purpose, steps)
    <job>/events.log      this pipeline's stdout (the TUI tails it)
    <job>/reaper/         REAPer's job dir: data/, work/, logs/, out/, .done/
    <job>/out/            links to the finished artifacts
    <job>/receipt.json    what ran and what it measured
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, fields
from pathlib import Path

from blackhole_pipeline import DEFAULTS


def params(cfg: dict) -> dict:
    """Run parameters: DEFAULTS, overridden by the schema steps' params and top-level keys."""
    p = dict(DEFAULTS)
    for step in cfg.get("steps") or []:
        sp = step.get("params") or {}
        if step.get("kind") == "reap" and "prune" in sp:
            p["prune"] = float(sp["prune"])
        for k in DEFAULTS:
            if k in sp and k != "prune":
                p[k] = type(DEFAULTS[k])(sp[k])
    for k in DEFAULTS:
        if k in cfg:
            p[k] = type(DEFAULTS[k])(cfg[k])
    return p


class BHJob:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.config = json.loads((self.root / "config.json").read_text())
        self.params = params(self.config)
        (self.root / "out").mkdir(exist_ok=True)

    @property
    def reaper_root(self) -> Path:
        return self.root / "reaper"

    def reaper(self):
        """REAPer's Job for this run, created on first use from the TUI settings."""
        from reaper_pipeline.job import Config, Job
        cfg_p = self.reaper_root / "config.json"
        if not cfg_p.exists():
            c = Config(model=self.config["model"], purpose=self.config.get("purpose", ""),
                       prune=self.params["prune"], calib_samples=self.params["calib_samples"],
                       quant="fp8-experts")
            for f in fields(Config):  # REAPer settings given in the job config (calib, heldout, max_seq, ...)
                if f.name in self.config and f.name not in ("model", "purpose", "prune", "quant", "calib_samples"):
                    setattr(c, f.name, self.config[f.name])
            self.reaper_root.mkdir(parents=True, exist_ok=True)
            cfg_p.write_text(json.dumps(asdict(c), indent=2) + "\n")
        return Job(self.reaper_root)

    def gptq_dir(self, rj) -> Path:
        return rj.out_dir.with_name(rj.out_dir.name + "-GPTQ-W4A16")

    def reaq_dir(self, rj) -> Path:
        return rj.out_dir.with_name(rj.out_dir.name + "-REAQ-codebook")

    def read(self, rel: str):
        p = self.root / rel
        return json.loads(p.read_text()) if p.exists() else None

    def write(self, rel: str, obj) -> None:
        p = self.root / rel
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(obj, indent=2) + "\n")
        os.replace(tmp, p)

    def link(self, target: Path) -> None:
        dst = self.root / "out" / target.name
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        dst.symlink_to(target)
