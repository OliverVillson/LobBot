"""Robot pipeline stages: site, demos, finetune, compress, simeval, report.

    python -m robot.stages <stage> --job <job_dir>

Each stage reads <job>/robotspec.json, prints progress lines (common/progress.py)
and marks itself done in <job>/.done/<stage>. robot_pipeline.py runs them one
subprocess each. The modules they drive (farmsim/*, robot/compress.py, ...) are
imported inside the stage functions so a stage only loads what it needs.
See docs/robot-mvp.md for the contract.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from common.progress import emit
from common.robotspec import RobotSpec
from stages._util import DRY_RUN, Job

STAGES = ["site", "demos", "finetune", "compress", "simeval", "report"]


@dataclass
class RobotConfig:
    """Backend knobs for the robot pipeline. Override per job with <job>/config.json."""

    img_size: int = 224  # cab camera, what the policy sees
    clip_width: int = 640  # chase-camera clips and the training video
    clip_height: int = 360
    fps: int = 10  # clips are written at the control rate
    demo_seed: int = 0  # demos use seeds demo_seed.. ; eval uses eval_seed.. (never overlapping)
    eval_seed: int = 100_000
    base_model: str = os.environ.get("LOBBOT_GR00T_MODEL", "nvidia/GR00T-N1.7-3B")
    candidates: list[str] | None = None  # keep only these candidate names (None: all from compress)
    video_seconds: float = 45.0  # rough length of out/training.mp4
    row_length_m: float = 80.0  # length of the field rows each episode drives (FarmEnv)


class RobotJob(Job):
    """stages._util.Job with the robot config and spec."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        for sub in ("data", "work", "out", ".done"):
            (self.root / sub).mkdir(exist_ok=True)
        cfg_path = self.root / "config.json"
        overrides = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
        known = {f.name for f in fields(RobotConfig)}
        self.config = RobotConfig(**{k: v for k, v in overrides.items() if k in known})

    @property
    def spec(self) -> RobotSpec:  # type: ignore[override]
        return RobotSpec.load(self.root / "robotspec.json")

    def site(self):
        from farmsim.site import load_site

        info = json.loads(self.path("work", "site.json").read_text())
        return load_site(info["dir"])

    def read(self, *parts: str) -> dict | list:
        return json.loads(self.path(*parts).read_text())

    def write(self, obj, *parts: str) -> Path:
        p = self.path(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(obj, indent=2, default=_jsonable))
        return p


def _jsonable(o):
    if hasattr(o, "item"):
        return o.item()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


# ----------------------------------------------------------------- site


def stage_site(job: RobotJob) -> None:
    from farmsim.site import build_site, load_site

    spec = job.spec
    if spec.site_id:
        emit("site", "running", 10, f"loading site {spec.site_id}")
        site = load_site(spec.site_id)
    else:
        p = spec.site
        emit("site", "running", 10, f"building {p.size_m:.0f} m site at {p.lat:.4f}, {p.lon:.4f} ({p.source})")
        site = build_site(p.lat, p.lon, p.size_m, name=p.name or spec.task_name, source=p.source)
    info = {"id": site.id, "dir": str(site.dir), "name": site.name, "lat": site.lat, "lon": site.lon,
            "size_m": site.size_m, "source": site.source, "summary": site.summary}
    job.write(info, "work", "site.json")
    job.mark_done("site", {"id": site.id, "source": site.source})
    emit("site", "done", 100, f"{site.id} from {site.source}")


# ----------------------------------------------------------------- demos


def make_env(job: RobotJob, site, seed: int = 0, img_size: int | None = None):
    """img_size=0 skips the cab camera, for policies that do not look (the expert)."""
    from farmsim.env import FarmEnv

    img = job.config.img_size if img_size is None else img_size
    return FarmEnv(site, seed=seed, img_size=img, row_length_m=job.config.row_length_m)


def make_expert(env):
    from farmsim.expert import ExpertPolicy

    return ExpertPolicy(env)


def record_clip(job: RobotJob, site, policy, seed: int, path: Path) -> tuple[dict, Path]:
    """One chase-camera episode of policy to an MP4; returns its metrics."""
    from farmsim.video import rollout, write_mp4

    env = make_env(job, site, seed)
    metrics, frames = rollout(env, policy, seed, max_steps=job.spec.max_steps, frames=True,
                              width=job.config.clip_width, height=job.config.clip_height)
    path.parent.mkdir(parents=True, exist_ok=True)
    env.close()
    write_mp4(frames, path, fps=job.config.fps)
    return metrics, path


def stage_demos(job: RobotJob) -> None:
    from farmsim.lerobot import export_episodes

    spec, cfg, site = job.spec, job.config, job.site()
    out = job.path("data", "lerobot")
    emit("demos", "running", 5, f"expert drives {spec.n_demos} episodes")
    stats = export_episodes(site, None, spec.n_demos, out, seed=cfg.demo_seed, img_size=cfg.img_size,
                            row_length_m=cfg.row_length_m,
                            progress=lambda i, n: emit("demos", "running", 5 + 70 * i / n, f"episode {i}/{n}"))
    emit("demos", "running", 80, "recording the demo clip")
    metrics, clip = record_clip(job, site, make_expert(None), cfg.demo_seed, job.path("out", "clips", "expert.mp4"))
    job.write({"stats": stats, "clip": str(clip.relative_to(job.root)), "clip_metrics": metrics}, "work", "demos.json")
    job.mark_done("demos", {"episodes": spec.n_demos})
    emit("demos", "done", 100, f"{spec.n_demos} episodes in data/lerobot")


# ----------------------------------------------------------------- finetune


def stage_finetune(job: RobotJob) -> None:
    from robot import finetune

    spec, cfg = job.spec, job.config
    emit("finetune", "running", 1, "DRY RUN: no fine-tune" if DRY_RUN else
         f"GR00T N1.7 NEW_EMBODIMENT fine-tune, {spec.finetune_steps} steps")
    result = finetune.run(job, job.path("data", "lerobot"), job.path("work", "gr00t"), steps=spec.finetune_steps,
                          base_model=cfg.base_model, dry_run=DRY_RUN)  # writes work/finetune.json
    job.mark_done("finetune", {"checkpoint": result.get("checkpoint"), "dry_run": DRY_RUN})
    emit("finetune", "done", 100, "DRY RUN: placeholder checkpoint" if DRY_RUN else f"checkpoint {result['checkpoint']}")


# ----------------------------------------------------------------- compress


def stage_compress(job: RobotJob) -> None:
    from robot.compress import build_candidates

    emit("compress", "running", 1, "building candidates" + (" (DRY RUN: synthetic sensitivities)" if DRY_RUN else ""))
    ckpt = job.read("work", "finetune.json").get("checkpoint")  # may be a checkpoint-N subfolder
    cands = [asdict(c) for c in build_candidates(job, checkpoint=ckpt, dry_run=DRY_RUN)]  # writes work/candidates.json
    if job.config.candidates:
        keep = set(job.config.candidates) | {"teacher"}
        cands = [c for c in cands if c["name"] in keep]
        job.write(cands, "work", "candidates.json")
    job.mark_done("compress", {"candidates": [c["name"] for c in cands], "dry_run": DRY_RUN})
    emit("compress", "done", 100, ", ".join(f"{c['name']} {c['footprint']['size_gb']:.2f} GB" for c in cands))


# ----------------------------------------------------------------- simeval / report


def stage_simeval(job: RobotJob) -> None:
    from robot.simeval import run_simeval

    report = run_simeval(job, dry_run=DRY_RUN)
    job.mark_done("simeval", {"candidates": len(report["candidates"]), "dry_run": report["dry_run"]})
    best = max(report["candidates"], key=lambda c: (c["success_rate"], -c["size_gb"]))
    emit("simeval", "done", 100, f"{len(report['candidates'])} candidates; best {best['name']} "
         f"{best['success_rate']:.0%}" + (" (DRY RUN)" if report["dry_run"] else ""))


def stage_report(job: RobotJob) -> None:
    from robot.report import build_report

    summary = build_report(job)
    job.mark_done("report", {"video": summary["video"], "graph": summary["graph"]})
    emit("report", "done", 100, summary["video"])


RUNNERS = {"site": stage_site, "demos": stage_demos, "finetune": stage_finetune,
           "compress": stage_compress, "simeval": stage_simeval, "report": stage_report}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=STAGES)
    ap.add_argument("--job", required=True)
    a = ap.parse_args(argv)
    RUNNERS[a.stage](RobotJob(a.job))
    return 0


if __name__ == "__main__":
    sys.exit(main())
