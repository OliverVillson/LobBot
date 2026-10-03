"""Score every compression candidate in the farm sim, on the same seeded episodes.

Every candidate (and the scripted expert, as the reference) drives episodes
with seeds eval_seed .. eval_seed + eval_episodes - 1, so their results are
paired: a candidate that fails where the teacher succeeds failed on the very
same row, weather and start. The first episode of each is recorded as a clip.
Writes out/eval.json (docs/robot-mvp.md).

In a dry run there is no GR00T checkpoint: each candidate is the expert with
steering noise and control lag scaled by how compressed it is, and the report
says dry_run: true. Those numbers only exercise the pipeline; never present
them as results.
"""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path

import numpy as np

from common.progress import emit


class TimedPolicy:
    """Wraps a policy and records the wall time of each act() call."""

    def __init__(self, policy):
        self.policy = policy
        self.name = getattr(policy, "name", type(policy).__name__)
        self.size_label = getattr(policy, "size_label", None)
        self.times_ms: list[float] = []

    def bind(self, env):
        if hasattr(self.policy, "bind"):
            self.policy.bind(env)
        return self

    def reset(self) -> None:
        self.policy.reset()

    def act(self, obs: dict) -> np.ndarray:
        t = time.perf_counter()
        a = self.policy.act(obs)
        self.times_ms.append((time.perf_counter() - t) * 1000)
        return a


class DegradedPolicy:
    """DRY RUN stand-in for a compressed policy: the expert plus a slowly wandering
    steering error (like a quantised head's drift), per-step jitter and control
    lag, all growing with the compression severity (0..1)."""

    def __init__(self, base, severity: float, name: str, seed: int = 0, latency_ms: float | None = None):
        self.base = base
        self.severity = float(np.clip(severity, 0.0, 1.0))
        self.name = name
        self.drift_std = 0.06 * self.severity  # rad per step into an AR(1) drift
        self.noise_std = 0.05 + 0.2 * self.severity  # rad of per-step steering jitter
        self.lag = int(round(6 * self.severity))  # control steps (0.1 s each)
        self.seed = seed
        self.episode = 0
        if latency_ms is not None:
            self.latency_ms = latency_ms
        self.reset()

    def bind(self, env):
        if hasattr(self.base, "bind"):
            self.base.bind(env)
        return self

    def reset(self) -> None:
        self.base.reset()
        # Seeded per episode so every rerun of the eval sees the same noise.
        self.rng = np.random.default_rng([self.seed, self.episode])
        self.drift = 0.0
        self.queue: deque = deque()

    def act(self, obs: dict) -> np.ndarray:
        a = np.asarray(self.base.act(obs), dtype=np.float32).copy()
        self.drift = 0.97 * self.drift + self.rng.normal(0.0, self.drift_std)
        a[0] += self.drift + self.rng.normal(0.0, self.noise_std)
        a[1] *= 1.0 + self.rng.normal(0.0, 0.1 * self.severity)
        if not self.queue:  # until the lag has passed, the wheels stay straight
            self.queue.extend([np.array([0.0, a[1]], np.float32)] * self.lag)
        self.queue.append(a)
        out = self.queue.popleft()
        return np.array([np.clip(out[0], -0.6, 0.6), np.clip(out[1], 0.0, 3.0)], dtype=np.float32)


def severity(cand: dict, teacher: dict) -> float:
    """How compressed a candidate is relative to the teacher, 0 (teacher) to 1."""
    t_size = max(teacher["footprint"]["size_gb"], 1e-6)
    size = 1.0 - cand["footprint"]["size_gb"] / t_size
    steps = 1.0 - cand.get("denoise_steps", 4) / max(teacher.get("denoise_steps", 4), 1)
    blocks = len(cand.get("drop_dit_blocks") or []) / 16
    return float(np.clip(1.05 * size + 0.4 * steps + 0.6 * blocks, 0.0, 1.0))


def dry_latency_ms(cand: dict) -> float:
    """Rough GPU latency estimate for a dry run: backbone once, DiT per denoising step."""
    size = cand["footprint"]["size_gb"]
    return round(6.0 + size * (1.5 + 0.9 * cand.get("denoise_steps", 4)), 1)


def make_policy(job, cand: dict, env, dry_run: bool, seed: int = 0):
    from robot.stages import make_expert

    if dry_run:
        cands = job.read("work", "candidates.json")
        teacher = next((c for c in cands if c["name"] == "teacher"), cands[0])
        p = DegradedPolicy(make_expert(env), severity(cand, teacher), cand["name"],
                           seed=job.config.eval_seed, latency_ms=dry_latency_ms(cand))
        p.size_label = f"{cand['footprint']['size_gb']:.2f} GB, DRY RUN"
        p.episode = seed  # noise keyed by the episode seed: paired across candidates and reruns
        return p
    from robot.footprint import Candidate
    from robot.gr00t_policy import Gr00tPolicy

    keys = Candidate.__dataclass_fields__.keys()
    c = Candidate(**{k: v for k, v in cand.items() if k in keys})
    global _gr00t
    if _gr00t is None:  # one checkpoint load; set_candidate switches the compression in place
        _gr00t = Gr00tPolicy(job.read("work", "finetune.json")["checkpoint"], c)
    else:
        _gr00t.set_candidate(c)
    _gr00t.name = cand["name"]
    if isinstance(getattr(_gr00t, "latency_ms", None), list):
        _gr00t.latency_ms.clear()
    return _gr00t


_gr00t = None


def reported_latency(policy) -> float | None:
    """policy.latency_ms as a number: a value, a method, or a list of per-inference times (median)."""
    v = getattr(policy, "latency_ms", None)
    if callable(v):
        v = v()
    if isinstance(v, (list, tuple)):
        return float(np.median(v)) if v else None
    return float(v) if v is not None else None


def evaluate(job, site, make, seeds: list[int], clip: Path | None, label: str, pct: tuple[float, float],
             reuse: bool = False, vision: bool = True):
    """Run one policy (built by make(env, seed), per episode unless reuse) on the given seeds.

    Returns (aggregate metrics, measured latency in ms or None, policy)."""
    from farmsim.video import rollout, write_mp4
    from robot.stages import make_env

    max_steps = job.spec.max_steps
    rows, policy, times = [], None, []
    # One env for all episodes (rollout resets it per seed): under osmesa a second
    # live FarmEnv in the process makes renders go stale. Closed at the end.
    env = make_env(job, site, seeds[0], None if vision else 0)
    for i, seed in enumerate(seeds):
        if policy is None or not reuse:
            # Dry-run stand-ins are rebuilt per episode so their noise is keyed by the
            # episode seed; a GR00T checkpoint is loaded once.
            policy = make(env, seed)
        timed = TimedPolicy(policy)
        timed.times_ms = times
        want_frames = clip is not None and i == 0
        m, frames = rollout(env, timed, seed, max_steps=max_steps, frames=want_frames,
                            width=job.config.clip_width, height=job.config.clip_height)
        rows.append(m)
        if want_frames:
            write_mp4(frames, clip, fps=job.config.fps)
        lo, hi = pct
        emit("simeval", "running", lo + (hi - lo) * (i + 1) / len(seeds),
             f"{label}: episode {i + 1}/{len(seeds)}")
    env.close()
    agg = {
        "success_rate": round(float(np.mean([bool(r.get("success")) for r in rows])), 4),
        "cte_mean_m": round(float(np.mean([r.get("cte_mean_m", 0.0) for r in rows])), 4),
        "cte_max_m": round(float(np.max([r.get("cte_max_m", 0.0) for r in rows])), 4),
        "progress": round(float(np.mean([r.get("progress", 0.0) for r in rows])), 4),
        "per_episode": [{"seed": s, **{k: _num(v) for k, v in r.items()}} for s, r in zip(seeds, rows)],
    }
    measured = float(np.median(times)) if times else None
    return agg, measured, policy


def _num(v):
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    return v


def run_simeval(job, dry_run: bool) -> dict:
    from robot.stages import make_expert

    spec, cfg, site = job.spec, job.config, job.site()
    cands = job.read("work", "candidates.json")
    seeds = [cfg.eval_seed + i for i in range(spec.eval_episodes)]
    clips = job.path("out", "clips")
    clips.mkdir(parents=True, exist_ok=True)
    n = len(cands) + 1
    span = 100.0 / n

    # The expert and the dry-run stand-ins never look at the camera, so their envs skip it
    # (clips still get the chase view and front inset). GR00T candidates see it.
    expert, _, _ = evaluate(job, site, lambda env, seed: make_expert(env), seeds, None, "expert", (0, span),
                            vision=False)
    out = []
    for k, cand in enumerate(cands, start=1):
        clip = clips / f"{cand['name']}.mp4"
        agg, measured, policy = evaluate(job, site, lambda env, seed, c=cand: make_policy(job, c, env, dry_run, seed),
                                         seeds, clip, cand["name"], (k * span, (k + 1) * span), reuse=not dry_run,
                                         vision=not dry_run)
        reported = reported_latency(policy)
        latency = reported if reported is not None else measured
        fp = cand.get("footprint", {})
        out.append({
            "name": cand["name"],
            "size_gb": round(float(fp.get("size_gb", 0.0)), 3),
            "params": fp.get("params"),
            "bits_per_weight": fp.get("bits_per_weight"),
            "denoise_steps": cand.get("denoise_steps"),
            "note": cand.get("note", ""),
            "latency_ms": round(latency, 2) if latency is not None else None,
            "latency_source": ("dry_run_estimate" if dry_run else "policy") if reported is not None else "timed_act",
            "hz": round(1000.0 / latency, 1) if latency else None,
            "fits": bool(fp.get("size_gb", 1e9) <= spec.target.max_size_gb
                         and (latency is None or 1000.0 / max(latency, 1e-6) >= spec.target.min_hz)),
            **{k: v for k, v in agg.items() if k != "per_episode"},
            "video": str(clip.relative_to(job.root)),
            "per_episode": agg["per_episode"],
        })
    report = {
        "kind": "robot",
        "dry_run": bool(dry_run),
        "site": site.id,
        "episodes": spec.eval_episodes,
        "seeds": seeds,
        "target": {"max_size_gb": spec.target.max_size_gb, "min_hz": spec.target.min_hz},
        "candidates": out,
        "expert": {k: v for k, v in expert.items() if k != "per_episode"},
    }
    if dry_run:
        report["note"] = "DRY RUN: candidates are the scripted expert with noise and lag; not model results"
    job.write(report, "out", "eval.json")
    return report
