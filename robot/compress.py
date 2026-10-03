"""Compression candidates for the fine-tuned GR00T N1.7 policy, scored in the sim.

    teacher   bf16, 4 denoising steps (the fine-tuned checkpoint as is)
    steps2/1  fewer flow-matching denoising steps
    fp8       every layer group at FP8
    nvfp4     NVIDIA's Thor recipe: NVFP4 everywhere except o_proj, down_proj
              and DiT attention (FP8) and the embedding/projectors (FP8)
    mixed     sim-in-the-loop precision: per-group action error measured on
              sim states, then a greedy allocation under a size budget
    pruned    mixed plus the DiT blocks whose removal moves the action least

The precision search is stages/bits.py ported from llama.cpp quant types to
FP8/NVFP4 and from weight MSE to the error that matters here: how far the
predicted action chunk moves from the teacher's on the same observation (and
the same flow-matching noise). States come from expert and teacher rollouts in
the farm sim, so the error is measured where the policy actually drives.

DRY RUN builds the same list from the default shape table with synthetic
sensitivities (labelled dry_run), so the pipeline and graph run without a GPU.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, replace

from common.progress import emit
from robot.footprint import (
    BPW,
    DEFAULT_SHAPES,
    DIT_BLOCKS,
    Candidate,
    allowed,
    footprint,
    size_gb,
)
from stages._util import DRY_RUN

STAGE = "compress"
# Normalise action error by the action range so steering and speed weigh alike.
ACTION_SCALE = (0.6, 3.0)  # rad, m/s
# Mixed-precision budget when the spec gives none tighter: this fraction of the
# bf16 teacher (the recipe floor is ~0.38, all-FP8 ~0.50).
DEFAULT_BUDGET_FRAC = 0.42
N_PRUNE = 4  # DiT blocks dropped for "pruned" (of 16)


# ---------------------------------------------------------------------------
# Pure allocation (no GPU)
# ---------------------------------------------------------------------------
def floor_precision(shapes: dict[str, int]) -> dict[str, str]:
    """Cheapest allowed precision per group ("nvfp4" candidate)."""
    return {g: allowed(g)[0] for g in shapes if g != "other"}


def allocate(
    shapes: dict[str, int],
    sens: dict[str, dict[str, float]],
    budget_gb: float,
    drop: list[int] | None = None,
) -> dict[str, str]:
    """Greedy rate-distortion allocation (as stages/bits.allocate): start every
    group at its cheapest allowed precision, then repeatedly buy the upgrade
    with the largest drop in measured action error per extra byte, until the
    next upgrade would break the budget. The most sensitive groups upgrade first.

    sens[g][p]: action error with group g alone at precision p (bf16 is 0).
    Groups without a measurement stay at the floor.
    """
    drop = drop or []
    dropped = {f"dit.{j}.{k}" for j in drop for k in ("attn", "ff")}
    prec = floor_precision(shapes)
    current = size_gb(shapes, prec, drop)
    if current > budget_gb + 1e-9:
        raise ValueError(f"budget {budget_gb:.3f} GB is below the floor size {current:.3f} GB")

    def err(g: str, p: str) -> float:
        return 0.0 if p == "bf16" else float(sens.get(g, {}).get(p, 0.0))

    while True:
        options = []
        for g, p in prec.items():
            if g in dropped or g not in sens:
                continue
            ladder = allowed(g)
            i = ladder.index(p)
            if i + 1 >= len(ladder):
                continue
            q = ladder[i + 1]
            extra = shapes[g] * (BPW[q] - BPW[p]) / 8 / 1e9
            gain = max(err(g, p) - err(g, q), 0.0)
            options.append((gain / extra, gain, g, q, extra))
        bought = False
        for _, _, g, q, extra in sorted(options, reverse=True):
            if current + extra > budget_gb:
                continue  # a cheaper upgrade may still fit
            prec[g] = q
            current += extra
            bought = True
            break
        if not bought:
            return prec


def predicted_error(sens: dict[str, dict[str, float]], prec: dict[str, str]) -> float:
    """Additive estimate of the action error of an allocation."""
    return sum(sens.get(g, {}).get(p, 0.0) for g, p in prec.items() if p != "bf16")


def default_budget(shapes: dict[str, int], max_size_gb: float | None = None) -> float:
    teacher = size_gb(shapes, {})
    b = DEFAULT_BUDGET_FRAC * teacher
    if max_size_gb is not None:
        b = min(b, max_size_gb)
    return max(b, size_gb(shapes, floor_precision(shapes)))


def assemble(
    shapes: dict[str, int],
    sens: dict[str, dict[str, float]],
    block_scores: list[float],
    budget_gb: float,
    n_prune: int = N_PRUNE,
    note: str = "",
) -> list[Candidate]:
    """The candidate list from measured (or synthetic) sensitivities."""
    groups = [g for g in shapes if g != "other"]
    mixed = allocate(shapes, sens, budget_gb)
    order = sorted(range(len(block_scores)), key=lambda j: block_scores[j])
    drop = sorted(order[:n_prune])
    sfx = f" ({note})" if note else ""
    cands = [
        Candidate("teacher", note="bf16, 4 denoising steps" + sfx),
        Candidate("steps2", denoise_steps=2, note="bf16, 2 denoising steps" + sfx),
        Candidate("steps1", denoise_steps=1, note="bf16, 1 denoising step" + sfx),
        Candidate("fp8", precision={g: "fp8" for g in groups}, note="all groups FP8 e4m3, per-channel scale" + sfx),
        Candidate("nvfp4", precision=floor_precision(shapes),
                  note="Thor recipe: NVFP4, o/down/DiT-attn/embed/projectors FP8" + sfx),
        Candidate("mixed", precision=mixed,
                  note=f"sim-scored per-group FP8/NVFP4 under {budget_gb:.2f} GB" + sfx),
        Candidate("pruned", precision=mixed, drop_dit_blocks=drop,
                  note=f"mixed, minus DiT blocks {drop} (least action change)" + sfx),
    ]
    return [replace(c, footprint=footprint(shapes, c)) for c in cands]


# ---------------------------------------------------------------------------
# Synthetic sensitivities (DRY RUN)
# ---------------------------------------------------------------------------
def synthetic_sensitivities(shapes: dict[str, int]) -> tuple[dict, list[float]]:
    """Plausible, deterministic stand-ins. NOT measurements."""
    sens: dict[str, dict[str, float]] = {}
    for g in shapes:
        if g == "other":
            continue
        parts = g.split(".")
        if g == "vision":
            w = 1.0
        elif g == "llm.embed":
            w = 0.3
        elif g == "proj":
            w = 3.0
        elif parts[0] == "llm":
            i = int(parts[1])
            w = (1.0 + 0.6 * math.cos(math.pi * i / 11) ** 2) * (1.5 if parts[2] in ("o", "down") else 1.0)
        else:
            j = int(parts[1])
            w = (2.0 if parts[2] == "attn" else 1.2) * (1.3 - 0.6 * j / 15)
        e4 = 0.01 * w
        sens[g] = {p: (e4 if p == "nvfp4" else e4 / 16) for p in allowed(g) if p != "bf16"}
    blocks = [0.02 * (1 + (j * 7 % 5) / 2) * (2.0 if j in (0, DIT_BLOCKS - 1) else 1.0) for j in range(DIT_BLOCKS)]
    return sens, blocks


# ---------------------------------------------------------------------------
# Sim-in-the-loop measurement (GPU)
# ---------------------------------------------------------------------------
def collect_states(env_factory, drivers: list, n_states: int = 48, stride: int = 5,
                   max_steps: int = 400, seed0: int = 50_000, max_episodes: int = 40) -> list[dict]:
    """Observations from rollouts, alternating the drivers per episode
    (expert and teacher), one obs every `stride` steps."""
    states: list[dict] = []
    ep = 0
    while len(states) < n_states and ep < max_episodes:
        driver = drivers[ep % len(drivers)]
        env = env_factory(seed0 + ep)
        if hasattr(driver, "bind"):
            driver.bind(env)
        obs = env.reset(seed=seed0 + ep)
        driver.reset()
        for t in range(max_steps):
            if t % stride == 0:
                states.append({k: (v.copy() if hasattr(v, "copy") else v) for k, v in obs.items()})
                if len(states) >= n_states:
                    break
            obs, done, _ = env.step(driver.act(obs))
            if done:
                break
        ep += 1
    return states


def action_error(policy, states: list[dict], ref: list, k: int = 8) -> float:
    """Mean normalised squared error of the first k actions of the chunk vs the
    teacher's on the same obs."""
    import numpy as np

    scale = np.asarray(ACTION_SCALE, dtype=np.float32)
    errs = []
    for obs, r in zip(states, ref):
        a = policy.predict_chunk(obs)[:k]
        errs.append(float((((a - r[:k]) / scale) ** 2).mean()))
    return float(np.mean(errs))


def measure(policy, states: list[dict], shapes: dict[str, int], k: int = 8) -> tuple[dict, list[float], list]:
    """Per-group error at each lower precision, then per-DiT-block drop error.
    `policy` must support set_candidate and predict_chunk (Gr00tPolicy)."""
    policy.set_candidate(Candidate("teacher"))
    ref = [policy.predict_chunk(o) for o in states]
    groups = [g for g in shapes if g != "other"]
    probes = [(g, p) for g in groups for p in allowed(g) if p != "bf16"]
    sens: dict[str, dict[str, float]] = {g: {} for g in groups}
    for n, (g, p) in enumerate(probes):
        policy.set_candidate(Candidate(f"probe:{g}:{p}", precision={g: p}))
        sens[g][p] = action_error(policy, states, ref, k)
        if n % 8 == 0:
            emit(STAGE, pct=10 + 60 * n / len(probes), msg=f"sensitivity {g} {p}: {sens[g][p]:.2e}")
    return sens, ref


def measure_blocks(policy, states, ref, precision: dict[str, str], n_blocks: int, k: int = 8) -> list[float]:
    """S_u: action change when DiT block j alone is dropped (on top of `precision`)."""
    scores = []
    for j in range(n_blocks):
        policy.set_candidate(Candidate(f"probe:drop{j}", precision=precision, drop_dit_blocks=[j]))
        scores.append(action_error(policy, states, ref, k))
        emit(STAGE, pct=72 + 20 * j / n_blocks, msg=f"DiT block {j} drop: {scores[-1]:.2e}")
    return scores


def _max_size(job) -> float | None:
    try:
        return float(job.spec.target.max_size_gb)
    except Exception:
        return None


def build_candidates(
    job,
    policy_factory=None,
    env_factory=None,
    budget: float | None = None,
    *,
    checkpoint: str | None = None,
    dry_run: bool | None = None,
    n_states: int = 48,
    n_prune: int = N_PRUNE,
    expert=None,
) -> list[Candidate]:
    """Candidates with footprints; writes work/sensitivity.json and
    work/candidates.json (asdict(Candidate), footprint included).

    policy_factory(cand) -> Gr00tPolicy (default: Gr00tPolicy(checkpoint, cand)).
    env_factory(seed) -> FarmEnv (default: FarmEnv(job.site(), seed=seed)).
    budget: size budget in GB for "mixed" (default: DEFAULT_BUDGET_FRAC of the
    teacher, capped by robotspec target.max_size_gb, floored at the recipe floor).
    expert: the demonstrator driving half the state-collection episodes
    (default farmsim.expert.ExpertPolicy, bound to each env).
    """
    dry = DRY_RUN if dry_run is None else dry_run
    if dry:
        shapes = dict(DEFAULT_SHAPES)
        sens, blocks = synthetic_sensitivities(shapes)
        budget = budget or default_budget(shapes, _max_size(job))
        cands = assemble(shapes, sens, blocks, budget, n_prune, note="dry_run")
        info = {"dry_run": True, "note": "synthetic sensitivities, not measurements"}
    else:
        if policy_factory is None:
            from robot.gr00t_policy import Gr00tPolicy

            if checkpoint is None:
                checkpoint = json.loads(job.path("work", "finetune.json").read_text())["checkpoint"]
            policy_factory = lambda cand: Gr00tPolicy(checkpoint, cand)  # noqa: E731
        if env_factory is None:
            from farmsim.env import FarmEnv

            site = job.site()
            img = getattr(job.config, "img_size", 224)
            env_factory = lambda seed: FarmEnv(site, seed=seed, img_size=img)  # noqa: E731
        if expert is None:
            from farmsim.expert import ExpertPolicy

            expert = ExpertPolicy()
        emit(STAGE, pct=2, msg="loading teacher")
        policy = policy_factory(Candidate("teacher"))
        shapes = policy.shapes
        emit(STAGE, pct=5, msg=f"collecting {n_states} sim states (expert + teacher rollouts)")
        states = collect_states(env_factory, [expert, policy], n_states=n_states)
        sens, ref = measure(policy, states, shapes)
        budget = budget or default_budget(shapes, _max_size(job))
        mixed = allocate(shapes, sens, budget)
        from robot.gr00t_policy import dit_blocks

        n_blocks = len(dit_blocks(policy.model))
        blocks = measure_blocks(policy, states, ref, mixed, n_blocks)
        cands = assemble(shapes, sens, blocks, budget, n_prune)
        # Measured error of the composite candidates (the additive estimate can be off).
        measured = {}
        for c in cands:
            policy.set_candidate(c)
            measured[c.name] = action_error(policy, states, ref)
        policy.set_candidate(Candidate("teacher"))
        info = {"dry_run": False, "n_states": len(states), "candidate_action_error": measured}

    out = {
        **info,
        "metric": "mean normalised squared error of the first 8 actions vs teacher, same obs and noise",
        "budget_gb": round(budget, 4),
        "shapes": shapes,
        "groups": sens,
        "dit_block_drop": blocks,
        "predicted_error": {c.name: predicted_error(sens, c.precision) for c in cands},
    }
    job.path("work").mkdir(parents=True, exist_ok=True)
    job.path("work", "sensitivity.json").write_text(json.dumps(out, indent=2))
    job.path("work", "candidates.json").write_text(json.dumps([asdict(c) for c in cands], indent=2))
    emit(STAGE, pct=98, msg=", ".join(f"{c.name} {c.footprint['size_gb']:.2f} GB" for c in cands))
    return cands
