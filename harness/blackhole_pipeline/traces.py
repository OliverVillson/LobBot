"""Verified shortest-trace distillation data.

The teacher solves competitive-programming problems (deepmind/code_contests: pre-2022, so not
in LiveCodeBench's newer releases) `samples` times each, with its own sampling settings; every
answer's code runs against the problem's tests (bench.run_problem) and the SHORTEST passing
answer is kept as the student's training sequence. The student learns to reason the way the
teacher does when the teacher is both right and brief, which targets what breaks compressed
reasoning models first: long, looping thoughts that never reach an answer.
"""

from __future__ import annotations

import os
import random
from concurrent.futures import ThreadPoolExecutor

MAX_TESTS = 12
MAX_CHARS = 6000


def contest_problems(n: int, seed: int = 1234) -> list[dict]:
    """n code_contests problems with stdin/stdout tests, in bench.py's problem format."""
    from datasets import load_dataset
    ds = load_dataset("deepmind/code_contests", split="train", streaming=True).shuffle(seed=seed, buffer_size=2000)
    out = []
    for r in ds:
        desc = r.get("description") or ""
        tests = []
        for key in ("public_tests", "private_tests", "generated_tests"):
            t = r.get(key) or {}
            tests += [{"input": i, "output": o} for i, o in zip(t.get("input", []), t.get("output", []))]
        if not desc or len(desc) > MAX_CHARS or len(tests) < 2:
            continue
        out.append({"id": r.get("name", str(len(out))), "title": r.get("name", ""), "question": desc, "starter": "",
                    "func": None, "difficulty": str(r.get("difficulty", "")), "date": "", "platform": "codecontests",
                    "tests": tests[:MAX_TESTS]})
        if len(out) >= n:
            break
    return out


def shortest_passing(problems: list[dict], samples: int, generate, check, on_progress=None) -> tuple[list[dict], dict]:
    """generate(list of (problem, sample index)) -> list of (prompt ids, answer ids, reply text);
    check(problem, reply text) -> bool. Returns training rows {"ids", "start"} (one per solved
    problem, its shortest passing answer) and statistics."""
    jobs = [(p, s) for p in problems for s in range(samples)]
    outs = generate(jobs)
    done = [0]

    def one(a):
        r = check(a[0][0], a[1][2]) if a[1][2] else False   # unfinished answers are empty: nothing to run
        done[0] += 1
        if on_progress and done[0] % 64 == 0:
            on_progress(done[0], len(jobs))
        return r
    with ThreadPoolExecutor(max(1, (os.cpu_count() or 4) - 1)) as ex:
        ok = list(ex.map(one, zip(jobs, outs)))
    best: dict[int, tuple] = {}
    lens_all, lens_pass = [], []
    for (p, _), (pids, aids, _), passed in zip(jobs, outs, ok):
        lens_all.append(len(aids))
        if not passed:
            continue
        lens_pass.append(len(aids))
        k = id(p)
        if k not in best or len(aids) < len(best[k][1]):
            best[k] = (pids, aids)
    rows = [{"ids": list(pids) + list(aids), "start": len(pids)} for pids, aids in best.values()]
    random.Random(0).shuffle(rows)
    kept = [len(r["ids"]) - r["start"] for r in rows]
    mean = lambda v: round(sum(v) / len(v), 1) if v else 0.0  # noqa: E731
    stats = {"problems": len(problems), "samples": samples, "answers": len(jobs), "passed": sum(ok),
             "solved": len(rows), "solve_rate": round(len(rows) / max(1, len(problems)), 4),
             "tokens_mean_all": mean(lens_all), "tokens_mean_passing": mean(lens_pass), "tokens_mean_kept": mean(kept),
             "shorter_than_passing": round(1 - mean(kept) / mean(lens_pass), 4) if lens_pass else 0.0}
    return rows, stats
