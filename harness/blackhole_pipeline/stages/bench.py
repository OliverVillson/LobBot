"""Stage bench: LiveCodeBench (bench.py) on the original model and on the shipped GPTQ int4
model, so the compression's cost shows up as solved coding problems, not only KL.

Each model runs in its own process (the GPU is freed between them); a model whose
work/bench/<label>/summary.json exists is not run again. bench = "none" skips the stage.
Writes work/bench.json. Clears package's done mark so the model card picks the result up.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from blackhole_pipeline.common import emit, stage_main

STAGE = "bench"


def run(bh, rj) -> None:
    p = bh.params
    if p["bench"] in ("", "none"):
        rj.mark_done(STAGE, {"skipped": True})
        emit(STAGE, "skipped", 100, "bench: none")
        return
    release = p["bench"].split("-")[-1] if p["bench"].startswith("lcb") else "v6"
    dirs = {"original": rj.model_dir, "int4": bh.gptq_dir(rj), "reaq": bh.reaq_dir(rj),
            "mlx": Path((rj.read("work/mlx.json") or {}).get("out", rj.path("work", "no-mlx-model")))}
    models = [(m, dirs[m]) for m in p["bench_models"].split(",") if m]
    res = {}
    for i, (label, path) in enumerate(models):
        out = rj.path("work", "bench", label)
        if label == "original":  # the same for every job on this model: one run per model and settings
            from blackhole_tui.jobs import home
            slug = "".join(c if c.isalnum() or c in "-." else "-" for c in str(bh.config["model"]))[-80:]
            out = home() / "bench" / f"{slug}-{p['bench']}-{p['bench_max_tokens']}-{p['bench_limit']}"
        old = json.loads((out / "summary.json").read_text()) if (out / "summary.json").exists() else {}
        if old.get("max_tokens") != p["bench_max_tokens"] or (label != "original" and old.get("model") != str(path)):
            # missing, run with another token limit, or the model folder changed (qheal)
            if not (path / "config.json").exists():
                raise RuntimeError(f"no model to bench in {path}")
            cmd = [sys.executable, "-u", "-m", "blackhole_pipeline.bench", "--model", str(path), "--out", str(out),
                   "--release", release, "--max-tokens", str(p["bench_max_tokens"]), "--stage", STAGE,
                   "--pct", f"{100 * i / len(models)},{100 * (i + 1) / len(models)}", "--label", label]
            if p["bench_limit"]:
                cmd += ["--limit", str(p["bench_limit"])]
            if p["bench_file"]:
                cmd += ["--file", p["bench_file"]]
            rc = subprocess.call(cmd)
            if rc != 0:
                raise RuntimeError(f"bench of the {label} model failed (exit {rc})")
        res[label] = json.loads((out / "summary.json").read_text())
        s = res[label]
        print(f"bench: {label} pass@1 {s['pass@1']:.1%} ({s['problems']} problems, {s['bench']}), "
              f"{s.get('tok_s', 0):.0f} tok/s", flush=True)
    first = res[models[0][0]]
    summary = {"bench": first["bench"], "problems": first["problems"], "runs": res,
               **{m: r["pass@1"] for m, r in res.items()}, "tok_s": {m: r.get("tok_s") for m, r in res.items()}}
    comp = [m for m in res if m != "original"]
    if "original" in res and comp and res["original"]["pass@1"]:
        summary["retained"] = round(res[comp[-1]]["pass@1"] / res["original"]["pass@1"], 4)
    rj.write("work/bench.json", summary)
    rj.path(".done", "package").unlink(missing_ok=True)
    rj.mark_done(STAGE, {k: v for k, v in summary.items() if k != "runs"})
    msg = f"{summary['bench']} pass@1: " + " · ".join(f"{m} {r['pass@1']:.1%}" for m, r in res.items()) \
        + f" on {summary['problems']} problems"
    emit(STAGE, "done", 100, msg, problems=summary["problems"], **{f"lcb_{m}": r["pass@1"] for m, r in res.items()})


if __name__ == "__main__":
    stage_main(run, STAGE)
