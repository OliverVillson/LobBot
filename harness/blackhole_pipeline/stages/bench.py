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

from blackhole_pipeline.common import emit, stage_main

STAGE = "bench"


def run(bh, rj) -> None:
    p = bh.params
    if p["bench"] in ("", "none"):
        rj.mark_done(STAGE, {"skipped": True})
        emit(STAGE, "skipped", 100, "bench: none")
        return
    release = p["bench"].split("-")[-1] if p["bench"].startswith("lcb") else "v6"
    models = [("original", rj.model_dir), ("int4", bh.gptq_dir(rj))]
    res = {}
    for i, (label, path) in enumerate(models):
        out = rj.path("work", "bench", label)
        if not (out / "summary.json").exists():
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
        print(f"bench: {label} pass@1 {s['pass@1']:.1%} ({s['problems']} problems, {s['bench']})", flush=True)
    o, q = res["original"]["pass@1"], res["int4"]["pass@1"]
    summary = {"bench": res["original"]["bench"], "problems": res["original"]["problems"],
               "original": o, "int4": q, "retained": round(q / o, 4) if o else None, "runs": res}
    rj.write("work/bench.json", summary)
    rj.path(".done", "package").unlink(missing_ok=True)
    rj.mark_done(STAGE, {k: v for k, v in summary.items() if k != "runs"})
    msg = f"{summary['bench']}: original {o:.1%} → int4 {q:.1%} pass@1 on {summary['problems']} problems"
    emit(STAGE, "done", 100, msg, lcb_original=o, lcb_int4=q, problems=summary["problems"])


if __name__ == "__main__":
    stage_main(run, STAGE)
