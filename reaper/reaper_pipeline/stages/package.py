"""Stage package: model card and REAP record next to the pruned weights."""

from __future__ import annotations

import shutil

from reaper_pipeline.events import emit, size
from reaper_pipeline.job import Job
from reaper_pipeline.stages._common import stage_main

STAGE = "package"


def card(job: Job) -> str:
    cfg = job.config
    r = job.read("work/reap.json") or {}
    e = job.read("work/eval.json") or {}
    t = job.read("data/topics.json") or {}
    lines = [
        "---", "tags: [moe, reap, expert-pruning]", f"base_model: {cfg.model}", "---", "",
        f"# {job.out_dir.name}", "",
        f"{cfg.model} with {r.get('experts_orig')} → {r.get('experts_kept')} routed experts per MoE layer "
        f"({cfg.prune:.0%} pruned with REAP, router-weighted expert activation pruning), made with REAPer.", "",
        f"- Purpose: {cfg.purpose or 'general'}",
        f"- Calibration: {r.get('calib_samples')} samples, {r.get('tokens', 0):,} tokens. {t.get('note', '')}",
        f"- Size: {size(r.get('bytes_before', 0))} → {size(r.get('bytes_after', 0))}",
        "- Same tensor layout and quantization as the original, so it loads wherever the original does.",
    ]
    if t.get("topics"):
        lines += ["", "Sub-tasks the teacher derived from the purpose:", ""] + [f"- {x}" for x in t["topics"]]
    if "top1_agreement" in e:
        lines += ["", "## Against the unpruned model (held-out teacher answers)", "",
                  "| top-1 agreement | KL (teacher ‖ pruned) | ppl teacher | ppl pruned |", "|---|---|---|---|",
                  f"| {e['top1_agreement']:.1%} | {e['kl']:.4f} | {e['ppl_teacher']:.3f} | {e['ppl_pruned']:.3f} |"]
    lines += ["", "Pruned without any fine-tuning; the kept experts' weights are unchanged. "
              "`reap_saliency.json` has every expert's score and the ids kept per layer.", ""]
    return "\n".join(lines)


def run(job: Job) -> None:
    out = job.out_dir
    if not (out / "config.json").exists():
        raise RuntimeError(f"no pruned model in {out}")
    (out / "README.md").write_text(card(job))
    shutil.copy2(job.path("work", "saliency.json"), out / "reap_saliency.json")
    if job.path("work", "eval.json").exists():
        shutil.copy2(job.path("work", "eval.json"), out / "reap_eval.json")
    job.mark_done(STAGE, {"out": str(out)})
    emit(STAGE, "done", 100, f"{out} ({size(sum(p.stat().st_size for p in out.glob('*.safetensors')))})")


if __name__ == "__main__":
    stage_main(run, STAGE)
