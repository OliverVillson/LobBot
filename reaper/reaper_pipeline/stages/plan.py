"""Stage plan: what this machine is, whether the model is a MoE it can run,
and how every later stage adapts (dtype, GPU placement, batch sizes, how the
teacher is served). Writes hardware.json, plan.json and model.json."""

from __future__ import annotations

import json
from pathlib import Path

from reaper_pipeline import hardware
from reaper_pipeline.events import emit, log
from reaper_pipeline.job import Job
from reaper_pipeline.moe import EXPERT_COUNT_KEYS
from reaper_pipeline.stages._common import stage_main

STAGE = "plan"


def model_config(job: Job) -> dict:
    local = job.model_dir / "config.json"
    if local.exists():
        return json.loads(local.read_text())
    from huggingface_hub import hf_hub_download
    return json.loads(Path(hf_hub_download(job.config.model, "config.json")).read_text())


def moe_shape(cfg: dict) -> dict:
    c = cfg.get("text_config") or cfg
    E = next((c[k] for k in EXPERT_COUNT_KEYS if isinstance(c.get(k), int) and c[k] > 1), 0)
    return {"model_type": c.get("model_type") or cfg.get("model_type"), "experts": E,
            "top_k": c.get("num_experts_per_tok") or c.get("moe_k") or 0,
            "layers": c.get("num_hidden_layers", 0), "hidden": c.get("hidden_size", 0),
            "quantization": (cfg.get("quantization_config") or {}).get("quant_method")}


def run(job: Job) -> None:
    cfg = job.config
    if not 0 < cfg.prune < 1:
        raise ValueError(f"prune must be between 0 and 1, got {cfg.prune}")
    emit(STAGE, pct=10, msg=f"reading {cfg.model} config")
    shape = moe_shape(model_config(job))
    if shape["experts"] < 2:
        raise RuntimeError(f"{cfg.model} is not a mixture-of-experts model (no expert count in its config)")
    job.write("model.json", shape)
    log(f"model: {shape}")

    emit(STAGE, pct=40, msg="detecting hardware")
    src = str(job.model_dir) if job.model_dir.exists() else cfg.model
    hw, prof = hardware.profile(src, cfg.prune, cfg.max_seq)
    job.write("hardware.json", hw)
    job.write("plan.json", prof)
    log(f"hardware ({prof['source']}): {prof.get('summary', '')}")
    for n in prof.get("notes", []):
        log(f"plan: {n}")
    if not prof.get("ok", True):
        raise RuntimeError("; ".join(prof.get("problems") or ["the planner refused this model on this machine"]))
    job.mark_done(STAGE)
    emit(STAGE, "done", 100, f"{shape['model_type']}: {shape['experts']} experts top-{shape['top_k']} · "
                             f"{prof.get('summary') or prof['device']}")


if __name__ == "__main__":
    stage_main(run, STAGE)
