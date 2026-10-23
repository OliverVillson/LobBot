"""Stage detect: what the source model is (family, dense or MoE, experts, size) from its
config.json and weight index, before anything big is downloaded; and whether the disk can hold
the run. This pipeline compresses MoE models (REAP needs routed experts)."""

from __future__ import annotations

import shutil
from pathlib import Path

from blackhole_pipeline.common import emit, stage_main

STAGE = "detect"


def run(bh, rj) -> None:
    from blackhole_tui import detect

    emit(STAGE, pct=10, msg="reading config.json")
    local = rj.model_dir if (rj.model_dir / "config.json").exists() else None
    d = detect.detect(str(local) if local else bh.config["model"])
    shape = (f"{d.family} · moe {d.total_b:.1f}B, {d.experts} experts top-{d.top_k}" if d.experts
             else f"{d.family} · dense {d.total_b:.1f}B")
    if not d.experts:
        raise RuntimeError(f"{shape}: this schema needs a mixture-of-experts model (dense models: dense-prune-int4)")
    p = bh.params
    src_gb = d.total_b * 2  # bf16 source
    need = (0 if local else src_gb) + src_gb * (1 - 0.9 * p["prune"]) * (1 + 0.5 + 0.6 + 0.6)  # bf16, fp8, int4 + parts, codebook + parts
    where = Path(rj.config.models_dir).expanduser() if not local else local
    while not where.exists():
        where = where.parent
    free = shutil.disk_usage(where).free / 1e9
    info = {"family": d.family, "experts": d.experts, "top_k": d.top_k, "total_b": d.total_b, "layers": d.layers,
            "where": "local" if local else d.where, "disk_free_gb": round(free), "disk_need_gb": round(need)}
    bh.write("detect.json", info)
    if free < need:
        raise RuntimeError(f"needs ~{need:.0f} GB free disk, {free:.0f} GB free")
    rj.mark_done(STAGE, info)
    emit(STAGE, "done", 100, f"{shape} · {'on disk' if local else 'to download'} · disk {free:.0f} GB free")


if __name__ == "__main__":
    stage_main(run, STAGE)
