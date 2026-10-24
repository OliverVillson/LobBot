"""Stage package: model cards, the receipt and links in the TUI job's out/.

The shippable model is the GPTQ int4 checkpoint; the bf16 and fp8 REAP folders are kept as
REAPer writes them, and the REAQ codebook folder is the lab measurement with its saved weights."""

from __future__ import annotations

import time

from blackhole_pipeline.common import emit, stage_main

STAGE = "package"


def row(name: str, r: dict, size_key: str = "size_gb") -> str:
    kl = f"{r['kl']:.4f}" if "kl" in r else "–"
    t1 = f"{r['top1']:.1%}" if "top1" in r else "–"
    pp = f"{r['ppl_student']:.3f}" if "ppl_student" in r else "–"
    gb = f"{r[size_key]:.1f} GB" if r.get(size_key) is not None else "–"
    eg = f"{r['expert_gb']:.1f} GB" if r.get("expert_gb") is not None else "–"
    bpw = f"{r['bpw']:.3g}" if r.get("bpw") else "16"
    return f"| {name} | {bpw} | {eg} | {gb} | {kl} | {t1} | {pp} |"


def run(bh, rj) -> None:
    from reaper_pipeline.stages import package as reaper_package

    emit(STAGE, pct=10, msg="REAPer model card")
    reaper_package.run(rj)  # README + saliency next to the bf16 and fp8 folders (prints its own done event)
    ev = rj.read("work/eval.json") or {}
    g = rj.read("work/gptq.json") or {}
    q = rj.read("work/reaq.json") or {}
    reap = rj.read("work/reap.json") or {}
    base = {"top1": ev.get("top1_agreement"), "kl": ev.get("kl"), "ppl_student": ev.get("ppl_pruned"),
            "size_gb": reap.get("bytes_after", 0) / 1024 ** 3 if reap.get("bytes_after") else None}
    base = {k: v for k, v in base.items() if v is not None}
    p = bh.params
    table = ["| model | expert bpw | experts | model size | KL vs teacher | top-1 | ppl |", "|---|---|---|---|---|---|---|",
             row(f"REAP {p['prune']:.0%}, bf16", base),
             row("REAP + GPTQ int4 (shipped)", g),
             row("REAP + REAQ codebooks (lab)", q)]
    b = rj.read("work/bench.json") or {}
    if b.get("runs"):
        names = {"original": "original", "int4": "GPTQ int4", "reaq": "REAQ codebooks"}
        table += ["", f"**{b['bench']}** ({b['problems']} problems, pass@1, one sample, up to "
                      f"{next(iter(b['runs'].values())).get('max_tokens', '?')} tokens):", "",
                  "| model | pass@1 | tok/s (batched, PyTorch) |", "|---|---|---|"]
        table += [f"| {names.get(m, m)} | {r['pass@1']:.1%} | {r.get('tok_s') or '–'} |" for m, r in b["runs"].items()]
    ppl_t = ev.get("ppl_teacher")
    card = [
        f"# {bh.gptq_dir(rj).name}", "",
        f"{bh.config['model']} compressed by BlackHole-Harness (schema `{bh.config.get('schema')}`, lab): "
        f"REAP {reap.get('experts_orig')} → {reap.get('experts_kept')} routed experts per layer, then GPTQ int4 "
        f"(group 128) on the routed experts; attention, routers, shared experts, embeddings and the LM head stay bf16.", "",
        f"Purpose: {rj.config.purpose or 'general'} · calibration {reap.get('calib_samples')} samples, "
        f"{reap.get('tokens', 0):,} tokens · GPTQ Hessians from {g.get('calib_seqs')} sequences.", "",
        "## Against the unpruned model (held-out teacher answers)", "", *table, "",
        f"Teacher ppl on the same answers: {ppl_t:.3f}." if ppl_t else "", "",
        "Format: compressed-tensors `pack-quantized` W4A16 (what llm-compressor writes). Whether a given vLLM build "
        "serves it depends on that build's model class for this architecture supporting quantized MoE experts.", "",
        "The REAQ codebook row is a measured lab format (per-expert learned 16/8-level tables chosen by REAP score); "
        "no inference kernel exists yet. Its packed weights are in the `-REAQ-codebook` folder (see REAQ_LAYOUT.txt).", ""]
    gd, qd = bh.gptq_dir(rj), bh.reaq_dir(rj)
    if gd.exists():
        (gd / "README.md").write_text("\n".join(card))
    if qd.exists():
        (qd / "README.md").write_text("\n".join(card).replace(f"# {gd.name}", f"# {qd.name}", 1))
    links = [d for d in (rj.out_dir, rj.quant_dir, gd, qd) if d.exists()]
    for d in links:
        bh.link(d)
    final = gd if gd.exists() else rj.final_dir
    receipt = {"schema": bh.config.get("schema"), "model_in": bh.config["model"], "model": str(final),
               "finished": time.strftime("%Y-%m-%d %H:%M:%S"), "params": p,
               "size_gb": g.get("size_gb"), "results": {"reap_bf16": base, "gptq_int4": g, "reaq_codebook": q,
                           "bench": {k: v for k, v in b.items() if k != "runs"}},
               "artifacts": [str(d) for d in links], "gates_passed": True}
    bh.write("receipt.json", receipt)
    (bh.root / "RESULTS.md").write_text("\n".join(card))
    rj.mark_done(STAGE, {"out": str(final)})
    emit(STAGE, "done", 100, f"{final.name} · receipt.json · RESULTS.md")


if __name__ == "__main__":
    stage_main(run, STAGE)
