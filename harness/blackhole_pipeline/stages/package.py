"""Stage package: model cards, the receipt and links in the TUI job's out/.

The shippable model is the GPTQ int4 checkpoint; the bf16 and fp8 REAP folders are kept as
REAPer writes them, and the REAQ codebook folder is the lab measurement with its saved weights."""

from __future__ import annotations

import time
from pathlib import Path

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
    h = rj.read("work/heal.json") or {}
    x = rj.read("work/mlx.json") or {}
    qh = rj.read("work/qheal.json") or {}
    base = {"top1": ev.get("top1_agreement"), "kl": ev.get("kl"), "ppl_student": ev.get("ppl_pruned"),
            "size_gb": reap.get("bytes_after", 0) / 1024 ** 3 if reap.get("bytes_after") else None}
    base = {k: v for k, v in base.items() if v is not None}
    p = bh.params
    table = ["| model | expert bpw | experts | model size | KL vs teacher | top-1 | ppl |", "|---|---|---|---|---|---|---|",
             row(f"REAP {p['prune']:.0%}, bf16", base),
             *([row("+ distillation heal, bf16", {"kl": h["kl_after"], "top1": h["top1_after"]})]
               if h.get("kept") else []),
             *([row("REAP + GPTQ int4 (shipped)", g)] if g else []),
             *([row("REAP + REAQ codebooks (lab)", q)] if q else []),
             *([row(f"MLX for Apple silicon ({x.get('format', '')})",
                    {**x, "kl": x.get("unhealed_kl", x.get("kl")), "top1": x.get("unhealed_top1", x.get("top1"))})]
               if x else []),
             *([row("+ heal around the quantized MLX model", x)] if x.get("healed_after_quant") else [])]
    b = rj.read("work/bench.json") or {}
    if b.get("runs"):
        names = {"original": "original", "int4": "GPTQ int4", "reaq": "REAQ codebooks", "mlx": "MLX (Mac)"}
        table += ["", f"**{b['bench']}** ({b['problems']} problems, pass@1, one sample, up to "
                      f"{next(iter(b['runs'].values())).get('max_tokens', '?')} tokens):", "",
                  "| model | pass@1 | tok/s (batched, PyTorch) |", "|---|---|---|"]
        table += [f"| {names.get(m, m)} | {r['pass@1']:.1%} | {r.get('tok_s') or '–'} |" for m, r in b["runs"].items()]
    ppl_t = ev.get("ppl_teacher")
    gd, qd = bh.gptq_dir(rj), bh.reaq_dir(rj)
    xd = Path(x["out"]) if x.get("out") else None
    final = xd if xd and xd.exists() else gd if gd.exists() else rj.final_dir
    steps = [f"REAP {reap.get('experts_orig')} → {reap.get('experts_kept')} routed experts per layer "
             f"(calibration: {rj.config.purpose or rj.config.calib})"]
    if h.get("kept"):
        steps.append(f"distillation heal from the unpruned teacher ({h['sequences']} teacher answers, "
                     f"{h['steps']} steps, {h['trainable_params'] / 1e9:.2f}B trainable; routed experts frozen)")
    if g:
        steps.append("GPTQ int4 (group 128) on the routed experts, compressed-tensors W4A16 "
                     "(attention, routers, shared experts, embeddings, LM head bf16)")
    if q:
        steps.append("REAQ per-expert codebooks (16/8-level tables by REAP rank), a measured lab format "
                     "with no inference kernel yet (packed weights in the -REAQ-codebook folder)")
    if x:
        steps.append(f"MLX for Apple silicon: {x.get('format')}; attention / shared experts {p['mlx_attn_bits']}-bit, "
                     f"embedding {p['mlx_embed_bits']}-bit, LM head {p['mlx_head_bits']}-bit; routers and norms bf16")
    if x.get("healed_after_quant"):
        steps.append(f"heal again around the quantized MLX model ({qh.get('steps')} steps; experts frozen at 3-bit, "
                     "attention trained through its rounding)")
    card = [f"# {final.name}", "",
            f"{bh.config['model']} compressed by BlackHole-Harness (schema `{bh.config.get('schema')}`):", "",
            *[f"- {t}" for t in steps], "",
            "## Against the unpruned model (held-out teacher answers)", "", *table, "",
            f"Teacher ppl on the same answers: {ppl_t:.3f}." if ppl_t else ""]
    if xd and xd.exists():
        card += ["", "## Run it on a Mac", "", "```", "pip install mlx-lm",
                 f"mlx_lm.server --model {xd.name} --trust-remote-code    # OpenAI API on :8080 (opencode, pi, ...)",
                 f"mlx_lm.generate --model {xd.name} --trust-remote-code --prompt 'Write a binary search in Python'",
                 "```", "", "If macOS refuses to fit it in GPU memory, raise the limit: "
                 "`sudo sysctl iogpu.wired_limit_mb=12288` (16 GB Mac; resets on reboot)."]
    card.append("")
    for d in (gd, qd, xd):
        if d and d.exists():
            (d / "README.md").write_text("\n".join(card).replace(f"# {final.name}", f"# {d.name}", 1))
    links = [d for d in (rj.out_dir, rj.quant_dir, gd, qd, bh.healed_dir(rj), xd) if d and d.exists()]
    for d in links:
        bh.link(d)
    size = x.get("size_gb") if x else g.get("size_gb")
    receipt = {"schema": bh.config.get("schema"), "model_in": bh.config["model"], "model": str(final),
               "finished": time.strftime("%Y-%m-%d %H:%M:%S"), "params": p, "size_gb": size,
               "results": {"reap_bf16": base, "heal": h, "gptq_int4": g, "reaq_codebook": q, "mlx": x, "qheal": qh,
                           "bench": {k: v for k, v in b.items() if k != "runs"}},
               "artifacts": [str(d) for d in links], "gates_passed": True}
    bh.write("receipt.json", receipt)
    (bh.root / "RESULTS.md").write_text("\n".join(card))
    rj.mark_done(STAGE, {"out": str(final)})
    emit(STAGE, "done", 100, f"{final.name} · receipt.json · RESULTS.md")


if __name__ == "__main__":
    stage_main(run, STAGE)
