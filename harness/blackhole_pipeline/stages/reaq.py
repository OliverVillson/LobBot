"""Stage reaq: REAQ per-expert codebooks on the REAP-pruned model (lab, measured only).

Every routed expert matrix (gate+up as one [2I, H] matrix, down as [H, I]) gets its own learned
level table, an fp32 matrix scale and one fp8 scale per `reaq_block` inputs. REAQ (router-weighted
expert activation quantization) picks the table size from REAP: in every layer the top
`reaq_top` of the kept experts by REAP score get 16 levels, the rest 8. Tables are a weighted
k-means on block-normalised weights (weights = each input channel's activation energy); rounding
to the tables is GPTQ, sequential over chunks of layers. The exact recipe of the B200 run on
2026-10-23 (gpu_perf/bench_b200.py headers --headers-gptq) that gave KL 0.033 at 3.75 bpw on
Qwen3-30B-A3B, beating GPTQ int4 (0.034 at 4.125 bpw).

No kernel serves this format yet, so the stage measures it (KL / top-1 / ppl vs the teacher, and
the size) and saves the packed indices, tables and scales for a future kernel or a bf16 loader.
Writes work/reaq.json and out/<name>-REAQ-codebook/.
"""

from __future__ import annotations

import shutil
import time

from blackhole_pipeline.common import Pruned, emit, expert_keys, gib, stage_main, write_checkpoint
from blackhole_pipeline.pack import (block_codes, bpw, codebook_dequant, expert_scale, pack_nibbles,
                                     table_indices)

STAGE = "reaq"
LAYOUT = """REAQ per-expert codebook experts (BlackHole-Harness, lab format v0)

Per MoE layer and tier (tier16 = 16-level tables, tier8 = 8-level), for each matrix m in
{gate_up, down} (gate_up = gate rows then up rows, [2I, H]; down [H, I]):
  <experts>.tier<L>.experts          int32 [n]          kept expert ids (new numbering) in this tier
  <experts>.tier<L>.<m>.idx          uint8 [n, out, in/2] 4-bit level indices, low nibble first
  <experts>.tier<L>.<m>.block_scale  float8_e4m3fn [n, out, in/block]
  <experts>.tier<L>.<m>.expert_scale float32 [n]
  <experts>.tier<L>.<m>.levels       float32 [n, L]     sorted level table in [-1, 1]
weight[e, o, i] = levels[e, idx[e, o, i]] * float(block_scale[e, o, i // block]) * expert_scale[e]
Everything else (attention, routers, shared experts, embeddings, LM head) is bf16 as in the
REAP-pruned model. blackhole_pipeline.pack.codebook_dequant() rebuilds bf16 experts.
"""


def run(bh, rj) -> None:
    import torch
    from gptq import gptq_table
    from headers import header_parts
    from reaper_pipeline.load import free
    from safetensors.torch import save_file

    p = bh.params
    block, top = p["reaq_block"], p["reaq_top"]
    emit(STAGE, pct=1, msg=f"loading {rj.out_dir.name}")
    m = Pruned(rj, p["gptq_calib"])
    sal = {L["layer"]: [L["saliency"][j] for j in L["kept"]] for L in rj.read("work/saliency.json")["layers"]}
    emit(STAGE, pct=3, msg="activation energy per input channel (k-means weights)")
    energy = m.hessians(m.blocks, diag=True)
    tier_of = {}
    for blk in m.blocks:
        sv = sal[blk.layer]
        if len(sv) != blk.num_experts:
            raise RuntimeError(f"layer {blk.layer}: {len(sv)} REAP scores for {blk.num_experts} experts")
        order = sorted(range(len(sv)), key=lambda j: -sv[j])
        cut = int(round(len(sv) * top))
        tier_of[blk.layer] = {j: (16 if r < cut else 8) for r, j in enumerate(order)}
    work = rj.path("work", "reaq")
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    nL, done, t0, worst, off, total = len(m.blocks), 0, time.time(), 0.0, 0, 0
    for chunk in m.chunks(p["layer_chunk"]):
        emit(STAGE, pct=5 + 75 * done / nL, msg=f"Hessians for layers {chunk[0].layer}–{chunk[-1].layer}")
        hs = m.hessians(chunk)
        for blk in chunk:
            Hin, Hdn = hs.pop(blk.layer)
            ein, edn = energy[blk.layer]
            prefix = m.names[blk.layer][0]["gate"].rsplit(".", 2)[0]
            out = {}
            for L in (16, 8):
                js = [j for j in range(blk.num_experts) if tier_of[blk.layer][j] == L]
                if not js:
                    continue
                jj = torch.tensor(js, device=ein.device)
                gu, dn = m.stacked(blk.layer, js, ein.device)
                qs = {}
                for mat, W, Hm, en in (("gate_up", gu, Hin, ein), ("down", dn, Hdn, edn)):
                    S, Lv = header_parts(W, "expert", L, block, en[jj], False)
                    Q = gptq_table(W, Hm[jj.to(Hm.device)], S, Lv)
                    es = expert_scale(W)
                    idx = table_indices(Q, S, Lv)
                    codes = block_codes(S, es, block)
                    packed = pack_nibbles(idx)
                    back = codebook_dequant(packed, W.shape[-1], codes, es, Lv, block)
                    # Elements GPTQ zeroed (dead input columns: no calibration token ever lit them) and
                    # all-zero blocks have no exact code in the saved format; measure the saved weights,
                    # so the reported KL is the KL of what is on disk, and count how many differ.
                    diff = (back - Q).abs() > 1e-3 * Q.abs().amax(dim=(1, 2), keepdim=True).clamp(min=1e-30)
                    off += int(diff.sum())
                    total += diff.numel()
                    worst = max(worst, float((back - Q).abs().max() / Q.abs().max().clamp(min=1e-30)))
                    Q = back
                    t = f"{prefix}.tier{L}.{mat}"
                    out[f"{t}.idx"] = packed.cpu()
                    out[f"{t}.block_scale"] = codes.cpu()
                    out[f"{t}.expert_scale"] = es.float().cpu()
                    out[f"{t}.levels"] = Lv.float().cpu()
                    qs[mat] = Q
                    del S, idx, codes, back
                out[f"{prefix}.tier{L}.experts"] = torch.tensor(js, dtype=torch.int32)
                m.put(blk.layer, js, qs["gate_up"], qs["down"])
                del gu, dn, qs
            save_file({k: v.contiguous() for k, v in out.items()}, str(work / f"layer_{blk.layer:03d}.safetensors"))
            del Hin, Hdn, out
            done += 1
            emit(STAGE, pct=5 + 75 * done / nL, msg=f"layer {done}/{nL} · codebooks 16/8 · {time.time() - t0:.0f}s")
        del hs
        free()
    quant_s = time.time() - t0
    frac = off / max(total, 1)
    print(f"reaq: {off:,} of {total:,} weights ({frac:.4%}) differ from the GPTQ result in the saved format "
          f"(dead columns / zero blocks); max rel diff {worst:.2e}; the measurement uses the saved weights", flush=True)
    if frac > 0.02:
        raise RuntimeError(f"saved codebooks differ from the GPTQ result on {frac:.2%} of weights: a packing bug")
    emit(STAGE, pct=81, msg="REAQ codebook model vs teacher on held-out answers")
    r = m.compare(lambda i, n: emit(STAGE, pct=81 + 12 * i / n, msg=f"held-out {i}/{n} vs teacher"))
    replace = expert_keys(m.names)
    params, n_calib = m.params, m.n_calib
    n_per = {L: sum(1 for t in tier_of[m.blocks[0].layer].values() if t == L) for L in (16, 8)}
    del m
    free()

    def new_tensors():
        from safetensors.torch import load_file
        for f in sorted(work.glob("layer_*.safetensors")):
            yield f"experts-{f.stem}.safetensors", load_file(str(f))

    dst = bh.reaq_dir(rj)
    qcfg = {"quantization_config": {"quant_method": "blackhole-reaq-codebook", "version": 0, "block": block,
                                    "tiers": {"16": n_per[16], "8": n_per[8]}, "rank": "REAP saliency",
                                    "serving": "none yet: needs a tiered per-expert lookup kernel"}}
    total = write_checkpoint(rj.out_dir, dst, replace, new_tensors, qcfg,
                             lambda i, n, what: emit(STAGE, pct=94 + 5 * i / n, msg=f"writing {dst.name}: {what}"))
    shutil.rmtree(work, ignore_errors=True)  # the per-layer parts now live in dst
    (dst / "REAQ_LAYOUT.txt").write_text(LAYOUT)
    n = n_per[16] + n_per[8]
    b = (n_per[16] * bpw(16, block) + n_per[8] * bpw(8, block)) / n
    g = rj.read("work/gptq.json") or {}
    nonexpert_gb = g["size_gb"] - g["expert_gb"] if g.get("size_gb") else None
    res = {"format": f"REAQ codebooks: top {top:.0%} of experts by REAP 16-level, rest 8-level, b{block}, GPTQ",
           "bpw": round(b, 3), "expert_gb": round(gib(params * b / 8), 2), "file_gb": round(gib(total), 2),
           "tiers": n_per, "out": str(dst), "quant_s": round(quant_s, 1), "roundtrip_max_rel_err": worst, "roundtrip_diff_frac": frac,
           "calib_seqs": n_calib}
    if nonexpert_gb is not None:
        res["size_gb"] = round(res["expert_gb"] + nonexpert_gb, 2)  # at the format's real bit width
    if r:
        res.update(top1=round(r["top1_agreement"], 4), kl=round(r["kl"], 4), ppl_teacher=round(r["ppl_teacher"], 3),
                   ppl_student=round(r["ppl_pruned"], 3))
    if r and "kl" in g:
        res["vs_int4"] = {"kl_change": round(res["kl"] / g["kl"] - 1, 4) if g["kl"] else None,
                          "expert_gb_change": round(res["expert_gb"] - g["expert_gb"], 2)}
    rj.write("work/reaq.json", res)
    rj.mark_done(STAGE, res)
    msg = f"{res['bpw']:.2f} bpw, experts {res['expert_gb']:.1f} GB"
    if r:
        msg += f" · KL {res['kl']:.3f} · top-1 {res['top1']:.0%}"
        if "vs_int4" in res and res["vs_int4"]["kl_change"] is not None:
            msg += f" · {res['vs_int4']['kl_change']:+.0%} KL vs int4"
    emit(STAGE, "done", 100, msg, **{k: res[k] for k in ("kl", "top1", "expert_gb", "bpw") if k in res})


if __name__ == "__main__":
    stage_main(run, STAGE)
