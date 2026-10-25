"""Stage gptq: GPTQ int4 (symmetric, group 128) on every routed expert of the REAP-pruned model.

Sequential over chunks of MoE layers: each chunk's per-expert Hessians come from the tokens the
model's own router sends to each expert, with the earlier chunks already quantized (the recipe
measured on B200, gpu_perf/bench_b200.py frontier). Attention, routers, shared experts,
embeddings and the LM head stay bf16. Then the quantized model is compared with the teacher on
the held-out answers (the same KL / top-1 / ppl as eval) and written as a compressed-tensors
pack-quantized W4A16 checkpoint next to the bf16 one.
Writes work/gptq.json, work/gptq/layer_<L>.safetensors and out/<name>-GPTQ-W4A16/.
"""

from __future__ import annotations

import shutil
import time

from blackhole_pipeline.common import Pruned, emit, expert_keys, gib, stage_main, write_checkpoint
from blackhole_pipeline.pack import int_codes, pack_int4

STAGE = "gptq"
FMT, GROUP, BITS = "int4_g128", 128, 4


def ct_config() -> dict:
    return {"quantization_config": {
        "quant_method": "compressed-tensors", "format": "pack-quantized", "quantization_status": "compressed",
        "config_groups": {"group_0": {
            "targets": [r"re:.*\.experts\.\d+\.(gate|up|down)_proj$"],
            "weights": {"num_bits": BITS, "type": "int", "symmetric": True, "strategy": "group",
                        "group_size": GROUP, "dynamic": False, "actorder": None, "observer": "minmax"},
            "input_activations": None, "output_activations": None, "format": "pack-quantized"}},
        "ignore": ["lm_head"]}}


def run(bh, rj) -> None:
    import torch
    from gptq import gptq_batched
    from reaper_pipeline.load import free
    from safetensors.torch import save_file

    p = bh.params
    emit(STAGE, pct=1, msg=f"loading {rj.out_dir.name}")
    m = Pruned(rj, p["gptq_calib"])
    if m.fused:
        raise RuntimeError("fused expert tensors (e.g. Qwen3.5 / 3.6) are not supported by this export yet; use the mlx stage")
    layers_dir = rj.path("work", "gptq")
    shutil.rmtree(layers_dir, ignore_errors=True)
    layers_dir.mkdir(parents=True)
    dtype = next(m.model.parameters()).dtype
    nL, done, t0 = len(m.blocks), 0, time.time()
    for chunk in m.chunks(p["layer_chunk"]):
        emit(STAGE, pct=2 + 78 * done / nL, msg=f"Hessians for layers {chunk[0].layer}–{chunk[-1].layer}")
        hs = m.hessians(chunk)
        for blk in chunk:
            Hin, Hdn = hs.pop(blk.layer)
            js = list(range(blk.num_experts))
            gu, dn = m.stacked(blk.layer, js, Hin.device)
            qg, sg = gptq_batched(gu, Hin, FMT, return_scales=True)
            qd, sd = gptq_batched(dn, Hdn, FMT, return_scales=True)
            m.put(blk.layer, js, qg, qd)
            cg, cd = int_codes(qg, sg, GROUP).cpu(), int_codes(qd, sd, GROUP).cpu()
            sg, sd = sg.cpu(), sd.cpu()
            I = gu.shape[1] // 2
            out = {}
            for j in js:
                nm = m.names[blk.layer][j]
                for key, codes, sc in (("gate", cg[j, :I], sg[j, :I]), ("up", cg[j, I:], sg[j, I:]),
                                       ("down", cd[j], sd[j])):
                    out[f"{nm[key]}.weight_packed"] = pack_int4(codes)
                    out[f"{nm[key]}.weight_scale"] = sc.to(dtype).contiguous()
                    out[f"{nm[key]}.weight_shape"] = torch.tensor(list(codes.shape), dtype=torch.int64)
            save_file(out, str(layers_dir / f"layer_{blk.layer:03d}.safetensors"))
            del Hin, Hdn, gu, dn, qg, qd, cg, cd, out
            done += 1
            emit(STAGE, pct=2 + 78 * done / nL, msg=f"layer {done}/{nL} · {blk.num_experts} experts · "
                                                     f"{time.time() - t0:.0f}s")
        del hs
        free()
    quant_s = time.time() - t0
    emit(STAGE, pct=81, msg="int4 model vs teacher on held-out answers")
    r = m.compare(lambda i, n: emit(STAGE, pct=81 + 12 * i / n, msg=f"held-out {i}/{n} vs teacher"))
    replace = expert_keys(m.names)
    params, n_calib = m.params, m.n_calib
    del m
    free()

    def new_tensors():
        for f in sorted(layers_dir.glob("layer_*.safetensors")):
            from safetensors.torch import load_file
            yield f"experts-{f.stem}.safetensors", load_file(str(f))

    dst = bh.gptq_dir(rj)
    total = write_checkpoint(rj.out_dir, dst, replace, new_tensors, ct_config(),
                             lambda i, n, what: emit(STAGE, pct=94 + 5 * i / n, msg=f"writing {dst.name}: {what}"))
    shutil.rmtree(layers_dir, ignore_errors=True)  # the per-layer parts now live in dst
    bpw = BITS + 16 / GROUP
    res = {"format": "GPTQ int4 g128 (compressed-tensors W4A16)", "bpw": bpw, "expert_params": params,
           "expert_gb": round(gib(params * bpw / 8), 2), "size_gb": round(gib(total), 2), "out": str(dst),
           "calib_seqs": n_calib, "quant_s": round(quant_s, 1)}
    if r:
        res.update(top1=round(r["top1_agreement"], 4), kl=round(r["kl"], 4), ppl_teacher=round(r["ppl_teacher"], 3),
                   ppl_student=round(r["ppl_pruned"], 3))
    rj.write("work/gptq.json", res)
    rj.mark_done(STAGE, res)
    msg = f"int4 experts {res['expert_gb']:.1f} GB, model {res['size_gb']:.1f} GB"
    if r:
        msg += f" · KL {res['kl']:.3f} · top-1 {res['top1']:.0%}"
    emit(STAGE, "done", 100, msg, **{k: res[k] for k in ("kl", "top1", "size_gb", "expert_gb") if k in res})


if __name__ == "__main__":
    stage_main(run, STAGE)
