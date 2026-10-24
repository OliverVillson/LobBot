"""Stage mlx: the REAP-pruned (and healed, when heal ran) model as an MLX folder for Apple silicon.

Routed experts get GPTQ at `mlx_bits` (group `mlx_group`), sequential over chunks of MoE layers
like the gptq stage; with mlx_hot > 0 the most salient share of each layer's experts (REAP
score, Oliver's REAQ idea as bit widths) gets `mlx_hot_bits` and is stored as its own stacked
tensors. GPTQ's symmetric codes map exactly onto MLX's affine format, so the Mac runs the
weights GPTQ chose. Attention, shared experts and dense MLPs get round-to-nearest at
`mlx_attn_bits`, the embedding at `mlx_embed_bits` and the LM head at `mlx_head_bits`; routers
and norms stay bf16. The quantized model is compared with the teacher (KL / top-1) and written
with flint_mlx.py, so `mlx_lm.server --model DIR --trust-remote-code` serves it.
Writes work/mlx.json and out/<name>-MLX-<size>/.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from blackhole_pipeline.common import Pruned, emit, gib, stage_main

STAGE = "mlx"
MODEL_FILE = Path(__file__).resolve().parents[1] / "mlx" / "flint_mlx.py"


def gptq_fmt(bits: int, group: int) -> str:
    return f"int{bits}_g{group}"


def run(bh, rj) -> None:
    import torch
    from gptq import GRID, gptq_batched
    from reaper_pipeline.ckpt import SKIP_DIRS, WEIGHT_SUFFIXES
    from reaper_pipeline.load import free
    from safetensors.torch import save_file

    from blackhole_pipeline.pack import int_codes_bits, mlx_bpw, mlx_dequant, mlx_from_symmetric, mlx_pack, mlx_rtn

    p = bh.params
    bits, group, hot_bits, hot_share = p["mlx_bits"], p["mlx_group"], p["mlx_hot_bits"], p["mlx_hot"]
    for b in {bits, hot_bits}:
        GRID.setdefault(gptq_fmt(b, group), (b, group))
    src = bh.healed_dir(rj) if bh.healed_dir(rj).exists() else rj.out_dir
    emit(STAGE, pct=1, msg=f"loading {src.name}")
    m = Pruned(rj, p["gptq_calib"], src=src)
    cfg = json.loads((src / "config.json").read_text())
    if cfg.get("model_type") != "flint":
        raise RuntimeError(f"MLX export is written for Flint; this model is {cfg.get('model_type')!r}")
    dtype = torch.bfloat16
    sal = {}
    if hot_share > 0:
        sal = {L["layer"]: [L["saliency"][j] for j in L["kept"]] for L in rj.read("work/saliency.json")["layers"]}

    out: dict[str, torch.Tensor] = {}
    quant: dict[str, dict] = {}
    tiers: dict[str, int] = {}
    nbytes = {"experts": 0, "rest": 0}

    def put_q(path: str, u, s, b, nb: int, kind: str) -> None:
        out[f"{path}.weight"] = mlx_pack(u, nb).cpu()
        out[f"{path}.scales"] = s.contiguous().cpu()
        out[f"{path}.biases"] = b.contiguous().cpu()
        quant[path] = {"group_size": group, "bits": nb}
        nbytes[kind] += sum(out[f"{path}.{x}"].numel() * out[f"{path}.{x}"].element_size()
                            for x in ("weight", "scales", "biases"))

    nL, done, t0 = len(m.blocks), 0, time.time()
    for chunk in m.chunks(p["layer_chunk"]):
        emit(STAGE, pct=2 + 70 * done / nL, msg=f"Hessians for layers {chunk[0].layer}–{chunk[-1].layer}")
        hs = m.hessians(chunk)
        for blk in chunk:
            Hin, Hdn = hs.pop(blk.layer)
            E = blk.num_experts
            if hot_share > 0:
                sv = sal[blk.layer]
                if len(sv) != E:
                    raise RuntimeError(f"layer {blk.layer}: {len(sv)} REAP scores for {E} experts")
                order = sorted(range(E), key=lambda j: -sv[j])
                n_hot = int(round(E * hot_share))
            else:
                order, n_hot = list(range(E)), 0
            prefix = f"model.layers.{blk.layer}.mlp"
            for name, js, nb in (("experts_hot", order[:n_hot], hot_bits), ("experts", order[n_hot:], bits)):
                if not js:
                    continue
                gu, dn = m.stacked(blk.layer, js, Hin.device)
                qg, sg = gptq_batched(gu, Hin[js], gptq_fmt(nb, group), return_scales=True)
                qd, sd = gptq_batched(dn, Hdn[js], gptq_fmt(nb, group), return_scales=True)
                ug, sgm, bgm = mlx_from_symmetric(int_codes_bits(qg, sg, group, nb), sg, nb, dtype)
                ud, sdm, bdm = mlx_from_symmetric(int_codes_bits(qd, sd, group, nb), sd, nb, dtype)
                m.put(blk.layer, js, mlx_dequant(ug, sgm, bgm, group), mlx_dequant(ud, sdm, bdm, group))
                I = gu.shape[1] // 2
                put_q(f"{prefix}.{name}.gate_proj", ug[:, :I], sgm[:, :I], bgm[:, :I], nb, "experts")
                put_q(f"{prefix}.{name}.up_proj", ug[:, I:], sgm[:, I:], bgm[:, I:], nb, "experts")
                put_q(f"{prefix}.{name}.down_proj", ud, sdm, bdm, nb, "experts")
                del gu, dn, qg, qd, ug, ud
            if n_hot:
                tiers[str(blk.layer)] = n_hot
            # the router's rows follow the experts' new order (hot first)
            out[f"{prefix}.gate.weight"] = blk.router.weight.detach()[order].to(dtype).cpu()
            del Hin, Hdn
            done += 1
            emit(STAGE, pct=2 + 70 * done / nL, msg=f"layer {done}/{nL} · {E} experts ({n_hot} on {hot_bits}-bit) · "
                                                     f"{time.time() - t0:.0f}s")
        del hs
        free()

    emit(STAGE, pct=73, msg="attention, shared experts, embeddings")
    expert_mods = {n for per in m.names.values() for e in per for n in e.values()}
    with torch.no_grad():
        for name, mod in m.model.named_modules():
            if name in expert_mods:
                continue
            if isinstance(mod, torch.nn.Embedding):
                nb = p["mlx_embed_bits"]
            elif isinstance(mod, torch.nn.Linear):
                nb = p["mlx_head_bits"] if name == "lm_head" else p["mlx_attn_bits"]
            else:
                continue
            u, s, b = mlx_rtn(mod.weight, nb, group, dtype)
            mod.weight.copy_(mlx_dequant(u, s, b, group).to(mod.weight.dtype))
            put_q(name, u, s, b, nb, "rest")
        for name, prm in m.model.named_parameters():  # routers (done above), norms, sinks: unquantized
            mod_name = name.rsplit(".", 1)[0]
            if name in out or mod_name in quant or mod_name in expert_mods:
                continue
            t = prm.detach().to(torch.float32 if name.endswith("softmax_offset") else dtype).cpu()
            out[name] = t
            nbytes["rest"] += t.numel() * t.element_size()

    emit(STAGE, pct=80, msg="MLX model vs teacher on held-out answers")
    r = m.compare(lambda i, n: emit(STAGE, pct=80 + 12 * i / n, msg=f"held-out {i}/{n} vs teacher"))
    expert_params = m.params
    del m
    free()

    total = nbytes["experts"] + nbytes["rest"]
    dst = rj.out_dir.with_name(f"{rj.out_dir.name}-MLX-{gib(total):.1f}GB")
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    emit(STAGE, pct=93, msg=f"writing {dst.name}")
    shard, size, n, index = {}, 0, 0, {}

    def flush():
        nonlocal shard, size, n
        if shard:
            n += 1
            fname = f"model-{n:05d}.safetensors"
            save_file(shard, str(tmp / fname), metadata={"format": "mlx"})
            index.update({k: fname for k in shard})
        shard, size = {}, 0

    for k in sorted(out):
        shard[k] = out[k]
        size += out[k].numel() * out[k].element_size()
        if size > 4 * 1024 ** 3:
            flush()
    flush()
    (tmp / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": total}, "weight_map": index}, indent=2) + "\n")
    for f in src.iterdir():
        if f.is_dir() or f.name in SKIP_DIRS or f.name.endswith(WEIGHT_SUFFIXES) \
                or f.name in ("model.safetensors.index.json", "config.json", "README.md"):
            continue
        shutil.copy2(f, tmp / f.name)
    shutil.copy2(MODEL_FILE, tmp / MODEL_FILE.name)
    cfg.pop("quantization_config", None)
    cfg.update({"model_file": MODEL_FILE.name, "expert_tiers": tiers,
                "quantization": {"group_size": group, "bits": bits, **quant}})
    (tmp / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    shutil.rmtree(dst, ignore_errors=True)
    tmp.rename(dst)
    bh.link(dst)

    avg = (bits * (1 - hot_share) + hot_bits * hot_share)
    res = {"format": f"MLX affine, experts GPTQ {bits}-bit" + (f" (top {hot_share:.0%} by REAP {hot_bits}-bit)"
                                                               if hot_share else "") + f" g{group}",
           "bpw": round(avg + 32 / group, 3), "expert_params": expert_params,
           "expert_gb": round(gib(nbytes["experts"]), 2), "size_gb": round(gib(total), 2), "out": str(dst),
           "healed": src != rj.out_dir}
    if r:
        res.update(top1=round(r["top1_agreement"], 4), kl=round(r["kl"], 4), ppl_teacher=round(r["ppl_teacher"], 3),
                   ppl_student=round(r["ppl_pruned"], 3))
    rj.write("work/mlx.json", res)
    rj.mark_done(STAGE, res)
    msg = f"MLX {res['size_gb']:.1f} GB (experts {res['expert_gb']:.1f} GB)"
    if r:
        msg += f" · KL {res['kl']:.3f} · top-1 {res['top1']:.0%}"
    emit(STAGE, "done", 100, msg, **{k: res[k] for k in ("kl", "top1", "size_gb", "expert_gb") if k in res})


if __name__ == "__main__":
    stage_main(run, STAGE)
