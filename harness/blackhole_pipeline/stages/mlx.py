"""Stage mlx: the REAP-pruned (and healed, when heal ran) model as an MLX folder for Apple silicon.

Routed experts get GPTQ at `mlx_bits` (group `mlx_group`), sequential over chunks of MoE layers
like the gptq stage; with mlx_hot > 0 the most salient share of each layer's experts (REAP
score, Oliver's REAQ idea as bit widths) gets `mlx_hot_bits` and is stored as its own stacked
tensors. GPTQ's symmetric codes map exactly onto MLX's affine format, so the Mac runs the
weights GPTQ chose. Attention, shared experts and dense MLPs get GPTQ too (`mlx_attn_gptq`; else
round-to-nearest) at `mlx_attn_bits` (q/o projections at `mlx_qo_bits` when set), the LM head at
`mlx_head_bits`, the embedding round-to-nearest at `mlx_embed_bits`; routers and norms stay bf16.
Output-only vocab trim (`mlx_head_vocab` = K > 0): the LM head keeps only K rows, the tokens the
teacher actually writes (its distillation answers and calibration answers first, then the lowest
ids, the tokenizer's most common merges); the model still reads every token, and the rest get
-inf logits. Fewer head rows are both a smaller file and less memory read per generated token. The quantized model is compared with the teacher (KL / top-1) and written
with its model file (flint_mlx.py for Flint, qwen_mlx.py for Qwen3.5 / 3.6 MoE; mlx_layout.py has
the names), so `mlx_lm.server --model DIR --trust-remote-code` serves it.
Writes work/mlx.json and out/<name>-MLX-<size>/.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from blackhole_pipeline.common import Pruned, emit, gib, stage_main

STAGE = "mlx"
HEX_BLOCK = 32


def gptq_fmt(bits: int, group: int) -> str:
    return f"int{bits}_g{group}"


def linear_hessians(model, names: list[str], batches) -> dict:
    """GPTQ Hessians (sum of x x^T over real tokens) of the inputs of the named Linear layers.
    Layers that share an input (q/k/v, gate/up) share one matrix."""
    import torch
    mods = dict(model.named_modules())
    share = {}
    for n in names:
        head, _, leaf = n.rpartition(".")
        src = f"{head}.q_proj" if leaf in ("k_proj", "v_proj") else f"{head}.gate_proj" if leaf == "up_proj" else n
        share[n] = src if src in mods else n
    Hs, state, hooks = {}, {"mask": None}, []
    for src in set(share.values()):
        def hook(mod, args, src=src):
            x = args[0].reshape(-1, args[0].shape[-1])
            m = state["mask"]
            if m is not None and m.numel() == x.shape[0]:
                x = x[m.to(x.device)]
            x = x.float()
            if src in Hs:
                Hs[src].addmm_(x.T, x)
            else:
                Hs[src] = x.T @ x
        hooks.append(mods[src].register_forward_pre_hook(hook))
    try:
        with torch.no_grad():
            for ids, mask in batches:
                state["mask"] = mask.reshape(-1).bool()
                model(input_ids=ids, attention_mask=mask, use_cache=False)
    finally:
        for h in hooks:
            h.remove()
    return {n: Hs[share[n]] for n in names if share[n] in Hs}


def head_vocab(rj, tok, vocab: int, k: int) -> list[int]:
    """The k token ids the LM head keeps: special tokens, then the tokens of the teacher's own
    answers (distillation data, calibration answers) by frequency, then the lowest ids."""
    from collections import Counter

    from reaper_pipeline.stages._common import read_jsonl
    keep = {i for i in getattr(tok, "all_special_ids", []) if i is not None and i < vocab}
    keep |= {int(i) for i in getattr(tok, "added_tokens_decoder", {}) if int(i) < vocab}
    cnt = Counter()
    for r in read_jsonl(rj.path("work", "heal_data.jsonl")):
        cnt.update(r["ids"][r["start"]:])
    for r in read_jsonl(rj.path("data", "calib.jsonl")):
        if r.get("answer"):
            cnt.update(tok(r["answer"], add_special_tokens=False)["input_ids"])
    for i, _ in cnt.most_common():
        if len(keep) >= k:
            break
        if i < vocab:
            keep.add(i)
    i = 0
    while len(keep) < k and i < vocab:
        keep.add(i)
        i += 1
    return sorted(keep)[:k] if len(keep) > k else sorted(keep)


def run(bh, rj) -> None:
    import torch
    from gptq import GRID, gptq_batched
    from reaper_pipeline.ckpt import SKIP_DIRS, WEIGHT_SUFFIXES
    from reaper_pipeline.load import free
    from safetensors.torch import save_file

    from blackhole_pipeline.pack import int_codes_bits, mlx_bpw, mlx_dequant, mlx_from_symmetric, mlx_pack, mlx_rtn
    from blackhole_pipeline.serve import mask_head
    from gptq import gptq_table
    from headers import header_parts

    from blackhole_pipeline.mlx_layout import MLX_DIR, layout_for
    from blackhole_pipeline.pack import block_codes, codebook_dequant, expert_scale, pack_nibbles, table_indices

    p = bh.params
    bits, group, hot_bits, hot_share = p["mlx_bits"], p["mlx_group"], p["mlx_hot_bits"], p["mlx_hot"]
    for b in {bits, hot_bits}:
        GRID.setdefault(gptq_fmt(b, group), (b, group))
    src = bh.healed_dir(rj) if bh.healed_dir(rj).exists() else rj.out_dir
    emit(STAGE, pct=1, msg=f"loading {src.name}")
    m = Pruned(rj, p["gptq_calib"], src=src)
    cfg = json.loads((src / "config.json").read_text())
    lay = layout_for(cfg)   # Flint (flint_mlx.py) or Qwen3.5 / 3.6 MoE (qwen_mlx.py): names and layouts in the folder
    dtype = torch.bfloat16
    sal = {}
    if hot_share > 0:
        sal = {L["layer"]: [L["saliency"][j] for j in L["kept"]] for L in rj.read("work/saliency.json")["layers"]}

    out: dict[str, torch.Tensor] = {}
    quant: dict[str, dict] = {}
    tiers: dict[str, int] = {}
    hexcfg: dict[str, dict] = {}
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
            prefix = lay.to_mlx(f"model.layers.{blk.layer}.mlp")
            for name, js, nb in (("experts_hot", order[:n_hot], hot_bits), ("experts", order[n_hot:], bits)):
                if not js:
                    continue
                gu, dn = m.stacked(blk.layer, js, Hin.device)
                I = gu.shape[1] // 2
                if p["mlx_hex"]:  # HEX: per-expert learned level tables, GPTQ rounding to them
                    hexcfg.setdefault(str(blk.layer), {})[name] = nb
                    jj = torch.tensor(js, device=Hin.device)
                    back = {}
                    for mat, W, Hm in (("gate_up", gu, Hin), ("down", dn, Hdn)):
                        Hj = Hm[jj]
                        S, Lv = header_parts(W, "expert", 2 ** nb, HEX_BLOCK, Hj.diagonal(dim1=-2, dim2=-1), False)
                        Q = gptq_table(W, Hj, S, Lv)
                        es = expert_scale(W)
                        idx = table_indices(Q, S, Lv)
                        codes = block_codes(S, es, HEX_BLOCK)
                        back[mat] = codebook_dequant(pack_nibbles(idx), W.shape[-1], codes, es, Lv, HEX_BLOCK)
                        n_, o_, i_ = idx.shape
                        words = mlx_pack(idx.long().reshape(n_, o_, i_ // HEX_BLOCK, HEX_BLOCK), nb)
                        words = words.reshape(n_, o_, -1)
                        parts = {"idx": words, "block_scale": codes.view(torch.uint8), "expert_scale": es.float(),
                                 "levels": Lv.float()}
                        rows = {"gate_proj": slice(0, I), "up_proj": slice(I, 2 * I)} if mat == "gate_up" \
                            else {"down_proj": slice(None)}
                        for proj, sl in rows.items():
                            for k_, t in parts.items():
                                t = t[:, sl] if t.dim() == 3 else t
                                out[f"{prefix}.{name}.{proj}.{k_}"] = t.cpu().clone().contiguous()
                                nbytes["experts"] += t.numel() * t.element_size()
                        del S, Q, idx, codes, words
                    m.put(blk.layer, js, back["gate_up"], back["down"])
                    del gu, dn, back
                    continue
                qg, sg = gptq_batched(gu, Hin[js], gptq_fmt(nb, group), return_scales=True)
                qd, sd = gptq_batched(dn, Hdn[js], gptq_fmt(nb, group), return_scales=True)
                ug, sgm, bgm = mlx_from_symmetric(int_codes_bits(qg, sg, group, nb), sg, nb, dtype)
                ud, sdm, bdm = mlx_from_symmetric(int_codes_bits(qd, sd, group, nb), sd, nb, dtype)
                m.put(blk.layer, js, mlx_dequant(ug, sgm, bgm, group), mlx_dequant(ud, sdm, bdm, group))
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

    expert_mods = {n for per in m.names.values() for e in per for n in e.values()}
    # small gates (Qwen's shared-expert gate, linear attention's a / b projections) stay bf16
    lin = [n for n, mod in m.model.named_modules() if isinstance(mod, torch.nn.Linear) and n not in expert_mods
           and not n.endswith(("shared_expert_gate", "linear_attn.in_proj_a", "linear_attn.in_proj_b"))]
    head_ids = None
    if p["mlx_head_vocab"] and "lm_head" in lin and p["mlx_head_vocab"] < m.model.lm_head.out_features:
        from reaper_pipeline.load import load_tokenizer
        head_ids = head_vocab(rj, load_tokenizer(src, rj.config.trust_remote_code), m.model.lm_head.out_features,
                              p["mlx_head_vocab"])
        head_ids += [0] * (-len(head_ids) % 32)   # MLX quantized matmuls like a multiple of 32 rows
    Hs = {}
    if p["mlx_attn_gptq"]:
        emit(STAGE, pct=73, msg="Hessians for attention, shared experts and the LM head")
        Hs = linear_hessians(m.model, lin, m.batches)
    emit(STAGE, pct=75, msg="attention, shared experts, LM head, embedding")
    with torch.no_grad():
        for name in lin:
            mod = m.model.get_submodule(name)
            leaf = name.rpartition(".")[2]
            if name == "lm_head":
                nb = p["mlx_head_bits"]
            elif leaf in ("q_proj", "o_proj") and p["mlx_qo_bits"]:
                nb = p["mlx_qo_bits"]
            else:
                nb = p["mlx_attn_bits"]
            W = mod.weight.float()
            if name == "lm_head" and head_ids is not None:
                W = W[torch.tensor(head_ids, device=W.device)]
            if name in Hs:
                GRID.setdefault(gptq_fmt(nb, group), (nb, group))
                H = Hs.pop(name).to(W.device)
                Q, S = gptq_batched(W[None], H[None], gptq_fmt(nb, group), return_scales=True)
                u, s_, b = mlx_from_symmetric(int_codes_bits(Q, S, group, nb), S, nb, dtype)
                u, s_, b = u[0], s_[0], b[0]
                del Q, S, H
            else:
                u, s_, b = mlx_rtn(W, nb, group, dtype)
            deq = mlx_dequant(u, s_, b, group).to(mod.weight.dtype)
            if name == "lm_head" and head_ids is not None:
                full = torch.zeros_like(mod.weight)
                full[torch.tensor(head_ids, device=full.device)] = deq
                mod.weight.copy_(full)
            else:
                mod.weight.copy_(deq)
            put_q(lay.to_mlx(name), u, s_, b, nb, "rest")
        del Hs
        for name, mod in m.model.named_modules():
            if isinstance(mod, torch.nn.Embedding):
                u, s_, b = mlx_rtn(mod.weight, p["mlx_embed_bits"], group, dtype)
                mod.weight.copy_(mlx_dequant(u, s_, b, group).to(mod.weight.dtype))
                put_q(lay.to_mlx(name), u, s_, b, p["mlx_embed_bits"], "rest")
        if head_ids is not None:
            out["lm_head_ids"] = torch.tensor(head_ids, dtype=torch.int32)
        for name, prm in m.model.named_parameters():  # routers (done above), norms, sinks: unquantized
            mod_name = name.rsplit(".", 1)[0]
            if lay.to_mlx(name) in out or lay.to_mlx(mod_name) in quant or mod_name in expert_mods:
                continue
            t = prm.detach().to(torch.float32 if name.endswith(("softmax_offset", "A_log")) else dtype).cpu()
            t = lay.tensor_to_mlx(name, t)
            out[lay.to_mlx(name)] = t
            nbytes["rest"] += t.numel() * t.element_size()

    emit(STAGE, pct=80, msg="MLX model vs teacher on held-out answers")
    r = m.compare(lambda i, n: emit(STAGE, pct=80 + 6 * i / n, msg=f"held-out {i}/{n} vs teacher, full vocab"))
    r_full = r
    if head_ids is not None:   # the trim scored apart: a trimmed teacher top-20 token alone inflates KL
        mask_head(m.model, head_ids)
        r = m.compare(lambda i, n: emit(STAGE, pct=86 + 6 * i / n, msg=f"held-out {i}/{n} vs teacher, trimmed vocab"))
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
    for f in lay.files:
        shutil.copy2(MLX_DIR / f, tmp / f)
    if hexcfg:
        shutil.copy2(MLX_DIR / "hex_check.py", tmp / "hex_check.py")
    cfg.pop("quantization_config", None)
    tc = cfg.get("text_config")
    if isinstance(tc, dict):   # Qwen3.5 / 3.6: what mlx-lm's text model reads
        tc.pop("quantization_config", None)
        types_ = tc.get("layer_types") or []
        full = [i for i, t in enumerate(types_) if t == "full_attention"]
        if full:
            iv = full[0] + 1
            if [i for i in range(len(types_)) if (i + 1) % iv == 0] != full:
                raise RuntimeError("mlx-lm's Qwen3.5 needs full attention every n-th layer; this model mixes them otherwise")
            tc["full_attention_interval"] = iv
        tc.setdefault("norm_topk_prob", True)   # transformers' Qwen3.5 router always renormalizes the top k
    if head_ids is not None:
        cfg["head_vocab"] = len(head_ids)
    if hexcfg:
        cfg["hex"] = hexcfg
    cfg.update({"model_file": lay.model_file, "expert_tiers": tiers, "blackhole_source": str(src),
                "quantization": {"group_size": group, "bits": bits, **quant}})
    (tmp / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    shutil.rmtree(dst, ignore_errors=True)
    tmp.rename(dst)
    bh.link(dst)

    avg = (bits * (1 - hot_share) + hot_bits * hot_share)
    if p["mlx_hex"]:
        fmt = (f"MLX, HEX experts: {2 ** bits}-level per-expert tables" +
               (f", top {hot_share:.0%} by REAP {2 ** hot_bits}-level" if hot_share else "") + f", block {HEX_BLOCK}")
        bpw_ = avg + 8 / HEX_BLOCK
    else:
        fmt = (f"MLX affine, experts GPTQ {bits}-bit" + (f" (top {hot_share:.0%} by REAP {hot_bits}-bit)"
                                                         if hot_share else "") + f" g{group}")
        bpw_ = avg + 32 / group
    res = {"format": fmt, "bpw": round(bpw_, 3), "expert_params": expert_params,
           "expert_gb": round(gib(nbytes["experts"]), 2), "size_gb": round(gib(total), 2), "out": str(dst),
           "healed": src != rj.out_dir, "attn": ("GPTQ" if p["mlx_attn_gptq"] else "RTN") +
           f" {p['mlx_attn_bits']}-bit" + (f", q/o {p['mlx_qo_bits']}-bit" if p["mlx_qo_bits"] else ""),
           "head_vocab": len(head_ids) if head_ids is not None else None}
    if r:
        if head_ids is not None:
            res.update(kl_full_vocab=round(r_full["kl"], 4), top1_full_vocab=round(r_full["top1_agreement"], 4))
        res.update(top1=round(r["top1_agreement"], 4), kl=round(r["kl"], 4), ppl_teacher=round(r["ppl_teacher"], 3),
                   ppl_student=round(r["ppl_pruned"], 3))
    rj.write("work/mlx.json", res)
    rj.mark_done(STAGE, res)
    msg = f"MLX {res['size_gb']:.1f} GB (experts {res['expert_gb']:.1f} GB)"
    if r:
        msg += f" · KL {res['kl']:.3f} · top-1 {res['top1']:.0%}"
        if "kl_full_vocab" in res:
            msg += f" (full vocab KL {res['kl_full_vocab']:.3f})"
    emit(STAGE, "done", 100, msg, **{k: res[k] for k in ("kl", "top1", "size_gb", "expert_gb") if k in res})


if __name__ == "__main__":
    stage_main(run, STAGE)
