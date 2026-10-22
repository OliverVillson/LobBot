"""Stages score and prune (one process, one model load).

score  forward the calibration set through the unpruned model with REAP hooks
       on every router (moe.Saliency), then record the teacher's own top
       tokens on the held-out answers for the eval stage.
prune  keep the n highest-saliency experts in every MoE layer (the same n
       everywhere, so the config stays uniform) and rewrite the checkpoint
       shard by shard in its original format (ckpt.py).
Writes work/saliency.json, work/reap.json, work/teacher_ref.pt and out/<name>/.
"""

from __future__ import annotations

import time

from reaper_pipeline import ckpt, moe
from reaper_pipeline.events import emit, log, size
from reaper_pipeline.job import Job
from reaper_pipeline.load import free, input_device, load_model, load_tokenizer
from reaper_pipeline.stages._common import batches, encode, pad_batch, profile, read_jsonl

TOPK_REF = 20


def teacher_reference(model, tok, rows, max_seq: int) -> list[dict]:
    """Per held-out answer token: the teacher's NLL and top-20 log-probs."""
    import torch
    dev = input_device(model)
    out = []
    for r in rows:
        ids, start = encode(tok, r["prompt"], r["answer"], max_seq)
        if len(ids) - start < 2:
            continue
        with torch.no_grad():
            logits = model(input_ids=torch.tensor([ids], device=dev), use_cache=False).logits[0, start - 1:-1]
        lp = logits.float().log_softmax(-1)
        tgt = torch.tensor(ids[start:], device=lp.device)
        v, i = lp.topk(TOPK_REF, dim=-1)
        out.append({"ids": ids, "start": start, "nll": (-lp.gather(1, tgt[:, None])[:, 0]).cpu(),
                    "top_v": v.cpu(), "top_i": i.cpu()})
    return out


def score(job: Job):
    import torch
    cfg = job.config
    prof = profile(job)
    emit("score", pct=1, msg="loading the model")
    tok = load_tokenizer(job.model_dir, cfg.trust_remote_code)
    model = load_model(job.model_dir, prof, cfg.trust_remote_code)
    blocks = moe.find_blocks(model)
    if not blocks:
        raise RuntimeError("no MoE blocks found in the loaded model")
    E = blocks[0].num_experts
    k = moe.top_k(blocks[0], model.config)
    groups = moe.n_groups(model.config)
    tg = getattr(moe.text_config(model.config), "topk_group", None)
    n_keep = moe.experts_to_keep(E, cfg.prune, k, groups, tg)
    log(f"REAP: {len(blocks)} MoE layers x {E} experts (top-{k}"
        + (f", {groups} groups" if groups > 1 else "") + f"), keeping {n_keep}")

    rows = read_jsonl(job.path("data", "calib.jsonl"))
    enc = [encode(tok, r["prompt"], r.get("answer"), cfg.max_seq)[0] for r in rows]
    bs = max(1, int((prof.get("reap") or {}).get("batch_seqs", 1)))
    groups_idx = list(batches(enc, bs, bs * cfg.max_seq))
    col = moe.Saliency(model, blocks)
    dev = input_device(model)
    t0, done, ntok = time.time(), 0, 0
    with torch.no_grad():
        for gi, idx in enumerate(groups_idx):
            ids, mask = pad_batch([enc[i] for i in idx], tok.pad_token_id, dev)
            col.mask = mask.reshape(-1).bool()
            try:  # only the router hooks matter here; skip the [batch, seq, vocab] logits
                model(input_ids=ids, attention_mask=mask, use_cache=False, logits_to_keep=1)
            except TypeError:
                model(input_ids=ids, attention_mask=mask, use_cache=False)
            done += len(idx)
            ntok += int(mask.sum())
            if gi % max(1, len(groups_idx) // 100) == 0 or done == len(enc):
                rate = ntok / max(time.time() - t0, 1e-6)
                emit("score", pct=3 + 85 * done / len(enc), msg=f"calibrating {done}/{len(enc)} · {rate:,.0f} tok/s")
    col.mask = None
    col.remove()
    log(f"REAP: scored {ntok} tokens in {time.time() - t0:.0f}s; routing read from "
        f"{sorted(set(col.routed_from.values()))}")

    held = read_jsonl(job.path("data", "heldout.jsonl"))
    if held:
        emit("score", pct=90, msg=f"teacher reference on {len(held)} held-out answers")
        torch.save(teacher_reference(model, tok, held, cfg.max_seq), job.path("work", "teacher_ref.pt"))

    res = col.results()
    layers = []
    for b in blocks:
        s, f = res[b.layer]["saliency"], res[b.layer]["freq"]
        keep = moe.choose_keep(s, f, n_keep, groups)
        layers.append({"layer": b.layer, "module": b.name, "kept": keep, "saliency": s, "freq": f,
                       "unused_experts": sum(1 for x in f if x == 0)})
    num_layers = getattr(moe.text_config(model.config), "num_hidden_layers", max(b.layer for b in blocks) + 1)
    job.write("work/saliency.json", {"model": cfg.model, "purpose": cfg.purpose, "experts": E, "kept": n_keep,
                                     "tokens": ntok, "layers": layers})
    job.write("work/reap.json", {"experts_orig": E, "experts_kept": n_keep, "top_k": k, "groups": groups,
                                 "moe_layers": len(blocks), "num_layers": num_layers, "tokens": ntok,
                                 "calib_samples": len(enc)})
    job.mark_done("score")
    unused = sum(l["unused_experts"] for l in layers) / len(layers)
    emit("score", "done", 100, f"{ntok:,} tokens · {unused:.1f} experts/layer never routed")
    del model, col, blocks
    free()
    return {l["layer"]: l["kept"] for l in layers}, E, n_keep, num_layers


def prune(job: Job, keep: dict, E: int, n_keep: int, num_layers: int) -> None:
    out = job.out_dir
    emit("prune", pct=2, msg=f"{E} → {n_keep} experts per layer, writing {out.name}")
    info = ckpt.prune_checkpoint(job.model_dir, out, keep, E, num_layers,
                                 on_progress=lambda i, n, f: emit("prune", pct=2 + 96 * i / n, msg=f"shard {i}/{n}"))
    for n in info["notes"]:
        log(f"prune: {n}")
    reap = job.read("work/reap.json") or {}
    reap.update(bytes_before=info["bytes_before"], bytes_after=info["bytes_after"], out=str(out))
    job.write("work/reap.json", reap)
    job.mark_done("prune", {"out": str(out)})
    emit("prune", "done", 100, f"{E} → {n_keep} experts/layer · {size(info['bytes_before'])} → "
                               f"{size(info['bytes_after'])} · {out}")


def run(job: Job) -> None:
    if job.is_done("score") and (s := job.read("work/saliency.json")):
        keep = {l["layer"]: l["kept"] for l in s["layers"]}
        r = job.read("work/reap.json")
        emit("score", "skipped", 100, "cached")
        prune(job, keep, s["experts"], s["kept"], r["num_layers"])
    else:
        prune(job, *score(job))


def main() -> None:
    import argparse
    import sys
    import traceback
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    job = Job(ap.parse_args().job)
    try:
        run(job)
    except Exception as e:
        traceback.print_exc()
        emit("prune" if job.is_done("score") else "score", "error", msg=f"{type(e).__name__}: {e}"[:400])
        sys.exit(1)


if __name__ == "__main__":
    main()
