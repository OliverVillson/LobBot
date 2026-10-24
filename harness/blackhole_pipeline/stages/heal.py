"""Stage heal: distillation of the REAP-pruned model from the unpruned teacher.

1. The teacher (the original model, bf16 on the same GPU) writes answers to `heal_prompts`
   fresh prompts from the job's calibration set (seeded apart from the calibration prompts).
2. The pruned student learns the teacher's full next-token distribution on those sequences
   (forward KL over the answer tokens, teacher logits computed on the fly). Trained: attention,
   q/k norms, sinks, routers, shared experts, dense MLPs and norms (fp32 master copies, AdamW,
   cosine). Frozen: the routed experts (the quant stages GPTQ them afterwards) and the
   embedding / LM head. Decoder layers are checkpointed.
3. KL / top-1 on the held-out teacher answers every `heal_eval_every` steps; training stops when
   it has not improved for three checks and the weights go back to the best point. The healed
   model is written to <pruned>-healed only if that beats the start (the quant stages then use it).
Writes work/heal.json and work/heal_data.jsonl.
"""

from __future__ import annotations

import math
import random
import shutil
import time

from blackhole_pipeline.common import emit, stage_main

STAGE = "heal"


def trainable(name: str) -> bool:
    if ".experts." in name or "embed_tokens" in name or name.startswith("lm_head"):
        return False
    return True


def run(bh, rj) -> None:
    import torch
    import torch.nn.functional as F
    from reaper_pipeline.calib import load_prompts
    from reaper_pipeline.load import free, input_device, load_model, load_tokenizer
    from reaper_pipeline.stages._common import encode, profile, read_jsonl, write_jsonl
    from reaper_pipeline.stages.eval import compare

    p = bh.params
    t_start = time.time()
    tok = load_tokenizer(rj.model_dir, rj.config.trust_remote_code)
    prof = profile(rj)
    data_path = rj.path("work", "heal_data.jsonl")
    rows = read_jsonl(data_path)
    emit(STAGE, pct=1, msg="loading the teacher")
    teacher = load_model(rj.model_dir, prof, rj.config.trust_remote_code).eval()
    for q in teacher.parameters():
        q.requires_grad_(False)
    dev = input_device(teacher)
    if len(rows) < p["heal_prompts"]:
        calib = rj.config.calib or "mix"
        prompts, note = load_prompts(calib, p["heal_prompts"], seed=1234)
        emit(STAGE, pct=3, msg=f"teacher answers for {len(prompts)} {note}")
        enc = [encode(tok, q, None, rj.config.max_seq)[0] for q in prompts]
        ge = teacher.generation_config.eos_token_id
        eos = sorted({t for t in [tok.eos_token_id, *(ge if isinstance(ge, list) else [ge])] if t is not None})
        done = [0]
        answers = [None] * len(enc)

        def on_done(i, r):
            answers[i] = r
            done[0] += 1
            if done[0] % 16 == 0:
                emit(STAGE, pct=3 + 27 * done[0] / len(enc), msg=f"teacher answered {done[0]}/{len(enc)}")
        if hasattr(teacher, "generate_many"):
            teacher.generate_many(enc, p["heal_max_tokens"], 0.7, 0.95, eos, slots=min(len(enc), 128), on_done=on_done)
        else:  # any other model: plain HF generate, 16 prompts at a time
            for b0 in range(0, len(enc), 16):
                grp = enc[b0:b0 + 16]
                T = max(map(len, grp))
                ids = torch.tensor([[tok.pad_token_id or 0] * (T - len(e)) + e for e in grp], device=dev)
                am = torch.tensor([[0] * (T - len(e)) + [1] * len(e) for e in grp], device=dev)
                with torch.no_grad():
                    g = teacher.generate(input_ids=ids, attention_mask=am, max_new_tokens=p["heal_max_tokens"],
                                         do_sample=True, temperature=0.7, top_p=0.95)
                for j, r in enumerate(g[:, T:].tolist()):
                    cut = next((i + 1 for i, t in enumerate(r) if t in eos), len(r))
                    on_done(b0 + j, r[:cut])
        rows = [{"ids": e + a, "start": len(e)} for e, a in zip(enc, answers) if a]
        write_jsonl(data_path, rows)
    free()

    emit(STAGE, pct=31, msg=f"loading the student {rj.out_dir.name}")
    student = load_model(rj.out_dir, prof, rj.config.trust_remote_code)
    refs_path = rj.path("work", "teacher_ref.pt")
    refs = torch.load(refs_path, weights_only=False) if refs_path.exists() else []
    student.eval()
    before = compare(student, refs, lambda i, n: emit(STAGE, pct=31 + 4 * i / n, msg=f"KL before {i}/{n}")) \
        if refs else {}

    params = [(n, q) for n, q in student.named_parameters() if trainable(n)]
    for n, q in student.named_parameters():
        q.requires_grad_(False)
    for _, q in params:
        q.requires_grad_(True)
    master = [q.detach().float().clone().requires_grad_(True) for _, q in params]
    opt = torch.optim.AdamW(master, lr=p["heal_lr"], betas=(0.9, 0.95), weight_decay=0.0)
    n_train = sum(q.numel() for _, q in params)

    m = student.model
    random.Random(0).shuffle(rows)
    steps = max(1, math.ceil(len(rows) * p["heal_epochs"] / p["heal_accum"]))
    limit = p["heal_minutes"] * 60
    T_CHUNK = 512

    def hidden(model, ids, grad: bool):
        x = model.model.embed_tokens(ids)
        pos = torch.arange(ids.shape[1], device=ids.device)[None]
        mask = torch.ones_like(ids, dtype=torch.bool)
        for layer in model.model.layers:
            if grad:
                x = torch.utils.checkpoint.checkpoint(layer, x, pos, mask, use_reentrant=False)
            else:
                x = layer(x, pos, mask)
        return model.model.norm(x)

    for mdl in (teacher, student):  # batched experts (two bmm per layer) instead of the per-expert loop
        if hasattr(mdl, "stack_experts"):
            mdl.stack_experts()
    student.train()
    emit(STAGE, pct=36, msg=f"distilling: {n_train / 1e9:.2f}B trainable, {len(rows)} sequences, {steps} steps")
    t0, step, seen, loss_avg = time.time(), 0, 0, None
    best = {"kl": before.get("kl", float("inf")), "step": 0, "state": None, "r": None}
    history = []
    k = 0
    stopped = False
    while step < steps and not stopped:
        for r in rows:
            ids = torch.tensor([r["ids"][:rj.config.max_seq]], device=dev)
            a = r["start"]
            with torch.no_grad():
                ht = hidden(teacher, ids, False)[0, a - 1:-1]
            hs = hidden(student, ids, True)[0, a - 1:-1]
            n_tok = hs.shape[0]
            total = 0.0
            # the loss runs in [chunk, vocab] pieces against a detached copy of the hidden states,
            # then one backward pass through the network (not one per chunk)
            hd = hs.detach().requires_grad_(True)
            for c in range(0, n_tok, T_CHUNK):
                with torch.no_grad():
                    lt = teacher.lm_head(ht[c:c + T_CHUNK]).float().log_softmax(-1)
                ls = student.lm_head(hd[c:c + T_CHUNK]).float().log_softmax(-1)
                kl = F.kl_div(ls, lt, log_target=True, reduction="sum") / max(1, n_tok) / p["heal_accum"]
                kl.backward()
                total += kl.item() * p["heal_accum"]
            hs.backward(hd.grad)
            del hd
            del ht, hs
            loss_avg = total if loss_avg is None else 0.95 * loss_avg + 0.05 * total
            seen += 1
            k += 1
            if k % p["heal_accum"]:
                continue
            ok = all(q.grad is None or torch.isfinite(q.grad).all() for _, q in params)
            if ok:
                for (_, q), mq in zip(params, master):
                    mq.grad = q.grad.float() if q.grad is not None else None
                torch.nn.utils.clip_grad_norm_(master, 1.0)
                lr = p["heal_lr"] * (min(1.0, (step + 1) / max(1, steps // 20)) *
                                     0.5 * (1 + math.cos(math.pi * step / steps)))
                for g in opt.param_groups:
                    g["lr"] = lr
                opt.step()
                with torch.no_grad():
                    for (_, q), mq in zip(params, master):
                        q.copy_(mq.to(q.dtype))
            opt.zero_grad(set_to_none=True)
            for _, q in params:
                q.grad = None
            step += 1
            el = time.time() - t0
            emit(STAGE, pct=36 + 50 * step / steps,
                 msg=f"step {step}/{steps} · KL {loss_avg:.4f} · {seen} seqs · {el / 60:.0f} min"
                     + ("" if ok else " · skipped a non-finite step"))
            if refs and (step % p["heal_eval_every"] == 0 or step >= steps):
                student.eval()
                r_ = compare(student, refs)
                student.train()
                history.append({"step": step, "kl": round(r_["kl"], 4), "top1": round(r_["top1_agreement"], 4)})
                if r_["kl"] < best["kl"]:
                    best.update(kl=r_["kl"], step=step, r=r_,
                                state=[q.detach().to("cpu", copy=True) for _, q in params])
                emit(STAGE, pct=36 + 50 * step / steps,
                     msg=f"step {step}: held-out KL {r_['kl']:.3f} (start {before['kl']:.3f}, best {best['kl']:.3f}"
                         f" at step {best['step']})")
                if step - best["step"] >= 3 * p["heal_eval_every"]:  # no better point for a while
                    stopped = True
                    break
            if step >= steps:
                break
            if limit and el > limit:
                stopped = True
                break
    if best.get("state") is not None:  # back to the best point seen on the held-out answers
        with torch.no_grad():
            for (_, q), saved in zip(params, best["state"]):
                q.copy_(saved.to(q.device))
    elif refs:  # never better than the start: the unhealed weights stand (nothing is written)
        best["r"] = None
    del opt, master
    for _, q in params:
        q.requires_grad_(False)
    student.eval()
    del teacher
    free()
    after = (best["r"] or before) if refs else {}
    better = not refs or best["r"] is not None
    dst = bh.healed_dir(rj)
    shutil.rmtree(dst, ignore_errors=True)
    if better:
        emit(STAGE, pct=93, msg=f"writing {dst.name}")
        student.save_pretrained(dst, safe_serialization=True)
        tok.save_pretrained(dst)
        for f in rj.out_dir.iterdir():
            if f.suffix in (".py", ".jinja") or f.name == "generation_config.json":
                shutil.copy2(f, dst / f.name)
    res = {"trainable_params": n_train, "sequences": len(rows), "steps": step, "minutes": round((time.time() - t0) / 60, 1),
           "kl_before": round(before.get("kl", float("nan")), 4), "kl_after": round(after.get("kl", float("nan")), 4),
           "top1_before": round(before.get("top1_agreement", float("nan")), 4),
           "top1_after": round(after.get("top1_agreement", float("nan")), 4), "kept": better,
           "best_step": best["step"], "history": history,
           "total_minutes": round((time.time() - t_start) / 60, 1)}
    rj.write("work/heal.json", res)
    rj.mark_done(STAGE, res)
    msg = (f"KL {res['kl_before']:.3f} → {res['kl_after']:.3f}, top-1 {res['top1_before']:.0%} → {res['top1_after']:.0%}"
           if refs else f"{step} steps")
    emit(STAGE, "done", 100, msg + ("" if better else " · no gain, kept the unhealed model"),
         kl=res["kl_after"], top1=res["top1_after"])


if __name__ == "__main__":
    stage_main(run, STAGE)
