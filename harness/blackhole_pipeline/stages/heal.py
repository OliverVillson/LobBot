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

Stage qheal is the same distillation after the mlx stage, around the quantized model: routed
experts, embedding and LM head are the MLX folder's own (dequantized, frozen); attention, shared
experts and dense MLPs train in bf16 behind a straight-through round-to-nearest at the MLX bit
width (so the student is exactly the MLX model and the gains survive re-quantization); routers,
norms and sinks train as they are. So the heal repairs the 3-bit error as well as the pruning.
If it beats the start it writes <mlx folder>-healed and points work/mlx.json at it (the bench
then measures it). Writes work/qheal.json.
"""

from __future__ import annotations

import json
import math
import random
import shutil
import time
from pathlib import Path

from blackhole_pipeline.common import emit, stage_main

STAGE = "heal"


def trainable(name: str) -> bool:
    if ".experts." in name or "embed_tokens" in name or name.startswith("lm_head"):
        return False
    return True


class FakeQuant:
    """Parametrization: the weight as MLX round-to-nearest stores it, gradient straight through."""

    def __new__(cls, bits: int, group: int):
        import torch

        from blackhole_pipeline.pack import mlx_dequant, mlx_rtn

        class _FQ(torch.nn.Module):
            def forward(self, w):
                with torch.no_grad():
                    u, s, b = mlx_rtn(w, bits, group, torch.bfloat16)  # the mlx stage's scale dtype
                    q = mlx_dequant(u, s, b, group).to(w.dtype)
                return w + (q - w).detach()
        return _FQ()


def quantize_like(student, mlx_dir: Path) -> dict:
    """Makes the bf16 student the MLX model: frozen tensors (routed experts, routers in the MLX
    expert order, embedding, LM head) from the MLX folder, fake-quant on the other quantized
    Linear layers. Returns {module name: {"bits", "group_size"}} of the fake-quantized ones."""
    import torch
    from torch.nn.utils import parametrize

    from blackhole_pipeline.serve import _mlx_state
    cfg = json.loads((mlx_dir / "config.json").read_text())
    q = cfg.get("quantization") or {}
    dtype = next(student.parameters()).dtype
    st = _mlx_state(mlx_dir, dtype)
    keep = {k: v for k, v in st.items()
            if ".experts." in k or k.endswith("mlp.gate.weight") or "embed_tokens" in k or k.startswith("lm_head")}
    student.load_state_dict(keep, strict=False)
    names = {n for n, _ in student.named_parameters()}
    lost = [k for k in keep if k not in names]
    if lost:
        raise RuntimeError(f"MLX tensors with no place in the model: {lost[:4]}")
    fq = {}
    for name, mod in student.named_modules():
        if (isinstance(mod, torch.nn.Linear) and name in q and isinstance(q[name], dict) and ".experts." not in name
                and name != "lm_head"):
            spec = {"bits": q[name]["bits"], "group_size": q[name]["group_size"]}
            parametrize.register_parametrization(mod, "weight", FakeQuant(spec["bits"], spec["group_size"]))
            fq[name] = spec
    return fq


def write_mlx(student, fq: dict, src: Path, dst: Path) -> None:
    """A copy of the MLX folder src with the student's trained tensors: fake-quantized layers
    re-quantized (same codes the student ran with), routers / norms / sinks as they are."""
    import torch
    from safetensors.torch import load_file, save_file
    from torch.nn.utils import parametrize

    from blackhole_pipeline.pack import mlx_pack, mlx_rtn
    new = {}
    for name, spec in fq.items():
        mod = student.get_submodule(name)
        parametrize.remove_parametrizations(mod, "weight", leave_parametrized=False)
        u, s, b = mlx_rtn(mod.weight.detach(), spec["bits"], spec["group_size"], torch.bfloat16)
        new.update({f"{name}.weight": mlx_pack(u, spec["bits"]).cpu(), f"{name}.scales": s.contiguous().cpu(),
                    f"{name}.biases": b.contiguous().cpu()})
    for n, prm in student.named_parameters():
        if ".experts." in n or "embed_tokens" in n or n.startswith("lm_head") or n.rsplit(".", 1)[0] in fq:
            continue
        new[n] = prm.detach().to(torch.float32 if n.endswith("softmax_offset") else prm.dtype).cpu().contiguous()
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(src, tmp)
    where = json.loads((tmp / "model.safetensors.index.json").read_text())["weight_map"]
    extra = [k for k in new if k not in where]
    if extra:
        raise RuntimeError(f"trained tensors the MLX folder does not have: {extra[:4]}")
    for f in sorted(set(where[k] for k in new)):
        t = load_file(str(tmp / f))
        t.update({k: v.to(t[k].dtype) if v.is_floating_point() else v for k, v in new.items() if where[k] == f})
        save_file(t, str(tmp / f), metadata={"format": "mlx"})
    shutil.rmtree(dst, ignore_errors=True)
    tmp.rename(dst)


def run(bh, rj, quantized: bool = False) -> None:
    import torch
    import torch.nn.functional as F
    from reaper_pipeline.calib import load_prompts
    from reaper_pipeline.load import free, input_device, load_model, load_tokenizer
    from reaper_pipeline.stages._common import encode, profile, read_jsonl, write_jsonl
    from reaper_pipeline.stages.eval import compare

    p = bh.params
    stage = "qheal" if quantized else STAGE
    t_start = time.time()
    tok = load_tokenizer(rj.model_dir, rj.config.trust_remote_code)
    prof = profile(rj)
    data_path = rj.path("work", "heal_data.jsonl")
    calib = p["heal_calib"] or rj.config.calib or "mix"
    meta = {"calib": calib, "max_tokens": p["heal_max_tokens"], "samples": p["heal_samples"]}
    rows = read_jsonl(data_path) if (rj.read("work/heal_data_meta.json") or meta) == meta else []
    emit(stage, pct=1, msg="loading the teacher")
    teacher = load_model(rj.model_dir, prof, rj.config.trust_remote_code).eval()
    for q in teacher.parameters():
        q.requires_grad_(False)
    dev = input_device(teacher)
    ge = teacher.generation_config.eos_token_id
    eos = sorted({t for t in [tok.eos_token_id, *(ge if isinstance(ge, list) else [ge])] if t is not None})

    def teacher_answers(enc, temperature, top_p, top_k, what):
        done = [0]
        answers = [None] * len(enc)

        def on_done(i, r):
            answers[i] = r
            done[0] += 1
            if done[0] % 16 == 0:
                emit(stage, pct=3 + 27 * done[0] / len(enc), msg=f"teacher {what} {done[0]}/{len(enc)}")
        if hasattr(teacher, "generate_many"):
            from blackhole_pipeline.bench import fit_batch
            slots = fit_batch(teacher, min(len(enc), 256), max(map(len, enc)) + p["heal_max_tokens"])
            teacher.generate_many(enc, p["heal_max_tokens"], temperature, top_p, eos, slots=slots, on_done=on_done,
                                  top_k=top_k)
        else:  # any other model: plain HF generate, 16 prompts at a time
            for b0 in range(0, len(enc), 16):
                grp = enc[b0:b0 + 16]
                T = max(map(len, grp))
                ids = torch.tensor([[tok.pad_token_id or 0] * (T - len(e)) + e for e in grp], device=dev)
                am = torch.tensor([[0] * (T - len(e)) + [1] * len(e) for e in grp], device=dev)
                with torch.no_grad():
                    g = teacher.generate(input_ids=ids, attention_mask=am, max_new_tokens=p["heal_max_tokens"],
                                         do_sample=True, temperature=temperature, top_p=top_p, top_k=top_k or None)
                for j, r in enumerate(g[:, T:].tolist()):
                    cut = next((i + 1 for i, t in enumerate(r) if t in eos), len(r))
                    on_done(b0 + j, r[:cut])
        return [a or [] for a in answers]

    samples = p["heal_samples"]
    if not rows or (samples <= 1 and len(rows) < p["heal_prompts"]):
        if samples > 1:  # verified shortest-trace distillation (traces.py)
            from blackhole_pipeline import traces
            from blackhole_pipeline.bench import extract_code, prompt, run_problem
            if calib != "contest":
                raise RuntimeError("heal_samples > 1 needs heal_calib = contest (problems with tests)")
            emit(stage, pct=2, msg=f"loading {p['heal_prompts']} code_contests problems")
            probs = traces.contest_problems(p["heal_prompts"])
            gc = teacher.generation_config
            sampled = getattr(gc, "do_sample", False)
            temp = (gc.temperature or 1.0) if sampled else 1.0
            top_p = (gc.top_p or 0.95) if sampled else 0.95
            top_k = int(gc.to_diff_dict().get("top_k") or 20)

            def gen(jobs):
                texts = [tok.apply_chat_template(prompt(pr), tokenize=False, add_generation_prompt=True) for pr, _ in jobs]
                enc = [tok(t, add_special_tokens=False)["input_ids"][:rj.config.max_seq] for t in texts]
                emit(stage, pct=3, msg=f"teacher solving {len(probs)} problems × {samples}")
                ans = teacher_answers(enc, temp, top_p, top_k, "answers")
                return [(e, a, tok.decode(a, skip_special_tokens=True)) for e, a in zip(enc, ans)]
            rows, trace_stats = traces.shortest_passing(
                probs, samples, gen, lambda pr, text: run_problem(pr, extract_code(text))["passed"])
            rj.write("work/heal_traces.json", trace_stats)
            emit(stage, pct=30, msg=f"teacher solved {trace_stats['solved']}/{len(probs)} problems; kept answers are "
                                    f"{trace_stats['shorter_than_passing']:.0%} shorter than its average passing one")
        else:
            prompts, note = load_prompts(calib, p["heal_prompts"], seed=1234)
            emit(stage, pct=3, msg=f"teacher answers for {len(prompts)} {note}")
            enc = [encode(tok, q, None, rj.config.max_seq)[0] for q in prompts]
            answers = teacher_answers(enc, 0.7, 0.95, 0, "answered")
            rows = [{"ids": e + a, "start": len(e)} for e, a in zip(enc, answers) if a]
        if not rows:
            raise RuntimeError("the teacher produced no usable training sequences")
        write_jsonl(data_path, rows)
        rj.write("work/heal_data_meta.json", meta)
    free()

    fq, mlx_info = {}, {}
    if quantized:
        mlx_info = rj.read("work/mlx.json") or {}
        mlx_dir = Path(mlx_info.get("unhealed_out") or mlx_info.get("out", ""))
        if not (mlx_dir / "config.json").exists():
            raise RuntimeError("no MLX model to heal: run the mlx stage first")
        src = bh.healed_dir(rj) if bh.healed_dir(rj).exists() else rj.out_dir
        emit(stage, pct=31, msg=f"loading {src.name} as {mlx_dir.name}")
        student = load_model(src, prof, rj.config.trust_remote_code)
        fq = quantize_like(student, mlx_dir)
    else:
        emit(stage, pct=31, msg=f"loading the student {rj.out_dir.name}")
        student = load_model(rj.out_dir, prof, rj.config.trust_remote_code)
    refs_path = rj.path("work", "teacher_ref.pt")
    refs = torch.load(refs_path, weights_only=False) if refs_path.exists() else []
    student.eval()
    before = compare(student, refs, lambda i, n: emit(stage, pct=31 + 4 * i / n, msg=f"KL before {i}/{n}")) \
        if refs else {}

    for mdl in (teacher, student):  # batched experts (two bmm per layer) instead of the per-expert loop
        if hasattr(mdl, "stack_experts"):
            mdl.stack_experts()
    rank = 0 if quantized else p["heal_expert_rank"]
    if rank and hasattr(student, "add_expert_lora"):  # the routed experts learn too, as low-rank updates
        student.add_expert_lora(rank)
    params = [(n, q) for n, q in student.named_parameters() if trainable(n)]
    for n, q in student.named_parameters():
        q.requires_grad_(False)
    for _, q in params:
        q.requires_grad_(True)
    master = [q.detach().float().clone().requires_grad_(True) for _, q in params]
    lora = [i for i, (n, _) in enumerate(params) if ".lora_" in n]
    groups = [{"params": [mq for i, mq in enumerate(master) if i not in set(lora)], "base": p["heal_lr"]}]
    if lora:
        groups.append({"params": [master[i] for i in lora], "base": p["heal_lora_lr"]})
    opt = torch.optim.AdamW(groups, lr=p["heal_lr"], betas=(0.9, 0.95), weight_decay=0.0)
    seq = p["heal_seq"] or rj.config.max_seq
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

    student.train()
    emit(stage, pct=36, msg=f"distilling: {n_train / 1e9:.2f}B trainable, {len(rows)} sequences, {steps} steps")
    t0, step, seen, loss_avg = time.time(), 0, 0, None
    best = {"kl": before.get("kl", float("inf")), "step": 0, "state": None, "r": None}
    history = []
    k = 0
    stopped = False
    while step < steps and not stopped:
        for r in rows:
            ids = torch.tensor([r["ids"][:seq]], device=dev)
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
                sched = min(1.0, (step + 1) / max(1, steps // 20)) * 0.5 * (1 + math.cos(math.pi * step / steps))
                for g in opt.param_groups:
                    g["lr"] = g["base"] * sched
                opt.step()
                with torch.no_grad():
                    for (_, q), mq in zip(params, master):
                        q.copy_(mq.to(q.dtype))
            opt.zero_grad(set_to_none=True)
            for _, q in params:
                q.grad = None
            step += 1
            el = time.time() - t0
            emit(stage, pct=36 + 50 * step / steps,
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
                emit(stage, pct=36 + 50 * step / steps,
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
    if quantized:
        dst = mlx_dir.with_name(mlx_dir.name + "-healed")
        shutil.rmtree(dst, ignore_errors=True)
        if better:
            emit(stage, pct=93, msg=f"writing {dst.name}")
            write_mlx(student, fq, mlx_dir, dst)
            bh.link(dst)
            info = {**mlx_info, "unhealed_out": str(mlx_dir), "out": str(dst), "healed_after_quant": True,
                    "unhealed_kl": mlx_info.get("unhealed_kl", mlx_info.get("kl")),
                    "unhealed_top1": mlx_info.get("unhealed_top1", mlx_info.get("top1"))}
            if refs:
                info.update(kl=round(after["kl"], 4), top1=round(after["top1_agreement"], 4))
            rj.write("work/mlx.json", info)
        elif mlx_info.get("unhealed_out"):  # an earlier qheal's folder is gone: the unhealed one stands
            rj.write("work/mlx.json", {**mlx_info, "out": mlx_info["unhealed_out"], "healed_after_quant": False,
                                       "kl": mlx_info.get("unhealed_kl"), "top1": mlx_info.get("unhealed_top1")})
    else:
        dst = bh.healed_dir(rj)
        shutil.rmtree(dst, ignore_errors=True)
        if better:
            emit(stage, pct=93, msg=f"writing {dst.name}")
            if hasattr(student, "merge_expert_lora"):
                student.merge_expert_lora()
            student.save_pretrained(dst, safe_serialization=True)
            tok.save_pretrained(dst)
            for f in rj.out_dir.iterdir():
                if f.suffix in (".py", ".jinja") or f.name == "generation_config.json":
                    shutil.copy2(f, dst / f.name)
    res = {"trainable_params": n_train, "sequences": len(rows), "steps": step, "minutes": round((time.time() - t0) / 60, 1),
           "kl_before": round(before.get("kl", float("nan")), 4), "kl_after": round(after.get("kl", float("nan")), 4),
           "top1_before": round(before.get("top1_agreement", float("nan")), 4),
           "top1_after": round(after.get("top1_agreement", float("nan")), 4), "kept": better,
           "best_step": best["step"], "history": history, "traces": rj.read("work/heal_traces.json") or None,
           "total_minutes": round((time.time() - t_start) / 60, 1)}
    rj.write(f"work/{stage}.json", res)
    rj.mark_done(stage, res)
    msg = (f"KL {res['kl_before']:.3f} → {res['kl_after']:.3f}, top-1 {res['top1_before']:.0%} → {res['top1_after']:.0%}"
           if refs else f"{step} steps")
    emit(stage, "done", 100, msg + ("" if better else " · no gain, kept the unhealed model"),
         kl=res["kl_after"], top1=res["top1_after"])


def run_quantized(bh, rj) -> None:
    run(bh, rj, quantized=True)


if __name__ == "__main__":
    stage_main(run, STAGE)
