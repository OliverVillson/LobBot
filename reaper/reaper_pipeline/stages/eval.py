"""Stage eval: the pruned model against the teacher on held-out answers.

The teacher's top tokens on each held-out answer were recorded by the score
stage, so only the pruned model is loaded here (the two never need to share
the GPU). On every answer token:
  top1    the pruned model's first choice equals the teacher's
  kl      KL(teacher || pruned) over the teacher's top-20 tokens (both renormalised there)
  nll     the pruned model's loss on the teacher's actual answer (vs the teacher's own)
plus a few greedy samples side by side with the teacher's answers.
Writes work/eval.json."""

from __future__ import annotations

import math

from reaper_pipeline.events import emit
from reaper_pipeline.job import Job
from reaper_pipeline.load import free, input_device, load_model, load_tokenizer
from reaper_pipeline.stages._common import profile, read_jsonl, stage_main

STAGE = "eval"
N_SAMPLES = 3


def compare(model, refs: list[dict], on_progress=None) -> dict:
    import torch
    dev = input_device(model)
    agree = kl = nll_s = nll_t = 0.0
    n = 0
    for i, r in enumerate(refs):
        ids, start = r["ids"], r["start"]
        with torch.no_grad():
            logits = model(input_ids=torch.tensor([ids], device=dev), use_cache=False).logits[0, start - 1:-1]
        lp = logits.float().log_softmax(-1).cpu()
        tgt = torch.tensor(ids[start:])
        tv, ti = r["top_v"], r["top_i"]
        pt = tv.exp()
        pt = pt / pt.sum(-1, keepdim=True)
        ls = lp.gather(1, ti)
        ls = ls - ls.logsumexp(-1, keepdim=True)  # both renormalised over the teacher's top 20
        kl += (pt * (pt.log() - ls)).sum().item()
        agree += (lp.argmax(-1) == ti[:, 0]).sum().item()
        nll_s += -lp.gather(1, tgt[:, None]).sum().item()
        nll_t += r["nll"].sum().item()
        n += len(tgt)
        if on_progress:
            on_progress(i + 1, len(refs))
    n = max(n, 1)
    return {"tokens": n, "top1_agreement": agree / n, "kl": kl / n,
            "ppl_teacher": math.exp(nll_t / n), "ppl_pruned": math.exp(nll_s / n)}


def samples(model, tok, rows: list[dict], max_new: int = 160) -> list[dict]:
    import torch

    from reaper_pipeline.stages._common import _render
    out = []
    for r in rows[:N_SAMPLES]:
        text = _render(tok, [{"role": "user", "content": r["prompt"]}], True)
        enc = tok(text, return_tensors="pt", add_special_tokens=False).to(input_device(model))
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        out.append({"prompt": r["prompt"], "teacher": r.get("answer", "")[:2000],
                    "pruned": tok.decode(g[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)})
    return out


def run(job: Job) -> None:
    import torch
    cfg = job.config
    prof = profile(job)
    out = job.out_dir
    ref_path = job.path("work", "teacher_ref.pt")
    emit(STAGE, pct=2, msg=f"loading {out.name}")
    tok = load_tokenizer(out, cfg.trust_remote_code)
    model = load_model(out, prof, cfg.trust_remote_code)  # also proves the pruned checkpoint loads
    res: dict = {"loads": True}
    if ref_path.exists():
        refs = torch.load(ref_path, weights_only=False)
        res.update(compare(model, refs, lambda i, n: emit(STAGE, pct=5 + 80 * i / n, msg=f"held-out {i}/{n}")))
    held = read_jsonl(job.path("data", "heldout.jsonl"))
    if held:
        emit(STAGE, pct=88, msg="sample answers")
        res["samples"] = samples(model, tok, held)
    job.write("work/eval.json", res)
    del model
    free()
    job.mark_done(STAGE, {k: v for k, v in res.items() if k != "samples"})
    if "top1_agreement" in res:
        msg = (f"top-1 agreement {res['top1_agreement']:.0%} · ppl {res['ppl_teacher']:.2f} → "
               f"{res['ppl_pruned']:.2f} · KL {res['kl']:.3f}")
    else:
        msg = "pruned model loads (no held-out answers to compare)"
    emit(STAGE, "done", 100, msg, **{k: v for k, v in res.items() if k != "samples"})


if __name__ == "__main__":
    stage_main(run, STAGE)
