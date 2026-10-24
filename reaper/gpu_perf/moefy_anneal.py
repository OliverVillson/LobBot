"""Dense model -> MoE that starts lossless and learns to route while fewer experts stay active.

    python reaper/gpu_perf/moefy_anneal.py                                  # Qwen/Qwen3-8B, 64 experts, 64 -> 16
    python reaper/gpu_perf/moefy_anneal.py --schedule 64,48,32,24,16 --minutes 180

Oliver's idea (2026-10-24): do the MoEfication without forcing a top-k at first, then post-train
the model to route. moefy_bench measured the one-shot cut (Qwen3-8B, 32 of 64 experts: KL 0.53,
oracle router 0.39); this measures what training buys on top.

1. split    per gated MLP, the `--shared` share of neurons with the largest E|h| * ||W_down col||
            becomes an always-on shared expert; the rest is cut into `--experts` equal experts by
            balanced k-means on co-activation (moefy_bench). Neurons are permuted so every expert is
            a contiguous block (what a real MoE checkpoint needs later).
2. router   one linear router per layer, initialised by ridge regression to each expert's output
            size (moefy_bench), kept in fp32.
3. anneal   distillation from the frozen dense original: forward KL over the full vocabulary on
            packed chat / code / math conversations (`--mix`). Every expert is on at the start, so
            the student IS the dense model; top-k then steps down along `--schedule`, each step
            getting an equal share of the time budget. The router learns from two signals:
              straight-through gate: each chosen expert's output is scaled by 1 + z - stop_grad(z),
                so the value stays exactly 1 and z gets the first-order usefulness of that expert
              auxiliary loss: cross-entropy towards the experts with the largest output-size
                estimate at the NEXT step's k (the router learns the cut before it happens)
            Trained: the MLPs (experts + shared) and routers. Frozen: attention, norms,
            embeddings, LM head (`--train-attn` trains attention and norms too).
4. eval     KL / top-1 / ppl vs the dense model on the REAPer job's held-out answers: first the
            untrained MoE at every k of the schedule (the zero-shot floor), then the trained one at
            the end of each step.
Execution is simulated with masks on the full MLP, so this measures quality, not speed; the
active share of FFN weights per token is reported. Results: <job>/work/moefy_anneal_<model>.json;
the trained MLPs, routers and layout: <job>/work/moefy_anneal_<model>/.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from moefy_bench import act_of, balanced_kmeans, collect, expert_norms, find_mlps, fit_router  # noqa: E402
from reaper_pipeline.job import home  # noqa: E402

SOURCES = {
    "code": ("theblackcat102/evol-codealpaca-v1", None, "train",
             lambda r: [("user", r["instruction"]), ("assistant", r["output"])]),
    "chat": ("HuggingFaceH4/ultrachat_200k", None, "train_sft",
             lambda r: [(m["role"], m["content"]) for m in r["messages"]]),
    "math": ("openai/gsm8k", "main", "train", lambda r: [("user", r["question"]), ("assistant", r["answer"])]),
}


# --------------------------------------------------------------------------- MoE with a trainable router


class AddAux(torch.autograd.Function):
    """Returns y unchanged; the backward pass also sends `weight` into the auxiliary loss, so a
    loss computed inside a (checkpointed) layer trains without being returned from it."""

    @staticmethod
    def forward(ctx, y, aux, weight):
        ctx.weight = weight
        ctx.save_for_backward(aux)
        return y.view_as(y)

    @staticmethod
    def backward(ctx, gy):
        (aux,) = ctx.saved_tensors
        return gy, torch.full_like(aux, ctx.weight), None


class RoutedMLPs:
    """Patches every gated MLP into shared expert + E routed experts (contiguous neuron blocks)
    with a linear router. `k` (experts on per token) and `k_aux` (target of the auxiliary loss)
    are set from outside; k >= E means every expert is on."""

    def __init__(self, model, mlps, layout: dict, aux_weight: float = 0.01):
        self.mlps, self.layout, self.aux_weight = mlps, layout, aux_weight
        self.E, self.n_shared, self.size = layout["experts"], layout["shared"], layout["expert_size"]
        self.k = self.k_aux = self.E
        self.orig = {}
        H = mlps[0][1].gate_proj.in_features
        dev = mlps[0][1].gate_proj.weight.device
        self.routers = torch.nn.ModuleDict()
        for name, mlp in mlps:
            key = name.replace(".", "_")
            self.routers[key] = torch.nn.Linear(H, self.E, bias=True, device=dev, dtype=torch.float32)
            self.orig[name] = mlp.forward
            mlp.forward = self._forward(mlp, act_of(mlp, model), self.routers[key])

    def _forward(self, mlp, act, router):
        E, ns, s = self.E, self.n_shared, self.size

        def fwd(x):
            h = act(mlp.gate_proj(x)) * mlp.up_proj(x)
            shp = h.shape
            xf, hf = x.reshape(-1, x.shape[-1]), h.reshape(-1, shp[-1])
            T = xf.shape[0]
            z = router(xf.float())                                         # [T, E]
            if self.k < E:
                on = torch.zeros_like(z).scatter_(1, z.topk(self.k, dim=-1).indices, 1.0)
            else:
                on = torch.ones_like(z)
            g = on * (1 + z - z.detach())                                  # value: the 0/1 mask
            mult = torch.cat([torch.ones(T, ns, device=z.device, dtype=z.dtype), g.repeat_interleave(s, dim=1)], 1)
            y = mlp.down_proj((hf * mult.to(hf.dtype)).reshape(shp))
            if torch.is_grad_enabled() and self.aux_weight > 0 and self.k_aux < E:
                with torch.no_grad():
                    col = mlp.down_proj.weight.float().norm(dim=0)[ns:]     # [E*s]
                    est = (hf[:, ns:].float().abs() * col).view(T, E, s).sum(-1)
                    tgt = torch.zeros_like(est).scatter_(1, est.topk(self.k_aux, dim=-1).indices, 1.0 / self.k_aux)
                za = router(xf.detach().float())
                aux = -(tgt * za.log_softmax(-1)).sum(-1).mean()
                y = AddAux.apply(y, aux, self.aux_weight)
            return y
        return fwd

    def init_routers(self, Ws: dict) -> None:
        with torch.no_grad():
            for name, _ in self.mlps:
                r = self.routers[name.replace(".", "_")]
                W = Ws[name].to(r.weight.device, torch.float32)               # [H+1, E]
                r.weight.copy_(W[:-1].T)
                r.bias.copy_(W[-1])

    def active_share(self, k: int) -> float:
        I = self.n_shared + self.E * self.size
        return (self.n_shared + min(k, self.E) * self.size) / I

    def remove(self):
        for name, mlp in self.mlps:
            mlp.forward = self.orig[name]


def split_layout(I: int, E: int, shared: float) -> dict:
    """Shared-expert size chosen so the routed neurons divide into E equal experts."""
    size = (I - int(round(I * shared))) // E
    return {"experts": E, "expert_size": size, "shared": I - E * size, "neurons": I}


@torch.no_grad()
def moefy(mlps, stats, layout: dict, dev) -> dict:
    """Permutes every MLP's neurons into [shared | expert 0 | expert 1 | ...] in place and
    returns the ridge routers {name: W [H+1, E]} fitted on the permuted layout."""
    E, ns = layout["experts"], layout["shared"]
    Ws = {}
    for name, mlp in mlps:
        x, h = stats[name]
        Wd = mlp.down_proj.weight.data
        score = h.float().abs().mean(0) * Wd.float().norm(dim=0).cpu()      # [I]
        shared_idx = score.topk(ns).indices if ns else torch.zeros(0, dtype=torch.long)
        rest = torch.ones(h.shape[1], dtype=torch.bool)
        rest[shared_idx] = False
        routed_idx = rest.nonzero()[:, 0]
        assign = balanced_kmeans(h[:, routed_idx].T.to(dev).abs(), E)
        order = torch.cat([shared_idx.sort().values, routed_idx[torch.argsort(assign, stable=True)]])
        o = order.to(Wd.device)
        mlp.gate_proj.weight.data = mlp.gate_proj.weight.data[o].contiguous()
        mlp.up_proj.weight.data = mlp.up_proj.weight.data[o].contiguous()
        mlp.down_proj.weight.data = Wd[:, o].contiguous()
        hp = h[:, order]
        new_assign = torch.arange(E).repeat_interleave(layout["expert_size"])
        y = expert_norms(hp[:, ns:], mlp.down_proj.weight.data[:, ns:], new_assign, E)
        Ws[name] = fit_router(x, y)
    return Ws


# --------------------------------------------------------------------------- training data


def packed_stream(tok, mix: dict, seq: int, seed: int, fallback: list[list[int]]):
    """Endless [seq] token lists: chat-formatted conversations from the `mix` sources (name ->
    weight), packed back to back. Offline, the REAPer job's calibration texts are cycled."""
    its, weights = {}, {}
    try:
        from datasets import load_dataset
        for name, w in mix.items():
            repo, cfg, split, conv = SOURCES[name]
            ds = load_dataset(repo, cfg, split=split, streaming=True).shuffle(seed=seed, buffer_size=10_000)
            its[name] = (iter(ds), conv, ds)
            weights[name] = w
        print(f"[anneal] training text: {', '.join(f'{n} {w:.0%}' for n, w in weights.items())}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[anneal] datasets unavailable ({type(e).__name__}: {e}); cycling the calibration texts", flush=True)
        its = {}
    rng = random.Random(seed)
    buf: list[int] = []
    fb = itertools.cycle(fallback)
    while True:
        if its:
            name = rng.choices(list(weights), weights=list(weights.values()))[0]
            it, conv, ds = its[name]
            try:
                r = next(it)
            except StopIteration:
                its[name] = (iter(ds), conv, ds)
                continue
            msgs = [{"role": role, "content": c} for role, c in conv(r) if c]
            if not msgs:
                continue
            ids = tok.apply_chat_template(msgs, tokenize=True)
            ids = ids["input_ids"] if isinstance(ids, dict) else list(ids)
        else:
            ids = list(next(fb))
        buf.extend(ids)
        while len(buf) >= seq:
            yield buf[:seq]
            buf = buf[seq:]


# --------------------------------------------------------------------------- distillation


def kl_step(teacher, student, ids, accum: int, chunk: int = 2048) -> float:
    """Forward KL(teacher || student) over every position, full vocabulary; one backward."""
    with torch.no_grad():
        ht = teacher.model(input_ids=ids, use_cache=False).last_hidden_state.flatten(0, 1)
    hs = student.model(input_ids=ids, use_cache=False).last_hidden_state.flatten(0, 1)
    n = hs.shape[0]
    hd = hs.detach().requires_grad_(True)
    total = 0.0
    for c in range(0, n, chunk):
        with torch.no_grad():
            lt = teacher.lm_head(ht[c:c + chunk]).float().log_softmax(-1)
        ls = student.lm_head(hd[c:c + chunk]).float().log_softmax(-1)
        kl = F.kl_div(ls, lt, log_target=True, reduction="sum") / n / accum
        kl.backward()
        total += kl.item() * accum
    hs.backward(hd.grad)
    return total


def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from reaper_pipeline.stages._common import batches as make_batches
    from reaper_pipeline.stages._common import encode, pad_batch, read_jsonl
    from reaper_pipeline.stages.eval import compare
    from reaper_pipeline.stages.reap import teacher_reference

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--data", default=None, help="REAPer job dir with data/calib.jsonl + heldout.jsonl")
    ap.add_argument("--experts", type=int, default=64)
    ap.add_argument("--shared", type=float, default=0.125, help="share of neurons in the always-on shared expert")
    ap.add_argument("--schedule", default="64,48,32,24,16", help="experts on per token, one training step each")
    ap.add_argument("--minutes", type=float, default=180, help="training time budget, split equally over the schedule")
    ap.add_argument("--mix", default="code:0.5,chat:0.3,math:0.2")
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--micro", type=int, default=4, help="sequences per forward")
    ap.add_argument("--accum", type=int, default=4, help="forwards per optimizer step")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--router-lr", type=float, default=1e-3)
    ap.add_argument("--aux", type=float, default=0.01, help="weight of the router's auxiliary loss")
    ap.add_argument("--train-attn", action="store_true")
    ap.add_argument("--calib", type=int, default=64, help="calibration sequences for the split and router init")
    ap.add_argument("--tokens", type=int, default=8192, help="sampled tokens per layer for the split")
    ap.add_argument("--heldout", type=int, default=32)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    data = Path(a.data) if a.data else next(iter(sorted((home() / "jobs").glob("*/data/calib.jsonl"))), None)
    if data is None:
        sys.exit("no calibration data: pass --data <REAPer job dir>")
    data = data if data.is_dir() else data.parent.parent
    slug = a.model.rstrip("/").split("/")[-1]
    out_path = Path(a.out) if a.out else data / "work" / f"moefy_anneal_{slug}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results: dict = {"runs": {}}

    def save(key, val, show=None):
        results[key] = val
        out_path.write_text(json.dumps(results, indent=1))
        print(f"[anneal] {key}: {json.dumps(val if show is None else show)[:300]}", flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if dev == "cuda" else torch.float32
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    teacher = AutoModelForCausalLM.from_pretrained(a.model, dtype=dtype).to(dev).eval()
    for q in teacher.parameters():
        q.requires_grad_(False)
    student = AutoModelForCausalLM.from_pretrained(a.model, dtype=dtype).to(dev).eval()

    rows = read_jsonl(data / "data" / "calib.jsonl")
    random.Random(0).shuffle(rows)
    enc = [encode(tok, r["prompt"], r.get("answer"), a.seq)[0] for r in rows]
    held = read_jsonl(data / "data" / "heldout.jsonl")[: a.heldout]
    refs = teacher_reference(teacher, tok, held, a.seq)
    calib = enc[: a.calib]
    batches = [pad_batch([calib[i] for i in idx], tok.pad_token_id, dev) for idx in make_batches(calib, 8, 8 * a.seq)]

    mlps = find_mlps(student)
    if not mlps:
        sys.exit("no gated MLPs (gate_proj/up_proj/down_proj) found")
    E = a.experts
    schedule = [min(int(v), E) for v in a.schedule.split(",")]
    layout = split_layout(mlps[0][1].down_proj.in_features, E, a.shared)
    stats = collect(student, mlps, batches, a.tokens)
    Ws = moefy(mlps, stats, layout, dev)
    del stats
    routed = RoutedMLPs(student, mlps, layout, a.aux)
    routed.init_routers(Ws)
    mlp_params = sum(sum(p.numel() for p in m.parameters()) for _, m in mlps)
    total_params = sum(p.numel() for p in student.parameters())
    save("setup", {"model": a.model, **layout, "layers": len(mlps), "schedule": schedule, "minutes": a.minutes,
                   "mix": a.mix, "ffn_share_of_params": round(mlp_params / total_params, 3),
                   "prep_s": round(time.time() - t0, 1)})

    runs = results["runs"]

    def evaluate(name: str, k: int, extra: dict | None = None):
        routed.k = k
        student.eval()
        t1 = time.time()
        with torch.no_grad():
            r = compare(student, refs)
        act = routed.active_share(k)
        runs[name] = {"k": k, "top1": round(r["top1_agreement"], 4), "kl": round(r["kl"], 4),
                      "ppl": round(r["ppl_pruned"], 3), "ffn_active_per_token": round(act, 3),
                      "params_active_per_token": round(1 - (1 - act) * mlp_params / total_params, 3),
                      "eval_s": round(time.time() - t1, 1), **(extra or {})}
        save("runs", runs, show={name: runs[name]})

    for k in sorted(set(schedule) | {E}, reverse=True):     # the untrained MoE; k = E must equal the dense model
        evaluate(f"zero-shot top-{k} of {E}", k)

    # ---------------------------------------------------------------- training
    def trainable(n: str) -> bool:
        if ".mlp." in n:
            return True
        return a.train_attn and ("self_attn" in n or "norm" in n) and "embed" not in n

    params = [(n, q) for n, q in student.named_parameters() if trainable(n)]
    for _, q in student.named_parameters():
        q.requires_grad_(False)
    for _, q in params:
        q.requires_grad_(True)
    master = [q.detach().float().clone().requires_grad_(True) for _, q in params]
    rparams = list(routed.routers.parameters())
    opt = torch.optim.AdamW([{"params": master, "lr": a.lr}, {"params": rparams, "lr": a.router_lr}],
                            betas=(0.9, 0.95), weight_decay=0.0)
    base_lr = [a.lr, a.router_lr]
    student.config.use_cache = False
    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    stream = packed_stream(tok, dict((kv.split(":")[0], float(kv.split(":")[1])) for kv in a.mix.split(",")),
                           a.seq, 0, enc[a.calib:] or enc)
    budget = a.minutes * 60
    t_train, step, tokens, loss_avg = time.time(), 0, 0, None
    eval_s = 0.0
    print(f"[anneal] training {sum(q.numel() for _, q in params) / 1e9:.2f}B params + routers, "
          f"{len(schedule)} steps of {a.minutes / len(schedule):.0f} min", flush=True)
    for si, k in enumerate(schedule):
        routed.k = k
        routed.k_aux = schedule[si + 1] if si + 1 < len(schedule) else k
        student.train()
        stage_end = budget * (si + 1) / len(schedule)
        stage_steps, stage_loss = 0, []
        while time.time() - t_train - eval_s < stage_end:
            for _ in range(a.accum):
                ids = torch.tensor([next(stream) for _ in range(a.micro)], device=dev)
                l = kl_step(teacher, student, ids, a.accum)
                tokens += ids.numel()
                loss_avg = l if loss_avg is None else 0.95 * loss_avg + 0.05 * l
                stage_loss.append(l)
            for (_, q), mq in zip(params, master):
                mq.grad = q.grad.float() if q.grad is not None else None
            ok = all(g is None or torch.isfinite(g).all() for g in [mq.grad for mq in master] +
                     [p.grad for p in rparams])
            if ok:
                torch.nn.utils.clip_grad_norm_(master + rparams, 1.0)
                frac = (time.time() - t_train - eval_s) / budget
                scale = min(1.0, frac / 0.02) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(frac, 1.0))))
                for g, lr in zip(opt.param_groups, base_lr):
                    g["lr"] = lr * scale
                opt.step()
                with torch.no_grad():
                    for (_, q), mq in zip(params, master):
                        q.copy_(mq.to(q.dtype))
            opt.zero_grad(set_to_none=True)
            for _, q in params:
                q.grad = None
            step += 1
            stage_steps += 1
            if step % 10 == 0:
                el = time.time() - t_train
                print(f"[anneal] top-{k} step {step} · train KL {loss_avg:.4f} · {tokens / 1e6:.1f}M tokens · "
                      f"{tokens / max(el - eval_s, 1):.0f} tok/s · {el / 60:.0f} min"
                      + ("" if ok else " · skipped a non-finite step"), flush=True)
        t1 = time.time()
        tail = stage_loss[-max(1, len(stage_loss) // 5):]
        evaluate(f"trained top-{k} of {E}", k, {"steps": stage_steps, "tokens_m": round(tokens / 1e6, 1),
                                                 "train_kl_end": round(sum(tail) / len(tail), 4)})
        eval_s += time.time() - t1

    # ---------------------------------------------------------------- save
    from safetensors.torch import save_file
    dst = out_path.with_suffix("")
    dst.mkdir(parents=True, exist_ok=True)
    state = {n: q.detach().cpu().contiguous() for n, q in student.named_parameters() if ".mlp." in n}
    state.update({f"router.{k}.{n}": t.detach().cpu().contiguous() for k, r in routed.routers.items()
                  for n, t in r.state_dict().items()})
    save_file(state, str(dst / "moe.safetensors"))
    (dst / "layout.json").write_text(json.dumps({"model": a.model, **layout, "schedule": schedule,
                                                 "router_keys": list(routed.routers)}, indent=1))
    save("trained", {"tokens_m": round(tokens / 1e6, 1), "steps": step, "out": str(dst),
                     "minutes": round((time.time() - t_train) / 60, 1)})

    print("\n================ moefy-anneal summary ================")
    print(json.dumps(results["setup"]))
    print(f"{'config':32s} top1    KL      ppl     FFN active  params active")
    for name, r in runs.items():
        print(f"{name:32s} {r['top1']:.3f}  {r['kl']:.4f}  {r['ppl']:6.3f}  {r['ffn_active_per_token']:9.3f}  "
              f"{r['params_active_per_token']:9.3f}")
    print(f"\nall results: {out_path}")


if __name__ == "__main__":
    main()
