"""Dense model -> MoE (MoEfication) -> REAP, measured against the dense model itself.

    python reaper/gpu_perf/moefy_bench.py                      # Qwen/Qwen3-8B, data from the REAP job
    python reaper/gpu_perf/moefy_bench.py --model meta-llama/Llama-3.1-8B-Instruct

What it does, per decoder layer's gated MLP (gate_proj / up_proj / down_proj):
  stats     forward calibration sequences, keep a token sample of the MLP input x and of the
            intermediate activation h = act(gate x) * up x
  cluster   split the I intermediate neurons into E equal experts by balanced k-means on
            their co-activation pattern |h| over the sample (MoEfication, Zhang et al. 2022)
  router    least-squares linear router from x to each expert's output norm ||W_down,e h_e||
            (closed form, no training)
  evaluate  KL / top-1 / ppl against the dense model on held-out answers for:
              MoE, learned router, top-k of E           (k/E of the FFN active per token)
              MoE, oracle router (true top-k by norm)    (ceiling for any router)
              REAP on the MoE: drop the experts with the lowest mean routed output norm, keep
                the rest dense (a smaller dense FFN that runs anywhere) or routed
              baselines at equal kept width: neuron-level activation pruning
                (E|h_i| * ||W_down[:, i]||) and random expert groups
Execution is simulated with masks on the full FFN, so it measures quality, not speed; active
FFN FLOPs per token are reported (k/E). Nothing is trained or healed: these are zero-shot
numbers, the floor a short distillation heal would start from.
Results: <job>/work/moefy_<model>.json and a table at the end.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]

import torch  # noqa: E402

from reaper_pipeline.job import home  # noqa: E402


def find_mlps(model):
    out = []
    for name, mod in model.named_modules():
        if all(isinstance(getattr(mod, p, None), torch.nn.Linear) for p in ("gate_proj", "up_proj", "down_proj")) \
                and not hasattr(mod, "experts"):
            out.append((name, mod))
    return out


def act_of(mlp, model):
    fn = getattr(mlp, "act_fn", None)
    if fn is None:
        from transformers.activations import ACT2FN
        fn = ACT2FN[model.config.get_text_config().hidden_act]
    return fn


# --------------------------------------------------------------------------- statistics


@torch.no_grad()
def collect(model, mlps, batches, per_layer: int, seed: int = 0):
    """Token samples per MLP: x [T, H] and h [T, I] (CPU, bf16), padding excluded."""
    g = torch.Generator().manual_seed(seed)
    xs = {n: [] for n, _ in mlps}
    hs = {n: [] for n, _ in mlps}
    state = {"mask": None}
    hooks = []
    total = sum(int(m.sum()) for _, m in batches)
    keep_p = min(1.0, 1.5 * per_layer / max(total, 1))

    for name, mlp in mlps:
        act = act_of(mlp, model)

        def hook(mod, args, name=name, act=act):
            x = args[0].reshape(-1, args[0].shape[-1])
            m = state["mask"]
            if m is not None:
                x = x[m.to(x.device)]
            sel = torch.rand(x.shape[0], generator=g) < keep_p
            x = x[sel.to(x.device)]
            h = act(mod.gate_proj(x)) * mod.up_proj(x)
            xs[name].append(x.to("cpu", torch.bfloat16))
            hs[name].append(h.to("cpu", torch.bfloat16))
        hooks.append(mlp.register_forward_pre_hook(hook))
    for ids, mask in batches:
        state["mask"] = mask.reshape(-1).bool()
        model(input_ids=ids, attention_mask=mask, use_cache=False, logits_to_keep=1)
    for h in hooks:
        h.remove()
    out = {}
    for name, _ in mlps:
        x, h = torch.cat(xs[name]), torch.cat(hs[name])
        out[name] = (x[:per_layer], h[:per_layer])
    return out


# --------------------------------------------------------------------------- MoEfication


def balanced_kmeans(feat: torch.Tensor, E: int, iters: int = 15, seed: int = 0) -> torch.Tensor:
    """Equal-size clusters of the rows of feat [I, D] (cosine). Returns expert id per row."""
    I = feat.shape[0]
    size = I // E
    f = torch.nn.functional.normalize(feat.float(), dim=1)
    g = torch.Generator(device="cpu").manual_seed(seed)
    cent = f[torch.randperm(I, generator=g)[:E].to(f.device)].clone()
    assign = torch.zeros(I, dtype=torch.long, device=f.device)
    for _ in range(iters):
        sim = f @ cent.T                                   # [I, E]
        # balanced assignment: neurons in order of how strongly they prefer their best cluster
        order = torch.argsort(sim.max(1).values - sim.median(1).values, descending=True)
        cap = [size + (e < I - size * E) for e in range(E)]
        pref = torch.argsort(sim, dim=1, descending=True).cpu().tolist()
        a = [0] * I
        for i in order.tolist():
            for e in pref[i]:
                if cap[e] > 0:
                    a[i] = e
                    cap[e] -= 1
                    break
        assign = torch.tensor(a, device=f.device)
        cent = torch.zeros_like(cent).index_add_(0, assign, f)
        cent = torch.nn.functional.normalize(cent, dim=1)
    return assign.cpu()


def expert_norms(h: torch.Tensor, Wd: torch.Tensor, assign: torch.Tensor, E: int, chunk: int = 512) -> torch.Tensor:
    """||W_down[:, e] h[:, e]|| for every token and expert: [T, E] float32."""
    dev = Wd.device
    groups = [(assign == e).nonzero()[:, 0].to(dev) for e in range(E)]
    Wg = [Wd[:, idx].float() for idx in groups]
    out = []
    for i in range(0, h.shape[0], chunk):
        hc = h[i:i + chunk].to(dev, torch.float32)
        cols = []
        for idx, W in zip(groups, Wg):
            cols.append((hc[:, idx] @ W.T).norm(dim=-1))
        out.append(torch.stack(cols, 1).cpu())
    return torch.cat(out)


def fit_router(x: torch.Tensor, y: torch.Tensor, lam: float = 1e-2) -> torch.Tensor:
    """Ridge regression x [T, H] -> y [T, E], with a bias column. Returns W [H+1, E]."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    X = torch.cat([x.float(), torch.ones(x.shape[0], 1)], 1).to(dev)
    Y = y.float().to(dev)
    A = X.T @ X
    A += lam * A.diagonal().mean() * torch.eye(A.shape[0], device=dev)
    return torch.linalg.solve(A, X.T @ Y).cpu()


# --------------------------------------------------------------------------- simulated execution


class Sim:
    """Patches every MLP's forward. Per layer: neuron -> expert map, optional router, kept
    expert set, top-k (None = all kept experts on), or a neuron keep-mask."""

    def __init__(self, model, mlps):
        self.mlps, self.cfg, self.orig = mlps, {}, {}
        for name, mlp in mlps:
            self.orig[name] = mlp.forward
            act = act_of(mlp, model)

            def fwd(x, name=name, mlp=mlp, act=act):
                c = self.cfg.get(name)
                if c is None:
                    return self.orig[name](x)
                h = act(mlp.gate_proj(x)) * mlp.up_proj(x)
                if "neuron_mask" in c:
                    return mlp.down_proj(h * c["neuron_mask"].to(h.dtype))
                assign, kept = c["assign"], c["kept"]                     # [I], [E] bool
                shp = h.shape
                hf = h.reshape(-1, shp[-1])
                if c.get("k") is None:
                    on = kept.unsqueeze(0).expand(hf.shape[0], -1)
                else:
                    if c["router"] == "oracle":
                        score = c["norm_fn"](hf)
                    else:
                        xf = x.reshape(-1, x.shape[-1]).float()
                        W = c["W"]
                        score = xf @ W[:-1] + W[-1]
                    score = score.masked_fill(~kept.unsqueeze(0), float("-inf"))
                    top = score.topk(c["k"], dim=-1).indices
                    on = torch.zeros_like(score, dtype=torch.bool).scatter_(1, top, True)
                hf = hf * on[:, assign].to(hf.dtype)
                return mlp.down_proj(hf.reshape(shp))
            mlp.forward = fwd

    def set(self, cfg: dict | None):
        self.cfg = cfg or {}

    def remove(self):
        for name, mlp in self.mlps:
            mlp.forward = self.orig[name]


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
    ap.add_argument("--topk", default="32,16,8")
    ap.add_argument("--prunes", default="0.25,0.5")
    ap.add_argument("--calib", type=int, default=64)
    ap.add_argument("--tokens", type=int, default=8192, help="sampled tokens per layer for clustering/router")
    ap.add_argument("--max-seq", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    data = Path(a.data) if a.data else next(iter(sorted((home() / "jobs").glob("*/data/calib.jsonl"))), None)
    if data is None:
        sys.exit("no calibration data: pass --data <REAPer job dir>")
    data = data if data.is_dir() else data.parent.parent
    out_path = Path(a.out) if a.out else data / "work" / f"moefy_{a.model.split('/')[-1]}.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else {}

    def save(key, val, show=None):
        results[key] = val
        out_path.write_text(json.dumps(results, indent=1))
        print(f"[moefy] {key}: {json.dumps(val if show is None else show)[:300]}", flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if dev == "cuda" else torch.float32
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dtype).to(dev).eval()
    load_s = time.time() - t0

    rows = read_jsonl(data / "data" / "calib.jsonl")
    random.Random(0).shuffle(rows)
    enc = [encode(tok, r["prompt"], r.get("answer"), a.max_seq)[0] for r in rows[: a.calib]]
    held = read_jsonl(data / "data" / "heldout.jsonl")
    batches = [pad_batch([enc[i] for i in idx], tok.pad_token_id, dev)
               for idx in make_batches(enc, a.batch, a.batch * a.max_seq)]

    t = time.time()
    refs = teacher_reference(model, tok, held, a.max_seq)
    mlps = find_mlps(model)
    if not mlps:
        sys.exit("no gated MLPs (gate_proj/up_proj/down_proj) found")
    stats = collect(model, mlps, batches, a.tokens)
    E = a.experts
    I = mlps[0][1].down_proj.in_features
    lay = {}
    for name, mlp in mlps:
        x, h = stats[name]
        assign = balanced_kmeans(h.T.to(dev).abs(), E)
        Wd = mlp.down_proj.weight.data
        y = expert_norms(h, Wd, assign, E)
        ntrain = int(0.8 * y.shape[0])
        W = fit_router(x[:ntrain], y[:ntrain])
        # router quality on the held-out 20% of sampled tokens: recall of the true top-k
        pred = torch.cat([x[ntrain:].float(), torch.ones(x.shape[0] - ntrain, 1)], 1) @ W
        k = E // 4
        recall = (torch.zeros_like(pred).scatter_(1, pred.topk(k, 1).indices, 1)
                  * torch.zeros_like(pred).scatter_(1, y[ntrain:].topk(k, 1).indices, 1)).sum(1).mean().item() / k
        # REAP saliency on the MoE: mean routed output norm with the learned router at k=E/4
        route = torch.cat([x.float(), torch.ones(x.shape[0], 1)], 1) @ W
        on = torch.zeros_like(route).scatter_(1, route.topk(k, 1).indices, 1)
        reap = (y * on).sum(0) / on.sum(0).clamp(min=1)
        dense_norm = y.mean(0)                                          # all tokens, no routing
        neuron = h.float().abs().mean(0) * Wd.float().norm(dim=0).cpu()  # [I] activation x weight
        lay[name] = {"assign": assign.to(dev), "W": W.to(dev), "reap": reap, "dense_norm": dense_norm,
                     "neuron": neuron, "recall": recall}
    prep_s = time.time() - t
    del stats
    torch.cuda.empty_cache() if dev == "cuda" else None
    mlp_params = sum(sum(p.numel() for p in m.parameters()) for _, m in mlps)
    total_params = sum(p.numel() for p in model.parameters())
    save("setup", {"model": a.model, "experts": E, "neurons": I, "layers": len(mlps), "load_s": round(load_s, 1),
                   "prep_s": round(prep_s, 1), "ffn_share_of_params": round(mlp_params / total_params, 3),
                   "router_recall_at_E/4": round(sum(v["recall"] for v in lay.values()) / len(lay), 3)})

    sim = Sim(model, mlps)
    runs = results.get("runs", {})

    def run(name, cfg, ffn_kept=1.0, ffn_active=1.0):
        sim.set(cfg)
        t1 = time.time()
        r = compare(model, refs)
        runs[name] = {"top1": round(r["top1_agreement"], 4), "kl": round(r["kl"], 4), "ppl": round(r["ppl_pruned"], 3),
                      "ffn_kept": round(ffn_kept, 3), "ffn_active_per_token": round(ffn_active, 3),
                      "params_kept": round(1 - (1 - ffn_kept) * mlp_params / total_params, 3),
                      "eval_s": round(time.time() - t1, 1)}
        save("runs", runs, show={name: runs[name]})

    def moe_cfg(kept_of, k, router):
        cfg = {}
        for name, mlp in mlps:
            L = lay[name]
            kept = torch.zeros(E, dtype=torch.bool, device=dev)
            kept[kept_of(L)] = True
            c = {"assign": L["assign"], "kept": kept, "k": k, "router": router, "W": L["W"]}
            if router == "oracle":
                Wd = mlp.down_proj.weight
                assign = L["assign"]

                def norm_fn(hf, Wd=Wd, assign=assign):
                    cols = [(hf[:, assign == e] @ Wd[:, assign == e].T).float().norm(dim=-1) for e in range(E)]
                    return torch.stack(cols, 1)
                c["norm_fn"] = norm_fn
            cfg[name] = c
        return cfg

    every = lambda L: torch.arange(E)  # noqa: E731
    run("dense", None)
    for k in [int(v) for v in a.topk.split(",")]:
        run(f"MoE E{E} top-{k}, learned router", moe_cfg(every, k, "learned"), 1.0, k / E)
        run(f"MoE E{E} top-{k}, oracle router", moe_cfg(every, k, "oracle"), 1.0, k / E)
    g = torch.Generator().manual_seed(0)
    for pr in [float(v) for v in a.prunes.split(",")]:
        n = E - int(round(E * pr))
        top_reap = lambda L, n=n: L["reap"].topk(n).indices  # noqa: E731
        top_dense = lambda L, n=n: L["dense_norm"].topk(n).indices  # noqa: E731
        rand = lambda L, n=n: torch.randperm(E, generator=g)[:n]  # noqa: E731
        f = n / E
        run(f"REAP {pr:.0%} of experts (routed saliency), rest dense", moe_cfg(top_reap, None, "learned"), f, f)
        run(f"expert-norm {pr:.0%} (unrouted saliency), rest dense", moe_cfg(top_dense, None, "learned"), f, f)
        run(f"random {pr:.0%} of experts, rest dense", moe_cfg(rand, None, "learned"), f, f)
        k = max(1, n // 2)
        run(f"REAP {pr:.0%} + learned router top-{k} of {n}", moe_cfg(top_reap, k, "learned"), f, k / E)
        cfg = {}
        for name, _ in mlps:
            keep_n = int(round(I * f))
            m = torch.zeros(I, device=dev)
            m[lay[name]["neuron"].topk(keep_n).indices.to(dev)] = 1
            cfg[name] = {"neuron_mask": m}
        run(f"neuron-level activation pruning {pr:.0%}", cfg, f, f)
    sim.remove()

    print("\n================ moefy summary ================")
    print(json.dumps(results["setup"]))
    print(f"{'config':58s} top1    KL      ppl     FFN kept  FFN active  params kept")
    for name, r in runs.items():
        print(f"{name:58s} {r['top1']:.3f}  {r['kl']:.4f}  {r['ppl']:6.3f}  {r['ffn_kept']:7.3f}  "
              f"{r['ffn_active_per_token']:9.3f}  {r['params_kept']:9.3f}")
    print(f"\nall results: {out_path}")


if __name__ == "__main__":
    main()
