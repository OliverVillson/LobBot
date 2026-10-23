"""Dense model shrink frontier: width pruning x depth pruning x heal x quant, against the dense model.

    python reaper/gpu_perf/dense_bench.py                      # Qwen/Qwen3-8B, data from the REAP job
    python reaper/gpu_perf/dense_bench.py --model meta-llama/Llama-3.1-8B-Instruct

The dense-model counterpart of bench_b200.py's MoE prune x bits frontier. Per config it reports
KL / top-1 / ppl against the unmodified model and the real model size in GB.

  width     drop the FFN neurons with the lowest E|h_i| * ||W_down[:, i]|| (best zero-shot dense
            scorer in moefy_bench.py), applied as zeroed gate/up rows and down columns
  heal      closed-form, no training: refit each layer's down_proj on the kept neurons by ridge
            least squares so the FFN output matches the dense FFN on sampled calibration tokens
            (the compensation step of FLAP / SparseGPT-style pruning)
  depth     skip the n decoder layers whose output is closest to their input
            (angular distance, Gromov et al. 2024); first and last layer are never dropped
  quant     fake quant of every decoder-layer Linear (attention + FFN): bf16, int8, int4, int3 by
            round-to-nearest (RTN), and int4 / int3 by GPTQ on the same grid (gptq.py, Hessians
            from the dense model's calibration activations); embeddings / lm_head stay bf16
Configs already in the results file are skipped (--redo to rerun), so a rerun only adds new rows.

Quality only, nothing is timed. Results: <job>/work/dense_<model>.json and a table at the end.
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

from fakequant import BPW, fake_quant  # noqa: E402
from gptq import gptq  # noqa: E402
from moefy_bench import collect, find_mlps  # noqa: E402
from reaper_pipeline.job import home  # noqa: E402


def decoder_layers(model):
    for mod in model.modules():
        if isinstance(mod, torch.nn.ModuleList) and len(mod) > 1 and any(
                isinstance(m, torch.nn.Linear) for m in mod[0].modules()):
            return list(mod)
    sys.exit("no decoder layer list found")


@torch.no_grad()
def layer_angles(layers, model, batches) -> list[float]:
    """Mean angular distance between each decoder layer's input and output hidden states."""
    acc = [0.0] * len(layers)
    cnt = [0] * len(layers)
    state = {"mask": None}
    hooks = []
    for i, layer in enumerate(layers):
        def hook(mod, args, kwargs, out, i=i):
            x = args[0] if args else kwargs["hidden_states"]
            y = out[0] if isinstance(out, tuple) else out
            m = state["mask"].to(x.device)
            c = torch.nn.functional.cosine_similarity(x.reshape(-1, x.shape[-1])[m].float(),
                                                      y.reshape(-1, y.shape[-1])[m].float(), dim=-1)
            acc[i] += torch.arccos(c.clamp(-1, 1)).sum().item() / torch.pi
            cnt[i] += c.numel()
        hooks.append(layer.register_forward_hook(hook, with_kwargs=True))
    for ids, mask in batches:
        state["mask"] = mask.reshape(-1).bool()
        model(input_ids=ids, attention_mask=mask, use_cache=False, logits_to_keep=1)
    for h in hooks:
        h.remove()
    return [a / max(c, 1) for a, c in zip(acc, cnt)]


SHARED = {"k_proj": "q_proj", "v_proj": "q_proj", "up_proj": "gate_proj"}   # same input, one Hessian


def hkey(name: str) -> str:
    head, _, leaf = name.rpartition(".")
    return f"{head}.{SHARED.get(leaf, leaf)}"


@torch.no_grad()
def hessians(lin: dict, model, batches) -> dict:
    """H = sum x x^T of each Linear's input over non-padding calibration tokens (fp32, on device)."""
    H = {}
    state = {"mask": None}
    hooks = []
    for n, m in lin.items():
        k = hkey(n)
        if k != n and k in lin:
            continue

        def hook(mod, args, k=k):
            x = args[0].reshape(-1, args[0].shape[-1])[state["mask"].to(args[0].device)].float()
            if k not in H:
                H[k] = torch.zeros(x.shape[1], x.shape[1], device=x.device)
            H[k].addmm_(x.T, x)
        hooks.append(m.register_forward_pre_hook(hook))
    for ids, mask in batches:
        state["mask"] = mask.reshape(-1).bool()
        model(input_ids=ids, attention_mask=mask, use_cache=False, logits_to_keep=1)
    for h in hooks:
        h.remove()
    return H


class Skip:
    """Forward hooks that replace a decoder layer's output with its input."""

    def __init__(self, layers):
        self.off = set()
        self.hooks = []
        for i, layer in enumerate(layers):
            def hook(mod, args, kwargs, out, i=i):
                if i not in self.off:
                    return None
                x = args[0] if args else kwargs["hidden_states"]
                return (x,) + tuple(out[1:]) if isinstance(out, tuple) else x
            self.hooks.append(layer.register_forward_hook(hook, with_kwargs=True))

    def remove(self):
        for h in self.hooks:
            h.remove()


def heal_down(h: torch.Tensor, Wd: torch.Tensor, keep: torch.Tensor, lam: float = 1e-3) -> torch.Tensor:
    """Ridge refit of down_proj on kept neurons: argmin_W ||h_keep W^T - h Wd^T||. Returns new Wd."""
    dev = Wd.device
    H = h.to(dev, torch.float32)
    Hk = H[:, keep]
    A = Hk.T @ Hk
    A += lam * A.diagonal().mean() * torch.eye(A.shape[0], device=dev)
    B = (Hk.T @ H) @ Wd.float().T                                        # [k, out]
    out = torch.zeros_like(Wd, dtype=torch.float32)
    out[:, keep] = torch.linalg.solve(A, B).T
    return out


def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from reaper_pipeline.stages._common import batches as make_batches
    from reaper_pipeline.stages._common import encode, pad_batch, read_jsonl
    from reaper_pipeline.stages.eval import compare
    from reaper_pipeline.stages.reap import teacher_reference

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--data", default=None, help="REAPer job dir with data/calib.jsonl + heldout.jsonl")
    ap.add_argument("--widths", default="0.125,0.25", help="fractions of FFN neurons to drop")
    ap.add_argument("--depths", default="4,8", help="numbers of decoder layers to drop")
    ap.add_argument("--formats", default="bf16,int8_g128,int4_g128,int3_g64")
    ap.add_argument("--depth-formats", default="bf16,int4_g128")
    ap.add_argument("--gptq-formats", default="int4_g128,int3_g64")
    ap.add_argument("--heal", action="store_true", help="also run the closed-form down_proj heal")
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--calib", type=int, default=64)
    ap.add_argument("--tokens", type=int, default=16384, help="sampled tokens per layer for scores / heal")
    ap.add_argument("--max-seq", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    data = Path(a.data) if a.data else next(iter(sorted((home() / "jobs").glob("*/data/calib.jsonl"))), None)
    if data is None:
        sys.exit("no calibration data: pass --data <REAPer job dir>")
    data = data if data.is_dir() else data.parent.parent
    out_path = Path(a.out) if a.out else data / "work" / f"dense_{a.model.split('/')[-1]}.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else {}

    def save(key, val, show=None):
        results[key] = val
        out_path.write_text(json.dumps(results, indent=1))
        print(f"[dense] {key}: {json.dumps(val if show is None else show)[:300]}", flush=True)

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
    layers = decoder_layers(model)
    angles = layer_angles(layers, model, batches)
    stats = collect(model, mlps, batches, a.tokens)
    hs, score = {}, {}
    for name, mlp in mlps:
        h = stats[name][1]
        hs[name] = h
        score[name] = h.float().abs().mean(0) * mlp.down_proj.weight.float().norm(dim=0).cpu()
    del stats
    prep_s = time.time() - t

    # every decoder-layer Linear gets quantized; keep a pristine copy to restore from
    owner = {id(m): i for i, L in enumerate(layers) for m in L.modules()}
    lin = {n: m for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and id(m) in owner}
    layer_of = {n: owner[id(m)] for n, m in lin.items()}
    orig = {n: m.weight.data.clone() for n, m in lin.items()}
    ffn = {f"{mn}.{p}" for mn, _ in mlps for p in ("gate_proj", "up_proj", "down_proj")}
    lin_params = sum(w.numel() for w in orig.values())
    total_params = sum(p.numel() for p in model.parameters())
    other_gb = (total_params - lin_params) * 2 / 1e9
    I = mlps[0][1].down_proj.in_features
    order = sorted(range(1, len(layers) - 1), key=lambda i: angles[i])
    save("setup", {"model": a.model, "layers": len(layers), "neurons": I, "load_s": round(load_s, 1),
                   "prep_s": round(prep_s, 1), "params_b": round(total_params / 1e9, 2),
                   "bf16_gb": round(total_params * 2 / 1e9, 2),
                   "layer_angles": [round(v, 4) for v in angles], "drop_order": order})

    skip = Skip(layers)
    runs = results.get("runs", {})

    H = hessians(lin, model, batches) if a.gptq_formats else {}

    def run(name, base: dict, fmt: str, drop: set, width: float, method: str = "rtn"):
        if name in runs and not a.redo:
            return
        t1 = time.time()
        for n, m in lin.items():
            w = base.get(n, orig[n])
            if fmt == "bf16":
                q = w
            elif method == "gptq":
                q = gptq(w, H[hkey(n)], fmt)
            else:
                q = fake_quant(w.float(), fmt)
            m.weight.data.copy_(q.to(w.dtype))
        quant_s = time.time() - t1
        skip.off = drop
        r = compare(model, refs)
        kept = 0.0
        for n, w in orig.items():
            if layer_of.get(n) in drop:
                continue
            kept += w.numel() * (1 - width if n in ffn else 1.0)
        gb = kept * BPW[fmt] / 8 / 1e9 + other_gb
        runs[name] = {"top1": round(r["top1_agreement"], 4), "kl": round(r["kl"], 4), "ppl": round(r["ppl_pruned"], 3),
                      "gb": round(gb, 2), "params_kept": round((kept + total_params - lin_params) / total_params, 3),
                      "quant_s": round(quant_s, 1), "eval_s": round(time.time() - t1 - quant_s, 1)}
        save("runs", runs, show={name: runs[name]})

    def width_base(frac: float, heal: bool) -> dict:
        base = {}
        for name, mlp in mlps:
            keep_n = I - int(round(I * frac))
            keep = score[name].topk(keep_n).indices.sort().values.to(dev)
            mask = torch.zeros(I, device=dev, dtype=orig[f"{name}.down_proj"].dtype)
            mask[keep] = 1
            base[f"{name}.gate_proj"] = orig[f"{name}.gate_proj"] * mask[:, None]
            base[f"{name}.up_proj"] = orig[f"{name}.up_proj"] * mask[:, None]
            Wd = orig[f"{name}.down_proj"]
            base[f"{name}.down_proj"] = (heal_down(hs[name], Wd, keep).to(Wd.dtype) if heal else Wd * mask[None, :])
        return base

    fmts = [f for f in a.formats.split(",") if f]
    gfmts = [f for f in a.gptq_formats.split(",") if f]
    for f in fmts:
        run(f"dense {f}", {}, f, set(), 0.0)
    for f in gfmts:
        run(f"dense gptq-{f}", {}, f, set(), 0.0, "gptq")
    for w in [float(v) for v in a.widths.split(",") if v]:
        for heal in ((False, True) if a.heal else (False,)):
            base = width_base(w, heal)
            tag = f"width -{w:.1%}{' +heal' if heal else ''}"
            for f in fmts:
                run(f"{tag} {f}", base, f, set(), w)
            for f in gfmts:
                run(f"{tag} gptq-{f}", base, f, set(), w, "gptq")
            del base
    for d in [int(v) for v in a.depths.split(",") if v]:
        for f in a.depth_formats.split(","):
            run(f"depth -{d} layers {f}", {}, f, set(order[:d]), 0.0)
    skip.remove()
    for n, m in lin.items():
        m.weight.data.copy_(orig[n])

    print("\n================ dense summary ================")
    s = dict(results["setup"])
    s.pop("layer_angles")
    print(json.dumps(s))
    print(f"{'config':44s} top1    KL      ppl      GB     params kept")
    for name, r in sorted(runs.items(), key=lambda kv: kv[1]["gb"]):
        print(f"{name:44s} {r['top1']:.3f}  {r['kl']:.4f}  {r['ppl']:7.3f}  {r['gb']:6.2f}  {r['params_kept']:9.3f}")
    print(f"\nall results: {out_path}")


if __name__ == "__main__":
    main()
