"""Dense model -> experts (MoEfication) -> REAP-ranked mixed precision, against the dense model.

    python reaper/gpu_perf/mixq_bench.py                      # Qwen/Qwen3-8B, data from the REAP job

Every FFN keeps all its weights; nothing is pruned or routed. The FFN neurons are grouped into E
experts (balanced k-means on co-activation, as in moefy_bench.py), each expert is scored, and the
experts get bit widths by rank: an interval lo-hi (e.g. 2-8 bits) is spread over the ranking,
most important expert at hi, least at lo, along b(r) = lo + (hi - lo)(1 - r)^p with p chosen so
the FFN averages a target bits per weight (default the int4 and int3 budgets, 4.125 and 3.25
including group scales). Scorers:
  reap      mean routed output norm with a fitted router at k = E/4 (REAP on the MoEfied model)
  expnorm   mean expert output norm over all tokens (no routing)
  neuron    no MoEfication: neurons ranked one by one by E|h_i| * ||W_down[:, i]||, cut into E bins
  random    the same expert groups in random order (control)
All quantization is GPTQ: gate/up rows per format, down_proj with a format per input column
(gptq_mixed). Attention is GPTQ int4 in every config; embeddings / lm_head stay bf16.
Results: <job>/work/mixq_<model>.json and a table at the end.
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

from dense_bench import decoder_layers, hessians, hkey  # noqa: E402
from fakequant import BPW  # noqa: E402
from gptq import gptq_batched, gptq_mixed  # noqa: E402
from moefy_bench import balanced_kmeans, collect, expert_norms, find_mlps, fit_router  # noqa: E402
from reaper_pipeline.job import home  # noqa: E402

FMT = {2: "int2_g64", 3: "int3_g64", 4: "int4_g128", 5: "int5_g128", 6: "int6_g128", 7: "int7_g128",
       8: "int8_g128"}


def ladder(E: int, lo: int, hi: int, target: float) -> list[int] | None:
    """Bits for ranks 0..E-1 (0 = most important) on lo + (hi-lo)(1-r)^p, p fitted to the target
    mean bits per weight. None when the target is outside the interval."""
    if not BPW[FMT[lo]] <= target <= BPW[FMT[hi]]:
        return None
    best = None
    for i in range(801):
        p = 2 ** (-6 + 12 * i / 800)
        bits = [round(lo + (hi - lo) * (1 - (r + 0.5) / E) ** p) for r in range(E)]
        avg = sum(BPW[FMT[b]] for b in bits) / E
        if best is None or abs(avg - target) < best[0]:
            best = (abs(avg - target), bits)
    return best[1]


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
    ap.add_argument("--targets", default="4.125,3.25", help="FFN mean bits per weight")
    ap.add_argument("--ranges", default="2-8,2-6,3-6,3-5,4-5")
    ap.add_argument("--scorers", default="reap,expnorm,neuron,random")
    ap.add_argument("--random-ranges", default="2-8,3-5", help="ranges the random control runs on")
    ap.add_argument("--attn", default="int4_g128")
    ap.add_argument("--calib", type=int, default=64)
    ap.add_argument("--tokens", type=int, default=8192, help="sampled tokens per layer for clustering/scores")
    ap.add_argument("--max-seq", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--layer-chunk", type=int, default=4, help="layers quantized together by batched GPTQ")
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    data = Path(a.data) if a.data else next(iter(sorted((home() / "jobs").glob("*/data/calib.jsonl"))), None)
    if data is None:
        sys.exit("no calibration data: pass --data <REAPer job dir>")
    data = data if data.is_dir() else data.parent.parent
    out_path = Path(a.out) if a.out else data / "work" / f"mixq_{a.model.split('/')[-1]}.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else {}

    def save(key, val, show=None):
        results[key] = val
        out_path.write_text(json.dumps(results, indent=1))
        print(f"[mixq] {key}: {json.dumps(val if show is None else show)[:300]}", flush=True)

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
    owner = {id(m): i for i, L in enumerate(layers) for m in L.modules()}
    lin = {n: m for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and id(m) in owner}
    H = hessians(lin, model, batches)
    stats = collect(model, mlps, batches, a.tokens)
    E = a.experts
    I = mlps[0][1].down_proj.in_features
    if I % E:
        sys.exit(f"{I} neurons do not split into {E} equal experts")
    g = torch.Generator().manual_seed(0)
    lay = []
    for name, mlp in mlps:
        x, h = stats.pop(name)
        assign = balanced_kmeans(h.T.to(dev).abs(), E)
        Wd = mlp.down_proj.weight.data
        y = expert_norms(h, Wd, assign, E)
        W = fit_router(x, y)
        k = E // 4
        route = torch.cat([x.float(), torch.ones(x.shape[0], 1)], 1) @ W
        on = torch.zeros_like(route).scatter_(1, route.topk(k, 1).indices, 1)
        neuron = h.float().abs().mean(0) * Wd.float().norm(dim=0).cpu()
        # rank of every neuron's unit (expert or neuron bin), 0 = most important
        unit_rank = {}
        for sc, score in (("reap", (y * on).sum(0) / on.sum(0).clamp(min=1)), ("expnorm", y.mean(0)),
                          ("random", torch.rand(E, generator=g))):
            rank = torch.empty(E, dtype=torch.long)
            rank[score.argsort(descending=True)] = torch.arange(E)
            unit_rank[sc] = rank[assign]
        nr = torch.empty(I, dtype=torch.long)
        nr[neuron.argsort(descending=True)] = torch.arange(I)
        unit_rank["neuron"] = nr * E // I
        lay.append(unit_rank)
    del stats
    prep_s = time.time() - t

    # attention (and anything else that is not FFN) is quantized once, GPTQ at --attn
    ffn = {f"{mn}.{p}" for mn, _ in mlps for p in ("gate_proj", "up_proj", "down_proj")}
    for n, m in lin.items():
        if n not in ffn:
            m.weight.data.copy_(gptq_batched(m.weight.data[None], H[hkey(n)][None], a.attn)[0].to(m.weight.dtype))
    orig = {n: lin[n].weight.data.clone() for n in ffn}
    total = sum(p.numel() for p in model.parameters())
    ffn_params = sum(w.numel() for w in orig.values())
    attn_params = sum(m.weight.numel() for n, m in lin.items() if n not in ffn)
    other_gb = (total - ffn_params - attn_params) * 2 / 1e9 + attn_params * BPW[a.attn] / 8 / 1e9
    save("setup", {"model": a.model, "experts": E, "neurons": I, "layers": len(mlps), "load_s": round(load_s, 1),
                   "prep_s": round(prep_s, 1), "attn": a.attn, "ffn_share_of_params": round(ffn_params / total, 3)})
    runs = results.get("runs", {})

    def apply(bits_of_unit: list[int], scorer: str):
        """Quantize every FFN with per-neuron bits = bits_of_unit[unit rank of that neuron]."""
        bu = torch.tensor(bits_of_unit)
        for c0 in range(0, len(mlps), a.layer_chunk):
            chunk = mlps[c0:c0 + a.layer_chunk]
            nb = [bu[lay[c0 + i][scorer]] for i in range(len(chunk))]            # [I] bits per neuron
            # gate / up: rows of one format together, all layers of the chunk in one batch
            for b in sorted(set(bits_of_unit), reverse=True):
                rows = [(x == b).nonzero()[:, 0].to(dev) for x in nb]
                Wf = torch.stack([torch.cat([orig[f"{mn}.gate_proj"][r], orig[f"{mn}.up_proj"][r]])
                                  for (mn, _), r in zip(chunk, rows)])
                Hs = torch.stack([H[f"{mn}.gate_proj"] for mn, _ in chunk])
                q = gptq_batched(Wf, Hs, FMT[b])
                for (mn, mlp), r, qi in zip(chunk, rows, q):
                    mlp.gate_proj.weight.data[r] = qi[: len(r)].to(dtype)
                    mlp.up_proj.weight.data[r] = qi[len(r):].to(dtype)
                del Wf, Hs, q
            # down: columns permuted so each format is one contiguous run, format per column
            perms = [torch.sort(-x, stable=True).indices.to(dev) for x in nb]
            col_fmt = [FMT[int(b)] for b in nb[0][perms[0].cpu()]]
            Wp = torch.stack([orig[f"{mn}.down_proj"][:, p] for (mn, _), p in zip(chunk, perms)])
            Hp = torch.stack([H[f"{mn}.down_proj"][p][:, p] for (mn, _), p in zip(chunk, perms)])
            q = gptq_mixed(Wp, Hp, col_fmt)
            for (mn, mlp), p, qi in zip(chunk, perms, q):
                mlp.down_proj.weight.data[:, p] = qi.to(dtype)
            del Wp, Hp, q

    def run(name, bits_of_unit, scorer):
        if name in runs and not a.redo:
            return
        t1 = time.time()
        apply(bits_of_unit, scorer)
        quant_s = time.time() - t1
        r = compare(model, refs)
        bpw = sum(BPW[FMT[b]] for b in bits_of_unit) / len(bits_of_unit)
        hist = {str(b): bits_of_unit.count(b) for b in sorted(set(bits_of_unit))}
        runs[name] = {"top1": round(r["top1_agreement"], 4), "kl": round(r["kl"], 4), "ppl": round(r["ppl_pruned"], 3),
                      "ffn_bpw": round(bpw, 3), "gb": round(ffn_params * bpw / 8 / 1e9 + other_gb, 2),
                      "experts_per_bits": hist, "quant_s": round(quant_s, 1),
                      "eval_s": round(time.time() - t1 - quant_s, 1)}
        save("runs", runs, show={name: runs[name]})

    for b in (4, 3):
        run(f"uniform int{b}", [b] * E, "neuron")
    scorers = a.scorers.split(",")
    rand_ranges = set(a.random_ranges.split(","))
    for target in [float(v) for v in a.targets.split(",")]:
        for rg in a.ranges.split(","):
            lo, hi = (int(v) for v in rg.split("-"))
            bits = ladder(E, lo, hi, target)
            if bits is None or max(bits.count(b) for b in bits) > 0.9 * E:   # (almost) uniform
                continue
            for sc in scorers:
                if sc == "random" and rg not in rand_ranges:
                    continue
                run(f"{rg} bits @{target:g} {sc}", bits, sc)

    print("\n================ mixq summary ================")
    print(json.dumps(results["setup"]))
    print(f"{'config':28s} FFN bpw  top1    KL      ppl      GB    experts per bit width")
    for name, r in sorted(runs.items(), key=lambda kv: (-round(kv[1]["ffn_bpw"] * 2) / 2, kv[1]["kl"])):
        h = " ".join(f"{k}b:{v}" for k, v in r["experts_per_bits"].items())
        print(f"{name:28s} {r['ffn_bpw']:6.3f}  {r['top1']:.3f}  {r['kl']:.4f}  {r['ppl']:7.3f}  {r['gb']:5.2f}  {h}")
    print(f"\nall results: {out_path}")


if __name__ == "__main__":
    main()
