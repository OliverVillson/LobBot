"""One-command GPU benchmark for the gpu_perf prototypes (BlackHole-Harness NVIDIA thread).

    source ~/reaper-venv/bin/activate          # or the Harness venv
    python reaper/gpu_perf/bench_b200.py       # Qwen3-30B-A3B-Instruct-2507, "Debugging only"
    python reaper/gpu_perf/bench_b200.py --job ~/.reaper/jobs/<existing job>   # reuse its data

Phases (each result lands in <job>/work/gpu_perf.json as soon as it is measured, and a
phase whose result is already there is skipped on a rerun):
  setup      REAPer's own plan, weights and calibrate stages (calibrate carries the
             teacher-batching patch on this branch), timed
  teacher    vLLM answers the same prompts in 64-prompt calls (old) and in big calls (new)
  score      reference moe.Saliency vs FastSaliency on the same batches (time + agreement),
             then a full FastSaliency pass with a sample reservoir, and the teacher
             reference on held-out answers
  prune      fast_prune writes the REAP-50% checkpoint, timed
  quality    loads the pruned model and compares, by KL / top-1 / ppl against the teacher:
             uniform int4 and int3 experts vs per-expert allocations at the same bytes
             (2 and 3 formats per layer, with and without a cap on bytes read per token)

Prints a summary table at the end. Nothing is uploaded anywhere.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REAPER = HERE.parent
sys.path[:0] = [str(HERE), str(REAPER)]

from reaper_pipeline.job import Config, Job  # noqa: E402

RESULTS = "work/gpu_perf.json"


def load_results(job: Job) -> dict:
    return job.read(RESULTS) or {}


def save(job: Job, key: str, value) -> None:
    r = load_results(job)
    r[key] = value
    job.write(RESULTS, r)
    print(f"[gpu_perf] {key}: {json.dumps(value)[:600]}", flush=True)


def env_for_children() -> dict:
    pp = os.pathsep.join([str(HERE), str(REAPER), os.environ.get("PYTHONPATH", "")])
    return {**os.environ, "PYTHONPATH": pp, "PYTHONUNBUFFERED": "1"}


# --------------------------------------------------------------------------- setup


def setup(job: Job) -> None:
    times = load_results(job).get("setup", {})
    for stage in ("plan", "weights", "calibrate"):
        if job.is_done(stage):
            continue
        t0 = time.time()
        rc = subprocess.call([sys.executable, "-m", "reaper_pipeline", "--job", str(job.root), "--only", stage],
                             env=env_for_children())
        if rc != 0:
            sys.exit(f"REAPer stage {stage} failed (rc {rc}); see {job.root}/logs/")
        times[stage + "_s"] = round(time.time() - t0, 1)
        save(job, "setup", times)


# --------------------------------------------------------------------------- teacher A/B (own process)


def teacher_ab(job_dir: str, n: int) -> None:
    from reaper_pipeline.stages._common import profile, read_jsonl
    from reaper_pipeline.stages.calibrate import CHUNK, answer_chunk
    from reaper_pipeline.teacher import make_teacher, pick_backend
    job = Job(job_dir)
    cfg, prof = job.config, profile(job)
    rows = (read_jsonl(job.path("data", "heldout.jsonl")) + read_jsonl(job.path("data", "calib.jsonl")))[:n]
    convs = [[{"role": "user", "content": r["prompt"]}] for r in rows]
    backend = pick_backend(prof, cfg.teacher_backend)
    if backend != "vllm":
        save(job, "teacher", {"skipped": f"teacher backend is {backend}, not vLLM"})
        return
    t0 = time.time()
    teacher = make_teacher(str(job.model_dir), prof, backend, cfg.trust_remote_code, cfg.seed)
    startup = time.time() - t0

    def timed(chunk):
        t, chars = time.time(), 0
        for i in range(0, len(convs), chunk):
            chars += sum(len(a) for a in teacher.chat(convs[i:i + chunk], cfg.answer_max_tokens, 0.6))
        return time.time() - t, chars

    big = answer_chunk(prof, backend, len(convs))
    teacher.chat(convs[:8], 32, 0.6)  # warm up
    new_s, new_c = timed(big)          # new first: any prefix-cache benefit goes to the old way
    old_s, old_c = timed(CHUNK)
    teacher.close()
    save(job, "teacher", {"prompts": len(convs), "startup_s": round(startup, 1),
                          "old_chunk": CHUNK, "old_s": round(old_s, 1), "new_chunk": big, "new_s": round(new_s, 1),
                          "speedup": round(old_s / max(new_s, 1e-6), 2), "chars_old": old_c, "chars_new": new_c})


# --------------------------------------------------------------------------- score


def score(job: Job, ref_batches: int, reservoir: int) -> None:
    import torch

    from fast_saliency import FastSaliency
    from reaper_pipeline import moe
    from reaper_pipeline.load import free, input_device, load_model, load_tokenizer
    from reaper_pipeline.stages._common import batches, encode, pad_batch, profile, read_jsonl
    from reaper_pipeline.stages.reap import teacher_reference

    cfg, prof = job.config, profile(job)
    t0 = time.time()
    tok = load_tokenizer(job.model_dir, cfg.trust_remote_code)
    model = load_model(job.model_dir, prof, cfg.trust_remote_code)
    load_s = time.time() - t0
    blocks = moe.find_blocks(model)
    E, k = blocks[0].num_experts, moe.top_k(blocks[0], model.config)
    n_keep = moe.experts_to_keep(E, cfg.prune, k, moe.n_groups(model.config),
                                 getattr(moe.text_config(model.config), "topk_group", None))
    rows = read_jsonl(job.path("data", "calib.jsonl"))
    enc = [encode(tok, r["prompt"], r.get("answer"), cfg.max_seq)[0] for r in rows]
    bs = max(1, int((prof.get("reap") or {}).get("batch_seqs", 1)))
    groups = list(batches(enc, bs, bs * cfg.max_seq))
    dev = input_device(model)

    def sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def run(col, idxs):
        sync()
        t, ntok = time.time(), 0
        with torch.no_grad():
            for idx in idxs:
                ids, mask = pad_batch([enc[i] for i in idx], tok.pad_token_id, dev)
                if col is not None:
                    col.mask = mask.reshape(-1).bool()
                try:
                    model(input_ids=ids, attention_mask=mask, use_cache=False, logits_to_keep=1)
                except TypeError:
                    model(input_ids=ids, attention_mask=mask, use_cache=False)
                ntok += int(mask.sum())
        sync()
        if col is not None:
            col.mask = None
        return time.time() - t, ntok

    # same batches, spread over the length range, for all three
    pick = groups[:: max(1, len(groups) // ref_batches)][:ref_batches]
    run(None, pick[:1])  # warm-up (kernels, allocator)
    plain_s, ntok = run(None, pick)
    ref = moe.Saliency(model, blocks)
    ref_s, _ = run(ref, pick)
    ref.remove()
    fast = FastSaliency(model, blocks)
    fast_s, _ = run(fast, pick)
    fast.remove()
    a, b = ref.results(), fast.results()
    max_rel = max(abs(x - y) / max(abs(x), 1e-9) for li in a for x, y in zip(a[li]["saliency"], b[li]["saliency"]) if x)
    same_freq = all(a[li]["freq"] == b[li]["freq"] for li in a)
    keep_overlap = sum(len(set(moe.choose_keep(a[li]["saliency"], a[li]["freq"], n_keep))
                           & set(moe.choose_keep(b[li]["saliency"], b[li]["freq"], n_keep))) for li in a) / (n_keep * len(a))
    save(job, "score_ab", {"batches": len(pick), "tokens": ntok, "batch_seqs": bs, "load_s": round(load_s, 1),
                           "no_hooks_s": round(plain_s, 2), "reference_s": round(ref_s, 2), "fast_s": round(fast_s, 2),
                           "speedup": round(ref_s / fast_s, 2), "fast_overhead_vs_plain": round(fast_s / plain_s, 2),
                           "max_rel_saliency_diff": max_rel, "same_freq": same_freq, "keep_overlap": keep_overlap,
                           "modes": sorted(set(fast.mode.values()))})
    del ref, a

    # full pass with the fast scorer and a reservoir for the bit probe
    full = FastSaliency(model, blocks, reservoir=reservoir)
    full_s, full_tok = run(full, groups)
    full.remove()
    res = full.results()
    layers = []
    for blk in blocks:
        s, f = res[blk.layer]["saliency"], res[blk.layer]["freq"]
        layers.append({"layer": blk.layer, "module": blk.name, "kept": moe.choose_keep(s, f, n_keep),
                       "saliency": s, "freq": f})
    num_layers = getattr(moe.text_config(model.config), "num_hidden_layers", max(b.layer for b in blocks) + 1)
    job.write("work/gpu_perf_saliency.json", {"model": cfg.model, "experts": E, "kept": n_keep, "top_k": k,
                                              "num_layers": num_layers, "tokens": full_tok, "layers": layers})
    torch.save({"x": full.samples, "g": full.sample_gates}, job.path("work", "gpu_perf_reservoir.pt"))
    est_ref = full_s * ref_s / fast_s
    out = {"tokens": full_tok, "fast_s": round(full_s, 1), "reference_estimate_s": round(est_ref, 1),
           "tok_per_s": round(full_tok / full_s)}
    held = read_jsonl(job.path("data", "heldout.jsonl"))
    if held and not job.path("work", "teacher_ref.pt").exists():
        t = time.time()
        torch.save(teacher_reference(model, tok, held, cfg.max_seq), job.path("work", "teacher_ref.pt"))
        out["teacher_ref_s"] = round(time.time() - t, 1)
    save(job, "score_full", out)
    del model, full
    free()


# --------------------------------------------------------------------------- prune


def pruned_dir(job: Job) -> Path:
    return job.path("work", "gpu_perf_pruned")


def prune(job: Job) -> None:
    import fast_prune
    s = job.read("work/gpu_perf_saliency.json")
    keep = {l["layer"]: l["kept"] for l in s["layers"]}
    t = time.time()
    info = fast_prune.prune_checkpoint(job.model_dir, pruned_dir(job), keep, s["experts"], s["num_layers"],
                                       workers=min(16, os.cpu_count() or 8))
    gb = 1024 ** 3
    save(job, "prune", {"s": round(time.time() - t, 1), "gb_before": round(info["bytes_before"] / gb, 1),
                        "gb_after": round(info["bytes_after"] / gb, 1)})


# --------------------------------------------------------------------------- quality


def expert_views(blk, j: int, H: int) -> dict:
    """{"gate","up","down"} as [out, in] views into the live weights (writable in place)."""
    ex = blk.experts
    if not blk.fused:
        e = ex[j]
        return {"gate": e.gate_proj.weight.data, "up": e.up_proj.weight.data, "down": e.down_proj.weight.data}
    gu = ex.gate_up_proj.data[j]
    if gu.shape[-1] != H:
        gu = gu.T
    I = gu.shape[0] // 2
    dn = ex.down_proj.data[j]
    if dn.shape[0] != H:
        dn = dn.T
    return {"gate": gu[:I], "up": gu[I:], "down": dn}


def quality(job: Job, tiers: list[int]) -> None:
    import torch

    import expert_bits as xb
    from fakequant import fake_quant
    from reaper_pipeline import moe
    from reaper_pipeline.load import free, load_model
    from reaper_pipeline.stages._common import profile
    from reaper_pipeline.stages.eval import compare

    menu = ["int2_g64", "int3_g64", "int4_g128", "int8_g128"]
    s = job.read("work/gpu_perf_saliency.json")
    resv = torch.load(job.path("work", "gpu_perf_reservoir.pt"), weights_only=False)
    refs = torch.load(job.path("work", "teacher_ref.pt"), weights_only=False)
    prof = profile(job)
    model = load_model(pruned_dir(job), prof, job.config.trust_remote_code)
    H = moe.text_config(model.config).hidden_size
    blocks = {b.layer: b for b in moe.find_blocks(model)}
    k = s["top_k"]

    t = time.time()
    experts, views, originals = [], {}, {}
    for L in s["layers"]:
        blk = blocks[L["layer"]]
        tot = sum(L["freq"]) or 1.0
        for jn, jo in enumerate(L["kept"]):
            v = expert_views(blk, jn, H)
            views[(blk.layer, jn)] = v
            originals[(blk.layer, jn)] = {n: w.detach().to("cpu", copy=True) for n, w in v.items()}
            params = sum(w.numel() for w in v.values())
            p = k * L["freq"][jo] / tot
            e = xb.Expert(blk.layer, jn, p, L["saliency"][jo], params)
            xs, gs = resv["x"][blk.layer][jo], resv["g"][blk.layer][jo]
            if xs:
                dev = v["gate"].device
                m = xb.probe({n: w.float() for n, w in v.items()}, torch.cat(xs).to(dev), torch.cat(gs).to(dev), menu)
                e.measured = {f: p * d for f, d in m.items()}
            experts.append(e)
    probe_s = time.time() - t

    def apply(choice):
        with torch.no_grad():
            for key, v in views.items():
                f = choice[key] if choice else None
                for n, w in v.items():
                    src = originals[key][n].to(w.device)
                    w.copy_(src if f is None else fake_quant(src, f).to(w.dtype))

    def run(name, choice, alloc=None):
        apply(choice)
        t0 = time.time()
        r = compare(model, refs)
        r = {k2: (round(v, 4) if isinstance(v, float) else v) for k2, v in r.items()}
        r["eval_s"] = round(time.time() - t0, 1)
        if alloc is not None:
            gb = 1024 ** 3
            r |= {"expert_gb": round(alloc.storage / gb, 2), "traffic_gb_per_tok": round(alloc.traffic / gb, 4),
                  "formats": {f: sum(1 for x in alloc.choice.values() if x == f) for f in menu}}
        out = load_results(job).get("quality", {"probe_s": round(probe_s, 1), "runs": {}})
        out["runs"][name] = r
        save(job, "quality", out)

    run("pruned bf16", None)
    for base in ("int4_g128", "int3_g64"):
        uni = {(e.layer, e.j): base for e in experts}
        ua = xb._evaluate(experts, uni)
        run(f"uniform {base}", uni, ua)
        for nt in tiers:
            a = xb.allocate(experts, menu, ua.storage, max_tiers=nt)
            run(f"per-expert @ {base} bytes, {nt} formats", a.choice, a)
            a = xb.allocate(experts, menu, ua.storage, traffic_bytes=ua.traffic, max_tiers=nt)
            run(f"per-expert @ {base} bytes + traffic cap, {nt} formats", a.choice, a)
    apply(None)
    del model
    free()


# --------------------------------------------------------------------------- summary


def summary(job: Job) -> None:
    r = load_results(job)
    print("\n================ gpu_perf summary ================")
    for key in ("setup", "teacher", "score_ab", "score_full", "prune"):
        if key in r:
            print(f"{key:11s} {json.dumps(r[key])}")
    q = r.get("quality", {})
    if q:
        print(f"\nquality (probe {q.get('probe_s')}s)   top1     KL      ppl     expert GB  GB/token")
        for name, x in q["runs"].items():
            print(f"  {name:52s} {x.get('top1_agreement', 0):.3f}  {x.get('kl', 0):.4f}  {x.get('ppl_pruned', 0):6.3f}"
                  f"  {x.get('expert_gb', ''):>8}  {x.get('traffic_gb_per_tok', '')}")
    print(f"\nall results: {job.path(*RESULTS.split('/'))}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", help="existing REAPer job directory (else one is created)")
    ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    ap.add_argument("--purpose", default="Debugging only")
    ap.add_argument("--calib-samples", type=int, default=256)
    ap.add_argument("--teacher-prompts", type=int, default=256)
    ap.add_argument("--ref-batches", type=int, default=6)
    ap.add_argument("--reservoir", type=int, default=32)
    ap.add_argument("--tiers", default="2,3")
    ap.add_argument("--skip", default="", help="comma list of phases to skip: teacher,score,prune,quality")
    ap.add_argument("--redo", default="", help="comma list of phases to rerun even if measured")
    ap.add_argument("--_teacher_ab", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()

    if a._teacher_ab:
        teacher_ab(a.job, a.teacher_prompts)
        return
    if a.job:
        job = Job(a.job)
    else:
        job = Job.create(Config(model=a.model, purpose=a.purpose, prune=0.5, calib_samples=a.calib_samples,
                                quant="none", out_name="gpu-perf-reference"))
    print(f"[gpu_perf] job {job.root}", flush=True)
    skip, redo = set(filter(None, a.skip.split(","))), set(filter(None, a.redo.split(",")))
    have = lambda k: k in load_results(job) and k.split("_")[0] not in redo  # noqa: E731

    setup(job)
    if "teacher" not in skip and not have("teacher"):
        py = os.environ.get("REAPER_VLLM_PY") or sys.executable
        subprocess.call([py, str(Path(__file__).resolve()), "--_teacher_ab", "--job", str(job.root),
                         "--teacher-prompts", str(a.teacher_prompts)], env=env_for_children())
    if "score" not in skip and not have("score_full"):
        score(job, a.ref_batches, a.reservoir)
    if "prune" not in skip and not have("prune"):
        prune(job)
    if "quality" not in skip and not have("quality"):
        quality(job, [int(x) for x in a.tiers.split(",")])
    summary(job)


if __name__ == "__main__":
    main()
