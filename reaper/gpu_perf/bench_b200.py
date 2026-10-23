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
    ref2 = moe.Saliency(model, blocks)  # the same batches again: how much bf16 run-to-run noise is there?
    run(ref2, pick)
    ref2.remove()
    fast = FastSaliency(model, blocks)
    fast_s, _ = run(fast, pick)
    fast.remove()

    def agree(a, b):
        tot = sum(sum(a[li]["freq"]) for li in a)
        return {"max_rel_saliency_diff": round(max(abs(x - y) / max(abs(x), 1e-9) for li in a
                                                   for x, y in zip(a[li]["saliency"], b[li]["saliency"]) if x), 4),
                "routed_pairs_moved": round(sum(abs(x - y) for li in a for x, y in zip(a[li]["freq"], b[li]["freq"]))
                                            / 2 / max(tot, 1), 6),
                "keep_overlap": round(sum(len(set(moe.choose_keep(a[li]["saliency"], a[li]["freq"], n_keep))
                                              & set(moe.choose_keep(b[li]["saliency"], b[li]["freq"], n_keep)))
                                          for li in a) / (n_keep * len(a)), 4)}

    a = ref.results()
    save(job, "score_ab", {"batches": len(pick), "tokens": ntok, "batch_seqs": bs, "load_s": round(load_s, 1),
                           "no_hooks_s": round(plain_s, 2), "reference_s": round(ref_s, 2), "fast_s": round(fast_s, 2),
                           "speedup": round(ref_s / fast_s, 2), "fast_overhead_vs_plain": round(fast_s / plain_s, 2),
                           "fast_vs_ref": agree(a, fast.results()), "ref_vs_ref": agree(a, ref2.results()),
                           "grouped_layers": sum(1 for v in fast._grouped_ok.values() if v),
                           "modes": sorted(set(fast.mode.values()))})
    del ref, ref2, a

    # full pass with the fast scorer and a reservoir for the bit probe
    full = FastSaliency(model, blocks, reservoir=reservoir)
    import random
    order = groups[:]
    random.Random(0).shuffle(order)  # reservoir samples from every length, not only the shortest batches
    full_s, full_tok = run(full, order)
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


def quality(job: Job, tiers: list[int], menu: list[str], bases: list[str]) -> None:
    import torch

    import expert_bits as xb
    from fakequant import fake_quant
    from reaper_pipeline import moe
    from reaper_pipeline.load import free, load_model
    from reaper_pipeline.stages._common import profile
    from reaper_pipeline.stages.eval import compare

    tag = "/".join(f.split("_")[0].replace("int", "") for f in menu)
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

    done = (load_results(job).get("quality") or {}).get("runs", {})
    if "pruned bf16" not in done:
        run("pruned bf16", None)
    for base in bases:
        uni = {(e.layer, e.j): base for e in experts}
        ua = xb._evaluate(experts, uni)
        if f"uniform {base}" not in done:
            run(f"uniform {base}", uni, ua)
        for nt in tiers:
            a = xb.allocate(experts, menu, ua.storage, max_tiers=nt)
            run(f"per-expert [{tag}] @ {base} bytes, {nt} formats", a.choice, a)
            a = xb.allocate(experts, menu, ua.storage, traffic_bytes=ua.traffic, max_tiers=nt)
            run(f"per-expert [{tag}] @ {base} bytes + traffic cap, {nt} formats", a.choice, a)
    apply(None)
    del model
    free()


# --------------------------------------------------------------------------- explore: more tests


def calib_breakdown(job: Job) -> None:
    """Where calibrate's wall time went, from the stage's own progress events."""
    ev, lines = [], []
    for src in (job.path("logs", "calibrate.log"), job.path("events.log")):
        if src.exists():
            lines = src.read_text(errors="replace").splitlines()
            break
    for line in lines:
        if line.startswith("{"):
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("stage") == "calibrate" and "ts" in e:
                ev.append(e)
    if not ev:
        save(job, "calib_breakdown", {"skipped": "no calibrate events"})
        return

    def first(pred):
        return next((e["ts"] for e in ev if pred(e)), None)

    t0, t_end = ev[0]["ts"], ev[-1]["ts"]
    t_write = first(lambda e: e.get("pct") == 5)
    t_written = first(lambda e: (e.get("pct") or 0) >= 40)
    out = {"total_s": round(t_end - t0, 1)}
    if t_write:
        out["teacher_startup_s"] = round(t_write - t0, 1)
    if t_write and t_written:
        out["prompt_writing_s"] = round(t_written - t_write, 1)
    if t_written:
        out["answering_s"] = round(t_end - t_written, 1)
    save(job, "calib_breakdown", out)


def teacher_variant(job_dir: str, variant: str, n_answers: int, n_written: int) -> None:
    """vLLM teacher in bf16 or fp8 (weights + KV cache): startup, prompt writing and answering."""
    from reaper_pipeline import calib
    from reaper_pipeline.stages._common import profile, read_jsonl
    from reaper_pipeline.stages.calibrate import answer_chunk
    from reaper_pipeline.teacher import make_teacher
    job = Job(job_dir)
    cfg, prof = job.config, dict(profile(job))
    prof["teacher"] = dict(prof.get("teacher") or {})
    if variant == "fp8":
        prof["teacher"].update(quantization="fp8", kv_cache_dtype="fp8")
    t0 = time.time()
    teacher = make_teacher(str(job.model_dir), prof, "vllm", cfg.trust_remote_code, cfg.seed)
    startup = time.time() - t0
    t = time.time()
    rows, _ = calib.build_prompts(teacher, cfg.purpose, cfg.calib, n_written, 0.0, cfg.seed,
                                  max_tokens=cfg.writer_max_tokens)
    write_s = time.time() - t
    prompts = (read_jsonl(job.path("data", "heldout.jsonl")) + read_jsonl(job.path("data", "calib.jsonl")))[:n_answers]
    convs = [[{"role": "user", "content": r["prompt"]}] for r in prompts]
    t, chars = time.time(), 0
    chunk = answer_chunk(prof, "vllm", len(convs))
    for i in range(0, len(convs), chunk):
        chars += sum(len(x) for x in teacher.chat(convs[i:i + chunk], cfg.answer_max_tokens, 0.6))
    answer_s = time.time() - t
    teacher.close()
    r = load_results(job).get("teacher_precision", {})
    r[variant] = {"startup_s": round(startup, 1), "write_s": round(write_s, 1), "prompts_written": len(rows),
                  "answer_s": round(answer_s, 1), "answers": len(convs), "chars": chars}
    save(job, "teacher_precision", r)


def calib_size(job: Job, sizes: list[int]) -> None:
    """How many calibration sequences does REAP need? Keep-set overlap with the full set when
    scoring on a subset, and the pruned model's quality when pruned from a small subset."""
    import random

    import torch

    import fast_prune
    from fast_saliency import FastSaliency
    from reaper_pipeline import moe
    from reaper_pipeline.load import free, input_device, load_model, load_tokenizer
    from reaper_pipeline.stages._common import encode, pad_batch, profile, read_jsonl
    from reaper_pipeline.stages.eval import compare

    cfg, prof = job.config, profile(job)
    full = job.read("work/gpu_perf_saliency.json")
    n_keep = full["kept"]
    tok = load_tokenizer(job.model_dir, cfg.trust_remote_code)
    model = load_model(job.model_dir, prof, cfg.trust_remote_code)
    blocks = moe.find_blocks(model)
    rows = read_jsonl(job.path("data", "calib.jsonl"))
    enc = [encode(tok, r["prompt"], r.get("answer"), cfg.max_seq)[0] for r in rows]
    order = list(range(len(enc)))
    random.Random(1).shuffle(order)
    dev = input_device(model)
    bs = max(1, int((prof.get("reap") or {}).get("batch_seqs", 1)))
    out, keeps = {}, {}
    for n in sizes:
        idx = sorted(order[:n], key=lambda i: len(enc[i]))
        col = FastSaliency(model, blocks)
        with torch.no_grad():
            for b in range(0, len(idx), bs):
                ids, mask = pad_batch([enc[i] for i in idx[b:b + bs]], tok.pad_token_id, dev)
                col.mask = mask.reshape(-1).bool()
                try:
                    model(input_ids=ids, attention_mask=mask, use_cache=False, logits_to_keep=1)
                except TypeError:
                    model(input_ids=ids, attention_mask=mask, use_cache=False)
        col.remove()
        res = col.results()
        keeps[n] = {L["layer"]: moe.choose_keep(res[L["layer"]]["saliency"], res[L["layer"]]["freq"], n_keep)
                    for L in full["layers"]}
        overlap = sum(len(set(keeps[n][L["layer"]]) & set(L["kept"])) for L in full["layers"]) / (n_keep * len(full["layers"]))
        out[str(n)] = {"keep_overlap_with_full": round(overlap, 4)}
    del model
    free()
    refs = torch.load(job.path("work", "teacher_ref.pt"), weights_only=False)
    for n in sizes:
        d = job.path("work", f"gpu_perf_calib{n}")
        fast_prune.prune_checkpoint(job.model_dir, d, keeps[n], full["experts"], full["num_layers"],
                                    workers=min(16, os.cpu_count() or 8))
        m = load_model(d, prof, cfg.trust_remote_code)
        r = compare(m, refs)
        out[str(n)].update(top1=round(r["top1_agreement"], 4), kl=round(r["kl"], 4), ppl=round(r["ppl_pruned"], 3))
        del m
        free()
        import shutil
        shutil.rmtree(d, ignore_errors=True)
        save(job, "calib_size", out)


def frontier(job: Job, prunes: list[float], formats: list[str]) -> None:
    """Prune ratio x expert format sweep: is a byte budget better spent on pruning or on bits?"""
    import shutil

    import torch

    import fast_prune
    from fakequant import BPW, fake_quant
    from reaper_pipeline import moe
    from reaper_pipeline.load import free, load_model
    from reaper_pipeline.stages._common import profile
    from reaper_pipeline.stages.eval import compare

    s = job.read("work/gpu_perf_saliency.json")
    refs = torch.load(job.path("work", "teacher_ref.pt"), weights_only=False)
    prof = profile(job)
    E, k = s["experts"], s["top_k"]
    out = load_results(job).get("frontier", {})
    for pr in prunes:
        n_keep = moe.experts_to_keep(E, pr, k)
        todo = [f for f in formats if f"{pr:.3f}/{f}" not in out]
        if not todo:
            continue
        keep = {L["layer"]: moe.choose_keep(L["saliency"], L["freq"], n_keep) for L in s["layers"]}
        d = job.path("work", f"gpu_perf_frontier_{int(pr * 1000)}")
        fast_prune.prune_checkpoint(job.model_dir, d, keep, E, s["num_layers"], workers=min(16, os.cpu_count() or 8))
        model = load_model(d, prof, job.config.trust_remote_code)
        H = moe.text_config(model.config).hidden_size
        views = [expert_views(b, j, H) for b in moe.find_blocks(model) for j in range(b.num_experts)]
        params = sum(w.numel() for v in views for w in v.values())
        orig = [{n: w.detach().to("cpu", copy=True) for n, w in v.items()} for v in views] \
            if any(f != "bf16" for f in todo) else None
        for f in todo:
            if orig is not None:
                with torch.no_grad():
                    for v, o in zip(views, orig):
                        for n, w in v.items():
                            src = o[n].to(w.device)
                            w.copy_(src if f == "bf16" else fake_quant(src, f).to(w.dtype))
            r = compare(model, refs)
            out[f"{pr:.3f}/{f}"] = {"prune": pr, "kept": n_keep, "format": f,
                                    "expert_gb": round(params * BPW[f] / 8 / 1024 ** 3, 2),
                                    "top1": round(r["top1_agreement"], 4), "kl": round(r["kl"], 4),
                                    "ppl": round(r["ppl_pruned"], 3)}
            save(job, "frontier", out)
        del model, views, orig
        free()
        shutil.rmtree(d, ignore_errors=True)


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
            print(f"  {name:62s} {x.get('top1_agreement', 0):.3f}  {x.get('kl', 0):.4f}  {x.get('ppl_pruned', 0):6.3f}"
                  f"  {x.get('expert_gb', ''):>8}  {x.get('traffic_gb_per_tok', '')}")
    for key in ("calib_breakdown", "teacher_precision", "calib_size"):
        if key in r:
            print(f"{key:17s} {json.dumps(r[key])}")
    fr = r.get("frontier", {})
    if fr:
        print("\nfrontier: prune  kept  format      expert GB   top1    KL      ppl")
        for x in sorted(fr.values(), key=lambda x: x["expert_gb"]):
            print(f"          {x['prune']:.3f}  {x['kept']:4d}  {x['format']:10s}  {x['expert_gb']:8.2f}  "
                  f"{x['top1']:.3f}  {x['kl']:.4f}  {x['ppl']:.3f}")
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
    ap.add_argument("--menu", default="int3_g64,int4_g128,int8_g128", help="formats the allocator may use")
    ap.add_argument("--bases", default="int4_g128", help="uniform formats whose size budgets are compared")
    ap.add_argument("--skip", default="", help="comma list of phases to skip: teacher,score,prune,quality")
    ap.add_argument("--redo", default="", help="comma list of phases to rerun even if measured")
    ap.add_argument("--explore", action="store_true",
                    help="more tests: calibrate breakdown, fp8 teacher, calibration size, prune x bits frontier")
    ap.add_argument("--prunes", default="0.25,0.375,0.5,0.625,0.75")
    ap.add_argument("--formats", default="bf16,int8_g128,int4_g128,int3_g64")
    ap.add_argument("--calib-sizes", default="32,64,128")
    ap.add_argument("--_teacher_ab", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--_teacher_variant", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args()

    if a._teacher_ab:
        teacher_ab(a.job, a.teacher_prompts)
        return
    if a._teacher_variant:
        teacher_variant(a.job, a._teacher_variant, a.teacher_prompts, 64)
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
        quality(job, [int(x) for x in a.tiers.split(",")], a.menu.split(","), a.bases.split(","))
    if a.explore:
        if not have("calib_breakdown"):
            calib_breakdown(job)
        py = os.environ.get("REAPER_VLLM_PY") or sys.executable
        for variant in ("bf16", "fp8"):
            if "teacher" not in skip and variant not in (load_results(job).get("teacher_precision") or {}):
                subprocess.call([py, str(Path(__file__).resolve()), "--_teacher_variant", variant, "--job",
                                 str(job.root), "--teacher-prompts", str(a.teacher_prompts)], env=env_for_children())
        if not have("calib_size"):
            calib_size(job, [int(x) for x in a.calib_sizes.split(",")])
        frontier(job, [float(x) for x in a.prunes.split(",")], a.formats.split(","))
    summary(job)


if __name__ == "__main__":
    main()
