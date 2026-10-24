"""LiveCodeBench code generation on a model folder: pass@1 with one sample per problem.

    python -m blackhole_pipeline.bench --model DIR --out OUTDIR [--release v6] [--max-tokens 4096]

Problems come from livecodebench/code_generation_lite (the test*.jsonl file of each release;
"v6" = the problems new in release v6, early 2025, the least likely to be in any training set),
or from --file (the same jsonl format). Prompts follow LCB's own code generation prompt; the last
```python block of the reply is the solution. Generation is batched on the GPU (Flint: REAPer's
cached fast_generate); the tests run in parallel subprocesses on every CPU, one process per
test with a timeout and memory cap, stopping at a problem's first failing test.
Writes OUTDIR/samples.jsonl (reply, code, verdict per problem) and OUTDIR/summary.json.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import pickle
import re
import resource
import subprocess
import sys
import tempfile
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = "livecodebench/code_generation_lite"
FILES = {"v1": "test.jsonl", "v2": "test2.jsonl", "v3": "test3.jsonl", "v4": "test4.jsonl", "v5": "test5.jsonl",
         "v6": "test6.jsonl"}
SYSTEM = ("You are an expert Python programmer. You will be given a question (problem specification) and will "
          "generate a correct Python program that matches the specification and passes all tests.")
WITH_STARTER = ("You will use the following starter code to write the solution to the problem and enclose your code "
                "within delimiters.")
NO_STARTER = ("Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on "
              "the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program "
              "runs, it reads the inputs, runs the algorithm and writes output to STDOUT.")
CODE_RE = re.compile(r"```(?:python|py|Python)?\s*\n(.*?)```", re.S)
TEST_TIMEOUT = 6.0


def emit(stage, pct, msg):
    if stage:
        print(json.dumps({"stage": stage, "status": "running", "pct": round(pct, 1), "msg": msg, "ts": time.time()}),
              flush=True)


# --------------------------------------------------------------------------- problems


def _tests(raw) -> list[dict]:
    if not raw:
        return []
    if isinstance(raw, list):
        return raw
    try:
        return json.loads(raw)
    except ValueError:
        return json.loads(pickle.loads(zlib.decompress(base64.b64decode(raw.encode("utf-8")))))


def load_problems(release: str = "v6", file: str | None = None, limit: int = 0) -> list[dict]:
    if file:
        path = Path(file).expanduser()
    else:
        from huggingface_hub import hf_hub_download
        names = [FILES[release]] if release in FILES else [FILES[f"v{i}"] for i in range(1, int(release[-1]) + 1)]
        rows = []
        for n in names:
            rows += [json.loads(l) for l in Path(hf_hub_download(REPO, n, repo_type="dataset")).read_text().splitlines()
                     if l.strip()]
        return _norm(rows, limit)
    return _norm([json.loads(l) for l in path.read_text().splitlines() if l.strip()], limit)


def _norm(rows: list[dict], limit: int) -> list[dict]:
    out = []
    for r in rows:
        meta = r.get("metadata") or {}
        meta = json.loads(meta) if isinstance(meta, str) else meta
        tests = _tests(r.get("public_test_cases")) + _tests(r.get("private_test_cases"))
        out.append({"id": r.get("question_id") or r.get("question_title"), "title": r.get("question_title", ""),
                    "question": r["question_content"], "starter": r.get("starter_code") or "",
                    "difficulty": r.get("difficulty", ""), "date": r.get("contest_date", ""),
                    "platform": r.get("platform", ""), "func": meta.get("func_name"), "tests": tests})
    out.sort(key=lambda p: (str(p["date"]), str(p["id"])))
    return out[:limit] if limit else out


def prompt(p: dict) -> list[dict]:
    q = f"### Question:\n{p['question']}\n\n"
    if p["starter"]:
        q += f"### Format: {WITH_STARTER}\n```python\n{p['starter']}\n```\n\n"
    else:
        q += f"### Format: {NO_STARTER}\n```python\n# YOUR CODE HERE\n```\n\n"
    q += "### Answer: (use the provided format with backticks)\n\n"
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}]


def extract_code(reply: str) -> str:
    reply = re.sub(r"<think>.*?</think>", "", reply, flags=re.S)
    blocks = CODE_RE.findall(reply)
    return blocks[-1] if blocks else ""


# --------------------------------------------------------------------------- tests


def _limits():
    resource.setrlimit(resource.RLIMIT_AS, (4 << 30, 4 << 30))
    resource.setrlimit(resource.RLIMIT_CPU, (int(TEST_TIMEOUT) + 2, int(TEST_TIMEOUT) + 2))
    os.setsid()


FUNC_DRIVER = """
import json, sys
sys.setrecursionlimit(100000)
from typing import *
from collections import *
from functools import *
from itertools import *
from heapq import *
from bisect import *
import math, string, re
exec(open(sys.argv[1]).read(), globals())
args = [json.loads(l) for l in open(sys.argv[2]).read().split("\\n") if l.strip()]
res = getattr(Solution(), sys.argv[3])(*args)
print(json.dumps(res))
"""
HEADER = "import sys\nsys.setrecursionlimit(100000)\nfrom typing import *\nfrom collections import *\nimport math\n"


def _same_stdout(got: str, want: str) -> bool:
    g = [l.strip() for l in got.strip().splitlines()]
    w = [l.strip() for l in want.strip().splitlines()]
    if g == w:
        return True
    if len(g) != len(w):
        return False
    for a, b in zip(g, w):
        if a == b:
            continue
        ta, tb = a.split(), b.split()
        if len(ta) != len(tb):
            return False
        for x, y in zip(ta, tb):
            if x == y:
                continue
            try:
                if abs(float(x) - float(y)) > 1e-6 * max(1.0, abs(float(y))):
                    return False
            except ValueError:
                return False
    return True


def _same_json(got: str, want: str) -> bool:
    try:
        return json.loads(got.strip()) == json.loads(want.strip())
    except ValueError:
        return got.strip() == want.strip()


def run_problem(p: dict, code: str) -> dict:
    """Run a solution against all of a problem's tests; stop at the first failure."""
    if not code.strip():
        return {"passed": False, "why": "no code block"}
    if not p["tests"]:
        return {"passed": False, "why": "no tests"}
    with tempfile.TemporaryDirectory() as d:
        sol = Path(d) / "sol.py"
        func = p["func"] if p["starter"] else None
        sol.write_text(code if func else HEADER + code)
        drv = Path(d) / "drv.py"
        drv.write_text(FUNC_DRIVER)
        for i, t in enumerate(p["tests"]):
            functional = func and t.get("testtype", "stdin") == "functional"
            try:
                if functional:
                    inp = Path(d) / "in.txt"
                    inp.write_text(t["input"])
                    r = subprocess.run([sys.executable, str(drv), str(sol), str(inp), func], capture_output=True,
                                       text=True, timeout=TEST_TIMEOUT, cwd=d, preexec_fn=_limits)
                else:
                    r = subprocess.run([sys.executable, str(sol)], input=t["input"], capture_output=True, text=True,
                                       timeout=TEST_TIMEOUT, cwd=d, preexec_fn=_limits)
            except subprocess.TimeoutExpired:
                return {"passed": False, "why": f"timeout on test {i + 1}/{len(p['tests'])}"}
            if r.returncode != 0:
                return {"passed": False, "why": f"error on test {i + 1}: {r.stderr.strip().splitlines()[-1:]}"}
            ok = _same_json(r.stdout, t["output"]) if functional else _same_stdout(r.stdout, t["output"])
            if not ok:
                return {"passed": False, "why": f"wrong answer on test {i + 1}/{len(p['tests'])}"}
    return {"passed": True, "why": f"{len(p['tests'])} tests"}


# --------------------------------------------------------------------------- generation


def fit_batch(model, batch: int, tokens: int) -> int:
    """Largest batch (up to `batch`) whose KV cache for `tokens` positions fits in 80% of free GPU
    memory. Sliding-window layers only keep their window."""
    import torch
    cfg = model.config
    if not torch.cuda.is_available() or not hasattr(cfg, "num_key_value_heads"):
        return batch
    torch.cuda.empty_cache()
    free = torch.cuda.mem_get_info()[0]
    types = getattr(cfg, "layer_types", None) or ["full_attention"] * cfg.num_hidden_layers
    win = getattr(cfg, "sliding_window", None) or tokens
    d = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    per_pos = 2 * cfg.num_key_value_heads * d * 2
    row = sum(per_pos * (min(win, tokens) if t == "sliding_attention" else tokens) for t in types)
    row += cfg.num_attention_heads * tokens * 4 * 2  # attention scores of one layer while decoding
    return max(1, min(batch, int(0.8 * free // row)))


def generate(model, tok, probs: list[dict], max_tokens: int, batch: int, stage: str | None, pct: tuple):
    import torch
    texts = [tok.apply_chat_template(prompt(p), tokenize=False, add_generation_prompt=True) for p in probs]
    gc = getattr(model, "generation_config", None)
    e = getattr(gc, "eos_token_id", None) if gc is not None else None
    eos = sorted({t for t in [tok.eos_token_id, *([e] if isinstance(e, int) else (e or []))] if t is not None})
    # the model's own sampling settings when its generation_config sets them, else 0.6 / 0.95
    sampled = gc is not None and getattr(gc, "do_sample", False)
    temp = float(gc.temperature) if sampled and gc.temperature else 0.6
    top_p = float(gc.top_p) if sampled and gc.top_p else 0.95
    # the model's own top_k if its generation_config sets one (not transformers' default 50), else 20
    top_k = int((gc.to_diff_dict() if gc is not None else {}).get("top_k") or 20)
    batch = fit_batch(model, batch, max(len(t) for t in texts) // 3 + max_tokens)
    print(f"bench: sampling at temperature {temp}, top_p {top_p}, top_k {top_k}, {batch} answers at a time, "
          f"up to {max_tokens} tokens each", flush=True)
    out = [""] * len(texts)
    t0, ntok, finished = time.time(), 0, 0

    def answer(r: list[int]) -> str:
        cut = next((k for k, t in enumerate(r) if t in eos), len(r))
        # the answer is what follows the reasoning (<think> can be a special token a plain decode drops)
        text = tok.decode(r[:cut], skip_special_tokens=False).rsplit("</think>", 1)[-1]
        for t in tok.all_special_tokens:
            text = text.replace(t, "")
        return text

    with torch.no_grad():
        if hasattr(model, "generate_many"):
            # continuous batching: a finished answer's slot takes the next problem at once
            # hard problems first: their long answers start early instead of forming the tail
            rank = {"hard": 0, "medium": 1, "easy": 2}
            order = sorted(range(len(texts)), key=lambda i: rank.get(probs[i].get("difficulty"), 1))
            enc = [tok(texts[i], add_special_tokens=False).input_ids for i in order]
            last = [0.0]

            def done(j, r):
                nonlocal ntok, finished
                out[order[j]] = answer(r)
                ntok += len(r)
                finished += 1
                if finished % 10 == 0 or finished == len(texts):
                    dt = time.time() - t0
                    print(f"bench: {finished}/{len(texts)} solutions written in {dt:.0f}s, {ntok:,} tokens, "
                          f"{ntok / dt:.0f} tok/s total", flush=True)

            def step(t, active):
                if time.time() - last[0] > 2:
                    last[0] = time.time()
                    emit(stage, pct[0] + (pct[1] - pct[0]) * finished / len(texts),
                         f"{finished}/{len(texts)} solutions written, {active} writing now, step {t}")
            model.generate_many(enc, max_tokens, temp, top_p, eos, slots=batch, on_step=step, on_done=done, top_k=top_k)
        else:
            tok.padding_side = "left"
            dev = model.get_input_embeddings().weight.device
            for b0 in range(0, len(texts), batch):
                idx = list(range(b0, min(b0 + batch, len(texts))))
                e = tok([texts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False)
                g = model.generate(**e.to(dev), max_new_tokens=max_tokens, do_sample=True, temperature=temp,
                                   top_p=top_p, top_k=top_k, pad_token_id=tok.pad_token_id)
                for i, r in zip(idx, g[:, e["input_ids"].shape[1]:].tolist()):
                    out[i] = answer(r)
                    ntok += next((k for k, t in enumerate(r) if t in eos), len(r))
                print(f"bench: {idx[-1] + 1}/{len(texts)} solutions written in {time.time() - t0:.0f}s", flush=True)
    return out, temp, ntok, batch


def run(model_dir: Path, out: Path, release: str, file: str | None, limit: int, max_tokens: int, batch: int,
        stage: str | None = None, pct: tuple = (0, 100), label: str = "") -> dict:
    from blackhole_pipeline import serve
    out.mkdir(parents=True, exist_ok=True)
    probs = load_problems(release, file, limit)
    if not probs:
        raise RuntimeError("no problems loaded")
    emit(stage, pct[0], f"{label}: loading {model_dir.name} for {len(probs)} problems")
    import torch
    model, tok = serve.load(model_dir, "cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    span = pct[1] - pct[0]
    replies, temp, ntok, batch = generate(model, tok, probs, max_tokens, batch, stage, (pct[0] + 0.03 * span, pct[0] + 0.9 * span))
    gen_s = time.time() - t0
    del model
    from reaper_pipeline.load import free
    free()
    emit(stage, pct[0] + 0.9 * span, f"{label}: running tests on {os.cpu_count()} CPUs")
    codes = [extract_code(r) for r in replies]
    with ThreadPoolExecutor(max(1, (os.cpu_count() or 4) - 1)) as ex:
        verdicts = list(ex.map(lambda a: run_problem(*a), zip(probs, codes)))
    rows = [{"id": p["id"], "title": p["title"], "difficulty": p["difficulty"], "date": p["date"],
             "passed": v["passed"], "why": v["why"], "code": c, "reply": r}
            for p, c, r, v in zip(probs, codes, replies, verdicts)]
    (out / "samples.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    by = {}
    for r in rows:
        d = by.setdefault(r["difficulty"] or "all", [0, 0])
        d[0] += r["passed"]
        d[1] += 1
    summary = {"model": str(model_dir), "bench": f"LiveCodeBench {release if not file else file}", "problems": len(rows),
               "pass@1": round(sum(r["passed"] for r in rows) / len(rows), 4),
               "by_difficulty": {k: {"passed": a, "total": n, "pass@1": round(a / n, 4)} for k, (a, n) in by.items()},
               "no_code": sum(1 for c in codes if not c.strip()), "max_tokens": max_tokens, "temperature": temp,
               "gen_s": round(gen_s, 1), "tokens": ntok, "tok_s": round(ntok / gen_s, 1),
               "batch": batch, "dates": [rows[0]["date"], rows[-1]["date"]]}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--release", default="v6")
    ap.add_argument("--file", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument("--batch", type=int, default=int(os.environ.get("BLACKHOLE_BENCH_BATCH", "256")))
    ap.add_argument("--stage", default=None, help="emit TUI progress events as this stage")
    ap.add_argument("--pct", default="0,100")
    ap.add_argument("--label", default="")
    a = ap.parse_args(argv)
    lo, hi = (float(x) for x in a.pct.split(","))
    model = Path(a.model).expanduser()
    if not model.exists():  # a Hugging Face id
        from huggingface_hub import snapshot_download
        model = Path(snapshot_download(a.model))
    s = run(model.resolve(), Path(a.out), a.release, a.file, a.limit, a.max_tokens, a.batch,
            a.stage, (lo, hi), a.label or Path(a.model).name)
    print(json.dumps(s), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
