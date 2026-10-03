"""Stage 5: evaluate teacher vs. candidates on the held-out set.

Each candidate GGUF is served with llama-server on the VM GPU and answers the
held-out inputs with the same system prompt the training data used. Every
answer is scored two ways:
  agreement  match with the teacher's reference answer (field-level for JSON
             outputs, token F1 otherwise). Needs no API key.
  judge      an LLM judge (Gemini by default, cfg.judge_model) scores each
             answer 0-10 against the TaskSpec criteria (teacher references
             included). Used as `score` when available.
Also records the decode speed measured on the VM. Writes out/eval.json, the
contract the scoreboard screen reads.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from common.progress import emit
from stages import data
from stages._util import DRY_RUN, Job, read_jsonl

STAGE = "eval"
PORT = int(os.environ.get("LOBBOT_EVAL_PORT", "8091"))
SLOTS = 8
# Must cover the longest teacher answer the data stage keeps
# (LOBBOT_DATA_ANSWER_MAX_TOKENS), or long correct answers are judged as truncated.
MAX_TOKENS = int(os.environ.get("LOBBOT_EVAL_MAX_TOKENS", max(2048, data.ANSWER_MAX_TOKENS)))
# Room per slot for the system prompt, an input of up to data.MAX_INPUT_CHARS
# (~2k tokens) and the answer.
CTX_PER_SLOT = int(os.environ.get("LOBBOT_EVAL_CTX", MAX_TOKENS + 4096))


def limits(cfg) -> tuple[int, int]:
    """(max answer tokens, context per slot) for this job: follows the job's
    data answer cap (Config data_answer_max_tokens) unless LOBBOT_EVAL_* is set."""
    max_tokens = int(os.environ.get("LOBBOT_EVAL_MAX_TOKENS", max(2048, data.answer_max_tokens(cfg))))
    return max_tokens, int(os.environ.get("LOBBOT_EVAL_CTX", max_tokens + 4096))


def serve(job: Job, gguf: str, name: str) -> subprocess.Popen:
    import httpx

    binary = Path(job.config.llama_cpp) / "build/bin/llama-server"
    log_path = job.path("work", f"llama-server-{name}.log")
    log = open(log_path, "w")
    proc = subprocess.Popen(
        [str(binary), "-m", gguf, "-ngl", "999", "--host", "127.0.0.1", "--port", str(PORT),
         "-c", str(SLOTS * limits(job.config)[1]), "-np", str(SLOTS), "--jinja"],
        stdout=log, stderr=subprocess.STDOUT)
    for _ in range(600):
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited for {name}:\n" + "\n".join(log_path.read_text().splitlines()[-20:]))
        try:
            if httpx.get(f"http://127.0.0.1:{PORT}/health", timeout=2).status_code == 200:
                return proc
        except httpx.HTTPError:
            pass
        time.sleep(1)
    proc.kill()
    raise RuntimeError(f"llama-server did not become healthy for {name}; see {log_path}")


def generate(system: str, inputs: list[str], max_tokens: int = MAX_TOKENS) -> tuple[list[str], float | None]:
    """Answers plus the median decode speed (tok/s) the server reported."""
    import httpx

    speeds: list[float] = []

    def one(text: str) -> str:
        for attempt in range(2):
            try:
                r = httpx.post(f"http://127.0.0.1:{PORT}/v1/chat/completions", timeout=600, json={
                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": text}],
                    "temperature": 0.0, "max_tokens": max_tokens,
                    "chat_template_kwargs": {"enable_thinking": False},
                })
                r.raise_for_status()
                body = r.json()
                tps = (body.get("timings") or {}).get("predicted_per_second")
                if tps:
                    speeds.append(float(tps))
                content = body["choices"][0]["message"].get("content") or ""
                return re.sub(r"<think>.*?</think>", "", content, flags=re.S).strip()
            except Exception as e:
                if attempt:
                    print(f"[eval] generation failed: {e}", flush=True)
        return ""

    with ThreadPoolExecutor(SLOTS) as ex:
        answers = list(ex.map(one, inputs))
    return answers, (round(statistics.median(speeds), 1) if speeds else None)


def _tokens(s: str) -> list[str]:
    return re.findall(r"\w+", s.lower())


def _f1(a: str, b: str) -> float:
    ta, tb = Counter(_tokens(a)), Counter(_tokens(b))
    common = sum((ta & tb).values())
    if not ta or not tb:
        return float(ta == tb)
    if common == 0:
        return 0.0
    p, r = common / sum(ta.values()), common / sum(tb.values())
    return 2 * p * r / (p + r)


def _as_json(s: str):
    s = s.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", s, flags=re.S)
    if m:
        s = m.group(1).strip()
    try:
        return json.loads(s)
    except (json.JSONDecodeError, ValueError):
        return None


def agreement(answer: str, reference: str) -> float:
    """1.0 means the answer matches the teacher. For JSON objects: short fields
    must match exactly, long text fields score by token F1; invalid JSON is 0."""
    ref = _as_json(reference)
    if isinstance(ref, dict) and ref:
        ans = _as_json(answer)
        if not isinstance(ans, dict):
            return 0.0
        scores = []
        for k, v in ref.items():
            a = ans.get(k)
            if isinstance(v, str) and len(v.split()) > 3:
                scores.append(_f1(str(a or ""), v))
            else:
                scores.append(float(str(a).strip().lower() == str(v).strip().lower()))
        return sum(scores) / len(scores)
    return _f1(answer, reference)


def judge_key(model: str) -> str | None:
    """The API key the judge model needs, or None if it is not set."""
    if model.startswith("gemini"):
        from stages.gemini import api_key

        return api_key()
    return os.environ.get("ANTHROPIC_API_KEY")


def judge(spec, model: str, inputs: list[str], answers: list[str]) -> float | None:
    """Mean judge score in [0, 1]; None if no answer could be judged. gemini-*
    models go to the Gemini API, claude-* to Anthropic; both through
    condense.chat when CONDENSE_API_KEY is set (stages/gemini.py, stages/condense.py)."""
    if model.startswith("gemini"):
        from stages import gemini

        ask = lambda prompt: gemini.chat(model, [{"role": "user", "content": prompt}])
    else:
        from stages.condense import anthropic_client

        client = anthropic_client(max_retries=6)
        ask = lambda prompt: client.messages.create(
            model=model, max_tokens=20, messages=[{"role": "user", "content": prompt}]).content[0].text

    broken = threading.Event()  # a 4xx (bad key or model id) will not fix itself: stop calling

    def one(pair) -> float | None:
        text, answer = pair
        if broken.is_set():
            return None
        try:
            reply = ask(
                f"Task: {spec.description}\nOutput format: {spec.output_format}\nCriteria: {spec.eval_criteria}\n\n"
                f"<input>\n{text}\n</input>\n<answer>\n{answer}\n</answer>\n\n"
                "Score how well the answer meets the criteria from 0 to 10. Reply with the number only.")
        except Exception as e:
            status = getattr(getattr(e, "response", None), "status_code", None) or getattr(e, "status_code", None)
            if isinstance(status, int) and 400 <= status < 500 and status != 429 and not broken.is_set():
                broken.set()
                print(f"[eval] judge disabled after HTTP {status}: {e}", flush=True)
            elif not broken.is_set():
                print(f"[eval] judge call failed: {e}", flush=True)
            return None
        m = re.search(r"\d+(\.\d+)?", reply)
        return min(10.0, float(m.group(0))) / 10 if m else None

    with ThreadPoolExecutor(16) as ex:
        scores = [s for s in ex.map(one, zip(inputs, answers)) if s is not None]
    return round(sum(scores) / len(scores), 3) if scores else None


def teacher_size_gb(job: Job) -> float:
    d = Path(job.model_path(job.config.teacher))
    if d.is_dir():
        n = sum(f.stat().st_size for f in d.glob("*.safetensors"))
        if n:
            return round(n / 1e9, 1)
    return 61.0  # Qwen3-30B-A3B in bf16


def run_stage(job: Job) -> None:
    from stages.data import system_prompt

    cfg, spec = job.config, job.spec
    alloc = json.loads(job.path("work", "allocation.json").read_text())
    held = read_jsonl(job.path("data", "heldout.jsonl"))
    inputs = [h["input"] for h in held]
    refs = [h["reference"] for h in held]
    system = system_prompt(spec)
    teacher_name = cfg.teacher.split("/")[-1]
    use_judge = bool(judge_key(cfg.judge_model)) and not DRY_RUN
    if not use_judge and not DRY_RUN:
        need = "GEMINI_API_KEY" if cfg.judge_model.startswith("gemini") else "ANTHROPIC_API_KEY"
        emit(STAGE, msg=f"{need} not set on the VM; scoring by agreement with the teacher")

    results: dict[str, dict] = {}
    teacher_judge = None
    if DRY_RUN:
        for i, name in enumerate(alloc["candidates"]):
            results[name] = {"agreement": 0.85 - 0.07 * i, "judge": None, "tok_s_vm": None, "samples": []}
    else:
        if use_judge:
            emit(STAGE, pct=5, msg=f"judging teacher references on {len(inputs)} held-out inputs")
            teacher_judge = judge(spec, cfg.judge_model, inputs, refs)
        n = len(alloc["candidates"])
        for i, (name, c) in enumerate(alloc["candidates"].items()):
            emit(STAGE, pct=10 + 85 * i / n, msg=f"running {name} on {len(inputs)} held-out inputs")
            proc = serve(job, c["path"], name)
            try:
                answers, tps = generate(system, inputs, limits(cfg)[0])
            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
            agree = round(sum(agreement(a, r) for a, r in zip(answers, refs)) / max(len(refs), 1), 3)
            emit(STAGE, pct=10 + 85 * (i + 0.5) / n, msg=f"{name}: {agree:.0%} agreement with teacher, {tps} tok/s on the VM")
            score = judge(spec, cfg.judge_model, inputs, answers) if use_judge else None
            results[name] = {
                "agreement": agree, "judge": score, "tok_s_vm": tps,
                "samples": [{"input": a, "output": b, "reference": r} for a, b, r in list(zip(inputs, answers, refs))[:3]],
            }

    # Judge score when every model has one, else agreement (teacher = 1.0 by definition).
    judged = teacher_judge is not None and all(r["judge"] is not None for r in results.values())
    method = "judge" if judged else "agreement"
    teacher_score = teacher_judge if judged else 1.0

    candidates = []
    for name, c in alloc["candidates"].items():
        r = results[name]
        candidates.append({
            "name": name, "size_gb": c["size_gb"], "score": r["judge"] if judged else r["agreement"],
            "agreement": r["agreement"], "judge_score": r["judge"],
            "tok_s_est": c["tok_s_est"], "tok_s_vm": r["tok_s_vm"],
            "meets_target": c["size_gb"] <= spec.target.max_size_gb and c["tok_s_est"] >= spec.target.min_tok_s,
        })
    eligible = [c for c in candidates if c["meets_target"]] or candidates
    best = max(eligible, key=lambda c: c["score"])
    # Prefer the compressed MoE when it is within 2 points of the best.
    moe = next((c for c in eligible if c["name"] == "lobbot-moe"), None)
    winner = moe if moe and moe["score"] >= best["score"] - 0.02 else best

    report = {
        "teacher": {"name": teacher_name, "size_gb": teacher_size_gb(job), "score": teacher_score},
        "candidates": candidates,
        "winner": winner["name"],
        "score_method": method,
        "bit_widths": alloc.get("bit_widths", []),
        "samples": {k: v["samples"] for k, v in results.items()},
    }
    job.path("out", "eval.json").write_text(json.dumps(report, indent=2))
    job.mark_done(STAGE, {"winner": winner["name"]})
    emit(STAGE, "done", 100, f"winner {winner['name']}: score {winner['score']} vs teacher {teacher_score} ({method})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    run_stage(Job(ap.parse_args().job))
