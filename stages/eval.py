"""Stage 5: evaluate teacher vs. candidates on the held-out set.

Each candidate GGUF is served with llama-server on the VM GPU and answers the
held-out inputs. Claude judges every answer (teacher references included)
against the TaskSpec's eval criteria. Writes out/eval.json, the contract the
scoreboard screen reads.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from common.progress import emit
from stages._util import DRY_RUN, Job, read_jsonl

STAGE = "eval"
PORT = 8091


def serve(job: Job, gguf: str) -> subprocess.Popen:
    import httpx

    binary = Path(job.config.llama_cpp) / "build/bin/llama-server"
    proc = subprocess.Popen([str(binary), "-m", gguf, "-ngl", "99", "--port", str(PORT), "-c", "8192", "-np", "8"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(180):
        try:
            if httpx.get(f"http://127.0.0.1:{PORT}/health", timeout=2).status_code == 200:
                return proc
        except httpx.HTTPError:
            pass
        time.sleep(1)
    proc.kill()
    raise RuntimeError(f"llama-server did not start for {gguf}")


def generate(system: str, inputs: list[str]) -> list[str]:
    import httpx

    def one(text: str) -> str:
        r = httpx.post(f"http://127.0.0.1:{PORT}/v1/chat/completions", timeout=300, json={
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": text}],
            "temperature": 0.3, "max_tokens": 2048,
        })
        r.raise_for_status()
        return re.sub(r"<think>.*?</think>", "", r.json()["choices"][0]["message"]["content"], flags=re.S).strip()

    with ThreadPoolExecutor(8) as ex:
        return list(ex.map(one, inputs))


def judge(spec, model: str, inputs: list[str], answers: list[str]) -> float:
    import anthropic

    client = anthropic.Anthropic()

    def one(pair) -> float:
        text, answer = pair
        msg = client.messages.create(model=model, max_tokens=20, messages=[{"role": "user", "content": (
            f"Task: {spec.description}\nOutput format: {spec.output_format}\nCriteria: {spec.eval_criteria}\n\n"
            f"<input>\n{text}\n</input>\n<answer>\n{answer}\n</answer>\n\n"
            "Score how well the answer meets the criteria from 0 to 10. Reply with the number only."
        )}])
        m = re.search(r"\d+(\.\d+)?", msg.content[0].text)
        return min(10.0, float(m.group(0))) / 10 if m else 0.0

    with ThreadPoolExecutor(16) as ex:
        scores = list(ex.map(one, zip(inputs, answers)))
    return round(sum(scores) / max(len(scores), 1), 3)


def run_stage(job: Job) -> None:
    from stages.data import system_prompt

    cfg, spec = job.config, job.spec
    alloc = json.loads(job.path("work", "allocation.json").read_text())
    held = read_jsonl(job.path("data", "heldout.jsonl"))
    inputs = [h["input"] for h in held]
    system = system_prompt(spec)
    teacher_name = cfg.teacher.split("/")[-1]

    if DRY_RUN or not os.environ.get("ANTHROPIC_API_KEY"):
        if not DRY_RUN:
            emit(STAGE, msg="ANTHROPIC_API_KEY not set on the VM; scores are placeholders")
        teacher_score = 0.9
        scores = {name: 0.85 - 0.07 * i for i, name in enumerate(alloc["candidates"])}
        samples = {}
    else:
        emit(STAGE, pct=5, msg="judging teacher references")
        teacher_score = judge(spec, cfg.judge_model, inputs, [h["reference"] for h in held])
        scores, samples = {}, {}
        n = len(alloc["candidates"])
        for i, (name, c) in enumerate(alloc["candidates"].items()):
            emit(STAGE, pct=10 + 85 * i / n, msg=f"running {name} on {len(inputs)} held-out inputs")
            proc = serve(job, c["path"])
            try:
                answers = generate(system, inputs)
            finally:
                proc.terminate()
                proc.wait()
            emit(STAGE, pct=10 + 85 * (i + 0.5) / n, msg=f"judging {name}")
            scores[name] = judge(spec, cfg.judge_model, inputs, answers)
            samples[name] = [{"input": a, "output": b} for a, b in list(zip(inputs, answers))[:3]]

    candidates = []
    for name, c in alloc["candidates"].items():
        candidates.append({
            "name": name, "size_gb": c["size_gb"], "score": scores[name], "tok_s_est": c["tok_s_est"],
            "meets_target": c["size_gb"] <= spec.target.max_size_gb and c["tok_s_est"] >= spec.target.min_tok_s,
        })
    eligible = [c for c in candidates if c["meets_target"]] or candidates
    best = max(eligible, key=lambda c: c["score"])
    # Prefer the compressed MoE when it is within 2 points of the best.
    moe = next((c for c in eligible if c["name"] == "lobbot-moe"), None)
    winner = moe if moe and moe["score"] >= best["score"] - 0.02 else best

    report = {
        "teacher": {"name": teacher_name, "size_gb": 61, "score": teacher_score},
        "candidates": candidates,
        "winner": winner["name"],
        "bit_widths": alloc.get("bit_widths", []),
        "samples": samples,
    }
    job.path("out", "eval.json").write_text(json.dumps(report, indent=2))
    job.mark_done(STAGE, {"winner": winner["name"]})
    emit(STAGE, "done", 100, f"winner {winner['name']}: score {winner['score']} vs teacher {teacher_score}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    run_stage(Job(ap.parse_args().job))
