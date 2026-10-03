"""Stage 1: teacher-generated task data.

Reads taskspec.json, asks the teacher (vLLM) for many new task inputs in the
style of the seeds, then has the teacher answer each one. Writes:
  data/train.jsonl    chat-format SFT data (also REAP calibration data)
  data/heldout.jsonl  held-out inputs with teacher reference answers
  data/calib.txt      plain text for llama.cpp imatrix
"""

from __future__ import annotations

import argparse
import json
import random
import re

from common.progress import emit
from stages._util import DRY_RUN, Job, write_jsonl

STAGE = "data"
BATCH = 20  # inputs requested per generation prompt


def system_prompt(spec) -> str:
    return (
        f"You are an expert at this task: {spec.description}\n"
        f"Input format: {spec.input_format}\nOutput format: {spec.output_format}\n"
        "Answer with the output only, no preamble."
    )


def gen_inputs_prompt(spec, rng: random.Random) -> list[dict]:
    shots = rng.sample(spec.seed_examples, k=min(5, len(spec.seed_examples)))
    shown = "\n\n".join(f"<input>\n{e.input}\n</input>" for e in shots)
    return [{
        "role": "user",
        "content": (
            f"Task: {spec.description}\nInput format: {spec.input_format}\n\n"
            f"Here are example inputs:\n\n{shown}\n\n"
            f"Write {BATCH} NEW, realistic and diverse inputs for this task. Vary length, "
            "tone, difficulty and edge cases. Do not copy the examples. "
            "Return a JSON array of strings and nothing else."
        ),
    }]


def answer_messages(spec, text: str) -> list[dict]:
    msgs = [{"role": "system", "content": system_prompt(spec)}]
    for e in spec.seed_examples[:4]:
        msgs += [{"role": "user", "content": e.input}, {"role": "assistant", "content": e.output}]
    msgs.append({"role": "user", "content": text})
    return msgs


def parse_array(text: str) -> list[str]:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    m = re.search(r"\[.*\]", text, flags=re.S)
    if not m:
        return []
    try:
        items = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    return [s.strip() for s in items if isinstance(s, str) and len(s.strip()) > 10]


def run_stage(job: Job) -> None:
    spec, cfg = job.spec, job.config
    rng = random.Random(0)
    total = cfg.n_generate + cfg.n_heldout

    if DRY_RUN:
        inputs = [e.input for e in spec.seed_examples] * (total // len(spec.seed_examples) + 1)
        answers = [e.output for e in spec.seed_examples] * (total // len(spec.seed_examples) + 1)
        inputs, answers = inputs[:total], answers[:total]
        emit(STAGE, pct=90, msg="dry run: reused seeds")
    else:
        from vllm import LLM, SamplingParams

        emit(STAGE, pct=1, msg=f"loading teacher {cfg.teacher}")
        llm = LLM(model=job.model_path(cfg.teacher), max_model_len=8192, gpu_memory_utilization=0.85)

        inputs: list[str] = []
        seen = {e.input for e in spec.seed_examples}
        rounds = 0
        while len(inputs) < total and rounds < 10:
            n_prompts = max(1, (total - len(inputs)) // BATCH + 5)
            convs = [gen_inputs_prompt(spec, rng) for _ in range(n_prompts)]
            outs = llm.chat(convs, SamplingParams(temperature=1.0, top_p=0.95, max_tokens=4096), use_tqdm=False)
            for o in outs:
                for s in parse_array(o.outputs[0].text):
                    if s not in seen:
                        seen.add(s)
                        inputs.append(s)
            rounds += 1
            emit(STAGE, pct=5 + 45 * min(1, len(inputs) / total), msg=f"{len(inputs)}/{total} inputs")
        inputs = inputs[:total]
        if len(inputs) < cfg.n_heldout * 2:
            raise RuntimeError(f"teacher produced only {len(inputs)} usable inputs")

        emit(STAGE, pct=55, msg="teacher answering")
        outs = llm.chat([answer_messages(spec, s) for s in inputs],
                        SamplingParams(temperature=0.3, max_tokens=2048), use_tqdm=False)
        answers = [re.sub(r"<think>.*?</think>", "", o.outputs[0].text, flags=re.S).strip() for o in outs]

    rows = [
        {"messages": [
            {"role": "system", "content": system_prompt(spec)},
            {"role": "user", "content": i},
            {"role": "assistant", "content": a},
        ]}
        for i, a in zip(inputs, answers) if a
    ]
    rng.shuffle(rows)
    n_held = min(cfg.n_heldout, len(rows) // 5)
    heldout, train = rows[:n_held], rows[n_held:]
    write_jsonl(job.path("data", "train.jsonl"), train)
    write_jsonl(job.path("data", "heldout.jsonl"), [
        {"input": r["messages"][1]["content"], "reference": r["messages"][2]["content"]} for r in heldout
    ])
    job.path("data", "calib.txt").write_text(
        "\n\n".join(r["messages"][1]["content"] + "\n" + r["messages"][2]["content"] for r in train[:1000])
    )
    job.mark_done(STAGE, {"train": len(train), "heldout": len(heldout)})
    emit(STAGE, "done", 100, f"{len(train)} train, {len(heldout)} held out")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    run_stage(Job(ap.parse_args().job))
