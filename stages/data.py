"""Stage 1: teacher-generated task data.

Reads taskspec.json and uses the teacher (vLLM) to:
  1. brainstorm scenarios the task's inputs come from (diversity),
  2. write new task inputs per scenario + style hint, deduplicated,
  3. answer every input (the distillation targets),
then drops bad answers (empty, truncated, invalid JSON when the output format
is JSON) and splits off a held-out set. Writes:
  data/train.jsonl    chat-format SFT data (also REAP calibration data); seeds included
  data/heldout.jsonl  held-out inputs with teacher reference answers (inputs written
                      by Gemini when Config.testgen_model and GEMINI_API_KEY are set)
  data/calib.txt      plain text for llama.cpp imatrix
  data/stats.json     counts, drop reasons, timings
  work/data_scenarios.json, work/data_inputs.jsonl   resume cache for a crashed run

Knobs (env vars, defaults in brackets): LOBBOT_DATA_SCENARIOS [40],
LOBBOT_DATA_PER_PROMPT [8], LOBBOT_DATA_MAX_LEN [8192], LOBBOT_DATA_GPU_UTIL [0.85],
LOBBOT_DATA_TP [1], LOBBOT_DATA_BATCH [256]. Sizes come from Config
(n_generate, n_heldout).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from dataclasses import dataclass

from common.progress import emit
from stages import testgen
from stages._util import DRY_RUN, Job, read_jsonl, write_jsonl

STAGE = "data"
N_SCENARIOS = int(os.environ.get("LOBBOT_DATA_SCENARIOS", 40))
PER_PROMPT = int(os.environ.get("LOBBOT_DATA_PER_PROMPT", 8))
MAX_MODEL_LEN = int(os.environ.get("LOBBOT_DATA_MAX_LEN", 8192))
GPU_UTIL = float(os.environ.get("LOBBOT_DATA_GPU_UTIL", 0.85))
TP = int(os.environ.get("LOBBOT_DATA_TP", 1))
BATCH = int(os.environ.get("LOBBOT_DATA_BATCH", 256))
# FlashInfer's B200 decode kernels are JIT-built at startup and need ninja plus a matching nvcc.
# FLASH_ATTN ships prebuilt. Empty string lets vLLM choose.
ATTN_BACKEND = os.environ.get("LOBBOT_DATA_ATTN_BACKEND", "FLASH_ATTN")
# FlashInfer's top-p/top-k sampler is JIT-built the same way during warmup.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
OVERSHOOT = 1.3      # extra inputs requested to cover dedupe and answer filtering
MAX_ROUNDS = 6
MAX_INPUT_CHARS = 6000  # keeps every answer prompt well inside MAX_MODEL_LEN
FEWSHOT = 6          # seed examples shown to the teacher when answering
SEED_REPEAT = 2      # human-written seeds appear this many times in train
ANSWER_MAX_TOKENS = 1536

STYLE_HINTS = [
    "short and terse", "long and detailed", "messy, with typos and informal language",
    "an unusual edge case", "ambiguous or tricky to handle", "from an expert user",
    "from a first-time user", "with irrelevant extra details", "emotional, in a hurry",
    "very polite and formal", "full of numbers, dates or identifiers",
    "mixing several requests or topics",
]


# --------------------------------------------------------------------------- prompts

def system_prompt(spec) -> str:
    """The task prompt the student is trained with. package.py puts it in the Modelfile."""
    return (
        f"You are an expert at this task: {spec.description}\n"
        f"Input format: {spec.input_format}\nOutput format: {spec.output_format}\n"
        "Answer with the output only, no preamble."
    )


def scenarios_prompt(spec, n: int) -> list[dict]:
    shown = "\n".join(f"- {_trunc(e.input, 300)}" for e in spec.seed_examples[:8])
    return [{"role": "user", "content": (
        f"Task: {spec.description}\nInput format: {spec.input_format}\n"
        f"Example inputs:\n{shown}\n\n"
        f"List {n} distinct real-world scenarios, sub-topics or situations that inputs for this task "
        "come from. Cover common cases, rare cases and edge cases. Each item is a short phrase. "
        "Return a JSON array of strings and nothing else."
    )}]


def gen_inputs_prompt(spec, rng: random.Random, scenario: str, style: str) -> list[dict]:
    shots = rng.sample(spec.seed_examples, k=min(3, len(spec.seed_examples)))
    shown = "\n\n".join(f"<input>\n{e.input}\n</input>" for e in shots)
    return [{"role": "user", "content": (
        f"Task: {spec.description}\nInput format: {spec.input_format}\n\n"
        f"Here are example inputs:\n\n{shown}\n\n"
        f"Write exactly {PER_PROMPT} NEW, realistic inputs for this task.\n"
        f"Scenario: {scenario}\nStyle: {style}\n"
        "Make them differ from the examples and from each other (length, wording, names, details). "
        "Write only the inputs, not the answers. Return a JSON array of strings and nothing else."
    )}]


def answer_messages(spec, shots, text: str) -> list[dict]:
    sys = system_prompt(spec)
    if spec.eval_criteria:
        sys += f"\nA good answer satisfies: {spec.eval_criteria}"
    msgs = [{"role": "system", "content": sys}]
    for e in shots:
        msgs += [{"role": "user", "content": e.input}, {"role": "assistant", "content": e.output}]
    msgs.append({"role": "user", "content": text})
    return msgs


# --------------------------------------------------------------------------- parsing & filters

_THINK = re.compile(r"<think>.*?</think>\s*", re.S)
_FENCE = re.compile(r"^```[\w-]*\s*\n?|\n?```\s*$")


def clean(text: str) -> str:
    return _FENCE.sub("", _THINK.sub("", text).strip()).strip()


def parse_array(text: str) -> list[str]:
    text = clean(text)
    a, b = text.find("["), text.rfind("]")
    if a < 0 or b <= a:
        return []
    try:
        items = json.loads(text[a:b + 1])
    except json.JSONDecodeError:
        return []
    out = []
    for it in items if isinstance(items, list) else []:
        if isinstance(it, dict):  # [{"input": "..."}] style
            it = it.get("input") or it.get("text") or json.dumps(it, ensure_ascii=False)
        if isinstance(it, str) and len(it.strip()) > 10:
            out.append(it.strip())
    return out


def expects_json(spec) -> bool:
    return "json" in f"{spec.output_format} {spec.eval_criteria}".lower()


def check_answer(text: str, finished: bool, want_json: bool) -> tuple[str | None, str]:
    """(cleaned answer, "") or (None, drop reason)."""
    if not finished:
        return None, "truncated"
    ans = clean(text)
    if not ans:
        return None, "empty"
    if want_json:
        try:
            json.loads(ans)
        except json.JSONDecodeError:
            m = re.search(r"(\{.*\}|\[.*\])", ans, re.S)  # salvage "Here it is: {...}"
            try:
                json.loads(m.group(1)) if m else None
            except json.JSONDecodeError:
                m = None
            if not m:
                return None, "invalid_json"
            ans = m.group(1)
    return ans, ""


def norm_key(s: str) -> str:
    return re.sub(r"\W+", " ", s.lower()).strip()[:160]


def _trunc(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n] + "..."


# --------------------------------------------------------------------------- teacher

@dataclass
class Gen:
    text: str
    finished: bool  # False if it hit max_tokens


class Teacher:
    def __init__(self, model_path: str):
        from vllm import LLM, SamplingParams

        self.SamplingParams = SamplingParams
        self.llm = LLM(model=model_path, max_model_len=MAX_MODEL_LEN, gpu_memory_utilization=GPU_UTIL,
                       tensor_parallel_size=TP, seed=0,
                       **({"attention_backend": ATTN_BACKEND} if ATTN_BACKEND else {}))

    def chat(self, convs: list[list[dict]], temperature: float, max_tokens: int) -> list[Gen]:
        sp = self.SamplingParams(temperature=temperature, top_p=0.95 if temperature > 0.5 else 0.9,
                                 max_tokens=max_tokens)
        try:  # Qwen3 hybrid checkpoints: no <think>; Instruct-2507 ignores the flag
            res = self.llm.chat(convs, sp, use_tqdm=False, chat_template_kwargs={"enable_thinking": False})
        except TypeError:
            res = self.llm.chat(convs, sp, use_tqdm=False)
        return [Gen(r.outputs[0].text, r.outputs[0].finish_reason != "length") for r in res]


def chat_batched(teacher, convs, temperature, max_tokens, lo, hi, label) -> list[Gen]:
    out: list[Gen] = []
    for i in range(0, len(convs), BATCH):
        out += teacher.chat(convs[i:i + BATCH], temperature, max_tokens)
        emit(STAGE, pct=lo + (hi - lo) * len(out) / len(convs), msg=f"{label} {len(out)}/{len(convs)}")
    return out


class DryRunTeacher:
    """No model: seed variations and seed answers, a few broken, for CPU tests."""

    def __init__(self, spec, rng: random.Random):
        self.spec, self.rng = spec, rng

    def chat(self, convs, temperature, max_tokens):
        out = []
        for conv in convs:
            prompt = conv[-1]["content"]
            if "real-world scenarios" in prompt:
                out.append(Gen(json.dumps([f"scenario number {i}" for i in range(N_SCENARIOS)]), True))
            elif "NEW, realistic inputs" in prompt:
                base = self.rng.choice(self.spec.seed_examples).input
                out.append(Gen("```json\n" + json.dumps(
                    [f"{base} (case {self.rng.randrange(10**9)})" for _ in range(PER_PROMPT)]) + "\n```", True))
            else:
                r = self.rng.random()
                ex = self.spec.seed_examples[len(prompt) % len(self.spec.seed_examples)].output
                out.append(Gen("{broken" if r < 0.03 else ex, r > 0.01))
        return out


# --------------------------------------------------------------------------- stage

def run_stage(job: Job) -> None:
    spec, cfg = job.spec, job.config
    rng = random.Random(0)
    t0 = time.monotonic()
    n_held = cfg.n_heldout
    target = int((cfg.n_generate + n_held) * OVERSHOOT)
    want_json = expects_json(spec)
    stats: dict = {"seeds": len(spec.seed_examples), "expects_json": want_json, "dry_run": DRY_RUN}

    if DRY_RUN:
        teacher = DryRunTeacher(spec, rng)
    else:
        model = job.model_path(cfg.teacher)
        emit(STAGE, pct=1, msg=f"loading teacher {model}")
        teacher = Teacher(model)
    stats["load_s"] = round(time.monotonic() - t0, 1)
    emit(STAGE, pct=10, msg="teacher loaded")

    # 1) scenarios (cached for resume)
    scen_path = job.path("work", "data_scenarios.json")
    if scen_path.exists():
        scenarios = json.loads(scen_path.read_text())
    else:
        g = teacher.chat([scenarios_prompt(spec, N_SCENARIOS)], 0.8, 2048)[0]
        scenarios = parse_array(g.text)[:N_SCENARIOS] or ["a typical everyday case"]
        scen_path.write_text(json.dumps(scenarios, indent=1))
    print(f"[data] {len(scenarios)} scenarios: {scenarios[:5]}", flush=True)
    emit(STAGE, pct=12, msg=f"{len(scenarios)} scenarios")

    # 2) inputs (cached for resume)
    inputs_path = job.path("work", "data_inputs.jsonl")
    seen = {norm_key(e.input) for e in spec.seed_examples}
    inputs: list[str] = []
    for r in read_jsonl(inputs_path) if inputs_path.exists() else []:
        if norm_key(r["input"]) not in seen:
            seen.add(norm_key(r["input"]))
            inputs.append(r["input"])
    rounds = 0
    while len(inputs) < target and rounds < MAX_ROUNDS:
        rounds += 1
        n_prompts = -(-(target - len(inputs)) // PER_PROMPT) + 2
        convs = [gen_inputs_prompt(spec, rng, rng.choice(scenarios), rng.choice(STYLE_HINTS))
                 for _ in range(n_prompts)]
        lo = 12 + 33 * len(inputs) / target
        gens = chat_batched(teacher, convs, 1.0, 4096, lo, 45, f"writing inputs (round {rounds})")
        added = 0
        for g in gens:
            for s in parse_array(g.text):
                k = norm_key(s)
                if k not in seen and len(s) <= MAX_INPUT_CHARS:
                    seen.add(k)
                    inputs.append(s)
                    added += 1
        write_jsonl(inputs_path, [{"input": s} for s in inputs])
        print(f"[data] round {rounds}: +{added} unique inputs, {len(inputs)}/{target}", flush=True)
        if added == 0:
            break
    inputs = inputs[:target]
    stats.update(input_rounds=rounds, inputs=len(inputs))
    if len(inputs) < 2 * n_held:
        raise RuntimeError(f"teacher produced only {len(inputs)} usable inputs (need {2 * n_held}+); "
                           "check the TaskSpec seeds and the teacher model")
    emit(STAGE, pct=45, msg=f"{len(inputs)} unique inputs")

    # 2b) optional: Gemini writes the held-out test inputs (stages/testgen.py); the teacher answers them
    ext: list[str] = []
    testgen_path = job.path("work", "data_testgen.jsonl")  # cached so a rerun keeps the same test set
    if testgen_path.exists():
        ext = [r["input"] for r in read_jsonl(testgen_path) if norm_key(r["input"]) not in seen]
        print(f"[data] {len(ext)} held-out inputs from cache {testgen_path.name}", flush=True)
    elif not DRY_RUN and testgen.enabled(cfg):
        emit(STAGE, pct=45, msg=f"{cfg.testgen_model} writing {n_held} held-out test inputs")
        ext = testgen.held_out_inputs(spec, cfg, n_held, seen, norm_key, parse_array, MAX_INPUT_CHARS)
        if ext:
            write_jsonl(testgen_path, [{"input": s} for s in ext])
        emit(STAGE, pct=46, msg=f"{len(ext)} held-out test inputs from {cfg.testgen_model}")

    # 3) answers
    shots = rng.sample(spec.seed_examples, k=min(FEWSHOT, len(spec.seed_examples)))
    convs = [answer_messages(spec, shots, s) for s in inputs + ext]
    gens = chat_batched(teacher, convs, 0.3, ANSWER_MAX_TOKENS, 45, 95, "teacher answering")
    sys = system_prompt(spec)
    rows, ext_rows, drops, ext_drops = [], [], {}, {}
    for i, (s, g) in enumerate(zip(inputs + ext, gens)):
        ans, why = check_answer(g.text, g.finished, want_json)
        is_ext = i >= len(inputs)
        if ans is None:
            d = ext_drops if is_ext else drops
            d[why] = d.get(why, 0) + 1
        else:
            (ext_rows if is_ext else rows).append(_row(sys, s, ans))
    stats.update(answered=len(rows), dropped=drops)
    if ext:
        stats.update(heldout_answered=len(ext_rows), heldout_dropped=ext_drops)
    print(f"[data] kept {len(rows)}/{len(inputs)} answers, dropped {drops}", flush=True)
    if len(rows) < 2 * n_held:
        raise RuntimeError(f"only {len(rows)} usable answers after filtering ({drops})")

    # 4) split: held-out from teacher rows only; human seeds always go to train
    rng.shuffle(rows)
    if ext_rows and len(ext_rows) >= n_held // 2:  # Gemini-written tests; all teacher rows can train
        heldout, train = ext_rows[:n_held], rows[:cfg.n_generate]
        stats["heldout_source"] = cfg.testgen_model
    else:
        n_held = min(n_held, len(rows) // 5)
        heldout, train = rows[:n_held], rows[n_held:n_held + cfg.n_generate]
        stats["heldout_source"] = "teacher"
    train += [_row(sys, e.input, e.output) for e in spec.seed_examples] * SEED_REPEAT
    rng.shuffle(train)
    write_jsonl(job.path("data", "train.jsonl"), train)
    write_jsonl(job.path("data", "heldout.jsonl"), [
        {"input": r["messages"][1]["content"], "reference": r["messages"][2]["content"]} for r in heldout
    ])
    job.path("data", "calib.txt").write_text(
        "\n\n".join(r["messages"][1]["content"] + "\n" + r["messages"][2]["content"] for r in train[:1000])
    )
    stats.update(train=len(train), heldout=len(heldout), elapsed_s=round(time.monotonic() - t0, 1))
    job.path("data", "stats.json").write_text(json.dumps(stats, indent=2))
    print(f"[data] stats {json.dumps(stats)}", flush=True)
    job.mark_done(STAGE, {"train": len(train), "heldout": len(heldout)})
    emit(STAGE, "done", 100, f"{len(train)} train, {len(heldout)} held out")


def _row(sys: str, inp: str, out: str) -> dict:
    return {"messages": [
        {"role": "system", "content": sys},
        {"role": "user", "content": inp},
        {"role": "assistant", "content": out},
    ]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    run_stage(Job(ap.parse_args().job))
