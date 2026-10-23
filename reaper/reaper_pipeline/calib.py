"""Calibration prompts. REAP keeps the experts that fire on the calibration
tokens, so these decide what the pruned model stays good at.

With a purpose ("Debugging only") the teacher writes the prompts itself:
  1. it lists the sub-tasks someone with that purpose brings to an assistant,
  2. it writes varied, concrete user requests for each sub-task,
  3. a share (Config.purpose_mix) comes from the closest public preset, so the
     set is anchored in real user text, not only in the teacher's own style.
Without a purpose, Config.calib picks a preset:
  chat  HuggingFaceH4/ultrachat_200k      first user turn of multi-turn chats
  code  theblackcat102/evol-codealpaca-v1 coding instructions (what the REAP paper calibrated on)
  math  openai/gsm8k                      grade-school word problems
  mix   equal parts of the three
  builtin  the small set below (offline)
or a path to a .jsonl ({"prompt": ...} or {"messages": [...]}) or .txt (one per line).
Presets stream a slice of the dataset; offline, the built-in set is used and the run says so.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

PRESETS = {
    "chat": ("HuggingFaceH4/ultrachat_200k", None, "train_sft", lambda r: r["messages"][0]["content"]),
    "code": ("theblackcat102/evol-codealpaca-v1", None, "train", lambda r: r["instruction"]),
    "math": ("openai/gsm8k", "main", "train", lambda r: r["question"]),
}

KEYWORDS = {
    "code": ("code", "coding", "debug", "bug", "program", "python", "javascript", "typescript", "rust", "java",
             "c++", "golang", "sql", "software", "refactor", "api", "devops", "script", "compiler", "stack trace",
             "unit test", "frontend", "backend", "kernel", "regex"),
    "math": ("math", "algebra", "arithmetic", "calculus", "equation", "proof", "statistic", "geometry",
             "probability", "word problem"),
    "chat": ("chat", "email", "writing", "write", "story", "summar", "translat", "support", "customer",
             "essay", "assistant", "conversation", "marketing"),
}

FALLBACK = [
    "Explain how a hash map handles collisions, with a short Python example.",
    "Write a SQL query that returns the top 3 customers by total order value per country.",
    "What causes the seasons on Earth? Answer for a 12 year old.",
    "Summarise the main arguments for and against a four-day work week.",
    "A train leaves at 9:40 and arrives at 13:15. How long is the trip in minutes?",
    "Write a bash one-liner that finds the 10 largest files under the current directory.",
    "Translate into French: 'The meeting has been moved to Thursday afternoon.'",
    "Refactor this function to be iterative: def fact(n): return 1 if n < 2 else n * fact(n - 1)",
    "What is the difference between TCP and UDP? Give one use case for each.",
    "Draft a polite email asking a supplier for an updated delivery date.",
    "Solve for x: 3x + 7 = 2x - 5. Show your steps.",
    "Write a Rust function that reverses the words in a sentence.",
    "Give three tips for writing clear documentation for an API.",
    "Explain gradient descent in two paragraphs without equations.",
    "If a shirt costs $24 after a 20% discount, what was the original price?",
    "Write a haiku about debugging at 3am.",
    "Compare PostgreSQL and SQLite for a small web app.",
    "How do vaccines train the immune system?",
    "Implement binary search in C and explain its time complexity.",
    "My Python script raises KeyError: 'id' when reading a JSON list of users. Why, and how do I fix it?",
    "What are the pros and cons of microservices?",
    "Convert 72 degrees Fahrenheit to Celsius and explain the formula.",
    "Write a regular expression that matches ISO 8601 dates.",
    "Explain what a mixture-of-experts language model is.",
]

STYLES = [
    "short and casual, the way people type into a chat box",
    "long and detailed, pasting in the concrete material it is about (code, logs, text, numbers)",
    "from a beginner who is unsure of the right terms",
    "from an expert, terse and precise",
    "about a tricky edge case or something that went wrong",
    "a multi-part request with constraints on the answer",
]


# --------------------------------------------------------------------------- presets and files


def _stream(name: str, config: str | None, split: str, get, n: int, seed: int) -> list[str]:
    from datasets import load_dataset
    ds = load_dataset(name, config, split=split, streaming=True).shuffle(seed=seed, buffer_size=10_000)
    out = []
    for row in ds:
        try:
            p = get(row).strip()
        except (KeyError, IndexError, TypeError):
            continue
        if 20 <= len(p) <= 6000:
            out.append(p)
        if len(out) >= n:
            break
    return out


def _from_file(path: Path) -> list[str]:
    if path.suffix == ".txt":
        return [l.strip() for l in path.read_text().splitlines() if l.strip()]
    out = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if "prompt" in r:
            out.append(r["prompt"])
        elif "messages" in r:
            out.append(next(m["content"] for m in r["messages"] if m["role"] == "user"))
    return out


def _builtin(n: int) -> list[str]:
    return (FALLBACK * (-(-n // len(FALLBACK))))[:n]


def load_prompts(calib: str, n: int, seed: int = 0) -> tuple[list[str], str]:
    """n prompts and a note saying where they came from."""
    path = Path(calib).expanduser()
    if path.exists():
        ps = _from_file(path)
        random.Random(seed).shuffle(ps)
        return ps[:n], f"{len(ps[:n])} prompts from {path}"
    if calib == "builtin":
        return _builtin(n), f"built-in prompts ({len(FALLBACK)} unique)"
    names = list(PRESETS) if calib == "mix" else [calib]
    if any(c not in PRESETS for c in names):
        raise ValueError(f"unknown calibration set {calib!r}: use mix, builtin, {', '.join(PRESETS)} or a file path")
    out: list[str] = []
    try:
        per = -(-n // len(names))
        for c in names:
            out += _stream(*PRESETS[c], n=per, seed=seed)
        random.Random(seed).shuffle(out)
        return out[:n], f"{min(n, len(out))} prompts from {calib} ({', '.join(PRESETS[c][0] for c in names)})"
    except Exception as e:  # offline, dataset moved, datasets not installed
        return _builtin(n), f"built-in prompts ({len(FALLBACK)} unique) because {calib} could not load: {e}"


# --------------------------------------------------------------------------- purpose


def preset_for(purpose: str) -> str | None:
    """The public preset closest to a purpose, by keywords; None when nothing matches."""
    p = purpose.lower()
    hits = {name: sum(1 for k in kws if k in p) for name, kws in KEYWORDS.items()}
    best = max(hits, key=hits.get)
    return best if hits[best] else None


def parse_list(text: str) -> list[str]:
    """Strings from a JSON array in a model answer, else from a numbered/bulleted list."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    a, b = text.find("["), text.rfind("]")
    if a != -1 and b > a:
        try:
            items = json.loads(text[a:b + 1])
            out = [(i if isinstance(i, str) else json.dumps(i)).strip() for i in items]
            return [o for o in out if o]
        except json.JSONDecodeError:
            pass
    out = []
    for line in text.splitlines():
        m = re.match(r"^\s*(?:\d+[.)]|[-*•])\s+(.*\S)", line)
        if m:
            out.append(m[1].strip().strip('"'))
    return out


def _norm(s: str) -> str:
    return re.sub(r"\W+", " ", s.lower()).strip()


def topics_prompt(purpose: str, n: int) -> list[dict]:
    return [{"role": "user", "content": (
        f"A language model is being specialised for this purpose: \"{purpose}\".\n"
        f"List {n} distinct sub-tasks, situations or kinds of requests that users with this purpose "
        "bring to an AI assistant. Cover the whole range, from common to rare. Each item is a short phrase.\n"
        "Answer with only a JSON array of strings.")}]


def requests_prompt(purpose: str, topic: str, style: str, k: int) -> list[dict]:
    return [{"role": "user", "content": (
        f"Write {k} realistic messages that a user would send to an AI assistant used for: \"{purpose}\".\n"
        f"Focus: {topic}.\nStyle: {style}.\n"
        "Each message must be self-contained: include the actual code, error output, data or text it refers to, "
        "and vary the languages, tools, domains and difficulty. Do not answer them.\n"
        "Answer with only a JSON array of strings, one string per message.")}]


def purpose_prompts(teacher, purpose: str, n: int, seed: int = 0, n_topics: int = 24,
                    per_request: int = 6, on_progress=None, max_tokens: int = 4096) -> tuple[list[str], list[str]]:
    """(prompts written by the teacher for the purpose, the sub-tasks it listed)."""
    rng = random.Random(seed)
    topics: list[str] = []
    for _ in range(3):
        topics = [t for t in parse_list(teacher.chat([topics_prompt(purpose, n_topics)], max_tokens, 0.7)[0])
                  if len(t) < 200]
        if len(topics) >= 4:
            break
    topics = list(dict.fromkeys(topics))[:n_topics] or [purpose]

    seen, out = set(), []
    for rnd in range(6):
        need = n - len(out)
        if need <= 0:
            break
        jobs = max(1, -(-int(need * 1.3) // per_request))
        convs = [requests_prompt(purpose, topics[(rnd * jobs + i) % len(topics)], rng.choice(STYLES), per_request)
                 for i in range(jobs)]
        for text in teacher.chat(convs, max_tokens * 3 // 2, 1.0):
            for p in parse_list(text):
                key = _norm(p)
                if len(p) >= 15 and key and key not in seen:
                    seen.add(key)
                    out.append(p)
        if on_progress:
            on_progress(min(len(out), n), n)
    rng.shuffle(out)
    return out[:n], topics


def build_prompts(teacher, purpose: str, calib: str, n: int, mix: float = 0.25, seed: int = 0,
                  on_progress=None, max_tokens: int = 4096) -> tuple[list[dict], dict]:
    """Rows {"prompt", "source"} and a summary of where they came from."""
    if not purpose.strip():
        ps, note = load_prompts(calib, n, seed)
        return [{"prompt": p, "source": calib} for p in ps], {"note": note}

    preset = preset_for(purpose)
    n_preset = int(round(n * mix)) if preset else 0
    generated, topics = ([], [])
    if teacher is not None:
        generated, topics = purpose_prompts(teacher, purpose, n - n_preset, seed, on_progress=on_progress,
                                            max_tokens=max_tokens)
    if not generated:
        topics = []  # the teacher wrote nothing usable; do not report its fallback topic
    rows = [{"prompt": p, "source": "teacher"} for p in generated]
    fill_from = preset or calib
    short = n - len(rows)
    if short > 0:
        ps, pnote = load_prompts(fill_from, short, seed)
        rows += [{"prompt": p, "source": fill_from} for p in ps]
    else:
        pnote = ""
    random.Random(seed).shuffle(rows)
    note = (f"{len(generated)} prompts written by the teacher for \"{purpose}\" over {len(topics)} sub-tasks"
            + (f", {short} from {fill_from} ({pnote})" if short > 0 else ""))
    return rows, {"note": note, "topics": topics, "preset": preset, "generated": len(generated)}
