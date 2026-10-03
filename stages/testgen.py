"""Held-out test inputs written by Gemini instead of the teacher.

With CONDENSE_API_KEY set, the example inputs Gemini is shown are compressed
by condense.chat first (stages/condense.py compress()), so it sees twice as many.

With the teacher writing both the training inputs and the test inputs, the
test set shares the teacher's blind spots and phrasing. Here Gemini writes the
held-out inputs, the
teacher still answers them as the references, and none of them go into
training. The data stage calls held_out_inputs(); if it returns too few, the
stage falls back to splitting the held-out set off the teacher's own data.

On when Config.testgen_model is set (default gemini-3.8-flash) and
GEMINI_API_KEY is set. testgen_model "" turns it off.
"""

from __future__ import annotations

import random
from concurrent.futures import ThreadPoolExecutor

PER_CALL = 10
STYLES = ["short and casual", "long and detailed", "messy, with typos", "formal",
          "an unusual edge case", "ambiguous or missing information", "a different domain or locale"]


def enabled(cfg) -> bool:
    from stages.gemini import api_key

    return bool(getattr(cfg, "testgen_model", "")) and bool(api_key())


def held_out_inputs(spec, cfg, n: int, seen: set[str], norm_key, parse_array,
                    max_chars: int = 6000, seed: int = 1) -> list[str]:
    """Up to n new task inputs, deduplicated against `seen` (normalised keys)."""
    from stages import condense, gemini

    rng = random.Random(seed)
    # With condense on, Gemini sees twice as many examples, compressed, for about the same tokens.
    examples = [e.input for e in spec.seed_examples[:8 if condense.enabled() else 4]]
    examples = condense.compress(examples)
    shown = "\n\n".join(f"<input>\n{x}\n</input>" for x in examples)

    def ask(i: int) -> list[str]:
        prompt = (
            f"Task: {spec.description}\nInput format: {spec.input_format}\n\n"
            f"Example inputs:\n\n{shown}\n\n"
            f"Write exactly {PER_CALL} NEW, realistic test inputs for this task. Style: {STYLES[i % len(STYLES)]}. "
            "They will be used to test a model, so include hard and tricky cases, not only easy ones. "
            "Make them differ from the examples and from each other. Write only the inputs, not the answers. "
            "Return a JSON array of strings and nothing else.")
        try:
            return parse_array(gemini.chat(cfg.testgen_model, [{"role": "user", "content": prompt}],
                                           temperature=1.0, max_tokens=8192))
        except Exception as e:
            print(f"[testgen] Gemini call failed: {e}", flush=True)
            return []

    out: list[str] = []
    calls = -(-n // PER_CALL) + 2
    with ThreadPoolExecutor(8) as ex:
        batches = list(ex.map(ask, range(calls)))
    for batch in batches:
        for s in batch:
            k = norm_key(s)
            if k not in seen and len(s) <= max_chars:
                seen.add(k)
                out.append(s)
    rng.shuffle(out)
    print(f"[testgen] {len(out)} held-out inputs from {cfg.testgen_model} via {gemini.last_route or 'nothing'}", flush=True)
    return out[:n]
