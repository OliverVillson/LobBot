"""/describe: turn an idea in plain words into a schema.

The real version sends the idea, the step vocabulary and the chosen model to a
frontier model (Claude or GPT, with the user's own key) and validates the schema
it writes. Until that is wired up, `draft_offline` builds one from keywords so
the flow can be tried end to end.
"""

from __future__ import annotations

import os
import re

from blackhole_tui.catalog import Schema, Step

PROVIDERS = {"claude": "ANTHROPIC_API_KEY", "gpt": "OPENAI_API_KEY"}


def provider() -> str | None:
    """The first frontier provider with a key in the environment."""
    return next((p for p, env in PROVIDERS.items() if os.environ.get(env)), None)


def draft_offline(idea: str) -> Schema:
    text = idea.lower()
    steps: list[Step] = []
    calib = "code" if re.search(r"\b(code|coding|debug|python|rust|sql)\b", text) else \
            "agent" if re.search(r"\b(tool|agent|function)\b", text) else "chat"
    if re.search(r"\b(docs?|documents?|wiki|manual|notes|pdfs?)\b", text):
        steps.append(Step("synth", "teacher writes Q&A from your documents", {"source": "./docs", "n": 20000}))
    if re.search(r"\b(persona|voice|character|style|tone|pirate)\b", text):
        steps.append(Step("synth", "teacher writes dialogues in that voice", {"n": 8000}))
    small = re.search(r"\b(laptop|tiny|small|phone|8 ?gb|16 ?gb)\b", text)
    prune = 0.62 if small else 0.5
    steps.append(Step("reap", f"drop {prune:.0%} of experts, {calib} calibration", {"prune": prune, "calib": calib}))
    steps.append(Step("heal", "LoRA heal from the unpruned teacher", {"tokens": "40M"}))
    if any(s.kind == "synth" for s in steps):
        steps.append(Step("sft", "LoRA on the generated data", {"lora_r": 64, "epochs": 2}))
    if re.search(r"\b(safe|polite|refus|align|prefer)\w*", text):
        steps.append(Step("dpo", "teacher-ranked preference pairs", {"pairs": 4000}))
    stop = {"a", "an", "the", "my", "that", "which", "for", "on", "from", "with", "to", "of", "and", "in",
            "runs", "run", "make", "want", "i", "it", "me", "model", "tiny", "small"}
    words = [w for w in re.findall(r"[a-z]+", text) if w not in stop][:2] or ["custom"]
    return Schema("my-" + "-".join(words), " ".join(words).capitalize(), "you (draft)",
                  f"Drafted from: “{idea.strip()}”", tuple(steps), quant="Q4_K" if small else "Q5_K",
                  tags=("draft",))
