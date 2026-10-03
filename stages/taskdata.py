"""Task data helpers for the REAP and heal stages: read <job>/data/train.jsonl
into chat messages and tokenize them with the prompt masked out of the loss."""
from __future__ import annotations

import json
from pathlib import Path


def load_examples(path: Path) -> list[dict]:
    """Read a jsonl file into a list of {"messages": [...]} where the
    last message is the assistant target. Accepts prompt/response, input/output,
    instruction/output and messages rows."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"task data not found: {path}")
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            msgs = to_messages(row)
            if msgs:
                out.append({"messages": msgs})
    if not out:
        raise ValueError(f"no usable examples in {path}")
    return out


def to_messages(row: dict) -> list[dict] | None:
    if "messages" in row:
        msgs = [m for m in row["messages"] if m.get("content")]
        if len(msgs) >= 2 and msgs[-1]["role"] == "assistant":
            return msgs
        return None
    for pk, rk in (("prompt", "response"), ("input", "output"),
                   ("instruction", "output"), ("question", "answer")):
        if pk in row and rk in row:
            msgs = []
            if row.get("system"):
                msgs.append({"role": "system", "content": row["system"]})
            msgs.append({"role": "user", "content": str(row[pk])})
            msgs.append({"role": "assistant", "content": str(row[rk])})
            return msgs
    return None


def _template(tok, msgs, **kw):
    try:
        return tok.apply_chat_template(msgs, tokenize=False, enable_thinking=False, **kw)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, **kw)


def tokenize_example(tok, msgs: list[dict], max_len: int) -> dict:
    """input_ids plus labels with the prompt masked to -100, so loss is only on
    the teacher's answer."""
    if tok.chat_template:
        full = _template(tok, msgs)
        prompt = _template(tok, msgs[:-1], add_generation_prompt=True)
    else:  # bare tokenizer (tests): plain concatenation
        prompt = "".join(m["content"] + "\n" for m in msgs[:-1])
        full = prompt + msgs[-1]["content"] + (tok.eos_token or "")
    ids = tok(full, add_special_tokens=False)["input_ids"]
    if full.startswith(prompt):
        n_prompt = len(tok(prompt, add_special_tokens=False)["input_ids"])
    else:
        n_prompt = 0
    # Qwen templates end with "<|im_end|>\n"; keep im_end as a target, drop the newline
    if tok.chat_template and full.endswith("\n"):
        ids = ids[:-1]
    ids = ids[:max_len]
    labels = [-100] * min(n_prompt, len(ids)) + ids[n_prompt:]
    return {"input_ids": ids, "labels": labels[: len(ids)]}
