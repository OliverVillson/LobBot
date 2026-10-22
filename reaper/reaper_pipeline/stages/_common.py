"""Shared bits of the stage processes."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from reaper_pipeline.events import emit
from reaper_pipeline.job import Job


def stage_main(run, stage: str) -> None:
    """`python -m reaper_pipeline.stages.<x> --job DIR` entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    job = Job(ap.parse_args().job)
    try:
        run(job)
    except Exception as e:
        traceback.print_exc()
        emit(stage, "error", msg=f"{type(e).__name__}: {e}"[:400])
        sys.exit(1)


def profile(job: Job) -> dict:
    p = job.read("plan.json")
    if p is None:
        raise RuntimeError("plan.json missing: run the plan stage first")
    return p


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    tmp.replace(path)


def _render(tok, conv: list[dict], gen_prompt: bool) -> str:
    if getattr(tok, "chat_template", None):
        return tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=gen_prompt)
    text = "\n\n".join(m["content"] for m in conv)
    return text + ("\n\n" if gen_prompt else "")


def encode(tok, prompt: str, answer: str | None, max_seq: int) -> tuple[list[int], int]:
    """Token ids of prompt (+ answer) in the chat format, and where the answer starts."""
    user = [{"role": "user", "content": prompt}]
    p_ids = tok(_render(tok, user, True), add_special_tokens=False)["input_ids"]
    if answer is None:
        return p_ids[:max_seq], min(len(p_ids), max_seq)
    try:
        full = tok(_render(tok, user + [{"role": "assistant", "content": answer}], False),
                   add_special_tokens=False)["input_ids"]
    except Exception:  # templates that reject a plain assistant turn
        full = p_ids + tok(answer, add_special_tokens=False)["input_ids"]
    start = 0
    while start < min(len(p_ids), len(full)) and p_ids[start] == full[start]:
        start += 1
    return full[:max_seq], min(start, max_seq)


def batches(seqs: list[list[int]], max_batch: int, max_tokens: int):
    """Indices grouped into batches of similar length (less padding)."""
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    cur: list[int] = []
    for i in order:
        longest = max([len(seqs[j]) for j in cur] + [len(seqs[i])])
        if cur and (len(cur) >= max_batch or longest * (len(cur) + 1) > max_tokens):
            yield cur
            cur = []
        cur.append(i)
    if cur:
        yield cur


def pad_batch(seqs: list[list[int]], pad_id: int, device):
    """Right-padded input_ids and attention_mask tensors."""
    import torch
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), L), dtype=torch.long)
    for r, s in enumerate(seqs):
        ids[r, :len(s)] = torch.tensor(s)
        mask[r, :len(s)] = 1
    return ids.to(device), mask.to(device)
