"""Time generation speed: `python -m blackhole_pipeline.tokps [MODEL] --batch 1,32,128`.

Loads MODEL (default: the newest job's int4 model) the way blackhole-serve does and, for each
batch size, decodes a fixed number of tokens from the same chat prompt, printing per-sequence
and aggregate tokens per second (prefill and CUDA graph capture are timed separately)."""

from __future__ import annotations

import argparse
import time
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    import torch

    from blackhole_pipeline.serve import load, newest_model
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", nargs="?")
    ap.add_argument("--batch", default="1,32,128")
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)
    path = Path(a.model) if a.model else newest_model()
    if path is None:
        raise SystemExit("no model found; pass a folder")
    t0 = time.time()
    model, tok = load(path, a.device)
    print(f"{path.name}: loaded in {time.time() - t0:.0f} s", flush=True)
    msgs = [{"role": "user", "content": "Write a Python function that returns the n-th prime, then explain it."}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    one = tok(text, return_tensors="pt").input_ids.to(a.device)

    def sync():
        if a.device == "cuda":
            torch.cuda.synchronize()

    for b in [int(x) for x in a.batch.split(",")]:
        ids = one.repeat(b, 1)
        mask = torch.ones_like(ids)
        steps = []
        model.fast_generate(ids, mask, max_new_tokens=16)  # warm-up (kernels, stacking the experts)
        sync()
        t = time.time()
        # eos None: every row decodes all tokens, so the count is exact
        out = model.fast_generate(ids, mask, max_new_tokens=a.tokens, on_step=lambda i, n: steps.append((i, time.time())))
        sync()
        dt = time.time() - t
        n = out.shape[1]
        rate = (steps[-1][0] - steps[0][0]) / (steps[-1][1] - steps[0][1]) if len(steps) > 2 else n / dt
        print(f"batch {b:4d}: {n} tokens in {dt:.1f} s · decode {rate:.1f} tok/s per sequence · "
              f"{rate * b:.0f} tok/s total", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
