"""Thinking budget for a BlackHole Qwen MLX folder (budget forcing, see qwen_mlx.py).

    python think_budget.py MODEL_DIR [--budget 32768]     # 0 turns it off

Writes `think_budget` and the token ids of <think>, </think> and the answer-now line into the
folder's config.json, and refreshes the folder's model files (qwen_mlx.py, flint_mlx.py) from
this checkout. Runs on the Mac (needs only transformers' tokenizer). The budget can also be set
per server run: BLACKHOLE_THINK_BUDGET=16384 mlx_lm.server ...
"""

import argparse
import json
import shutil
from pathlib import Path

# Qwen's own wording for a thinking budget (also what the bench scores budgets with)
FORCE_LINE = "\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>"


def think_config(tok, budget: int) -> dict:
    v = tok.get_vocab()
    if "<think>" not in v or "</think>" not in v:
        return {}
    force = tok(FORCE_LINE, add_special_tokens=False)["input_ids"]
    if not force or force[-1] != v["</think>"]:
        raise RuntimeError("the answer-now line does not end in the </think> token with this tokenizer")
    return {"think_budget": int(budget), "think_ids": {"start": v["<think>"], "end": v["</think>"], "force": force}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--budget", type=int, default=32768)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    d = Path(a.model).expanduser()
    cfg = json.loads((d / "config.json").read_text())
    if cfg.get("model_file") != "qwen_mlx.py":
        raise SystemExit(f"{d} is not a BlackHole Qwen MLX folder")
    tc = think_config(AutoTokenizer.from_pretrained(d), a.budget)
    if not tc:
        raise SystemExit("this tokenizer has no <think> tokens")
    cfg.update(tc)
    (d / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    here = Path(__file__).resolve().parent
    for f in ("qwen_mlx.py", "flint_mlx.py", "hex_check.py"):
        shutil.copy2(here / f, d / f)
    print(f"{d.name}: thinking budget {a.budget or 'off'} tokens; model files refreshed")


if __name__ == "__main__":
    main()
