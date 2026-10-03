"""Stage 2: REAP expert pruning, calibrated on the task's own data.

Experts the task rarely routes to (weighted by router gate and activation
norm) are removed, which is what specialises the model. Also measures a
per-layer importance score for the dynamic bit allocation in the quantize
stage. Writes:
  work/reaped/              pruned HF model (uncompressed safetensors)
  work/layer_importance.json
"""

from __future__ import annotations

import argparse
import json
import shutil

from common.progress import emit
from stages._util import DRY_RUN, Job, read_jsonl

STAGE = "reap"


def layer_importance(model, tokenizer, texts: list[str], max_len: int) -> list[float]:
    """Relative contribution of each MoE block: mean ||moe_out|| / ||moe_in||
    over calibration tokens. Layers whose experts move the residual stream
    more are treated as more sensitive to quantization."""
    import torch

    blocks = [m for m in model.modules() if type(m).__name__.endswith("SparseMoeBlock")]
    if not blocks:
        raise RuntimeError("no MoE blocks found; is the teacher a mixture-of-experts model?")
    sums = [0.0] * len(blocks)
    counts = [0] * len(blocks)

    def hook(idx):
        def fn(_mod, args, out):
            x = args[0]
            y = out[0] if isinstance(out, tuple) else out
            r = (y.float().norm(dim=-1) / (x.float().norm(dim=-1) + 1e-6)).flatten()
            sums[idx] += r.sum().item()
            counts[idx] += r.numel()
        return fn

    handles = [b.register_forward_hook(hook(i)) for i, b in enumerate(blocks)]
    try:
        with torch.no_grad():
            for t in texts:
                ids = tokenizer(t, return_tensors="pt", truncation=True, max_length=max_len).to(model.device)
                model(**ids)
    finally:
        for h in handles:
            h.remove()
    return [s / max(c, 1) for s, c in zip(sums, counts)]


def run_stage(job: Job) -> None:
    cfg = job.config
    out_dir = job.path("work", "reaped")
    rows = read_jsonl(job.path("data", "train.jsonl"))

    if DRY_RUN:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config.json").write_text(json.dumps({
            "num_hidden_layers": 48, "hidden_size": 2048, "moe_intermediate_size": 768,
            "num_experts": int(128 * (1 - cfg.reap_sparsity)), "num_experts_per_tok": 8,
            "vocab_size": 151936, "num_attention_heads": 32, "num_key_value_heads": 4, "head_dim": 128,
        }))
        imp = [1.0 + (i % 7) / 7 for i in range(48)]
    else:
        from compressed_tensors.offload import dispatch_model
        from datasets import Dataset
        from llmcompressor import oneshot
        from llmcompressor.modifiers.pruning import REAPPruningModifier
        from transformers import AutoModelForCausalLM, AutoTokenizer

        emit(STAGE, pct=2, msg=f"loading {cfg.teacher}")
        src = job.model_path(cfg.teacher)
        model = AutoModelForCausalLM.from_pretrained(src, torch_dtype="auto")
        tok = AutoTokenizer.from_pretrained(src)

        texts = [tok.apply_chat_template(r["messages"], tokenize=False) for r in rows[: cfg.reap_calib_samples]]
        ds = Dataset.from_dict({"text": texts}).map(
            lambda s: tok(s["text"], padding=False, truncation=True, max_length=cfg.reap_max_seq, add_special_tokens=False),
            remove_columns=["text"],
        )

        emit(STAGE, pct=10, msg=f"REAP {cfg.reap_sparsity:.0%} on {len(texts)} task samples")
        oneshot(
            model=model,
            dataset=ds,
            recipe=REAPPruningModifier(sparsity=cfg.reap_sparsity),
            max_seq_length=cfg.reap_max_seq,
            num_calibration_samples=len(texts),
            moe_calibrate_all_experts=False,
        )

        emit(STAGE, pct=75, msg="measuring per-layer importance")
        dispatch_model(model)
        imp = layer_importance(model, tok, texts[:64], cfg.reap_max_seq)

        emit(STAGE, pct=85, msg="saving pruned model")
        if out_dir.exists():
            shutil.rmtree(out_dir)
        model.save_pretrained(out_dir, save_compressed=False)
        tok.save_pretrained(out_dir)

    job.path("work", "layer_importance.json").write_text(json.dumps(imp))
    n_experts = json.loads((out_dir / "config.json").read_text()).get("num_experts")
    job.mark_done(STAGE, {"num_experts": n_experts})
    emit(STAGE, "done", 100, f"kept {n_experts} experts per layer")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    run_stage(Job(ap.parse_args().job))
