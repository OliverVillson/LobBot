"""Stage 2: REAP expert pruning, calibrated on the task's own data.

REAP (Router-weighted Expert Activation Pruning) scores every expert by
    S_j = mean over tokens routed to j of  g_j(x) * ||f_j(x)||_2
where g_j is the renormalised router weight and f_j the expert output, then
drops the lowest-scoring experts in every layer. The same number is dropped
in each layer so the expert count stays uniform (llama.cpp needs that).
Experts the task never routes to score 0 and go first, which is what
specialises the model. Also measures a per-layer importance score for the
dynamic bit allocation in the quantize stage. Writes:
  work/reaped/                pruned HF model (uncompressed safetensors)
  work/layer_importance.json  one score per layer (quantize stage input)
  work/reap_saliency.json     per-layer, per-expert REAP scores and kept ids
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import time

from common.progress import emit
from stages._util import DRY_RUN, Job

STAGE = "reap"


class SaliencyCollector:
    """Forward hooks that accumulate REAP scores without changing outputs."""

    def __init__(self, model):
        import torch

        from stages import moe_utils as mu

        self.config = model.config
        self.blocks = mu.find_moe_blocks(model)
        if not self.blocks:
            raise RuntimeError("no MoE blocks found; is the teacher a mixture-of-experts model?")
        self.sum, self.freq, self.handles = {}, {}, []
        for li, blk in self.blocks:
            E, dev = mu.num_experts(blk), mu.gate_weight(blk).device
            self.sum[li] = torch.zeros(E, dtype=torch.float64, device=dev)
            self.freq[li] = torch.zeros(E, dtype=torch.float64, device=dev)
            self.handles.append(blk.register_forward_hook(self._hook(li, blk)))

    def _hook(self, li, blk):
        import torch

        from stages import moe_utils as mu

        @torch.no_grad()
        def hook(_mod, args, _out):
            x = args[0].reshape(-1, args[0].shape[-1])
            w, idx = mu.route(blk, self.config, x)
            for j in torch.unique(idx).tolist():
                tok_pos, slot = torch.where(idx == j)
                out = mu.expert_forward(blk, self.config, j, x[tok_pos])
                self.sum[li][j] += (w[tok_pos, slot].float() * out.float().norm(dim=-1)).sum().double()
                self.freq[li][j] += tok_pos.numel()
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()

    def results(self) -> dict[int, tuple[list[float], list[float]]]:
        res = {}
        for li, _ in self.blocks:
            f = self.freq[li].cpu()
            res[li] = ((self.sum[li].cpu() / f.clamp(min=1)).tolist(), f.tolist())
        return res


def choose_keep(saliency: list[float], freq: list[float], n_keep: int) -> list[int]:
    order = sorted(range(len(saliency)), key=lambda j: (saliency[j], freq[j], -j), reverse=True)
    return sorted(order[:n_keep])


def layer_importance(model, batches) -> list[float]:
    """Relative contribution of each MoE block: mean ||moe_out|| / ||moe_in||
    over calibration tokens. Layers whose experts move the residual stream
    more are treated as more sensitive to quantization. Dense layers get the
    mean score so the list has one entry per decoder layer."""
    import torch

    from stages import moe_utils as mu

    blocks = mu.find_moe_blocks(model)
    sums = {li: 0.0 for li, _ in blocks}
    counts = {li: 0 for li, _ in blocks}

    def hook(li):
        def fn(_mod, args, out):
            x = args[0]
            y = out[0] if isinstance(out, tuple) else out
            r = (y.float().norm(dim=-1) / (x.float().norm(dim=-1) + 1e-6)).flatten()
            sums[li] += r.sum().item()
            counts[li] += r.numel()
        return fn

    handles = [b.register_forward_hook(hook(li)) for li, b in blocks]
    try:
        with torch.no_grad():
            for ids in batches:
                model(input_ids=ids, use_cache=False)
    finally:
        for h in handles:
            h.remove()
    scores = {li: sums[li] / max(counts[li], 1) for li in sums}
    mean = sum(scores.values()) / max(len(scores), 1)
    return [scores.get(i, mean) for i in range(model.config.num_hidden_layers)]


def run_stage(job: Job) -> None:
    cfg = job.config
    out_dir = job.path("work", "reaped")

    if DRY_RUN:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config.json").write_text(json.dumps({
            "num_hidden_layers": 48, "hidden_size": 2048, "moe_intermediate_size": 768,
            "num_experts": int(128 * (1 - cfg.reap_sparsity)), "num_experts_per_tok": 8,
            "vocab_size": 151936, "num_attention_heads": 32, "num_key_value_heads": 4, "head_dim": 128,
        }))
        imp = [1.0 + (i % 7) / 7 for i in range(48)]
        n_keep = int(128 * (1 - cfg.reap_sparsity))
    else:
        import torch
        from transformers import AutoTokenizer

        from stages import moe_utils as mu
        from stages.sft import load_causal_lm
        from stages.taskdata import load_examples, tokenize_example

        src = job.model_path(cfg.teacher)
        emit(STAGE, pct=1, msg=f"loading {src}")
        tok = AutoTokenizer.from_pretrained(src)
        model = load_causal_lm(src).eval()
        dev = next(model.parameters()).device

        examples = load_examples(job.path("data", "train.jsonl"))
        random.Random(0).shuffle(examples)
        examples = examples[: cfg.reap_calib_samples]
        seqs = [tokenize_example(tok, ex["messages"], cfg.reap_max_seq)["input_ids"] for ex in examples]

        col = SaliencyCollector(model)
        first = col.blocks[0][1]
        E = mu.num_experts(first)
        n_keep = max(mu.top_k(first, model.config), E - int(round(E * cfg.reap_sparsity)))
        print(f"REAP: {len(col.blocks)} MoE layers x {E} experts, keeping {n_keep}; "
              f"calibrating on {len(seqs)} task samples ({sum(map(len, seqs))} tokens)", flush=True)
        t0 = time.time()
        with torch.no_grad():
            for i, ids in enumerate(seqs):
                model(input_ids=torch.tensor([ids], device=dev), use_cache=False)
                if i % max(1, len(seqs) // 50) == 0 or i == len(seqs) - 1:
                    emit(STAGE, pct=5 + 70 * (i + 1) / len(seqs), msg=f"calibrating {i + 1}/{len(seqs)}")
        col.remove()
        print(f"REAP: calibration took {time.time() - t0:.0f}s", flush=True)

        layers = []
        scores = col.results()
        for li, blk in col.blocks:
            s, f = scores[li]
            keep = choose_keep(s, f, n_keep)
            kept = set(keep)
            layers.append({
                "layer": li, "kept": keep, "saliency": s, "freq": f,
                "layer_saliency_kept": sum(s[j] for j in keep),
                "layer_saliency_pruned": sum(s[j] for j in range(E) if j not in kept),
                "unused_experts": sum(1 for x in f if x == 0),
            })
            mu.prune_block(blk, keep)
        mu.set_config_experts(model.config, n_keep)

        emit(STAGE, pct=78, msg="measuring per-layer importance")
        imp = layer_importance(model, [torch.tensor([ids], device=dev) for ids in seqs[:64]])

        emit(STAGE, pct=85, msg=f"saving pruned model ({n_keep}/{E} experts per layer)")
        tmp = job.path("work", "reaped.tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        model.save_pretrained(tmp, safe_serialization=True, max_shard_size="5GB")
        tok.save_pretrained(tmp)
        mu.write_expert_count_alias(tmp)
        shutil.rmtree(out_dir, ignore_errors=True)
        tmp.rename(out_dir)
        job.path("work", "reap_saliency.json").write_text(json.dumps({
            "teacher": cfg.teacher, "num_experts_orig": E, "num_experts_kept": n_keep,
            "calib_samples": len(seqs), "layers": layers,
        }))

    job.path("work", "layer_importance.json").write_text(json.dumps(imp))
    job.mark_done(STAGE, {"num_experts": n_keep})
    emit(STAGE, "done", 100, f"kept {n_keep} experts per layer")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    run_stage(Job(ap.parse_args().job))
