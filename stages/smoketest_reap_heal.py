"""Smoke test for the reap and heal stages on tiny random Qwen3 models.

Builds a 16-expert Qwen3-MoE teacher and a tiny dense student (Qwen3, or
Gemma 4 with --gemma-tokenizer), runs the real reap and heal stages on
synthetic task data, and checks that the pruned model is exactly the original
with the pruned experts masked out of the router, that heal and the student
train the right weights on the right tokens, and (with --llama-cpp) that all
three outputs convert to GGUF and run.

    python -m stages.smoketest_reap_heal [--tokenizer PATH] [--gemma-tokenizer [PATH]] [--llama-cpp DIR]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F

from stages import moe_utils as mu
from common.progress import parse
from stages._util import Config


def tiny_gemma4(tok):
    """Tiny Gemma 4 with E4B's features: multimodal wrapper with vision and
    audio towers, per-layer embeddings, KV sharing, sliding/full layers."""
    from transformers import Gemma4Config, Gemma4ForConditionalGeneration
    text = dict(vocab_size=len(tok), vocab_size_per_layer_input=len(tok), hidden_size=64, intermediate_size=128,
                num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1, head_dim=16,
                global_head_dim=32, hidden_size_per_layer_input=8, num_kv_shared_layers=2, sliding_window=64,
                layer_types=["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
                final_logit_softcapping=30.0, max_position_embeddings=4096)
    vision = dict(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2,
                  num_key_value_heads=2, head_dim=16)
    audio = dict(hidden_size=32, num_hidden_layers=1, num_attention_heads=2, output_proj_dims=32,
                 subsampling_conv_channels=[8, 4])
    return Gemma4ForConditionalGeneration(Gemma4Config(text_config=text, vision_config=vision, audio_config=audio))


def build(root: Path, tok_path: str, gemma_tok: str | None = None):
    from transformers import (AutoTokenizer, Qwen3Config, Qwen3ForCausalLM,
                              Qwen3MoeConfig, Qwen3MoeForCausalLM)
    tok = AutoTokenizer.from_pretrained(tok_path)
    torch.manual_seed(0)
    common = dict(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_attention_heads=4,
                  num_key_value_heads=2, head_dim=16, max_position_embeddings=2048,
                  eos_token_id=tok.convert_tokens_to_ids("<|im_end|>"))
    moe = Qwen3MoeForCausalLM(Qwen3MoeConfig(num_hidden_layers=4, moe_intermediate_size=32, num_experts=16,
                                             num_experts_per_tok=4, norm_topk_prob=True,
                                             tie_word_embeddings=False, **common))
    moe.save_pretrained(root / "teacher")
    tok.save_pretrained(root / "teacher")
    if gemma_tok:
        stok = AutoTokenizer.from_pretrained(gemma_tok)
        dense = tiny_gemma4(stok)
    else:
        stok, dense = tok, Qwen3ForCausalLM(Qwen3Config(num_hidden_layers=2, tie_word_embeddings=True, **common))
    dense.save_pretrained(root / "student")
    stok.save_pretrained(root / "student")
    (root / "job" / "data").mkdir(parents=True)
    rng = random.Random(0)
    with open(root / "job" / "data" / "train.jsonl", "w") as f:
        for i in range(48):
            cat, pri = rng.choice(["billing", "bug", "account"]), rng.choice(["low", "high"])
            f.write(json.dumps({"messages": [
                {"role": "system", "content": "Turn support emails into JSON tickets."},
                {"role": "user", "content": f"Email: my {cat} problem #{i}, priority {pri}."},
                {"role": "assistant", "content": json.dumps({"category": cat, "priority": pri})}]}) + "\n")
    (root / "job" / "config.json").write_text(json.dumps({
        "teacher": str(root / "teacher"), "student": str(root / "student"),
        "reap_calib_samples": 32, "heal_epochs": 1.0, "heal_lr": 1e-3}))


def run_stage(mod: str, root: Path):
    env = {**os.environ, "LOBBOT_STUDENT_EPOCHS": "1"}
    p = subprocess.run([sys.executable, "-m", mod, "--job", str(root / "job")], env=env,
                       capture_output=True, text=True, cwd=Path(__file__).resolve().parent.parent)
    events = [e for e in map(parse, p.stdout.splitlines()) if e]
    if p.returncode != 0 or not events or events[-1].get("status") != "done":
        print(p.stdout[-3000:], p.stderr[-3000:])
        raise SystemExit(f"FAIL: {mod} did not finish")
    print(f"ok: {mod} -> {events[-1]['msg']}")


def check_equivalence(root: Path):
    from transformers import AutoModelForCausalLM
    # On a GPU box the reap stage loads and saves in bf16. Round the original's
    # weights the same way and compare both in fp32 on CPU, so only pruning
    # can make them differ.
    reaped = root / "job" / "work" / "reaped"
    saved = json.loads((reaped / "config.json").read_text())
    saved_dtype = getattr(torch, saved.get("dtype") or saved.get("torch_dtype") or "float32")
    t = AutoModelForCausalLM.from_pretrained(root / "teacher", dtype=torch.float32).eval()
    p = AutoModelForCausalLM.from_pretrained(reaped, dtype=torch.float32).eval()
    with torch.no_grad():
        for w in t.parameters():
            w.copy_(w.to(saved_dtype).float())
    sal = json.loads((root / "job" / "work" / "reap_saliency.json").read_text())
    for (_, blk), L in zip(mu.find_moe_blocks(t), sal["layers"]):
        drop = torch.ones(mu.num_experts(blk), dtype=torch.bool)
        drop[L["kept"]] = False
        _patch_router(blk, drop, t.config)  # -inf router logits == deleting the experts
    ids = torch.randint(0, 1000, (1, 24))
    with torch.no_grad():
        err = (t(ids).logits - p(ids).logits).abs().max().item()
    if err > 1e-3:
        raise SystemExit(f"FAIL: pruned model differs from masked original (max err {err})")
    print(f"ok: pruned model == original with pruned experts masked (max err {err:.1e})")


def _patch_router(blk, drop, config):
    gate = blk.gate
    k = mu.top_k(blk, config)

    def fwd(h):
        logits = F.linear(h, gate.weight)
        logits[:, drop] = float("-inf")
        probs = logits.softmax(-1, dtype=torch.float)
        w, idx = probs.topk(k, -1)
        if config.norm_topk_prob:
            w = w / w.sum(-1, keepdim=True)
        return logits, w.to(h.dtype), idx

    if isinstance(gate, torch.nn.Linear):  # transformers 4.x: the block does softmax/top-k on these logits
        def fwd(h):  # noqa: F811
            logits = F.linear(h, gate.weight)
            logits[:, drop] = float("-inf")
            return logits
    gate.forward = fwd


def check_label_mask(root: Path):
    """The student trains on the answer and end-of-turn marker only, with the
    prompt rendered as at inference."""
    from transformers import AutoTokenizer
    from stages.taskdata import load_examples, tokenize_example
    tok = AutoTokenizer.from_pretrained(root / "student")
    msgs = load_examples(root / "job" / "data" / "train.jsonl")[0]["messages"]
    d = tokenize_example(tok, msgs, 2048)
    trained = tok.decode([t for t, l in zip(d["input_ids"], d["labels"]) if l != -100])
    if not trained.startswith(msgs[-1]["content"]) or len(trained) > len(msgs[-1]["content"]) + 16:
        raise SystemExit(f"FAIL: student loss covers {trained!r}, expected the answer plus end-of-turn")
    print(f"ok: student trains on the answer only ({trained[len(msgs[-1]['content']):]!r} as end marker)")


def check_heal_trained(root: Path):
    """Router, attention and expert weights all moved during heal."""
    from safetensors.torch import load_file
    a = load_file(root / "job" / "work" / "reaped" / "model.safetensors")
    b = load_file(root / "job" / "work" / "healed" / "model.safetensors")
    changed = {k for k in a if not torch.equal(a[k], b[k])}
    for part in ("mlp.gate.weight", "experts.", "q_proj"):
        if not any(part in k for k in changed):
            raise SystemExit(f"FAIL: heal did not update {part} weights")
    print(f"ok: heal updated router, attention and expert weights ({len(changed)} tensors)")


def check_dense_trained(root: Path):
    """The student's text decoder moved; vision/audio towers (Gemma 4) did not."""
    from safetensors.torch import load_file
    a = load_file(root / "student" / "model.safetensors")
    b = load_file(root / "job" / "work" / "dense" / "model.safetensors")
    if set(a) != set(b):
        raise SystemExit(f"FAIL: dense student tensor names changed: {sorted(set(a) ^ set(b))[:5]}")
    changed = [k for k in a if not torch.equal(a[k], b[k].to(a[k].dtype))]
    towers = [k for k in changed if "vision" in k or "audio" in k]
    if not changed or towers:
        raise SystemExit(f"FAIL: dense student update wrong (changed {len(changed)}, towers {towers[:3]})")
    print(f"ok: dense student updated {len(changed)} text-decoder tensors, towers untouched")


def check_nonfinite_guard(root: Path):
    """A NaN loss or NaN gradient on one batch skips that update instead of
    poisoning the weights; training that keeps producing NaN saves nothing."""
    from stages import sft
    from stages.taskdata import load_examples
    examples = load_examples(root / "job" / "data" / "train.jsonl")
    load = sft.load_causal_lm

    def with_nan(calls):
        def loader(path):
            model, n = load(path), [0]

            def hook(_mod, _inp, out):
                n[0] += 1
                kind = calls(n[0])
                if kind == "loss":
                    return out * float("nan")
                if kind == "grad" and out.requires_grad:
                    out.register_hook(lambda g: g * float("nan"))
            model.get_output_embeddings().register_forward_hook(hook)
            return model
        return loader

    out = root / "nan-guard"
    try:
        sft.load_causal_lm = with_nan(lambda i: {2: "loss", 4: "grad"}.get(i))
        sft.train_sft(str(root / "student"), examples, out, epochs=1.0, max_tokens=256)
        base = _load_weights(root / "student")
        bad = [k for k, v in _load_weights(out).items() if not torch.isfinite(v).all() and torch.isfinite(base[k]).all()]
        if bad:
            raise SystemExit(f"FAIL: NaN batch poisoned the weights ({bad[:3]})")
        sft.load_causal_lm = with_nan(lambda i: "loss")
        try:
            sft.train_sft(str(root / "student"), examples, root / "nan-always", epochs=1.0, max_tokens=256)
            raise SystemExit("FAIL: training that only produces NaN was saved")
        except sft.NonFiniteTraining:
            pass
        if (root / "nan-always").exists():
            raise SystemExit("FAIL: diverged training left an output dir")
    finally:
        sft.load_causal_lm = load
    print("ok: NaN loss/gradient batches are skipped, diverged training saves nothing")


def _load_weights(d: Path) -> dict:
    from safetensors.torch import load_file
    w = {}
    for f in d.glob("*.safetensors"):
        w.update(load_file(f))
    return w


def gguf_check(root: Path, llama: Path):
    for name in ("reaped", "healed", "dense"):
        src = root / "job" / "work" / name
        out = root / f"{name}.gguf"
        subprocess.run([sys.executable, str(llama / "convert_hf_to_gguf.py"), str(src),
                        "--outfile", str(out), "--outtype", "f16"], check=True, capture_output=True)
        binary = next((llama / d / "llama-simple" for d in ("build/bin", "bin") if (llama / d / "llama-simple").exists()), None)
        if binary:
            r = subprocess.run([str(binary), "-m", str(out), "-n", "4", "Email: my"], capture_output=True, text=True)
            if r.returncode != 0:
                print(r.stderr[-2000:])
                raise SystemExit(f"FAIL: llama.cpp could not run {out}")
        print(f"ok: {name} converts to GGUF" + (" and runs in llama.cpp" if binary else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default=Config.teacher, help="any Qwen tokenizer (hub id or path)")
    ap.add_argument("--gemma-tokenizer", nargs="?", const="google/gemma-4-E4B-it",
                    help="make the student a tiny Gemma 4 with this tokenizer (default google/gemma-4-E4B-it)")
    ap.add_argument("--llama-cpp", help="llama.cpp checkout to test GGUF conversion")
    a = ap.parse_args()
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        build(root, a.tokenizer, a.gemma_tokenizer)
        check_label_mask(root)
        run_stage("stages.reap", root)
        check_equivalence(root)
        run_stage("stages.heal", root)
        check_heal_trained(root)
        check_dense_trained(root)
        check_nonfinite_guard(root)
        if a.llama_cpp:
            gguf_check(root, Path(a.llama_cpp).expanduser())
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
