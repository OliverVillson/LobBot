"""Build a tiny random Qwen3-MoE HF checkpoint plus a fake job dir, for CPU tests of
the quantize and eval stages without network access.

The tokenizer is rebuilt from llama.cpp's bundled ggml-vocab-qwen2.gguf, so
convert_hf_to_gguf.py recognises it exactly like the real Qwen3 tokenizer.

    python tests/make_tiny_moe.py /tmp/job
"""
import json
import os
import random
import sys
from pathlib import Path

import torch


def build_tokenizer(out: Path) -> None:
    sys.path.insert(0, str(Path(os.environ.get("LLAMA_CPP_DIR", "~/llama.cpp")).expanduser() / "gguf-py"))
    import gguf
    from tokenizers import Tokenizer, Regex, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab_path = Path(os.environ.get("LLAMA_CPP_DIR", "~/llama.cpp")).expanduser() / "models" / "ggml-vocab-qwen2.gguf"
    r = gguf.GGUFReader(str(vocab_path))

    def strings(key):
        f = r.fields[key]
        return [bytes(f.parts[i]).decode("utf-8") for i in f.data]

    tokens = strings("tokenizer.ggml.tokens")
    types = [int(r.fields["tokenizer.ggml.token_type"].parts[i][0]) for i in r.fields["tokenizer.ggml.token_type"].data]
    merges = [tuple(m.split(" ", 1)) for m in strings("tokenizer.ggml.merges")]
    chat_template = bytes(r.fields["tokenizer.chat_template"].parts[-1]).decode()

    normal = {t: i for i, (t, ty) in enumerate(zip(tokens, types)) if ty == 1}
    special = [(i, t) for i, (t, ty) in enumerate(zip(tokens, types)) if ty != 1 and not t.startswith("[PAD")]
    tk = Tokenizer(models.BPE(vocab=normal, merges=merges, byte_fallback=False))
    pat = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
    tk.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(pat), behavior="isolated", invert=False),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tk.decoder = decoders.ByteLevel()
    from tokenizers import AddedToken
    tk.add_special_tokens([AddedToken(t, special=True, normalized=False) for _, t in sorted(special)])
    fast = PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|im_end|>", pad_token="<|endoftext|>")
    fast.chat_template = chat_template
    fast.save_pretrained(str(out))


def build_model(out: Path, vocab_size: int) -> None:
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
    cfg = Qwen3MoeConfig(
        vocab_size=vocab_size, hidden_size=256, intermediate_size=512, moe_intermediate_size=256,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=64,
        num_experts=8, num_experts_per_tok=2, max_position_embeddings=2048,
        tie_word_embeddings=False, torch_dtype="bfloat16",
    )
    torch.manual_seed(0)
    m = Qwen3MoeForCausalLM(cfg).to(torch.bfloat16)
    m.save_pretrained(str(out), safe_serialization=True)


def build_job(job: Path) -> None:
    """Fake outputs of the data, reap and heal stages, in the scaffold's layout."""
    random.seed(0)
    for d in ("data", "work", "out", ".done"):
        (job / d).mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    spec = json.loads((root / "examples" / "support-tickets.taskspec.json").read_text())
    (job / "taskspec.json").write_text(json.dumps(spec, indent=2))
    sys.path.insert(0, str(root))
    from common.taskspec import TaskSpec
    from stages.data import system_prompt
    system = system_prompt(TaskSpec.from_dict(spec))
    cats = ["billing", "bug", "account", "feature"]

    def ex(i):
        c = cats[i % 4]
        return (f"Hi, ticket {i}: I have a {c} problem with my order #{1000 + i}.",
                json.dumps({"category": c, "priority": "high" if i % 3 == 0 else "low",
                            "summary": f"{c} issue on order {1000 + i}"}))
    with open(job / "data" / "train.jsonl", "w") as f:
        for i in range(64):
            a, b = ex(i)
            f.write(json.dumps({"messages": [{"role": "system", "content": system},
                                             {"role": "user", "content": a},
                                             {"role": "assistant", "content": b}]}) + "\n")
    with open(job / "data" / "heldout.jsonl", "w") as f:
        for i in range(100, 106):
            a, b = ex(i)
            f.write(json.dumps({"input": a, "reference": b}) + "\n")
    (job / "data" / "calib.txt").write_text("\n\n".join("\n".join(ex(i)) for i in range(64)))
    (job / "work" / "layer_importance.json").write_text(json.dumps([0.2, 0.5, 0.3, 0.9]))
    # Tiny model: budget = 1.0 - 0.942 = 58 MB, so the allocator has to trade bits.
    (job / "config.json").write_text(json.dumps({
        "llama_cpp": os.environ.get("LLAMA_CPP_DIR", os.path.expanduser("~/llama.cpp")),
        "size_margin_gb": 0.942, "dense_fallback": False,
    }))
    spec["target"]["max_size_gb"] = 1.0
    (job / "taskspec.json").write_text(json.dumps(spec, indent=2))


if __name__ == "__main__":
    job = Path(sys.argv[1])
    model_dir = job / "work" / "healed"
    model_dir.mkdir(parents=True, exist_ok=True)
    build_tokenizer(model_dir)
    build_model(model_dir, 151936)
    build_job(job)
    print("tiny job at", job)
