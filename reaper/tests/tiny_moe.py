"""Tiny random MoE checkpoints of real architectures, for CPU tests."""

from __future__ import annotations

import json
from pathlib import Path

COMMON = dict(vocab_size=320, hidden_size=64, intermediate_size=96, num_hidden_layers=2, num_attention_heads=4,
              num_key_value_heads=2, head_dim=16, max_position_embeddings=512, tie_word_embeddings=False)

ARCHS = {
    "qwen3_moe": dict(num_experts=8, num_experts_per_tok=2, moe_intermediate_size=32, norm_topk_prob=True,
                      decoder_sparse_step=1, mlp_only_layers=[]),
    "qwen3_next": dict(num_experts=8, num_experts_per_tok=2, moe_intermediate_size=32, norm_topk_prob=True,
                       shared_expert_intermediate_size=32, decoder_sparse_step=1, mlp_only_layers=[],
                       layer_types=["linear_attention", "full_attention"], linear_num_value_heads=4,
                       linear_num_key_heads=2, linear_key_head_dim=16, linear_value_head_dim=16,
                       linear_conv_kernel_dim=4, full_attention_interval=2),
    "mixtral": dict(num_local_experts=8, num_experts_per_tok=2),
    "olmoe": dict(num_experts=8, num_experts_per_tok=2, norm_topk_prob=True),
    "gpt_oss": dict(num_local_experts=8, num_experts_per_tok=2, layer_types=["sliding_attention", "full_attention"],
                    sliding_window=32),
    "deepseek_v3": dict(n_routed_experts=8, num_experts_per_tok=2, moe_intermediate_size=32, n_shared_experts=1,
                        n_group=1, topk_group=1, first_k_dense_replace=0, norm_topk_prob=True,
                        routed_scaling_factor=2.5, kv_lora_rank=16, q_lora_rank=32, qk_rope_head_dim=8,
                        qk_nope_head_dim=8, v_head_dim=16, head_dim=8),
    "glm4_moe": dict(n_routed_experts=8, num_experts_per_tok=2, moe_intermediate_size=32, n_shared_experts=1,
                     n_group=1, topk_group=1, first_k_dense_replace=1, norm_topk_prob=True,
                     routed_scaling_factor=2.5, num_hidden_layers=3),
    "minimax_m2": dict(num_local_experts=8, num_experts_per_tok=2, use_routing_bias=True),
    # Qwen3.5 / 3.6 MoE (Qwen3.6-35B-A3B): a vision-language checkpoint whose text model has Gated
    # DeltaNet linear-attention layers, a full-attention layer every 4th, fused routed experts and
    # a gated shared expert; built as the full image-text model like the real one
    "qwen3_5_moe": dict(num_experts=8, num_experts_per_tok=2, moe_intermediate_size=32,
                        shared_expert_intermediate_size=32, num_hidden_layers=4,
                        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
                        linear_num_value_heads=4, linear_num_key_heads=2, linear_key_head_dim=16,
                        linear_value_head_dim=16, linear_conv_kernel_dim=4),
    # REAPer's own PyTorch Flint (reaper_pipeline.models.flint): a dense first layer, sliding
    # layers with RoPE, a full layer without, attention sinks, sigmoid routing, shared expert
    "flint": dict(num_experts=8, num_experts_per_tok=2, moe_intermediate_size=32, shared_expert_intermediate_size=32,
                  num_hidden_layers=3, mlp_only_layers=[0], sliding_window=4, no_rope_layer_indices=[2],
                  layer_types=["sliding_attention", "sliding_attention", "full_attention"]),
}


def tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
    from transformers import PreTrainedTokenizerFast
    tk = Tokenizer(models.BPE(unk_token=None))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    special = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    trainer = trainers.BpeTrainer(vocab_size=300, special_tokens=special,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    text = ["def fix(bug): return bug.trace() # debugging python code error stack " * 4,
            "The quick brown fox jumps over the lazy dog. Explain, write, solve 3x + 7 = 2." * 4]
    tk.train_from_iterator(text, trainer)
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|im_end|>", pad_token="<|endoftext|>")
    tok.chat_template = ("{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n"
                         "{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
    return tok


def build(arch: str, path: Path, seed: int = 0, **over) -> Path:
    import torch
    from transformers import AutoModelForCausalLM
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    from reaper_pipeline.models import register_all
    register_all()
    torch.manual_seed(seed)
    kw = {**COMMON, **ARCHS[arch], **over}
    if arch == "qwen3_5_moe":
        from transformers import AutoModelForImageTextToText
        kw.pop("intermediate_size")
        vision = dict(depth=1, hidden_size=32, intermediate_size=32, num_heads=2, out_hidden_size=kw["hidden_size"],
                      patch_size=4, num_position_embeddings=16)
        cfg = CONFIG_MAPPING[arch](text_config=kw, vision_config=vision, tie_word_embeddings=False)
        model = AutoModelForImageTextToText.from_config(cfg).eval()
    else:
        cfg = CONFIG_MAPPING[arch](**kw)
        model = AutoModelForCausalLM.from_config(cfg).eval()
    with torch.no_grad():  # random routers and biases so routing is not degenerate
        for n, p in model.named_parameters():
            if "bias" in n and ("gate" in n or "router" in n or "correction" in n):
                p.normal_(0, 0.5)
            if "softmax_offset" in n or "norm" in n:  # attention sinks, norm scales
                p.normal_(1.0 if "norm" in n else 0.0, 0.5)
        for n, b in model.named_buffers():
            if "correction_bias" in n:
                b.normal_(0, 0.5)
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path, safe_serialization=True, max_shard_size="200KB")
    tokenizer().save_pretrained(path)
    return path


if __name__ == "__main__":
    import sys
    import tempfile
    for a in sys.argv[1:] or ARCHS:
        d = build(a, Path(tempfile.mkdtemp()) / a)
        cfg = json.loads((d / "config.json").read_text())
        keys = json.loads((d / "model.safetensors.index.json").read_text())["weight_map"] if (d / "model.safetensors.index.json").exists() else {}
        print(a, sorted({k.split("layers.0.")[-1] for k in keys if "layers.0." in k and ("expert" in k or "gate" in k or "router" in k or "bias" in k)})[:14])
