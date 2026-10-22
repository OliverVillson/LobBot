"""Size and shape of a MoE model, from its config.json and safetensors index
(a few KB from the Hub or a local folder, never the weights). Works for any
model whose config names its routed experts the usual ways.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

from reaper_hw.hardware import GB

EXPERT_KEYS = ("num_experts", "num_local_experts", "n_routed_experts", "moe_num_experts")
TOP_K_KEYS = ("num_experts_per_tok", "moe_k", "moe_topk", "top_k", "num_selected_experts")


@dataclass
class ModelInfo:
    id: str
    model_type: str = "?"
    params_b: float = 0.0        # total parameters, billions
    expert_params_b: float = 0.0  # routed expert weights, the part REAP prunes
    checkpoint_gb: float = 0.0   # bytes on disk
    quantized: str | None = None  # "fp8", "mxfp4", ... when the checkpoint is not 16-bit
    layers: int = 0
    hidden: int = 0
    experts: int = 0
    top_k: int = 0
    moe_layers: int = 0

    @property
    def bf16_gb(self) -> float:
        return self.params_b * 1e9 * 2 / GB

    def pruned_bf16_gb(self, prune: float) -> float:
        expert = self.expert_params_b or self.params_b * 0.9
        return (self.params_b - expert * prune) * 1e9 * 2 / GB

    @classmethod
    def from_sizes(cls, id: str, total_b: float, active_b: float, experts: int, top_k: int,
                   hidden: int = 0) -> "ModelInfo":
        """From headline numbers (e.g. a model card or the TUI catalog) when no config is at hand.
        Expert weights follow from total = dense + E*e and active = dense + k*e."""
        per = (total_b - active_b) / (experts - top_k) if experts > top_k else 0.0
        return cls(id, params_b=total_b, expert_params_b=round(per * experts, 2),
                   checkpoint_gb=round(total_b * 1e9 * 2 / GB, 1), hidden=hidden, experts=experts, top_k=top_k)

    def to_dict(self) -> dict:
        return asdict(self) | {"bf16_gb": round(self.bf16_gb, 1)}


def _text_config(cfg: dict) -> dict:
    return cfg.get("text_config") or cfg


def experts_in(cfg: dict) -> int:
    c = _text_config(cfg)
    for k in EXPERT_KEYS:
        if isinstance(c.get(k), int) and c[k] > 1:
            return c[k]
    return 0


def top_k_in(cfg: dict) -> int:
    c = _text_config(cfg)
    return next((c[k] for k in TOP_K_KEYS if isinstance(c.get(k), int)), 0)


def estimate_params(cfg: dict) -> tuple[float, float]:
    """(total, routed expert) parameters in billions from the config, for checkpoints
    whose size on disk is not 2 bytes per parameter. Within ~5% for memory planning."""
    c = _text_config(cfg)
    H, L, V = c["hidden_size"], c["num_hidden_layers"], c.get("vocab_size", 32000)
    heads = c.get("num_attention_heads", H // 128)
    hd = c.get("head_dim") or H // heads
    kv = c.get("num_key_value_heads", heads)
    if c.get("kv_lora_rank"):  # DeepSeek MLA
        nope, rope, v = c.get("qk_nope_head_dim", 128), c.get("qk_rope_head_dim", 64), c.get("v_head_dim", 128)
        q = c.get("q_lora_rank") or H
        attn = H * q + q * heads * (nope + rope) + H * (c["kv_lora_rank"] + rope) \
            + c["kv_lora_rank"] * heads * (nope + v) + heads * v * H
    else:
        attn = H * hd * (heads * 2 + kv * 2)
    E = experts_in(cfg)
    I_moe = c.get("moe_intermediate_size") or c.get("intermediate_size", 4 * H)
    I_dense = c.get("intermediate_size", 4 * H)
    shared = (c.get("n_shared_experts") or c.get("num_shared_experts") or 0) * 3 * H * I_moe
    if c.get("shared_expert_intermediate_size"):
        shared = 3 * H * c["shared_expert_intermediate_size"]
    moe_layers = _moe_layers(c)
    expert = moe_layers * E * 3 * H * I_moe
    total = L * attn + (L - moe_layers) * 3 * H * I_dense + expert + moe_layers * (E * H + shared) + 2 * V * H
    return total / 1e9, expert / 1e9


def _moe_layers(c: dict) -> int:
    L = c.get("num_hidden_layers", 0)
    if c.get("decoder_sparse_step", 1) > 1:
        return L // c["decoder_sparse_step"]
    if c.get("interleave_moe_layer_step", 1) > 1:
        return L // c["interleave_moe_layer_step"]
    return L - (c.get("first_k_dense_replace") or 0)


def model_info_from_config(model_id: str, cfg: dict, checkpoint_bytes: int = 0) -> ModelInfo:
    c = _text_config(cfg)
    q = cfg.get("quantization_config") or c.get("quantization_config")
    quant = str(q.get("quant_method") or q.get("fmt") or "quantized").lower() if q else None
    total, expert = estimate_params(cfg)
    if checkpoint_bytes and not quant:  # 16-bit checkpoint: the file size is exact
        expert *= checkpoint_bytes / 2 / 1e9 / total
        total = checkpoint_bytes / 2 / 1e9
    return ModelInfo(model_id, c.get("model_type", cfg.get("model_type", "?")), round(total, 2), round(expert, 2),
                     round((checkpoint_bytes or total * 2e9) / GB, 1), quant, c.get("num_hidden_layers", 0),
                     c.get("hidden_size", 0), experts_in(cfg), top_k_in(cfg), _moe_layers(c))


@lru_cache(maxsize=64)
def model_info(model_id: str, local_dir: str | None = None) -> ModelInfo:
    """Reads config.json and the safetensors index from local_dir, a local path given as
    model_id, or the Hugging Face Hub. Raises RuntimeError when no config can be read."""
    root = Path(local_dir or model_id).expanduser()

    def fetch(name: str) -> dict | None:
        if (root / name).is_file():
            return json.loads((root / name).read_text())
        try:
            from huggingface_hub import hf_hub_download
            return json.loads(Path(hf_hub_download(model_id, name)).read_text())
        except Exception:
            return None

    cfg = fetch("config.json")
    if cfg is None:
        raise RuntimeError(f"cannot read config.json for {model_id} (offline, gated, or not a model repo)")
    idx = fetch("model.safetensors.index.json") or {}
    size = int((idx.get("metadata") or {}).get("total_size") or 0)
    if not size and root.is_dir():
        size = sum(p.stat().st_size for p in root.glob("*.safetensors"))
    return model_info_from_config(model_id, cfg, size)
