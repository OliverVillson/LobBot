"""Auto-detect any model from a Hugging Face id or a local folder.

Reads only config.json and, when present, the safetensors index (for the exact
weight size). No weights are downloaded. MoE shape comes from whichever config
keys the family uses; size comes from the index, or is estimated from the
config when there is none.

    detect("Qwen/Qwen3-30B-A3B-Instruct-2507")   # hub
    detect("~/models/my-finetune")               # local folder with config.json
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from blackhole_tui.catalog import Model

EXPERT_KEYS = ("num_experts", "num_local_experts", "n_routed_experts", "moe_num_experts", "num_routed_experts")
TOPK_KEYS = ("num_experts_per_tok", "moe_topk", "num_experts_per_token", "top_k")
DTYPE_BYTES = {"bfloat16": 2, "float16": 2, "float32": 4, "float8_e4m3fn": 1, "fp8": 1}


class DetectError(Exception):
    pass


@dataclass
class Detected:
    source: str              # what the user typed, resolved
    where: str               # "local", "hub" or "demo"
    family: str              # model_type, e.g. qwen3_moe
    arch: str                # first entry of architectures
    experts: int = 0
    top_k: int = 0
    shared_experts: int = 0
    layers: int = 0
    hidden: int = 0
    total_b: float = 0.0
    active_b: float = 0.0
    dtype: str = "bfloat16"
    size_exact: bool = False  # from the weight index rather than estimated
    notes: list[str] = field(default_factory=list)

    @property
    def moe(self) -> bool:
        return self.experts > 0

    def to_model(self) -> Model:
        org = self.source.split("/")[0] if "/" in self.source and self.where == "hub" else "local"
        name = self.source.rstrip("/").split("/")[-1]
        return Model(self.source, name, org, round(self.total_b, 1), round(self.active_b, 1),
                     self.experts, self.top_k, "?")


def _cfg_get(cfg: dict, keys: tuple[str, ...]) -> int:
    for src in (cfg, cfg.get("text_config") or {}):
        for k in keys:
            v = src.get(k)
            if isinstance(v, int) and v > 0:
                return v
    return 0


def _estimate(cfg: dict, experts: int, top_k: int, shared: int) -> tuple[float, float]:
    """Params (total, active) in billions from the config alone."""
    c = {**(cfg.get("text_config") or {}), **cfg}
    h = c.get("hidden_size", 0)
    n = c.get("num_hidden_layers", 0)
    v = c.get("vocab_size", 0)
    if not (h and n):
        return 0.0, 0.0
    heads = c.get("num_attention_heads", 1)
    kv = c.get("num_key_value_heads", heads)
    hd = c.get("head_dim") or h // max(heads, 1)
    attn = h * hd * (heads * 2 + kv * 2)
    emb = v * h * (1 if c.get("tie_word_embeddings") else 2)
    if experts:
        ff = c.get("moe_intermediate_size") or c.get("intermediate_size", 0)
        per_expert = 3 * h * ff
        dense_first = c.get("first_k_dense_replace", 0)
        moe_layers = n - dense_first
        dense_ff = 3 * h * c.get("intermediate_size", 0)
        shared_p = shared * per_expert
        total = n * attn + moe_layers * (experts * per_expert + shared_p) + dense_first * dense_ff + emb
        active = n * attn + moe_layers * (top_k * per_expert + shared_p) + dense_first * dense_ff + emb
    else:
        total = active = n * (attn + 3 * h * c.get("intermediate_size", 0)) + emb
    return total / 1e9, active / 1e9


def from_config(cfg: dict, source: str, where: str, index: dict | None = None) -> Detected:
    experts = _cfg_get(cfg, EXPERT_KEYS)
    top_k = _cfg_get(cfg, TOPK_KEYS) if experts else 0
    shared = _cfg_get(cfg, ("n_shared_experts", "num_shared_experts"))
    if not shared and (cfg.get("shared_expert_intermediate_size") or 0):
        shared = 1
    total, active = _estimate(cfg, experts, top_k, shared)
    dtype = str(cfg.get("torch_dtype") or (cfg.get("text_config") or {}).get("torch_dtype") or "bfloat16")
    d = Detected(source, where, str(cfg.get("model_type", "?")), (cfg.get("architectures") or ["?"])[0],
                 experts, top_k, shared, cfg.get("num_hidden_layers", 0), cfg.get("hidden_size", 0),
                 total, active, dtype)
    size = (index or {}).get("metadata", {}).get("total_size")
    if size:
        exact = size / DTYPE_BYTES.get(dtype, 2) / 1e9
        if total:
            d.active_b *= exact / total
        d.total_b, d.size_exact = exact, True
    if experts and not top_k:
        d.notes.append("router top-k not in config; the plan stage reads it from the code")
    if cfg.get("quantization_config"):
        d.notes.append(f"already quantized ({cfg['quantization_config'].get('quant_method', '?')}); "
                       "a bf16 original prunes better")
    if not total:
        d.notes.append("size unknown until the weights stage")
    return d


def _fetch_json(url: str) -> dict | None:
    req = urllib.request.Request(url)
    tok = os.environ.get("HF_TOKEN")
    if tok:
        req.add_header("Authorization", f"Bearer {tok}")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise DetectError(f"Hugging Face said {e.code} for {url.split('/resolve')[0]}"
                          + (" (gated? set HF_TOKEN)" if e.code in (401, 403) else ""))
    except (urllib.error.URLError, TimeoutError) as e:
        raise DetectError(f"can't reach Hugging Face from this machine ({e})")


def detect(source: str) -> Detected:
    src = source.strip()
    path = Path(os.path.expanduser(src))
    if path.is_dir():
        cfg_p = path / "config.json"
        if not cfg_p.exists():
            raise DetectError(f"{path} has no config.json")
        idx_p = path / "model.safetensors.index.json"
        index = json.loads(idx_p.read_text()) if idx_p.exists() else None
        d = from_config(json.loads(cfg_p.read_text()), str(path), "local", index)
        if index is None:
            single = path / "model.safetensors"
            if single.exists():
                d.total_b, d.size_exact = single.stat().st_size / DTYPE_BYTES.get(d.dtype, 2) / 1e9, True
        return d
    if src.count("/") != 1 or src.startswith((".", "/", "~")):
        raise DetectError(f"“{src}” is not a folder here or an org/name Hugging Face id")
    base = f"https://huggingface.co/{src}/resolve/main"
    cfg = _fetch_json(f"{base}/config.json")
    if cfg is None:
        raise DetectError(f"no config.json at huggingface.co/{src} (typo, or not a transformers model?)")
    return from_config(cfg, src, "hub", _fetch_json(f"{base}/model.safetensors.index.json"))


# --demo has no network: a few real configs, trimmed to the keys detection reads
DEMO_CONFIGS = {
    "mistralai/Mixtral-8x7B-Instruct-v0.1": (
        {"model_type": "mixtral", "architectures": ["MixtralForCausalLM"], "hidden_size": 4096,
         "num_hidden_layers": 32, "intermediate_size": 14336, "num_local_experts": 8, "num_experts_per_tok": 2,
         "num_attention_heads": 32, "num_key_value_heads": 8, "vocab_size": 32000, "torch_dtype": "bfloat16"},
        {"metadata": {"total_size": 93405585408}}),
    "meta-llama/Llama-3.1-70B-Instruct": (
        {"model_type": "llama", "architectures": ["LlamaForCausalLM"], "hidden_size": 8192,
         "num_hidden_layers": 80, "intermediate_size": 28672, "num_attention_heads": 64,
         "num_key_value_heads": 8, "vocab_size": 128256, "torch_dtype": "bfloat16"},
        {"metadata": {"total_size": 141107412992}}),
    "ibm-granite/granite-4.0-h-tiny": (
        {"model_type": "granitemoehybrid", "architectures": ["GraniteMoeHybridForCausalLM"], "hidden_size": 1536,
         "num_hidden_layers": 40, "intermediate_size": 512, "num_local_experts": 64, "num_experts_per_tok": 6,
         "shared_intermediate_size": 1024, "num_attention_heads": 12, "num_key_value_heads": 4,
         "vocab_size": 100352, "torch_dtype": "bfloat16"},
        {"metadata": {"total_size": 13919354880}}),
}


def detect_demo(source: str) -> Detected:
    """Detection without network for --demo: known configs, else the real local path, else a clear miss."""
    src = source.strip()
    for k, (cfg, idx) in DEMO_CONFIGS.items():
        if k.lower() == src.lower():
            d = from_config(cfg, k, "demo", idx)
            d.notes.append("demo: config bundled, not fetched")
            return d
    if Path(os.path.expanduser(src)).is_dir():
        return detect(src)
    raise DetectError(f"demo mode has no network; try {', '.join(DEMO_CONFIGS)} or a local folder")
