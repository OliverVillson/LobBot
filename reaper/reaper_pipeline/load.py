"""Load a model and tokenizer the way the run profile says."""

from __future__ import annotations

import json
from pathlib import Path

from reaper_pipeline.hardware import torch_dtype


def checkpoint_config(path: str | Path) -> dict:
    p = Path(path) / "config.json"
    return json.loads(p.read_text()) if p.exists() else {}


def load_tokenizer(path, trust_remote_code: bool = False):
    from transformers import AutoTokenizer

    from reaper_pipeline.models import register_all
    register_all()
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(path, prof: dict, trust_remote_code: bool = False):
    """bf16/fp16 spread over the GPUs (spilling to RAM if the plan says so), or CPU."""
    import transformers
    from transformers import AutoModelForCausalLM

    from reaper_pipeline.models import register_all
    register_all()  # Flint and other architectures transformers does not ship yet

    kw: dict = {"trust_remote_code": trust_remote_code}
    dkey = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
    kw[dkey] = torch_dtype(prof.get("dtype", "bfloat16"))
    if prof.get("device") == "cuda":
        kw["device_map"] = "auto"
        r = prof.get("reap") or {}
        mm = r.get("max_memory")
        if r.get("offload_folder"):
            kw["offload_folder"] = r["offload_folder"]
        if mm:
            kw["max_memory"] = {(int(k) if str(k).isdigit() else k): v for k, v in mm.items()}
    q = checkpoint_config(path).get("quantization_config") or {}
    if str(q.get("quant_method", "")).lower() == "mxfp4":
        # gpt-oss: REAP hooks need the experts as plain tensors, so unpack MXFP4 to bf16.
        # The pruned checkpoint is written from the original MXFP4 files, not from this copy.
        from transformers import Mxfp4Config
        kw["quantization_config"] = Mxfp4Config(dequantize=True)
    model = AutoModelForCausalLM.from_pretrained(path, **kw)
    return model.eval()


def input_device(model):
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


def free() -> None:
    """Release memory after the caller dropped its references."""
    import gc

    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
