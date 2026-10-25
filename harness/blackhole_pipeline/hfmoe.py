"""Distillation helpers for MoE models that run on transformers' own code (Qwen3.5 / 3.6 and
others), where REAPer's Flint has its own: low-rank updates on fused routed experts, the
per-layer forward the heal trains through, and fast batch generation through vLLM.

Fused experts (transformers 5: one [E, 2I, H] gate_up_proj and one [E, H, I] down_proj per
layer) get rank-r updates B @ A per expert, kept on the MoE block (mlp.lora_gu_A, ...), so the
heal's "train everything but .experts." rule trains them and merge folds them into the weights.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import types
from pathlib import Path


def fused_blocks(model) -> list:
    """(mlp module, experts module) of every MoE layer whose experts are fused tensors."""
    out = []
    for layer in model.model.layers:
        mlp = getattr(layer, "mlp", None)
        ex = getattr(mlp, "experts", None)
        if ex is not None and hasattr(ex, "gate_up_proj") and getattr(ex.gate_up_proj, "dim", lambda: 0)() == 3:
            out.append((mlp, ex))
    return out


def _lora_forward(ex, mlp):
    """The experts' own forward (transformers picks grouped_mm or its loop) on W + B @ A: the
    updated weights of every expert are formed once per call (about 1 GB per Qwen3.6 layer,
    transient) and swapped in for the call."""
    import torch
    base = type(ex).forward

    def forward(self, hidden_states, top_k_index, top_k_weights):
        gu = self._parameters["gate_up_proj"]
        dn = self._parameters["down_proj"]
        try:
            self._parameters["gate_up_proj"] = gu + torch.bmm(mlp.lora_gu_B, mlp.lora_gu_A)
            self._parameters["down_proj"] = dn + torch.bmm(mlp.lora_dn_B, mlp.lora_dn_A)
            return base(self, hidden_states, top_k_index, top_k_weights)
        finally:
            self._parameters["gate_up_proj"], self._parameters["down_proj"] = gu, dn
    return types.MethodType(forward, ex)


def add_expert_lora(model, rank: int) -> int:
    """Rank-r updates on every fused expert (B starts at zero: the model is unchanged). Returns
    the number of MoE blocks that got them."""
    import torch
    g = torch.Generator().manual_seed(0)
    blocks = fused_blocks(model)
    for mlp, ex in blocks:
        E, I2, H = ex.gate_up_proj.shape
        dev, dt = ex.gate_up_proj.device, ex.gate_up_proj.dtype
        mlp.lora_gu_A = torch.nn.Parameter((torch.randn(E, rank, H, generator=g) / H ** 0.5).to(dev, dt))
        mlp.lora_gu_B = torch.nn.Parameter(torch.zeros(E, I2, rank, device=dev, dtype=dt))
        mlp.lora_dn_A = torch.nn.Parameter((torch.randn(E, rank, I2 // 2, generator=g) / (I2 // 2) ** 0.5).to(dev, dt))
        mlp.lora_dn_B = torch.nn.Parameter(torch.zeros(E, H, rank, device=dev, dtype=dt))
        ex.forward = _lora_forward(ex, mlp)
    return len(blocks)


def merge_expert_lora(model) -> None:
    import torch
    with torch.no_grad():
        for mlp, ex in fused_blocks(model):
            if not hasattr(mlp, "lora_gu_A"):
                continue
            ex.gate_up_proj.add_(torch.bmm(mlp.lora_gu_B.float(), mlp.lora_gu_A.float()).to(ex.gate_up_proj.dtype))
            ex.down_proj.add_(torch.bmm(mlp.lora_dn_B.float(), mlp.lora_dn_A.float()).to(ex.down_proj.dtype))
            for n in ("lora_gu_A", "lora_gu_B", "lora_dn_A", "lora_dn_B"):
                delattr(mlp, n)
            del ex.forward      # back to the class's forward


def hidden(model, ids, grad: bool, layers: bool = False):
    """Final (normed) hidden states [1, T, H] and, if asked, every layer's output, through the
    model's own forward (gradient checkpointing per decoder layer when training)."""
    if grad and not getattr(model, "is_gradient_checkpointing", False):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    out = model.model(input_ids=ids, use_cache=False, output_hidden_states=layers)
    return out.last_hidden_state, list(out.hidden_states[1:]) if layers else []


# --------------------------------------------------------------------------- generation


def vllm_python() -> str | None:
    """vLLM's interpreter: BLACKHOLE_VLLM_PY / REAPER_VLLM_PY, or this one if it has vLLM."""
    for k in ("BLACKHOLE_VLLM_PY", "REAPER_VLLM_PY"):
        if os.environ.get(k):
            return os.environ[k]
    try:
        import importlib.util
        return sys.executable if importlib.util.find_spec("vllm") else None
    except Exception:  # noqa: BLE001
        return None


def can_vllm() -> bool:
    import torch
    return torch.cuda.is_available() and vllm_python() is not None and not os.environ.get("BLACKHOLE_NO_VLLM")


def vllm_generate(model_dir: Path, prompts: list[list[int]], max_tokens: int, temperature: float, top_p: float,
                  top_k: int, stop: list[int], allowed: list[int] | None = None, on_line=None,
                  share: float | None = None) -> list[list[int]]:
    """Answers (token ids, ending with the stop token when the answer finished) from vLLM in its own process. `share` = fraction
    of the GPU's memory vLLM may use (default: 85% of what is free now)."""
    import torch
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info()
    util = share or max(0.05, min(0.92, 0.85 * free / total))
    work = Path(tempfile.mkdtemp(prefix="bh-vllm-"))
    job = {"model": str(model_dir), "prompts": prompts, "max_tokens": max_tokens, "temperature": temperature,
           "top_p": top_p, "top_k": top_k, "stop": stop, "allowed": allowed, "gpu_memory_utilization": round(util, 3),
           "max_model_len": max(map(len, prompts)) + max_tokens + 8, "out": str(work / "out.jsonl")}
    (work / "job.json").write_text(json.dumps(job))
    script = Path(__file__).with_name("vllm_gen.py")
    py = vllm_python()
    env = {**os.environ, "PATH": str(Path(py).parent) + os.pathsep + os.environ.get("PATH", "")}   # ninja for JIT
    p = subprocess.Popen([py, "-u", str(script), str(work / "job.json")], env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1, errors="replace")
    tail = []
    for line in p.stdout:
        tail = (tail + [line.rstrip()])[-40:]
        if on_line:
            on_line(line.rstrip())
    if p.wait() != 0:
        raise RuntimeError("vLLM generation failed:\n" + "\n".join(tail))
    ans = [[] for _ in prompts]
    for line in (work / "out.jsonl").read_text().splitlines():
        r = json.loads(line)
        ids = r["ids"]
        if r.get("finish") == "stop" and (not ids or ids[-1] not in stop) and stop:
            ids = ids + [r["stop"] if r.get("stop") is not None else stop[0]]
        ans[r["i"]] = ids
    return ans


def save_like(model, tok, src: Path, dst: Path) -> None:
    """Save the model in the layout of checkpoint src (its config.json, tensor names and any tensors
    the model does not hold, e.g. the vision tower of an image-text checkpoint whose text model was
    trained alone), so engines that load src (vLLM) load dst the same way."""
    import shutil

    from safetensors import safe_open
    from safetensors.torch import load_file, save_file
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    model.save_pretrained(tmp / "raw", safe_serialization=True)
    idx = src / "model.safetensors.index.json"
    src_keys = json.loads(idx.read_text())["weight_map"] if idx.exists() else \
        {k: "model.safetensors" for k in safe_open(str(src / "model.safetensors"), "pt").keys()}

    def name(k: str) -> str:
        if k in src_keys:
            return k
        alt = "model.language_model." + k[len("model."):] if k.startswith("model.") else None
        if alt in src_keys:
            return alt
        raise RuntimeError(f"{k} has no place in {src.name}'s layout")
    n, size, shard, wmap = 0, 0, {}, {}

    def flush():
        nonlocal n, size, shard
        if shard:
            n += 1
            f = f"model-{n:05d}.safetensors"
            save_file(shard, str(tmp / f), metadata={"format": "pt"})
            wmap.update({k: f for k in shard})
        shard, size = {}, 0
    for f in sorted((tmp / "raw").glob("*.safetensors")):
        for k, t in load_file(str(f)).items():
            shard[name(k)] = t
            size += t.numel() * t.element_size()
            if size > 4 * 2 ** 30:
                flush()
    for f in sorted(set(src_keys.values())):   # what the text model does not hold: vision tower, MTP head
        with safe_open(str(src / f), "pt") as h:
            for k in h.keys():
                if k not in wmap and k not in shard and not k.startswith(("model.language_model.", "lm_head")):
                    shard[k] = h.get_tensor(k)
    flush()
    shutil.rmtree(tmp / "raw")
    (tmp / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": wmap}, indent=2))
    for f in src.iterdir():
        if f.is_file() and not f.name.endswith(".safetensors") and f.name != "model.safetensors.index.json":
            shutil.copy2(f, tmp / f.name)
    if tok is not None:
        tok.save_pretrained(tmp)
    shutil.rmtree(dst, ignore_errors=True)
    tmp.rename(dst)


def hf_answers(model, tok, enc: list[list[int]], max_tokens: int, temperature: float, top_p: float, top_k: int,
               eos: list[int], on_done=None, batch: int = 16) -> list[list[int]]:
    """Plain transformers generate, `batch` prompts at a time (slow; CPU tests and fallbacks)."""
    import torch
    dev = model.get_input_embeddings().weight.device
    out = []
    for b0 in range(0, len(enc), batch):
        grp = enc[b0:b0 + batch]
        T = max(map(len, grp))
        ids = torch.tensor([[tok.pad_token_id or 0] * (T - len(e)) + e for e in grp], device=dev)
        am = torch.tensor([[0] * (T - len(e)) + [1] * len(e) for e in grp], device=dev)
        with torch.no_grad():
            g = model.generate(input_ids=ids, attention_mask=am, max_new_tokens=max_tokens, do_sample=True,
                               temperature=temperature, top_p=top_p, top_k=top_k or None, pad_token_id=tok.pad_token_id)
        for j, r in enumerate(g[:, T:].tolist()):
            cut = next((i + 1 for i, t in enumerate(r) if t in eos), len(r))
            out.append(r[:cut])
            if on_done:
                on_done(b0 + j, r[:cut])
    return out
