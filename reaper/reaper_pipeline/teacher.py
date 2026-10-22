"""The teacher: the unpruned model, generating on the local GPUs.

vLLM when it is installed and there is a GPU (fast, sized by the run profile's
teacher settings), else transformers generate (slow, but works anywhere,
including CPU tests on tiny models).
"""

from __future__ import annotations

import os

from reaper_pipeline.events import log


def pick_backend(prof: dict, wanted: str = "auto") -> str:
    if wanted in ("vllm", "hf"):
        return wanted
    if prof.get("device") != "cuda":
        return "hf"
    try:
        import vllm  # noqa: F401
        return "vllm"
    except ImportError:
        return "hf"


class VllmTeacher:
    def __init__(self, model_dir: str, prof: dict, trust_remote_code: bool = False, seed: int = 0):
        for k, v in (prof.get("env") or {}).items():
            os.environ.setdefault(k, str(v))
        from vllm import LLM

        t = prof.get("teacher") or {}
        kw = {k: t[k] for k in ("tensor_parallel_size", "gpu_memory_utilization", "quantization",
                                 "max_model_len", "kv_cache_dtype", "max_num_seqs") if t.get(k) is not None}
        kw["tensor_parallel_size"] = max(1, kw.get("tensor_parallel_size", 1))
        log(f"teacher: vLLM {kw}")
        self.llm = LLM(model=str(model_dir), dtype=prof.get("dtype", "auto"), trust_remote_code=trust_remote_code,
                       seed=seed, **kw)

    def chat(self, convs: list[list[dict]], max_tokens: int, temperature: float = 0.7) -> list[str]:
        from vllm import SamplingParams
        sp = SamplingParams(max_tokens=max_tokens, temperature=temperature, top_p=0.95)
        outs = self.llm.chat(convs, sp, use_tqdm=False)
        return [o.outputs[0].text for o in outs]

    def close(self) -> None:
        del self.llm


class HfTeacher:
    def __init__(self, model_dir: str, prof: dict, trust_remote_code: bool = False, seed: int = 0):
        import torch

        from reaper_pipeline.load import load_model, load_tokenizer
        torch.manual_seed(seed)
        self.tok = load_tokenizer(model_dir, trust_remote_code)
        self.tok.padding_side = "left"
        self.model = load_model(model_dir, prof, trust_remote_code)
        self.batch = max(1, int((prof.get("reap") or {}).get("batch_seqs", 1)) * 4)

    def _render(self, conv: list[dict]) -> str:
        if getattr(self.tok, "chat_template", None):
            return self.tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
        return "\n\n".join(m["content"] for m in conv) + "\n\n"

    def chat(self, convs: list[list[dict]], max_tokens: int, temperature: float = 0.7) -> list[str]:
        import torch

        from reaper_pipeline.load import input_device
        out: list[str] = []
        for i in range(0, len(convs), self.batch):
            texts = [self._render(c) for c in convs[i:i + self.batch]]
            enc = self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(input_device(self.model))
            with torch.no_grad():
                gen = self.model.generate(**enc, max_new_tokens=max_tokens, do_sample=temperature > 0,
                                          temperature=max(temperature, 1e-5), top_p=0.95,
                                          pad_token_id=self.tok.pad_token_id)
            new = gen[:, enc["input_ids"].shape[1]:]
            out += self.tok.batch_decode(new, skip_special_tokens=True)
        return out

    def close(self) -> None:
        del self.model


def make_teacher(model_dir: str, prof: dict, backend: str = "auto", trust_remote_code: bool = False, seed: int = 0):
    b = pick_backend(prof, backend)
    return (VllmTeacher if b == "vllm" else HfTeacher)(model_dir, prof, trust_remote_code, seed)
