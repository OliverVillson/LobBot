"""Batch generation with vLLM, run as its own process under vLLM's Python (BLACKHOLE_VLLM_PY,
usually ~/vllm-venv/bin/python), so vLLM and its torch never share a process with the training
code. Standalone on purpose: it imports nothing from the pipeline.

    python vllm_gen.py JOB.json

JOB.json: {"model": dir, "prompts": [[token ids], ...], "max_tokens", "temperature", "top_p",
"top_k", "stop": [eos ids], "allowed": [token ids] | null, "gpu_memory_utilization", "max_model_len",
"out": path}. Writes one JSON line per prompt to `out`: {"i": index, "ids": [answer token ids], "finish", "stop"},
and progress lines "vllm_gen: done N/M" on stdout.
"""

import json
import os
import sys


def main(path: str) -> None:
    job = json.loads(open(path).read())
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    cfg = json.loads(open(os.path.join(job["model"], "config.json")).read())
    kw = {"limit_mm_per_prompt": {"image": 0, "video": 0}} if "vision_config" in cfg else {}  # text only
    llm = LLM(model=job["model"], dtype="bfloat16", trust_remote_code=True, seed=job.get("seed", 0),
              gpu_memory_utilization=job["gpu_memory_utilization"], max_model_len=job["max_model_len"],
              enable_prefix_caching=True, **kw)
    sp = SamplingParams(max_tokens=job["max_tokens"], temperature=job["temperature"], top_p=job["top_p"],
                        top_k=job["top_k"] or -1, stop_token_ids=job["stop"], allowed_token_ids=job.get("allowed"))
    prompts = job["prompts"]
    step = max(1, min(len(prompts), 512))
    with open(job["out"], "w") as out:
        for b0 in range(0, len(prompts), step):
            batch = [TokensPrompt(prompt_token_ids=p) for p in prompts[b0:b0 + step]]
            for i, o in enumerate(llm.generate(batch, sp, use_tqdm=False)):
                c = o.outputs[0]
                out.write(json.dumps({"i": b0 + i, "ids": list(c.token_ids), "finish": c.finish_reason,
                                      "stop": c.stop_reason if isinstance(c.stop_reason, int) else None}) + "\n")
            out.flush()
            print(f"vllm_gen: done {min(b0 + step, len(prompts))}/{len(prompts)}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
