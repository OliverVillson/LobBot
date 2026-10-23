"""CPU micro-benchmark: REAP scoring pass, reference Saliency vs FastSaliency vs no hooks.
Relative numbers only; on a GPU the per-expert host syncs the fast path removes matter more."""
import sys, tempfile, time
from pathlib import Path
import torch
from tiny_moe import build
from transformers import AutoModelForCausalLM
from reaper_pipeline import moe
from fast_saliency import FastSaliency

torch.manual_seed(0)
torch.set_num_threads(4)
p = build("qwen3_moe", Path(tempfile.mkdtemp()) / "m", hidden_size=512, intermediate_size=1024,
          moe_intermediate_size=256, num_experts=64, num_experts_per_tok=8, num_hidden_layers=4,
          num_attention_heads=8, num_key_value_heads=4, head_dim=64)
m = AutoModelForCausalLM.from_pretrained(p, dtype=torch.float32).eval()
ids = torch.randint(5, 300, (4, 512))
mask = torch.ones_like(ids)

def t(col):
    best = 1e9
    for _ in range(3):
        if col is not None:
            col.mask = mask.reshape(-1).bool()
        t0 = time.perf_counter()
        with torch.no_grad():
            m(input_ids=ids, attention_mask=mask, use_cache=False, logits_to_keep=1)
        best = min(best, time.perf_counter() - t0)
    return best

base = t(None)
ref = moe.Saliency(m); r = t(ref); ref.remove()
fast = FastSaliency(m); f = t(fast); fast.remove()
print(f"no hooks {base*1e3:.0f} ms | reference Saliency {r*1e3:.0f} ms ({r/base:.2f}x) | FastSaliency {f*1e3:.0f} ms ({f/base:.2f}x) | speedup {r/f:.2f}x")
