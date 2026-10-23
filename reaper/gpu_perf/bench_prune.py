"""Disk benchmark: REAPer's ckpt.prune_checkpoint vs fast_prune on a synthetic 1.6 GB checkpoint."""
import json, os, shutil, sys, tempfile, time
from pathlib import Path
import torch
from safetensors.torch import save_file
from reaper_pipeline import ckpt
import fast_prune

root = Path(sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp())
src = root / "src"
L, E, H, I = 8, 32, 1024, 512
if not src.exists():
    src.mkdir(parents=True)
    wm = {}
    for l in range(L):
        t = {f"model.layers.{l}.mlp.experts.{j}.{p}.weight": torch.randn(*(s), dtype=torch.bfloat16)
             for j in range(E) for p, s in (("gate_proj", (I, H)), ("up_proj", (I, H)), ("down_proj", (H, I)))}
        t[f"model.layers.{l}.mlp.gate.weight"] = torch.randn(E, H, dtype=torch.bfloat16)
        t[f"model.layers.{l}.self_attn.q_proj.weight"] = torch.randn(H, H, dtype=torch.bfloat16)
        f = f"model-{l:05d}.safetensors"
        save_file(t, str(src / f), metadata={"format": "pt"})
        wm |= {k: f for k in t}
    (src / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": wm}))
    (src / "config.json").write_text(json.dumps({"num_hidden_layers": L, "num_experts": E, "num_experts_per_tok": 4}))
keep = {l: list(range(0, E, 2)) for l in range(L)}
size = sum(p.stat().st_size for p in src.glob("*.safetensors")) / 1024**3
os.system("sync")
t0 = time.time(); ckpt.prune_checkpoint(src, root / "ref", keep, E, L); a = time.time() - t0
t0 = time.time(); fast_prune.prune_checkpoint(src, root / "fast", keep, E, L, workers=8); b = time.time() - t0
print(f"{size:.2f} GB checkpoint, 50% of experts kept: ckpt.prune_checkpoint {a:.1f}s, fast_prune {b:.1f}s ({a / b:.1f}x)")
shutil.rmtree(root / "ref"); shutil.rmtree(root / "fast")
