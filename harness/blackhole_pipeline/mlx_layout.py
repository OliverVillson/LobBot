"""How a BlackHole MLX folder names and stores a model's tensors, per architecture.

The pipeline works on transformers' PyTorch model ("HF names"); the MLX folder holds the names
and layouts mlx-lm's model code loads directly ("MLX names"), next to the folder's model file:

  flint        flint_mlx.py; the same names as REAPer's PyTorch Flint
  qwen3_5_moe  qwen_mlx.py (mlx-lm's Qwen3.5 / 3.6 MoE with BlackHole's expert tiers and HEX);
               text model under language_model.*, zero-centred RMSNorm scales stored +1, the
               short convolution as [C, K, 1]; the vision tower is not exported (text only)

Routed experts are stored by tier ("<layer prefix>.experts_hot" / ".experts") in either case;
`to_hf_experts` says how serve._mlx_state rebuilds the PyTorch model's experts from the tiers.
"""

from __future__ import annotations

from pathlib import Path

MLX_DIR = Path(__file__).resolve().parent / "mlx"
ZERO_CENTRED = (".input_layernorm.weight", ".post_attention_layernorm.weight", ".q_norm.weight", ".k_norm.weight")


class Flint:
    model_type = "flint"
    model_file = "flint_mlx.py"
    files = ("flint_mlx.py",)
    fused_experts = False   # PyTorch Flint: one module per expert

    def to_mlx(self, name: str) -> str:
        return name

    def to_hf(self, name: str) -> str:
        return name

    def tensor_to_mlx(self, name: str, t):
        return t

    def tensor_to_hf(self, name: str, t):
        return t


class Qwen35Moe(Flint):
    model_type = "qwen3_5_moe"
    model_file = "qwen_mlx.py"
    files = ("qwen_mlx.py", "flint_mlx.py")   # qwen_mlx imports the tiers and HEX kernels from flint_mlx
    fused_experts = True    # transformers 5: [E, 2I, H] gate_up_proj and [E, H, I] down_proj

    def to_mlx(self, name: str) -> str:
        if name.startswith("model."):
            return "language_model.model." + name[len("model."):]
        if name.startswith("lm_head"):
            return "language_model." + name
        return name

    def to_hf(self, name: str) -> str:
        if name.startswith("language_model.model."):
            return "model." + name[len("language_model.model."):]
        if name.startswith("language_model.lm_head"):
            return name[len("language_model."):]
        return name

    @staticmethod
    def _centred(name: str) -> bool:
        return name.endswith(ZERO_CENTRED) or name == "model.norm.weight"

    def tensor_to_mlx(self, name: str, t):
        if self._centred(name):
            return t + 1.0
        if name.endswith("conv1d.weight") and t.dim() == 3 and t.shape[-1] != 1:
            return t.transpose(1, 2).contiguous()      # [C, 1, K] -> [C, K, 1]
        return t

    def tensor_to_hf(self, name: str, t):
        if self._centred(name):
            return t - 1.0
        if name.endswith("conv1d.weight") and t.dim() == 3 and t.shape[-1] == 1:
            return t.transpose(1, 2).contiguous()
        return t


LAYOUTS = {"flint": Flint(), "qwen3_5_moe": Qwen35Moe(), "qwen3_5_moe_text": Qwen35Moe()}


def layout_for(cfg: dict):
    mt = cfg.get("model_type")
    if mt not in LAYOUTS:
        raise RuntimeError(f"MLX export supports {sorted(set(LAYOUTS))}; this model is {mt!r}")
    return LAYOUTS[mt]


def layout_of_folder(cfg: dict):
    """The layout of an exported MLX folder (by its model file)."""
    for lay in LAYOUTS.values():
        if cfg.get("model_file") == lay.model_file:
            return lay
    return None
