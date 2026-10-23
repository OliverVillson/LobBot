"""PyTorch versions of MoE architectures transformers does not ship yet."""


def register_all() -> None:
    """Register them with the transformers Auto classes (no-op where transformers has its own)."""
    from reaper_pipeline.models import flint
    flint.register()
