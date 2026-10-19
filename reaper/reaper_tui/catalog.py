"""Providers and the MoE marketplace.

Draft data: model specs are from public model cards as best known and should be
checked before anything real depends on them.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Provider:
    id: str
    name: str
    kind: str  # "inference" or "compute"
    note: str


INFERENCE = [
    Provider("nebius", "Nebius AI Studio", "inference", "OpenAI-compatible, calibration data + judge"),
    Provider("together", "Together AI", "inference", "OpenAI-compatible"),
    Provider("fireworks", "Fireworks AI", "inference", "OpenAI-compatible"),
    Provider("openrouter", "OpenRouter", "inference", "router over many hosts"),
]

COMPUTE = [
    Provider("evroc", "Evroc", "compute", "sovereign EU cloud, GPU VMs"),
    Provider("nebius-cloud", "Nebius Cloud", "compute", "H100 / H200 VMs"),
    Provider("lambda", "Lambda", "compute", "on-demand H100"),
    Provider("runpod", "RunPod", "compute", "pods, community + secure"),
]

GPUS = ["1x H100 80GB", "2x H100 80GB", "4x H100 80GB", "8x H100 80GB", "8x H200 141GB"]


@dataclass(frozen=True)
class MoeModel:
    id: str          # Hugging Face repo
    name: str
    org: str
    total_b: float   # total params, billions
    active_b: float  # active params per token, billions
    experts: int     # routed experts per MoE layer
    top_k: int       # experts per token
    released: str
    gpus: str        # smallest box that fits bf16 weights + REAP scoring
    supported: bool  # REAPer has an adapter for its router

    @property
    def size(self) -> str:
        t = f"{self.total_b / 1000:.0f}T" if self.total_b >= 1000 else f"{self.total_b:g}B"
        return f"{t}-A{self.active_b:g}B"


MARKET = [
    MoeModel("Qwen/Qwen3-30B-A3B-Instruct-2507", "Qwen3 30B A3B", "Qwen", 30.5, 3.3, 128, 8, "2025-07", "1x H100 80GB", True),
    MoeModel("Qwen/Qwen3-Coder-30B-A3B-Instruct", "Qwen3 Coder 30B", "Qwen", 30.5, 3.3, 128, 8, "2025-07", "1x H100 80GB", True),
    MoeModel("Qwen/Qwen3-Next-80B-A3B-Instruct", "Qwen3 Next 80B", "Qwen", 80, 3, 512, 10, "2025-09", "2x H100 80GB", True),
    MoeModel("openai/gpt-oss-20b", "gpt-oss 20B", "OpenAI", 21, 3.6, 32, 4, "2025-08", "1x H100 80GB", True),
    MoeModel("openai/gpt-oss-120b", "gpt-oss 120B", "OpenAI", 117, 5.1, 128, 4, "2025-08", "2x H100 80GB", True),
    MoeModel("zai-org/GLM-4.5-Air", "GLM-4.5 Air", "Z.ai", 106, 12, 128, 8, "2025-07", "4x H100 80GB", True),
    MoeModel("zai-org/GLM-4.6", "GLM-4.6", "Z.ai", 355, 32, 160, 8, "2025-09", "8x H200 141GB", True),
    MoeModel("MiniMaxAI/MiniMax-M2", "MiniMax M2", "MiniMax", 230, 10, 256, 8, "2025-10", "8x H100 80GB", False),
    MoeModel("deepseek-ai/DeepSeek-V3.1", "DeepSeek V3.1", "DeepSeek", 671, 37, 256, 8, "2025-08", "8x H200 141GB", False),
    MoeModel("moonshotai/Kimi-K2-Instruct-0905", "Kimi K2", "Moonshot", 1000, 32, 384, 8, "2025-09", "8x H200 141GB", False),
]


def find_model(query: str) -> MoeModel | None:
    q = query.lower().strip()
    for m in MARKET:
        if q in (m.id.lower(), m.name.lower()):
            return m
    hits = [m for m in MARKET if q in m.id.lower() or q in m.name.lower()]
    return hits[0] if len(hits) == 1 else None


def find_provider(query: str, providers: list[Provider]) -> Provider | None:
    q = query.lower().strip()
    return next((p for p in providers if q in (p.id, p.name.lower()) or p.name.lower().startswith(q)), None)
