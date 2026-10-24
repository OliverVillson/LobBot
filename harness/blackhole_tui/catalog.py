"""What the harness offers: source models, the machines people want to run the
result on, and schemas (post-training recipes) from the marketplace.

Draft data. Model specs are from public model cards as best known (the MoE list
is REAPer's); schemas are illustrative until real runs back them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# bytes per parameter after the final quant step
QUANT_BYTES = {"bf16": 2.0, "fp8": 1.0, "int4": 0.52, "Q8_0": 1.06, "Q5_K": 0.69, "Q4_K": 0.56, "Q3_K": 0.43, "mixed-3bpw": 0.40,
               "mlx-3bit": 0.45}


@dataclass(frozen=True)
class Model:
    id: str          # Hugging Face repo
    name: str
    org: str
    total_b: float   # total params, billions
    active_b: float  # active params per token, billions
    experts: int     # routed experts per MoE layer (0 = dense)
    top_k: int
    released: str

    @property
    def moe(self) -> bool:
        return self.experts > 0

    @property
    def size(self) -> str:
        if not self.total_b:
            return "?"
        t = f"{self.total_b / 1000:.0f}T" if self.total_b >= 1000 else f"{self.total_b:g}B"
        return f"{t}-A{self.active_b:g}B" if self.moe else t

    @property
    def shape(self) -> str:
        return f"{self.experts} top-{self.top_k}" if self.moe else "dense"


MODELS = [
    Model("RecursiveSI/flint", "Flint", "Recursive", 39.8, 2.5, 256, 6, "2026-10"),
    Model("Qwen/Qwen3-30B-A3B-Instruct-2507", "Qwen3 30B A3B", "Qwen", 30.5, 3.3, 128, 8, "2025-07"),
    Model("Qwen/Qwen3-Coder-30B-A3B-Instruct", "Qwen3 Coder 30B", "Qwen", 30.5, 3.3, 128, 8, "2025-07"),
    Model("Qwen/Qwen3-Next-80B-A3B-Instruct", "Qwen3 Next 80B", "Qwen", 80, 3, 512, 10, "2025-09"),
    Model("openai/gpt-oss-20b", "gpt-oss 20B", "OpenAI", 21, 3.6, 32, 4, "2025-08"),
    Model("openai/gpt-oss-120b", "gpt-oss 120B", "OpenAI", 117, 5.1, 128, 4, "2025-08"),
    Model("zai-org/GLM-4.5-Air", "GLM-4.5 Air", "Z.ai", 106, 12, 128, 8, "2025-07"),
    Model("zai-org/GLM-4.6", "GLM-4.6", "Z.ai", 355, 32, 160, 8, "2025-09"),
    Model("MiniMaxAI/MiniMax-M2", "MiniMax M2", "MiniMax", 230, 10, 256, 8, "2025-10"),
    Model("deepseek-ai/DeepSeek-V3.1", "DeepSeek V3.1", "DeepSeek", 671, 37, 256, 8, "2025-08"),
    Model("moonshotai/Kimi-K2-Instruct-0905", "Kimi K2", "Moonshot", 1000, 32, 384, 8, "2025-09"),
]


@dataclass(frozen=True)
class Target:
    """A machine the finished model should run on."""
    key: str
    name: str
    mem_gb: float    # memory the weights may use (VRAM, or unified memory minus the OS)


TARGETS = [
    Target("rtx3060", "RTX 3060 12 GB", 11),
    Target("rtx4070", "RTX 4070 12 GB", 11),
    Target("rtx4090", "RTX 4090 24 GB", 22),
    Target("rtx5090", "RTX 5090 32 GB", 30),
    Target("mac16", "Mac 16 GB", 11),
    Target("mac32", "Mac 32 GB", 24),
    Target("mac64", "Mac 64 GB", 50),
    Target("cpu32", "CPU, 32 GB RAM", 26),
]


@dataclass(frozen=True)
class Step:
    kind: str        # synth · reap · heal · sft · dpo · qat
    what: str        # one line for the UI
    params: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Schema:
    """A post-training recipe. Runs between calibrate and eval; its last word is
    the final quant. Everything here round-trips to a schema file."""
    id: str
    name: str
    author: str
    summary: str
    steps: tuple[Step, ...]
    quant: str = "Q4_K"
    runs: int = 0
    proven: bool = False
    tags: tuple[str, ...] = ()
    stage_list: tuple[str, ...] = ()   # a real pipeline's own stage names; default: built from the steps

    @property
    def prune(self) -> float:
        return next((s.params.get("prune", 0.0) for s in self.steps if s.kind == "reap"), 0.0)

    def stages(self) -> list[str]:
        if self.stage_list:
            return list(self.stage_list)
        names, seen = [], {}
        for s in self.steps:
            seen[s.kind] = seen.get(s.kind, 0) + 1
            names.append(s.kind if seen[s.kind] == 1 else f"{s.kind}{seen[s.kind]}")
        # GPU formats stay safetensors; everything else ends as one GGUF for llama.cpp / Ollama
        final = "quant" if self.quant in ("fp8", "bf16") else "gguf"
        return ["detect", "plan", "weights", "calibrate", *names, "eval", final, "package"]


SCHEMAS = [
    Schema("reaq-codebook", "REAQ codebook (lab)", "lab", "Best method of 2026-10-23, experimental. REAP drops "
           "25% of routed experts; GPTQ int4 makes the shippable model; then REAQ per-expert codebooks (top half "
           "of each layer's experts by REAP score on 16-level tables, the rest 8-level) are measured against it "
           "and saved for a future kernel. Runs on the real pipeline (blackhole_pipeline).",
           (Step("reap", "drop 25% of routed experts by REAP score", {"prune": 0.25}),
            Step("gptq", "GPTQ int4 g128 on the routed experts (shipped)", {"gptq_calib": 64}),
            Step("reaq", "REAQ codebooks 16/8-level by REAP rank (lab, measured)", {"reaq_top": 0.5, "reaq_block": 32})),
           quant="int4", tags=("lab", "experimental", "moe"),
           stage_list=("detect", "plan", "weights", "calibrate", "score", "prune", "eval", "gptq", "reaq", "bench",
                       "quant", "package")),
    Schema("mac-16gb", "Mac 16 GB coder", "blackhole", "A coding model that fits a 16 GB Apple-silicon Mac "
           "(~9.5 GB): REAP drops half the routed experts, calibrated on code; the pruned model is distilled from "
           "the teacher on coding prompts; then an MLX folder with GPTQ 3-bit experts (top 12.5% by REAP on 4-bit, "
           "REAQ), 4-bit attention and embeddings, served by mlx_lm.server.",
           (Step("reap", "drop 50% of routed experts, calibrated on code", {"prune": 0.5, "calib": "code"}),
            Step("heal", "distil from the unpruned teacher on 1k coding prompts", {"heal_prompts": 1024}),
            Step("mlx", "MLX: GPTQ 3-bit experts, REAQ 4-bit hot tier, 4-bit attention",
                 {"mlx_bits": 3, "mlx_group": 64, "mlx_hot": 0.125, "mlx_hot_bits": 4, "bench_models": "original,mlx"}),
            Step("qheal", "distil again around the 3-bit model so the heal repairs the quantization too", {})),
           quant="mlx-3bit", tags=("mac", "code", "moe", "lab"),
           stage_list=("detect", "plan", "weights", "calibrate", "score", "prune", "eval", "heal", "mlx", "qheal", "bench",
                       "package")),
    Schema("mac-16gb-375", "Mac 16 GB coder, lighter prune", "blackhole", "Like Mac 16 GB coder but REAP drops "
           "37.5% of routed experts and the MLX groups are 128 wide (~11 GB): better quality, needs the Mac's GPU "
           "memory limit raised (sudo sysctl iogpu.wired_limit_mb=12288) and few other apps open.",
           (Step("reap", "drop 37.5% of routed experts, calibrated on code", {"prune": 0.375, "calib": "code"}),
            Step("heal", "distil from the unpruned teacher on 1k coding prompts", {"heal_prompts": 1024}),
            Step("mlx", "MLX: GPTQ 3-bit g128 experts, 4-bit attention",
                 {"mlx_bits": 3, "mlx_group": 128, "mlx_hot": 0.0, "bench_models": "original,mlx"}),
            Step("qheal", "distil again around the 3-bit model so the heal repairs the quantization too", {})),
           quant="mlx-3bit", tags=("mac", "code", "moe", "lab"),
           stage_list=("detect", "plan", "weights", "calibrate", "score", "prune", "eval", "heal", "mlx", "qheal", "bench",
                       "package")),
    Schema("reap-classic", "REAP classic", "blackhole", "REAPer's pipeline: prune experts by router-weighted "
           "activation on purpose-built calibration data, then store experts in fp8.",
           (Step("reap", "drop 50% of routed experts per layer", {"prune": 0.5}),), quant="fp8",
           runs=2, proven=True, tags=("moe", "fast")),
    Schema("lobbot-laptop", "LobBot laptop", "blackhole", "The hackathon chain: teacher writes task data, REAP "
           "50%, LoRA heal, mixed-bit GGUF with expert bits from REAP saliency × imatrix. Qwen3-30B went "
           "61 GB → 6.5 GB at 99% of teacher on its task.",
           (Step("synth", "teacher brainstorms scenarios, writes inputs, answers", {"n": 12000}),
            Step("reap", "drop 50% of experts on that data", {"prune": 0.5}),
            Step("heal", "LoRA heal on teacher answers, merged", {"epochs": 1})),
           quant="mixed-3bpw", runs=3, proven=True, tags=("laptop", "moe", "ollama")),
    Schema("pocket-coder", "Pocket coder", "blackhole", "A coding model for one GPU: REAP on code, then LoRA-heal "
           "with distillation from the unpruned teacher.",
           (Step("reap", "drop 50% of experts, calibrated on code", {"prune": 0.5, "calib": "code"}),
            Step("heal", "LoRA heal: 1 epoch of teacher logits on 40k code prompts", {"tokens": "60M"})),
           quant="Q4_K", tags=("code", "moe")),
    Schema("laptop-chat", "Laptop chat", "blackhole", "Aggressive cut for 16 GB laptops: heavy REAP, distil "
           "heal, a light DPO pass so it keeps its manners.",
           (Step("reap", "drop 62% of experts, chat calibration", {"prune": 0.62, "calib": "chat"}),
            Step("heal", "LoRA heal: teacher logits on 30k chat turns", {"tokens": "45M"}),
            Step("dpo", "teacher-ranked pairs, 4k", {"pairs": 4000})),
           quant="Q4_K", tags=("chat", "small")),
    Schema("domain-expert", "Domain expert", "blackhole", "Bring your documents: the teacher writes Q&A over "
           "them, REAP keeps the experts that answer it, SFT bakes it in.",
           (Step("synth", "teacher writes 20k Q&A from ./docs", {"source": "./docs", "n": 20000}),
            Step("reap", "drop 50% of experts on that Q&A", {"prune": 0.5}),
            Step("sft", "LoRA r=64, 2 epochs on the Q&A", {"lora_r": 64, "epochs": 2})),
           quant="Q5_K", tags=("rag", "docs")),
    Schema("event-horizon", "Event Horizon", "blackhole", "Expert mode: train inside the GGUF quant grid "
           "(QAT), pack the trained integers byte-exact, verify parity. Best sub-4-bit quality; hours, not minutes.",
           (Step("reap", "drop 50% of experts", {"prune": 0.5}),
            Step("heal", "LoRA heal with logit KD from the teacher", {"tokens": "80M"}),
            Step("qat", "Q3_K grid QAT: block-AP, then end-to-end KD", {"rounds": 3})),
           quant="Q3_K", tags=("expert", "sub-4-bit")),
    Schema("persona", "Persona", "community", "Describe a character; the teacher role-plays thousands of "
           "conversations; the student learns the voice.",
           (Step("synth", "teacher writes 8k persona dialogues", {"n": 8000}),
            Step("sft", "full fine-tune, 1 epoch", {"epochs": 1}),
            Step("reap", "drop 40% of experts on the dialogues", {"prune": 0.4})),
           quant="Q4_K", tags=("chat", "fun")),
    Schema("tiny-agent", "Tiny agent", "community", "Tool-calling agent: REAP on agent traces, heal, keep the "
           "function-call format exact.",
           (Step("reap", "drop 50% of experts on tool-use traces", {"prune": 0.5, "calib": "agent"}),
            Step("heal", "LoRA heal on 20k teacher tool calls", {"tokens": "35M"}),
            Step("sft", "format lock: 2k strict JSON calls", {"epochs": 1})),
           quant="Q5_K", tags=("agent", "tools")),
]


def find_model(query: str) -> Model | None:
    q = query.lower().strip()
    for m in MODELS:
        if q in (m.id.lower(), m.name.lower()):
            return m
    hits = [m for m in MODELS if q in m.id.lower() or q in m.name.lower()]
    return hits[0] if len(hits) == 1 else None


def custom_model(repo: str) -> Model:
    return Model(repo, repo.rstrip("/").split("/")[-1], repo.split("/")[0] if "/" in repo else "local",
                 0, 0, 0, 0, "?")


def model_from_config(cfg: dict) -> Model:
    """The model of a job, including a detected one that is not in the list."""
    sh = cfg.get("model_shape") or {}
    if not sh:
        return custom_model(cfg["model"])
    return Model(cfg["model"], sh["name"], sh["org"], sh["total_b"], sh["active_b"], sh["experts"], sh["top_k"], "?")


def find_schema(query: str) -> Schema | None:
    q = query.lower().strip()
    hits = [s for s in SCHEMAS if q in (s.id, s.name.lower())] or \
           [s for s in SCHEMAS if q in s.id or q in s.name.lower()]
    return hits[0] if len(hits) == 1 else None


def find_target(query: str) -> Target | None:
    q = query.lower().replace(" ", "")
    hits = [t for t in TARGETS if q == t.key] or [t for t in TARGETS if q in t.key or q in t.name.lower().replace(" ", "")]
    return hits[0] if len(hits) == 1 else None


def out_gb(model: Model, schema: Schema | None) -> float:
    """Rough size of the finished weights. Experts are most of an MoE's params;
    ~10% (attention, router, shared, embeddings) is never pruned."""
    if not model.total_b:
        return 0.0
    prune = schema.prune if schema else 0.0
    kept = model.total_b * (0.1 + 0.9 * (1 - prune)) if model.moe else model.total_b
    return kept * QUANT_BYTES[schema.quant if schema else "bf16"]


def fit_mark(model: Model, schema: Schema | None, target: Target) -> str:
    gb = out_gb(model, schema)
    if not gb:
        return "?"
    if gb <= target.mem_gb * 0.85:
        return "✓"
    return "~" if gb <= target.mem_gb * 1.15 else "✗"
