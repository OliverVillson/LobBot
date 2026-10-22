"""The run plan: how each stage adapts to the detected hardware and the model.

  teacher  vLLM serving the original model for calibration answers: tensor-parallel
           size, memory share, fp8 weights when bf16 does not fit, fp8 KV cache
  reap     transformers forward passes with REAP hooks on bf16 weights: device_map
           and max_memory (one GPU, spread over GPUs, CPU or disk offload), and
           calibration sequences per pass
  layout   whether teacher and REAP share the GPUs, split them, or run one after
           the other
  verify   tensor-parallel size for the pruned model

plan() refuses early (ok=False with problems) when the machine cannot run the model,
and lists every adaptation in notes for the TUI's hardware panel.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field

from reaper_hw.hardware import Hardware
from reaper_hw.model import ModelInfo, model_info

UTIL = 0.90           # vLLM gpu_memory_utilization when the teacher has the GPUs to itself
NATIVE_QUANT = ("fp8", "mxfp4", "compressed-tensors", "fbgemm_fp8")


@dataclass
class Plan:
    ok: bool = True
    dtype: str = "bfloat16"
    layout: str = "sequential"   # shared | split | sequential
    teacher: dict = field(default_factory=dict)
    reap: dict = field(default_factory=dict)
    verify: dict = field(default_factory=dict)
    env: dict = field(default_factory=dict)      # set before starting any stage
    torch_index: str = ""                        # pip --index-url for a torch that runs here
    model: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)     # what was adapted, for the UI
    problems: list[str] = field(default_factory=list)  # why it cannot run

    def fail(self, why: str) -> None:
        self.ok = False
        self.problems.append(why)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


def _ver(s: str) -> tuple[int, ...]:
    return tuple(int(x) for x in s.split(".")[:2] if x.isdigit()) if s else ()


def _pow2_tp(need_gb: float, gpus: int, per_gpu_gb: float) -> int | None:
    """Smallest power-of-two tensor-parallel size within gpus whose memory holds need_gb."""
    tp = 1
    while tp <= gpus:
        if tp * per_gpu_gb >= need_gb:
            return tp
        tp *= 2
    return None


def _software(hw: Hardware, p: Plan) -> None:
    """The CUDA stack has to know the GPU. Blackwell needs CUDA 12.8 builds everywhere."""
    if hw.blackwell:
        p.torch_index = "https://download.pytorch.org/whl/cu128"
        sm = hw.gpus[0].sm
        if hw.cuda and _ver(hw.cuda) < (12, 8):
            p.fail(f"driver {hw.driver} supports CUDA {hw.cuda}; {hw.gpus[0].family} needs a driver "
                   "with CUDA 12.8+ (R570 or newer)")
        if hw.torch:
            native = {sm, sm.replace("sm_", "compute_")} & set(hw.torch_arches)
            if _ver(hw.torch) < (2, 7) or (hw.torch_cuda and _ver(hw.torch_cuda) < (12, 8)) \
                    or (hw.torch_arches and not native):
                p.fail(f"torch {hw.torch} (CUDA {hw.torch_cuda or '?'}) has no {sm} kernels: "
                       f"pip install 'torch>=2.7' --index-url {p.torch_index}")
        # FlashInfer's Blackwell decode and sampler kernels JIT-build at startup and need a
        # matching nvcc (what broke LobBot's first B200 run): use the prebuilt backend.
        p.env |= {"VLLM_ATTENTION_BACKEND": "FLASH_ATTN", "VLLM_USE_FLASHINFER_SAMPLER": "0"}
        p.notes.append(f"Blackwell {sm}: CUDA 12.8 torch, prebuilt FLASH_ATTN backend, no JIT FlashInfer")
    elif hw.gpus:
        p.torch_index = "https://download.pytorch.org/whl/cu126"
        if hw.cuda and _ver(hw.cuda) < (12, 1):
            p.notes.append(f"driver supports only CUDA {hw.cuda}: install a torch built for it")
    if hw.gpus and not hw.torch:
        p.notes.append(f"torch not installed yet: pip install torch --index-url {p.torch_index}")
    p.env |= {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
              "OMP_NUM_THREADS": str(max(1, hw.cpus // max(1, hw.n_gpus)))}
    if hw.n_gpus > 1 and not hw.nvlink:
        p.notes.append("GPUs talk over PCIe (no NVLink): tensor parallel is slower, prefer fewer GPUs")


def plan(hardware: Hardware, model: str | ModelInfo, prune: float = 0.5, calib_seq: int = 2048) -> Plan:
    """How to run REAP on `model` (a Hub id, a local path, or a ModelInfo) at `prune`
    (share of routed experts dropped) on `hardware`. Never raises."""
    hw, p = hardware, Plan()
    if not 0 < prune < 1:
        p.fail(f"prune must be between 0 and 1, got {prune}")
    if isinstance(model, str):
        try:
            model = model_info(model)
        except Exception as e:
            p.fail(str(e))
            return p
    m = model
    p.model = m.to_dict()
    if not hw.gpus:
        p.fail("no NVIDIA GPU found (nvidia-smi and torch see none): run REAPer on the GPU VM")
        return p
    if m.experts < 2:
        p.fail(f"{m.id} is not a mixture-of-experts model (no routed expert count in its config)")
    _software(hw, p)

    if not hw.bf16:
        p.dtype = "float16"
        p.notes.append(f"{hw.arch} {hw.gpus[0].sm} has no bf16: running in fp16")
    n, mem = hw.n_gpus, hw.gpu_mem_gb
    W = m.bf16_gb
    if m.quantized:
        p.notes.append(f"checkpoint is {m.quantized}: REAP dequantizes to {p.dtype} ({W:.0f} GB)")

    # ---- teacher (vLLM) on its own
    kv_need = max(8.0, 0.04 * W)
    t: dict = {"dtype": p.dtype, "quantization": None, "kv_cache_dtype": "auto",
               "max_model_len": max(8192, calib_seq * 2)}
    t_weights = m.checkpoint_gb if m.quantized in NATIVE_QUANT else W
    tp = _pow2_tp(t_weights * 1.05 + kv_need, n, mem * UTIL)
    if tp is None and hw.fp8 and not m.quantized:
        t_weights = W / 2
        tp = _pow2_tp(t_weights * 1.05 + kv_need, n, mem * UTIL)
        if tp is not None:
            t["quantization"] = "fp8"
            p.notes.append(f"teacher: bf16 weights ({W:.0f} GB) do not fit, serving with on-the-fly fp8")
    if tp is None:
        p.fail(f"teacher needs ~{t_weights + kv_need:.0f} GB of GPU memory; this machine has {n} x {mem:.0f} GB")
        tp = n
    t_need = t_weights * 1.05 + kv_need  # what the teacher really needs across its tp GPUs

    # ---- REAP (transformers + hooks) on bf16 weights
    reserve = 6.0 + 0.02 * mem  # CUDA context, activations of one layer, hook scratch
    per_gpu = mem - reserve
    per_seq = calib_seq * max(m.hidden, 1024) * 2 * 48 / 1024**3 \
        + calib_seq * max(m.top_k, 1) * max(m.hidden, 1024) * 4 / 1024**3
    reap_gpus = max(1, math.ceil(W / per_gpu)) if per_gpu > 0 else n + 1

    # ---- layout: split GPUs > share them > one after the other
    share = min(UTIL, math.ceil((t_need / tp) / mem * 20) / 20 + 0.05)  # teacher's slice, 5% steps
    shared_mem = {i: per_gpu - (share * mem if i < tp else 0) for i in range(n)}
    if tp < n and reap_gpus <= n - tp:
        p.layout = "split"
        t_ids, r_ids = list(range(tp)), list(range(tp, n))
        t["gpu_memory_utilization"] = UTIL
        r_mem = {i: per_gpu for i in r_ids}
        p.notes.append(f"teacher on GPU {_ids(t_ids)} and REAP on GPU {_ids(r_ids)}, side by side")
    elif share < UTIL and min(shared_mem.values()) > 0 and sum(shared_mem.values()) >= W:
        p.layout = "shared"
        t_ids, r_ids = list(range(tp)), list(range(n))
        t["gpu_memory_utilization"] = round(share, 2)
        r_mem = shared_mem
        p.notes.append(f"teacher ({share:.0%} of GPU {_ids(t_ids)}) stays up next to REAP on the same GPUs")
    else:
        p.layout = "sequential"
        t_ids, r_ids = list(range(tp)), list(range(n))
        t["gpu_memory_utilization"] = UTIL
        r_mem = {i: per_gpu for i in r_ids}
        p.notes.append("teacher and REAP do not fit together: the teacher writes its answers, "
                       "shuts down, then REAP loads")
    t["tensor_parallel_size"] = tp
    t["gpus"] = t_ids
    free_after = tp * mem * t["gpu_memory_utilization"] - t_weights
    if hw.fp8 and free_after < 2 * kv_need:
        t["kv_cache_dtype"] = "fp8"
        p.notes.append("teacher: little room left for KV cache, using fp8 KV cache")
    t["max_num_seqs"] = int(max(16, min(512, free_after * 6)))
    if tp > 1:
        p.notes.append(f"teacher: tensor parallel over {tp} GPUs")
    p.teacher = t

    gpu_total = sum(r_mem.values())
    cpu_budget = hw.ram_gb * 0.8
    r: dict = {"dtype": p.dtype, "max_seq": calib_seq, "gpus": r_ids, "offload_folder": None,
               "threads": hw.cpus, "tf32": hw.bf16}
    if W <= max(r_mem.values()):
        r["placement"] = "gpu"
        r["device_map"] = f"cuda:{r_ids[0]}"
        headroom = max(r_mem.values()) - W
    elif W <= gpu_total:
        r["placement"] = "multi-gpu"
        r["device_map"] = "auto"
        headroom = (gpu_total - W) / len(r_ids)  # activations live on whichever GPU runs the layer
        p.notes.append(f"REAP: {W:.0f} GB of {p.dtype} weights spread over {len(r_ids)} GPUs")
    elif W <= gpu_total + cpu_budget:
        r["placement"] = "cpu-offload"
        r["device_map"] = "auto"
        headroom = 2.0
        p.notes.append(f"REAP: {W - gpu_total:.0f} GB of weights spill to CPU RAM (slow but works)")
    elif W <= gpu_total + cpu_budget + hw.disk_free_gb * 0.5:
        r["placement"] = "disk-offload"
        r["device_map"] = "auto"
        r["offload_folder"] = f"{hw.disk_path}/offload"
        headroom = 2.0
        p.notes.append(f"REAP: {W - gpu_total - cpu_budget:.0f} GB of weights spill to disk (very slow)")
    else:
        r["placement"] = "none"
        r["device_map"] = None
        headroom = 0.0
        p.fail(f"REAP needs ~{W:.0f} GB for {p.dtype} weights; GPUs + RAM + disk hold "
               f"{gpu_total + cpu_budget + hw.disk_free_gb * 0.5:.0f} GB")
    r["max_memory"] = {str(i): f"{max(1, int(v))}GiB" for i, v in r_mem.items()}
    r["max_memory"]["cpu"] = f"{max(1, int(cpu_budget))}GiB"
    batch = int(max(1, min(32, headroom / max(per_seq, 0.05) / 2))) if headroom > 0 else 1
    r["batch_seqs"] = 1 << (batch.bit_length() - 1)  # power of two
    if r["batch_seqs"] > 1:
        p.notes.append(f"REAP: {r['batch_seqs']} calibration sequences of {calib_seq} tokens per pass")
    p.reap = r

    # ---- verify: vLLM on the pruned model
    pruned = m.pruned_bf16_gb(prune)
    vtp = _pow2_tp(pruned * 1.05 + kv_need, n, mem * UTIL) or tp
    p.verify = {"tensor_parallel_size": vtp, "gpu_memory_utilization": UTIL, "dtype": p.dtype,
                "max_model_len": 4096, "pruned_gb": round(pruned, 1)}

    # ---- disk: original download + pruned copy (+ offloaded weights)
    need = m.checkpoint_gb + pruned + 10 + (W if r["placement"] == "disk-offload" else 0)
    if hw.disk_free_gb < need:
        p.fail(f"needs ~{need:.0f} GB free disk at {hw.disk_path}, has {hw.disk_free_gb:.0f} GB")
    if hw.ram_gb and hw.ram_gb < min(W, 64):
        p.notes.append(f"only {hw.ram_gb:.0f} GB RAM: weights load straight to GPU shard by shard")
    return p


def _ids(ids: list[int]) -> str:
    return f"{ids[0]}" if len(ids) == 1 else f"{ids[0]}-{ids[-1]}"
