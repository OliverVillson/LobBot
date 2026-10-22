import json

import pytest

from reaper_hw import ModelInfo, detect, plan
import reaper_hw.hardware as det
from reaper_hw.cli import main
from reaper_hw.model import model_info_from_config

B200_SMI = "\n".join(f"{i}, NVIDIA B200, 183359, 182900, 10.0, GPU-b200-{i}, 570.124.06" for i in range(8))
H100_SMI = "0, NVIDIA H100 80GB HBM3, 81559, 81000, 9.0, GPU-h100-0, 550.54.15"
H100X2_SMI = H100_SMI + "\n1, NVIDIA H100 80GB HBM3, 81559, 81000, 9.0, GPU-h100-1, 550.54.15"
TOPO_NVLINK = "GPU0\tX\tNV18\nGPU1\tNV18\tX\n"
HEAD = "| NVIDIA-SMI 570.124.06   Driver Version: 570.124.06   CUDA Version: 12.8 |"

QWEN3_30B = {"model_type": "qwen3_moe", "hidden_size": 2048, "num_hidden_layers": 48, "num_attention_heads": 32,
             "num_key_value_heads": 4, "head_dim": 128, "moe_intermediate_size": 768, "intermediate_size": 6144,
             "num_experts": 128, "num_experts_per_tok": 8, "vocab_size": 151936, "decoder_sparse_step": 1}
DEEPSEEK_V3 = {"model_type": "deepseek_v3", "hidden_size": 7168, "num_hidden_layers": 61, "num_attention_heads": 128,
               "q_lora_rank": 1536, "kv_lora_rank": 512, "qk_nope_head_dim": 128, "qk_rope_head_dim": 64,
               "v_head_dim": 128, "n_routed_experts": 256, "n_shared_experts": 1, "moe_intermediate_size": 2048,
               "intermediate_size": 18432, "first_k_dense_replace": 3, "vocab_size": 129280,
               "num_experts_per_tok": 8, "quantization_config": {"quant_method": "fp8"}}
QWEN = model_info_from_config("Qwen/Qwen3-30B-A3B-Instruct-2507", QWEN3_30B)
DSV3 = model_info_from_config("deepseek-ai/DeepSeek-V3.1", DEEPSEEK_V3, checkpoint_bytes=688_600_000_000)


def fake_machine(monkeypatch, tmp_path, smi: str | None, torch: tuple[str, str, list[str]] | None = ("2.8.0", "12.8", ["sm_90", "sm_100"])):
    def run(cmd, timeout=20):
        if smi is None:
            return None
        if cmd == ["nvidia-smi"]:
            return HEAD
        if cmd[:2] == ["nvidia-smi", "topo"]:
            return TOPO_NVLINK
        return smi

    def fake_torch(hw, probe):
        if torch:
            hw.torch, hw.torch_cuda, hw.torch_arches = torch
        return []

    monkeypatch.setattr(det, "_run", run)
    monkeypatch.setattr(det, "_torch", fake_torch)
    monkeypatch.setattr(det, "_meminfo", lambda: (2000.0, 1900.0))
    monkeypatch.setattr(det, "_disk_free", lambda path: 4000.0)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    return detect(str(tmp_path))


def test_detects_8x_b200(monkeypatch, tmp_path):
    hw = fake_machine(monkeypatch, tmp_path, B200_SMI)
    assert hw.n_gpus == 8 and hw.gpus[0].family == "B200" and hw.gpus[0].mem_gb == pytest.approx(179, abs=1)
    assert hw.blackwell and hw.fp8 and hw.fp4 and hw.arch == "Blackwell" and hw.nvlink
    assert hw.driver == "570.124.06" and hw.cuda == "12.8"
    assert hw.summary().startswith("8x B200 179 GB (Blackwell sm_100 NVLink)")
    json.loads(hw.to_json())


def test_cuda_visible_devices(monkeypatch, tmp_path):
    fake_machine(monkeypatch, tmp_path, B200_SMI)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    assert [g.index for g in detect(str(tmp_path)).gpus] == [2, 3]


def test_cc_from_name_on_old_driver(monkeypatch, tmp_path):
    old = "0, NVIDIA B200, 183359, 182900, GPU-x, 570.1"
    calls = iter([None, old])  # first query (with compute_cap) is rejected
    hw = fake_machine(monkeypatch, tmp_path, old)
    monkeypatch.setattr(det, "_run", lambda cmd, timeout=20: next(calls) if "--format" in " ".join(cmd) else "")
    hw = detect(str(tmp_path))
    assert hw.gpus[0].cc == 10.0 and hw.blackwell


def test_no_gpu_fails_gracefully(monkeypatch, tmp_path):
    hw = fake_machine(monkeypatch, tmp_path, None, torch=None)
    assert hw.gpus == [] and hw.summary().startswith("no GPU") and not hw.fp8
    p = plan(hw, QWEN)
    assert not p.ok and "no NVIDIA GPU" in p.problems[0]


def test_unreadable_model_fails_gracefully(monkeypatch, tmp_path):
    hw = fake_machine(monkeypatch, tmp_path, B200_SMI)
    p = plan(hw, str(tmp_path / "nope"))
    assert not p.ok and "config.json" in p.problems[0]


def test_model_sizes_from_config():
    assert QWEN.params_b == pytest.approx(30.5, rel=0.05) and QWEN.experts == 128 and QWEN.top_k == 8
    assert QWEN.expert_params_b / QWEN.params_b > 0.9 and QWEN.moe_layers == 48
    assert DSV3.params_b == pytest.approx(671, rel=0.05) and DSV3.quantized == "fp8" and DSV3.moe_layers == 58
    s = ModelInfo.from_sizes("x", 30.5, 3.3, 128, 8)
    assert s.expert_params_b == pytest.approx(29.0, abs=0.1)


def test_qwen_30b_on_8x_b200_splits_gpus(monkeypatch, tmp_path):
    p = plan(fake_machine(monkeypatch, tmp_path, B200_SMI), QWEN, prune=0.5)
    assert p.ok, p.problems
    assert p.dtype == "bfloat16" and p.layout == "split"
    assert p.teacher["tensor_parallel_size"] == 1 and p.teacher["gpus"] == [0]
    assert p.reap["placement"] == "gpu" and p.reap["device_map"] == "cuda:1" and p.reap["batch_seqs"] >= 16
    assert p.env["VLLM_ATTENTION_BACKEND"] == "FLASH_ATTN" and "cu128" in p.torch_index
    assert any("Blackwell" in n for n in p.notes)


def test_qwen_30b_on_1x_b200_shares_gpu(monkeypatch, tmp_path):
    p = plan(fake_machine(monkeypatch, tmp_path, B200_SMI.splitlines()[0]), QWEN)
    assert p.ok and p.layout == "shared" and p.teacher["gpu_memory_utilization"] < 0.6
    assert int(p.reap["max_memory"]["0"].rstrip("GiB")) > QWEN.bf16_gb


def test_deepseek_on_8x_b200(monkeypatch, tmp_path):
    p = plan(fake_machine(monkeypatch, tmp_path, B200_SMI), DSV3, prune=0.5)
    assert p.ok, p.problems
    assert p.teacher["tensor_parallel_size"] == 8 and p.teacher["quantization"] is None  # native fp8 checkpoint
    assert p.layout == "sequential" and p.reap["placement"] in ("multi-gpu", "cpu-offload")
    assert any("dequantizes" in n for n in p.notes)


def test_fp8_teacher_when_bf16_does_not_fit(monkeypatch, tmp_path):
    glm = ModelInfo.from_sizes("zai-org/GLM-4.5-Air", 106, 12, 128, 8, hidden=4096)
    p = plan(fake_machine(monkeypatch, tmp_path, H100X2_SMI, torch=("2.6.0", "12.4", ["sm_90"])), glm)
    assert p.ok, p.problems
    assert p.teacher["quantization"] == "fp8" and p.teacher["tensor_parallel_size"] == 2
    assert p.layout == "sequential" and p.reap["placement"] == "cpu-offload"


def test_old_torch_on_blackwell_is_a_problem(monkeypatch, tmp_path):
    p = plan(fake_machine(monkeypatch, tmp_path, B200_SMI, torch=("2.5.1", "12.4", ["sm_80", "sm_90"])), QWEN)
    assert not p.ok and any("sm_100" in x for x in p.problems)


def test_not_moe(monkeypatch, tmp_path):
    dense = ModelInfo("meta-llama/Llama-3.1-8B", params_b=8, hidden=4096)
    assert not plan(fake_machine(monkeypatch, tmp_path, H100_SMI), dense).ok


def test_cli_json(monkeypatch, tmp_path, capsys):
    fake_machine(monkeypatch, tmp_path, B200_SMI)
    assert main(["--sizes", "30.5,3.3,128,8", "--json", "--work-dir", str(tmp_path)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["hardware"]["fp8"] and out["plan"]["ok"]
    assert main(["--work-dir", str(tmp_path)]) == 0
    assert "8x B200" in capsys.readouterr().out
