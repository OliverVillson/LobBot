#!/usr/bin/env bash
# One-time setup of a fresh GPU VM (B200/B300, Ubuntu + NVIDIA driver) for the BlackHole pipeline:
# LobBot checkout on claude/flint-tui-run, ~/bh-venv with a CUDA build of torch matching the
# driver, REAPer + the harness (blackhole, blackhole-pipeline, blackhole-serve), mlx[cpu] for
# checking MLX exports, and vLLM in ~/vllm-venv (found there by the pipeline). Safe to rerun (it updates).
set -euo pipefail
if [ ! -d ~/LobBot ]; then
  git clone -b claude/flint-tui-run https://github.com/OliverVillson/LobBot.git ~/LobBot
fi
git -C ~/LobBot pull --ff-only
if [ ! -x ~/bh-venv/bin/pip ]; then
  rm -rf ~/bh-venv
  if ! python3 -m venv ~/bh-venv 2>/dev/null; then  # Ubuntu images without ensurepip
    rm -rf ~/bh-venv
    v=$(python3 -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
    sudo apt-get update -qq && sudo apt-get install -y -qq "python$v-venv" python3-venv
    python3 -m venv ~/bh-venv
  fi
fi
# Python headers + a C compiler: Triton builds a small CUDA helper with gcc on first use
v=$(python3 -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
if [ ! -f "/usr/include/python$v/Python.h" ] || ! command -v gcc >/dev/null; then
  sudo apt-get update -qq && sudo apt-get install -y -qq "python$v-dev" build-essential
fi
source ~/bh-venv/bin/activate
pip install -q -U pip
drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
idx=cu130; [ "${drv:-0}" -lt 580 ] && idx=cu128
python -c "import torch" 2>/dev/null || pip install -q torch --index-url https://download.pytorch.org/whl/$idx
pip install -q -e "$HOME/LobBot/reaper[pipeline]" -e "$HOME/LobBot/harness" "mlx[cpu]" mlx-lm || \
  pip install -q -e "$HOME/LobBot/reaper[pipeline]" -e "$HOME/LobBot/harness"
# Qwen3.5 / 3.6 linear attention (Gated DeltaNet) kernels for training; without them transformers
# runs a slow PyTorch reference
pip install -q flash-linear-attention || echo "flash-linear-attention not installed: Qwen3.5/3.6 training will be slow"
python -c "import torch; print('torch', torch.__version__, '·', torch.cuda.get_device_name(0), '·', \
round(torch.cuda.mem_get_info()[1] / 2**30), 'GB')"
# vLLM in its own venv (its own torch): the pipeline uses it for the teacher's answers, the
# student's on-policy answers and LiveCodeBench on models it supports (Qwen3.5 / 3.6; not Flint)
if [ ! -x ~/vllm-venv/bin/python ] || ! ~/vllm-venv/bin/python -c "import vllm" 2>/dev/null; then
  python3 -m venv ~/vllm-venv && ~/vllm-venv/bin/pip install -q -U pip && ~/vllm-venv/bin/pip install -q vllm ninja && \
  ~/vllm-venv/bin/pip install -q -e "$HOME/LobBot/reaper[pipeline]" || \
  echo "vLLM install failed: the pipeline falls back to (much slower) transformers generation"
fi
~/vllm-venv/bin/pip install -q ninja 2>/dev/null || true   # vLLM's JIT kernel builds need it on PATH
~/vllm-venv/bin/python -c "import vllm; from vllm import ModelRegistry as R; print('vllm', vllm.__version__, \
'· Qwen3.5/3.6 MoE', 'yes' if 'Qwen3_5MoeForConditionalGeneration' in R.get_supported_archs() else 'NO')" 2>/dev/null || true
echo "ready: source ~/bh-venv/bin/activate"
