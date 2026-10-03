#!/usr/bin/env bash
# One-time setup for the evroc B200 VM (Ubuntu 24.04).
#   NVME=/mnt/nvme bash scripts/setup_vm.sh
# Creates two venvs (vLLM and training, which pin different torch versions),
# builds llama.cpp with CUDA, and downloads the teacher and student weights to
# the local NVMe. Safe to re-run. Local NVMe is wiped if the VM is stopped.
set -euo pipefail

NVME=${NVME:-/mnt/nvme}
REPO=$(cd "$(dirname "$0")/.." && pwd)
TEACHER=${TEACHER:-Qwen/Qwen3-30B-A3B-Instruct-2507}
STUDENT=${STUDENT:-google/gemma-4-E4B-it}

say() { printf '\n==> %s\n' "$*"; }

say "GPU"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
mkdir -p "$NVME/models" "$NVME/jobs"
export HF_HOME="$NVME/hf-cache"

say "System packages"
sudo apt-get update -qq
sudo apt-get install -y -qq build-essential cmake git git-lfs python3-venv python3-dev rsync curl
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

say "Downloading weights in the background (log: $NVME/download.log)"
(
  uv tool run --from huggingface_hub hf download "$TEACHER" --local-dir "$NVME/models/${TEACHER##*/}"
  uv tool run --from huggingface_hub hf download "$STUDENT" --local-dir "$NVME/models/${STUDENT##*/}"
  echo DOWNLOADS_DONE
) > "$NVME/download.log" 2>&1 &

say "vLLM venv (data stage)"
uv venv -q --python 3.12 "$NVME/venv-vllm"
uv pip install -q --python "$NVME/venv-vllm/bin/python" vllm pyyaml
uv pip install -q --python "$NVME/venv-vllm/bin/python" -e "$REPO" --no-deps

say "Training venv (reap, heal, quantize, eval, package, API)"
uv venv -q --python 3.12 "$NVME/venv-train"
uv pip install -q --python "$NVME/venv-train/bin/python" \
  "torch>=2.7" "transformers>=5.5" "peft>=0.17" accelerate safetensors \
  anthropic httpx fastapi uvicorn sentencepiece gguf
uv pip install -q --python "$NVME/venv-train/bin/python" -e "$REPO" --no-deps

# Written before the llama.cpp build so the data stage works even if that step fails.
cat > "$REPO/.env.vm" <<ENV
export LOBBOT_MODELS=$NVME/models
export LOBBOT_LLAMA_CPP=$NVME/llama.cpp
export LOBBOT_VLLM_PY=$NVME/venv-vllm/bin/python
export HF_HOME=$NVME/hf-cache
export LOBBOT_JOBS=$NVME/jobs
export PATH=$NVME/venv-train/bin:\$PATH
ENV

say "vLLM smoke test"
"$NVME/venv-vllm/bin/python" -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, 'cuda ok:', torch.cuda.is_available())" || echo 'WARNING: vLLM import failed; the data stage will not run'

say "llama.cpp with CUDA"
if ! command -v nvcc >/dev/null && [ ! -x /usr/local/cuda/bin/nvcc ]; then
  echo "nvcc not found. Install the CUDA toolkit (12.8+ for B200), e.g.:"
  echo "  sudo apt-get install -y cuda-toolkit-12-8   # after adding NVIDIA's apt repo"
  exit 1
fi
export PATH="/usr/local/cuda/bin:$PATH"
[ -d "$NVME/llama.cpp" ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp "$NVME/llama.cpp"
cmake -S "$NVME/llama.cpp" -B "$NVME/llama.cpp/build" -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=100 -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build "$NVME/llama.cpp/build" -j"$(nproc)" --target llama-quantize llama-imatrix llama-server llama-cli
uv pip install -q --python "$NVME/venv-train/bin/python" -r "$NVME/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt" || true


say "Done. Weights still downloading: tail -f $NVME/download.log"
echo "Then: source .env.vm && python pipeline.py --job $NVME/jobs/demo  (after copying a taskspec.json there)"
