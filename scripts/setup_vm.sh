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

# evroc's GPU image installs the NVIDIA driver on first boot and then reboots.
command -v cloud-init >/dev/null && sudo cloud-init status --wait >/dev/null || true

say "GPU"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

say "Local NVMe at $NVME"
if ! mountpoint -q "$NVME"; then
  # The local disk arrives blank. Only a disk with no filesystem or partitions is formatted.
  DISK=$(lsblk -dbno NAME,SIZE,TYPE | awk '$3=="disk" && $2>1e12 {print "/dev/"$1}' | while read -r d; do
    [ -z "$(sudo blkid -o value -s TYPE "$d")" ] && [ "$(lsblk -no NAME "$d" | wc -l)" = 1 ] && echo "$d"; done | head -1)
  [ -n "$DISK" ] || { echo "No blank disk over 1 TB to mount at $NVME"; exit 1; }
  sudo mkfs.ext4 -q -L nvme -E nodiscard,lazy_itable_init=1,lazy_journal_init=1 "$DISK"
  sudo mkdir -p "$NVME"
  grep -q "LABEL=nvme" /etc/fstab || echo "LABEL=nvme $NVME ext4 defaults,noatime,nofail 0 2" | sudo tee -a /etc/fstab
  sudo mount "$NVME"
fi
sudo chown "$(id -u):$(id -g)" "$NVME"
df -h "$NVME"
mkdir -p "$NVME/models" "$NVME/jobs"
export HF_HOME="$NVME/hf-cache"
# The root disk is 14 GB, too small for the torch and vLLM wheels in uv's cache.
export UV_CACHE_DIR="$NVME/uv-cache"

say "System packages"
sudo apt-get update -qq
sudo apt-get install -y -qq build-essential cmake git git-lfs python3-venv python3-dev rsync curl
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

if grep -q DOWNLOADS_DONE "$NVME/download.log" 2>/dev/null; then
  say "Weights already downloaded"
elif pgrep -f "hf download" >/dev/null; then
  say "Weights download already running (log: $NVME/download.log)"
else
say "Downloading weights in the background (log: $NVME/download.log)"
(
  uv tool run --from huggingface_hub hf download "$TEACHER" --local-dir "$NVME/models/${TEACHER##*/}"
  uv tool run --from huggingface_hub hf download "$STUDENT" --local-dir "$NVME/models/${STUDENT##*/}"
  echo DOWNLOADS_DONE
) > "$NVME/download.log" 2>&1 &
fi

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
export UV_CACHE_DIR=$NVME/uv-cache
export LOBBOT_JOBS=$NVME/jobs
export PATH=$NVME/venv-train/bin:\$PATH
# API keys (GEMINI_API_KEY, ANTHROPIC_API_KEY) live outside the repo, mode 600.
if [ -f \$HOME/.lobbot-env ]; then . \$HOME/.lobbot-env; fi
ENV

say "vLLM smoke test"
"$NVME/venv-vllm/bin/python" -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, 'cuda ok:', torch.cuda.is_available())" || echo 'WARNING: vLLM import failed; the data stage will not run'

say "llama.cpp with CUDA"
if ! command -v nvcc >/dev/null && [ ! -x /usr/local/cuda/bin/nvcc ]; then
  # Just the compiler and the libraries llama.cpp links; the full toolkit does not fit the root disk.
  # evroc's image already has NVIDIA's apt repo; add it if not.
  if ! apt-cache show cuda-nvcc-12-9 >/dev/null 2>&1; then
    curl -fsSLo /tmp/cuda-keyring.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i /tmp/cuda-keyring.deb && sudo apt-get update -qq
  fi
  sudo apt-get install -y -qq --no-install-recommends cuda-nvcc-12-9 cuda-cudart-dev-12-9 libcublas-dev-12-9 cuda-nvrtc-dev-12-9
  sudo apt-get clean
fi
export PATH="/usr/local/cuda/bin:$PATH"
[ -d "$NVME/llama.cpp" ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp "$NVME/llama.cpp"
cmake -S "$NVME/llama.cpp" -B "$NVME/llama.cpp/build" -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=100 -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build "$NVME/llama.cpp/build" -j"$(nproc)" --target llama-quantize llama-imatrix llama-server llama-cli
# llama.cpp pins a CPU-only torch; installing it as-is replaces the venv's CUDA torch.
grep -v -E '^(torch|--extra-index-url)' "$NVME/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt" > /tmp/convert-reqs.txt
uv pip install -q --python "$NVME/venv-train/bin/python" -r /tmp/convert-reqs.txt || true
"$NVME/venv-train/bin/python" -c "import torch; assert torch.cuda.is_available(), torch.__version__; print('train venv torch', torch.__version__, 'cuda ok')"


say "Done. Weights still downloading: tail -f $NVME/download.log"
echo "Then: source .env.vm && python pipeline.py --job $NVME/jobs/demo  (after copying a taskspec.json there)"
