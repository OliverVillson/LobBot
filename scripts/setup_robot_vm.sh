#!/usr/bin/env bash
# One-time setup of the robot pipeline on the evroc B200 VM (Ubuntu 24.04).
#   NVME=/mnt/nvme HF_TOKEN=hf_... bash scripts/setup_robot_vm.sh
# Run scripts/setup_vm.sh first (it mounts the NVMe and installs uv). Safe to re-run.
#
# What it does, following the Isaac-GR00T README (dGPU x86_64 install):
#   1. apt: ffmpeg (torchcodec, the only GR00T video backend; FFmpeg 4-7 only),
#      libegl1/libgl1/libglvnd0 (headless MuJoCo on EGL), libosmesa6 (CPU
#      fallback), git-lfs.
#   2. Isaac-GR00T at $LOBBOT_GR00T_REPO: `uv sync --python 3.12` builds its
#      .venv (torch 2.9 cu128, flash-attn 2.8.3, transformers 4.57.3,
#      torchcodec 0.8). Submodules (LIBERO, SimplerEnv, robocasa) are only for
#      NVIDIA's sim benchmarks and are not needed.
#   3. The farm sim's packages into that same venv, constrained to the versions
#      uv sync installed (so numpy stays 1.26.4 and torch is never touched),
#      and LobBot on its path. compress and simeval run sim and policy in one
#      process with this python; robot/finetune.py launches launch_finetune.py
#      with it too (not `uv run`, which would re-sync the venv).
#   4. Checks: GPU, torchcodec decode, flash-attn kernel on this GPU, MuJoCo EGL.
#   5. Weights: nvidia/GR00T-N1.7-3B to $LOBBOT_MODELS and the backbone into
#      $HF_HOME.
#
# HF_TOKEN: the GR00T backbone nvidia/Cosmos-Reason2-2B is GATED on Hugging
# Face, and every GR00T checkpoint (base or fine-tuned) loads it by hub id at
# load time (launch_finetune.py hard-codes model_name="nvidia/Cosmos-Reason2-2B").
# Request access on https://huggingface.co/nvidia/Cosmos-Reason2-2B with the
# account whose token you use, then `export HF_TOKEN=hf_...` before running this
# script (or log in when asked). The pipeline later needs the same HF_HOME
# ($NVME/hf-cache), which robot/finetune.py sets when it is unset; source
# $NVME/robot.env (written at the end) in every shell that runs the pipeline.
set -euo pipefail

NVME=${NVME:-/mnt/nvme}
GR00T_REPO=${LOBBOT_GR00T_REPO:-$NVME/Isaac-GR00T}
GR00T_MODEL=${LOBBOT_GR00T_MODEL:-nvidia/GR00T-N1.7-3B}
BACKBONE=nvidia/Cosmos-Reason2-2B
MODELS=${LOBBOT_MODELS:-$NVME/models}
REPO=$(cd "$(dirname "$0")/.." && pwd)

say() { printf '\n==> %s\n' "$*"; }
warn() { printf 'WARNING: %s\n' "$*" >&2; }

export HF_HOME="$NVME/hf-cache"
export UV_CACHE_DIR="$NVME/uv-cache"   # the root disk is too small for the torch wheels
export PATH="$HOME/.local/bin:$PATH"
mkdir -p "$MODELS" "$NVME/sites" "$HF_HOME"

say "GPU"
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv

say "System packages"
sudo apt-get update -qq
sudo apt-get install -y -qq git git-lfs curl ffmpeg libegl1 libgl1 libglvnd0 libosmesa6 libaio-dev
git lfs install --skip-repo >/dev/null
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
FFMPEG_MAJOR=$(ffmpeg -version | head -1 | sed -E 's/^ffmpeg version n?([0-9]+).*/\1/')
if [ "${FFMPEG_MAJOR:-0}" -ge 8 ] 2>/dev/null; then
  warn "ffmpeg $FFMPEG_MAJOR found; torchcodec 0.8 needs FFmpeg 4-7 (Ubuntu 24.04 ships 6)"
fi
if [ ! -f /usr/share/glvnd/egl_vendor.d/10_nvidia.json ]; then
  warn "no NVIDIA EGL vendor file; MuJoCo EGL needs the driver's GL libs (apt install libnvidia-gl-<driver>-server)"
fi

say "Isaac-GR00T at $GR00T_REPO"
if [ -d "$GR00T_REPO/.git" ]; then
  git -C "$GR00T_REPO" pull --ff-only || echo "(pull skipped)"
else
  git clone https://github.com/NVIDIA/Isaac-GR00T "$GR00T_REPO"
fi
(cd "$GR00T_REPO" && uv sync --python 3.12)
VENV_PY="$GR00T_REPO/.venv/bin/python"
"$VENV_PY" -c 'import gr00t, torch; print("gr00t ok, torch", torch.__version__)'

say "Farm sim packages in the GR00T venv (pinned to what uv sync installed)"
CONSTRAINTS=$(mktemp)
uv pip freeze --python "$VENV_PY" | grep -v -e '^-e' -e ' @ ' -e '^gr00t' > "$CONSTRAINTS" || true
uv pip install --python "$VENV_PY" -q -c "$CONSTRAINTS" mujoco imageio imageio-ffmpeg pyarrow pyproj tifffile pytest
rm -f "$CONSTRAINTS"
SITE_PKGS=$("$VENV_PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
echo "$REPO" > "$SITE_PKGS/lobbot.pth"   # flat repo: on sys.path, no install needed
"$VENV_PY" -c 'import numpy, torch, farmsim.lerobot, robot.gr00t_policy; print("numpy", numpy.__version__, "torch", torch.__version__)'

say "torchcodec decode (GR00T's video loader)"
TMPV=$(mktemp -d)
ffmpeg -loglevel error -f lavfi -i testsrc=size=64x64:rate=10 -frames:v 5 -c:v libx264 -pix_fmt yuv420p "$TMPV/t.mp4"
"$VENV_PY" - "$TMPV/t.mp4" <<'EOF'
import sys, numpy as np
from gr00t.utils.video_utils import get_frames_by_indices
f = get_frames_by_indices(sys.argv[1], np.arange(5))
assert f.shape[0] == 5 and f.shape[-1] == 3, f.shape
print("torchcodec ok", f.shape)
EOF
rm -rf "$TMPV"

say "flash-attn kernel on this GPU"
if ! "$VENV_PY" - <<'EOF'
import torch
from flash_attn import flash_attn_func
q = torch.randn(1, 8, 2, 64, device="cuda", dtype=torch.bfloat16)
print("flash-attn ok", tuple(flash_attn_func(q, q, q).shape), torch.cuda.get_device_name(0))
EOF
then
  warn "flash-attn failed on this GPU. GR00T falls back to sdpa only when flash_attn cannot be imported:"
  warn "  uv pip uninstall --python $VENV_PY flash-attn"
fi

say "Headless MuJoCo render on EGL"
MUJOCO_GL=egl "$VENV_PY" - <<'EOF' || warn "EGL render failed; MUJOCO_GL=osmesa works (CPU, slower)"
import mujoco
m = mujoco.MjModel.from_xml_string(
    "<mujoco><worldbody><light pos='0 0 3'/><geom type='box' size='1 1 .1' rgba='.2 .6 .2 1'/></worldbody></mujoco>")
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
r = mujoco.Renderer(m, 64, 64)
r.update_scene(d)
img = r.render()
assert img.shape == (64, 64, 3) and img.max() > 0, "EGL render returned an empty image"
print("mujoco", mujoco.__version__, "EGL render ok, mean pixel", round(float(img.mean()), 1))
EOF

say "Hugging Face login (gated backbone $BACKBONE)"
if [ -n "${HF_TOKEN:-}" ]; then
  echo "using HF_TOKEN from the environment"
elif ! (cd "$GR00T_REPO" && "$GR00T_REPO/.venv/bin/hf" auth whoami >/dev/null 2>&1); then
  "$GR00T_REPO/.venv/bin/hf" auth login || warn "not logged in; the backbone download will fail"
fi

say "Weights: $GR00T_MODEL and $BACKBONE"
"$GR00T_REPO/.venv/bin/hf" download "$GR00T_MODEL" --local-dir "$MODELS/${GR00T_MODEL##*/}"
# Into the HF_HOME cache, not a local dir: GR00T loads it by hub id.
"$GR00T_REPO/.venv/bin/hf" download "$BACKBONE" \
  || warn "$BACKBONE download failed: request access on its HF page, export HF_TOKEN, re-run"

cat > "$NVME/robot.env" <<EOF
export LOBBOT_GR00T_REPO=$GR00T_REPO
export LOBBOT_MODELS=$MODELS
export HF_HOME=$HF_HOME
export MUJOCO_GL=egl
export PATH=$GR00T_REPO/.venv/bin:\$PATH
EOF

say "GR00T compatibility tests (CPU, tiny random model)"
(cd "$REPO" && LOBBOT_GR00T_REPO="$GR00T_REPO" MUJOCO_GL=egl "$VENV_PY" -m pytest -q tests/test_gr00t_compat.py) \
  || warn "tests/test_gr00t_compat.py failed: fix before the real run"

say "Done"
cat <<EOF
source $NVME/robot.env
python robot_pipeline.py --job <dir>     # python = $VENV_PY
EOF
