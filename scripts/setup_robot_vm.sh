#!/usr/bin/env bash
# One-time setup of the robot pipeline on the evroc B200 VM (Ubuntu 24.04).
#   NVME=/mnt/nvme bash scripts/setup_robot_vm.sh
# Run scripts/setup_vm.sh first (it mounts the NVMe and installs uv).
# Clones Isaac-GR00T, builds its uv venv (Python 3.12, torch 2.9), adds the
# farm sim's packages to that venv, checks headless MuJoCo rendering on EGL,
# and downloads nvidia/GR00T-N1.7-3B. Safe to re-run.
#
# The GR00T backbone nvidia/Cosmos-Reason2-2B is GATED on Hugging Face and
# every GR00T checkpoint (base or fine-tuned) loads it on first use. Request
# access on https://huggingface.co/nvidia/Cosmos-Reason2-2B, then either
# `export HF_TOKEN=<token>` before running this script or log in when asked.
set -euo pipefail

NVME=${NVME:-/mnt/nvme}
GR00T_REPO=${LOBBOT_GR00T_REPO:-$NVME/Isaac-GR00T}
GR00T_MODEL=${LOBBOT_GR00T_MODEL:-nvidia/GR00T-N1.7-3B}
MODELS=${LOBBOT_MODELS:-$NVME/models}
REPO=$(cd "$(dirname "$0")/.." && pwd)

say() { printf '\n==> %s\n' "$*"; }

export HF_HOME="$NVME/hf-cache"
export UV_CACHE_DIR="$NVME/uv-cache"
export PATH="$HOME/.local/bin:$PATH"
mkdir -p "$MODELS" "$NVME/sites"

say "System packages (EGL for headless MuJoCo, ffmpeg for video)"
sudo apt-get update -qq
sudo apt-get install -y -qq git git-lfs ffmpeg libegl1 libgl1 libglvnd0 curl
git lfs install --skip-repo >/dev/null
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh

say "Isaac-GR00T at $GR00T_REPO"
if [ -d "$GR00T_REPO/.git" ]; then
  git -C "$GR00T_REPO" pull --ff-only || echo "(pull skipped)"
else
  git clone https://github.com/NVIDIA/Isaac-GR00T "$GR00T_REPO"
fi
(cd "$GR00T_REPO" && uv sync --python 3.12)

say "Farm sim packages in the GR00T venv"
# compress and simeval run the sim and the policy in the same process.
VENV_PY="$GR00T_REPO/.venv/bin/python"
uv pip install --python "$VENV_PY" -q mujoco imageio imageio-ffmpeg pyarrow pyproj tifffile numpy pillow
# LobBot itself on the venv's path (no install needed for a flat repo).
SITE_PKGS=$("$VENV_PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
echo "$REPO" > "$SITE_PKGS/lobbot.pth"

say "Headless MuJoCo render on EGL"
MUJOCO_GL=egl "$VENV_PY" - <<'EOF'
import mujoco, numpy as np
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

say "Hugging Face login (gated backbone nvidia/Cosmos-Reason2-2B)"
if [ -n "${HF_TOKEN:-}" ]; then
  echo "using HF_TOKEN from the environment"
elif ! (cd "$GR00T_REPO" && uv run hf auth whoami >/dev/null 2>&1); then
  (cd "$GR00T_REPO" && uv run hf auth login) || echo "WARNING: not logged in; the backbone download will fail"
fi

say "Weights: $GR00T_MODEL and the backbone"
(cd "$GR00T_REPO" && uv run hf download "$GR00T_MODEL" --local-dir "$MODELS/${GR00T_MODEL##*/}")
(cd "$GR00T_REPO" && uv run hf download nvidia/Cosmos-Reason2-2B) \
  || echo "WARNING: Cosmos-Reason2-2B download failed: request access on its HF page and log in"

say "Done"
cat <<EOF
export LOBBOT_GR00T_REPO=$GR00T_REPO
export LOBBOT_MODELS=$MODELS
export HF_HOME=$HF_HOME
export MUJOCO_GL=egl
Run the robot pipeline with the GR00T venv: $VENV_PY robot_pipeline.py --job <dir>
EOF
