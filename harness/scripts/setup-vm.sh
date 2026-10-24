#!/usr/bin/env bash
# One-time setup of a fresh GPU VM (B200/B300, Ubuntu + NVIDIA driver) for the BlackHole pipeline:
# LobBot checkout on claude/flint-tui-run, ~/bh-venv with a CUDA build of torch matching the
# driver, REAPer + the harness (blackhole, blackhole-pipeline, blackhole-serve), mlx[cpu] for
# checking MLX exports. Safe to rerun (it updates).
set -euo pipefail
if [ ! -d ~/LobBot ]; then
  git clone -b claude/flint-tui-run https://github.com/OliverVillson/LobBot.git ~/LobBot
fi
git -C ~/LobBot pull --ff-only
[ -d ~/bh-venv ] || python3 -m venv ~/bh-venv
source ~/bh-venv/bin/activate
pip install -q -U pip
drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
idx=cu130; [ "${drv:-0}" -lt 580 ] && idx=cu128
python -c "import torch" 2>/dev/null || pip install -q torch --index-url https://download.pytorch.org/whl/$idx
pip install -q -e "$HOME/LobBot/reaper[pipeline]" -e "$HOME/LobBot/harness" "mlx[cpu]" mlx-lm || \
  pip install -q -e "$HOME/LobBot/reaper[pipeline]" -e "$HOME/LobBot/harness"
python -c "import torch; print('torch', torch.__version__, '·', torch.cuda.get_device_name(0), '·', \
round(torch.cuda.mem_get_info()[1] / 2**30), 'GB')"
echo "ready: source ~/bh-venv/bin/activate"
