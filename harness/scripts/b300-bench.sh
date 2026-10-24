#!/usr/bin/env bash
# LiveCodeBench v6 on a fresh GPU VM: original Flint vs the REAQ codebook model.
# Usage: copy flint-REAP-192e-REAQ-codebook to ~/flint/ on the VM, then
#   bash b300-bench.sh            (run it inside tmux: tmux new -s bench)
set -euo pipefail
REAQ=${REAQ:-~/flint/flint-REAP-192e-REAQ-codebook}
[ -f "$REAQ/config.json" ] || { echo "REAQ model not found at $REAQ"; exit 1; }
if [ ! -d ~/LobBot ]; then
  git clone -b claude/flint-tui-run https://github.com/OliverVillson/LobBot.git ~/LobBot
fi
git -C ~/LobBot pull --ff-only
if [ ! -d ~/bench-venv ]; then
  python3 -m venv ~/bench-venv
fi
source ~/bench-venv/bin/activate
pip install -q -U pip
# B300 (sm_103) needs a CUDA 13 build of torch (driver 580+); cu128 on older drivers
drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
idx=cu130; [ "${drv:-0}" -lt 580 ] && idx=cu128
python -c "import torch" 2>/dev/null || pip install -q torch --index-url https://download.pytorch.org/whl/$idx
pip install -q -e "$HOME/LobBot/reaper[pipeline]" -e "$HOME/LobBot/harness"
python -c "import torch; print('torch', torch.__version__, torch.cuda.get_device_name(0))"
mkdir -p ~/bench
for run in "reaq:$REAQ" "original:RecursiveSI/flint"; do
  label=${run%%:*}; model=${run#*:}
  echo "=== $label ($model) ==="
  python -m blackhole_pipeline.bench --model "$model" --out ~/bench/$label --release v6 --max-tokens 32768 \
    --label $label 2>&1 | grep --line-buffered -v '"status": "running"' | tee ~/bench/$label.log
done
python - <<'PY'
import json, pathlib
for l in ("original", "reaq"):
    s = json.loads((pathlib.Path.home() / "bench" / l / "summary.json").read_text())
    print(f"{l:9s} pass@1 {s['pass@1']:.1%}  {s['by_difficulty']}  no code {s['no_code']}  "
          f"{s['tok_s']} tok/s total, batch {s['batch']}, {s['gen_s'] / 60:.0f} min")
PY
