#!/usr/bin/env bash
# Start the LobBot HTTP API on the VM (localhost only; use an SSH tunnel).
set -euo pipefail
cd "$(dirname "$0")/.."
: "${LOBBOT_TOKEN:?set LOBBOT_TOKEN to a long random secret}"
export LOBBOT_JOBS=${LOBBOT_JOBS:-/mnt/nvme/jobs}
exec python -m uvicorn agent.server:app --host 127.0.0.1 --port "${PORT:-8700}"
