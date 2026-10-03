#!/usr/bin/env bash
# Smoke test of the quantize -> eval -> package stages on the B200, before REAP
# and heal are ready: the unpruned teacher stands in for the healed model.
#
#   source .env.vm
#   bash scripts/smoke_compress_eval.sh [job_dir]
#
# The unpruned 30B (128 experts) cannot fit 7 GB, so this job's target is
# 14 GB. Real runs keep the TaskSpec's 7 GB target.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
JOB=${1:-${LOBBOT_JOBS:-/mnt/nvme/jobs}/smoke-compress}
MODELS=${LOBBOT_MODELS:-/mnt/nvme/models}
TEACHER_DIR=$MODELS/Qwen3-30B-A3B-Instruct-2507

mkdir -p "$JOB/work" "$JOB/.done"
python - "$REPO/examples/support-tickets.taskspec.json" "$JOB/taskspec.json" <<'PY'
import json, sys
spec = json.load(open(sys.argv[1]))
spec["target"]["max_size_gb"] = 14
json.dump(spec, open(sys.argv[2], "w"), indent=2)
PY
echo '{"n_generate": 300, "n_heldout": 30, "dense_fallback": false}' > "$JOB/config.json"

cd "$REPO"
# 1. Real data stage (vLLM teacher), small.
python pipeline.py --job "$JOB" --only data
# 2. Stand-ins for reap and heal: the unpruned teacher, neutral layer importance.
ln -sfn "$TEACHER_DIR" "$JOB/work/healed"
echo '{}' > "$JOB/.done/reap"; echo '{}' > "$JOB/.done/heal"
# 3. The stages under test.
python pipeline.py --job "$JOB" --from quantize
python - "$JOB" <<'PY'
import json, sys
job = sys.argv[1]
a = json.load(open(f"{job}/work/allocation.json"))["candidates"]["lobbot-moe"]
e = json.load(open(f"{job}/out/eval.json"))
print(f"\nmodel.gguf {a['size_gb']} GB, {a['bytes_per_token_gb']} GB read/token, ~{a['tok_s_est']} tok/s est. on an M4 Air")
for c in e["candidates"]:
    print(f"{c['name']}: score {c['score']} ({e['score_method']}), agreement {c['agreement']}, {c['tok_s_vm']} tok/s on the B200")
print("teacher:", e["teacher"])
PY
ls -la "$JOB/out"
