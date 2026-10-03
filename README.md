# LobBot

Hackathon lobotomy machine for LLMs: describe one job, and get a small model
specialised for it that you download, run offline and own. A 30B
mixture-of-experts teacher becomes a ~6.5 GB GGUF that runs at 40+ tok/s on a
16 GB laptop.

## Layout

| Path | Owner | What |
|---|---|---|
| `common/` | shared | `TaskSpec` schema and the JSON progress protocol. Change only together. |
| `pipeline.py`, `stages/` | backend | Compression pipeline that runs on the GPU VM |
| `agent/server.py` | backend | HTTP API on the VM that the desktop app talks to |
| `scripts/` | backend | VM setup and API launcher |
| `lobbot/` | backend | Developer CLI (Oliver only) |
| `app/`, `web/` | frontend | Desktop app (wraps Ollama for local inference) and landing page |
| `examples/` | shared | Demo `TaskSpec` (support email to JSON ticket) |

## Pipeline

```
data      teacher (vLLM) writes ~2k task examples from the TaskSpec seeds
reap      REAP 50% expert pruning, calibrated on that task data; per-layer importance
heal      LoRA distillation of the pruned model on teacher answers (+ dense 4B fallback)
quantize  dynamic per-layer expert bit widths from importance + static types for the rest
eval      held-out answers judged by Gemini; size and tok/s estimate; pick the winner
package   out/model.gguf + out/Modelfile for Ollama
```

```bash
python pipeline.py --job <dir> [--from <stage>] [--only <stage>]
```

`<dir>/taskspec.json` must exist. Each stage prints one JSON line per update
(`{"stage", "status", "pct", "msg"}`; status is running, done, error or skipped)
and everything else is log output. Finished stages are cached in `<dir>/.done/`.
Per-job overrides of `stages/_util.py:Config` go in `<dir>/config.json`.

## Running on the VM

```bash
NVME=/mnt/nvme bash scripts/setup_vm.sh       # venvs, llama.cpp, weights
source .env.vm
mkdir -p /mnt/nvme/jobs/demo && cp examples/support-tickets.taskspec.json /mnt/nvme/jobs/demo/taskspec.json
export GEMINI_API_KEY=...                      # eval judge (Gemini, cfg.judge_model)
# optional: a claude-* judge_model uses ANTHROPIC_API_KEY, and goes through
# condense.chat when CONDENSE_API_KEY is set (check: python -m stages.condense)
python pipeline.py --job /mnt/nvme/jobs/demo
```

## HTTP API

```bash
source .env.vm
LOBBOT_TOKEN=<secret> bash scripts/serve_api.sh     # listens on 127.0.0.1:8700
ssh -L 8700:127.0.0.1:8700 <vm>                     # on the laptop
```

`GET /health`, `POST /jobs` (TaskSpec body), `GET /jobs`, `GET /jobs/{id}`,
`GET /jobs/{id}/events` (SSE progress lines, replayed from the start of the
latest run), `POST /jobs/{id}/resume`, `GET /jobs/{id}/eval`,
`GET /jobs/{id}/model` (GGUF, Range supported), `GET /jobs/{id}/modelfile`.
All but `/health` need `Authorization: Bearer $LOBBOT_TOKEN`.

## Without a GPU

`LOBBOT_DRY_RUN=1` walks every stage with placeholder outputs, so the TUI and
agent can be built against the real pipeline contract on a laptop.

```bash
LOBBOT_DRY_RUN=1 python pipeline.py --job /tmp/lobbot-job   # after copying a taskspec.json in
LOBBOT_DRY_RUN=1 LOBBOT_JOBS=/tmp/lobbot-jobs LOBBOT_TOKEN=dev uvicorn agent.server:app --port 8700
python -m pytest
```
