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
# CONDENSE_API_KEY also compresses the examples Gemini sees when it writes the
# held-out tests (check: python -m stages.condense --compress)
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
`GET /jobs/{id}/model` (GGUF, Range supported), `GET /jobs/{id}/modelfile`,
`POST /jobs/{id}/stop` (frees the GPU; state becomes `stopped`, resume continues).
One job runs at a time: starting or resuming another returns 409 with
`{"detail": ..., "running_job": "<id>"}`.
All but `/health` need `Authorization: Bearer $LOBBOT_TOKEN`.

## Developer CLI (`lobbot`)

Drives the pipeline on the VM from a laptop. Standard library only; it opens
its own SSH tunnel to the API and starts the API on the VM when needed.

```bash
uv tool install --editable .        # or: pipx install -e .
lobbot init                         # VM host (default evroc-user@194.14.81.33), key, checkout path
lobbot doctor                       # SSH, checkout, venvs, weights, llama.cpp, GPU, API
lobbot secret GEMINI_API_KEY        # stored in ~/.lobbot-env on the VM (mode 600) for the eval judge
lobbot new "turn support emails into JSON tickets"   # drafts a TaskSpec with Gemini (uses the VM's key if this Mac has none)
lobbot run my.taskspec.json --fast  # or --example; --long for code; live progress, Ctrl-C detaches
lobbot results <job>                # judge scores vs teacher, size, tok/s, held-out examples
lobbot status [job] | watch <job> | logs <job> [-s heal] [-f]
lobbot resume <job> [--from quantize] | stop <job>
lobbot save <job> --chat            # keep the model: see below
lobbot pull [--stash]               # git pull the VM checkout; --stash stashes local edits first
```

`lobbot chat <model>` and `lobbot ask <model> "..."` talk to a saved model in
the local Ollama. When the answer is JSON, short fields are shown as labelled
lines and multi-line fields as real code under a heading, with a file name
guessed from the include guard or `#include` (e.g. `observer.h`,
`observer.c`). `--out DIR` writes those files, and C code is checked with
`cc -std=c11 -Wall -fsyntax-only` so you see whether it compiles. Answers that
aren't JSON are shown as they come; `--raw` turns all of this off and
`chat --plain` runs plain `ollama run`. Each chat message is answered on its
own, like the model was trained; `--history` sends the whole conversation.

`lobbot save` (alias `install`; `run --save` chains it) copies the job's
report, spec, Modelfile and, if the small system disk has room, the GGUF to
`~/lobbot-saved/<task>-<job>/` on the VM. That disk survives a pause, unlike
`/mnt/nvme`, which is wiped when the VM is paused or stopped. Then it downloads
the GGUF to `~/lobbot-models/<task>-<job>/` (resumable), checks its sha256
against the VM's, writes `eval.json` and `lobbot.json` next to it, and runs
`ollama create`. API keys are never pasted anywhere: `lobbot secret NAME`
takes them from your local environment or a hidden prompt, and stores them in
`~/.lobbot-env` on the VM.

When a job finishes, `run`, `watch` and `status <job>` print the same results
report as `lobbot results`: the winner and why it won, a scoreboard of the
teacher and every candidate (size, estimated laptop tok/s, measured VM tok/s,
judge score and share of the teacher's), how many held-out tests were used and
who wrote and judged them, time per stage, the expert bit-width mix, and two
held-out examples compared field by field with the teacher.

`--fast` sets `n_generate=400, n_heldout=30, reap_calib_samples=128,
heal_max_minutes=20, dense_fallback=false`. `--long` is for tasks with long answers
such as code: it sets `data_answer_max_tokens=4096, data_max_len=16384,
heal_max_len=8192`, so the teacher's answers aren't cut off and dropped and heal
trains on whole examples. `lobbot new` suggests it when the drafted seed answers
run past ~800 tokens, and `run` warns when they do and `--long` is missing.
`--set key=value` overrides any `Config` field and wins over `--fast` and
`--long`. `run` refuses to start while another job is running (and writes nothing
then), or while something else is on the GPU unless you add `--force`, and refuses config keys the VM checkout doesn't know yet.
The API token lives in `~/.lobbot-token` on the VM (`lobbot token` prints it).

## Without a GPU

`LOBBOT_DRY_RUN=1` walks every stage with placeholder outputs, so the TUI and
agent can be built against the real pipeline contract on a laptop.

```bash
LOBBOT_DRY_RUN=1 python pipeline.py --job /tmp/lobbot-job   # after copying a taskspec.json in
LOBBOT_DRY_RUN=1 LOBBOT_JOBS=/tmp/lobbot-jobs LOBBOT_TOKEN=dev uvicorn agent.server:app --port 8700
python -m pytest
```
