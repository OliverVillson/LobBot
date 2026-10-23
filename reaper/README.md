# REAPer

Terminal UI for REAP (router-weighted expert activation pruning) of
mixture-of-experts models. It runs **on the GPU machine itself**: SSH into the
VM, start `reaper`, and it detects the hardware, runs the teacher on the local
GPU to write calibration data, prunes and checks the model. There are no API keys
or cloud providers to set up.

![REAPer reaping](docs/reaping.png)

## Run

```bash
# on the VM, once (B200 / Blackwell needs a torch built for CUDA 12.8)
git clone -b claude/project-thread-7e6em9 https://github.com/OliverVillson/LobBot.git && cd LobBot
python3 -m venv ~/reaper-venv && . ~/reaper-venv/bin/activate
pip install 'torch>=2.7' --index-url https://download.pytorch.org/whl/cu128
pip install -e 'reaper[pipeline]'
pip install vllm                               # optional: a much faster teacher
reaper hw Qwen/Qwen3-30B-A3B-Instruct-2507     # check: GPU, driver, torch kernels and the run plan

# from your laptop
ssh -t user@vm reaper                                       # open the TUI
ssh -t user@vm reaper "qwen3-30b-a3b" "Debugging only"      # find the model, prune it for that purpose
ssh -t user@vm reaper "org/Any-MoE" "SQL only" --prune 25   # any Hugging Face MoE id
reaper --demo                                               # simulated B200, no GPU needed
```

`reaper "<model>" "<purpose>"` resolves a fuzzy model name to a Hugging Face
repo, preferring the maker's own instruct model over quantized or pruned
re-uploads, and starts the run straight away. The other matches are shown, and
`/model <id>` switches. The purpose is written to the job's `config.json` and
becomes the calibration set, so the experts that purpose uses are the ones kept.

The run is detached from the terminal. Closing the TUI or losing SSH does not
stop it, and the next `reaper` reattaches to it.

| Key / command | What |
|---|---|
| `F2`, `/setup` | model, purpose, share of experts to prune, calibration set, samples |
| `F3`, `/market` | MoE marketplace, with whether each model fits this machine |
| `F5`, `/reap` | start the pipeline (`/reap --force` past a failed hardware check) |
| `/model <name or id>`, `/purpose <text>`, `/prune <pct>` | change the run inline |
| `/quant fp8-experts\|none`, `--quant` | after pruning, store the routed experts in fp8 (default) or leave the model in bf16 |
| `/hw` | detect the hardware again and show the plan |
| `/stop`, `/logs`, `/quit` | stop the run, show the last log lines, leave the TUI |

## How the pieces fit

| Package | What |
|---|---|
| `reaper_tui/` | this TUI and the `reaper` command |
| `reaper_hw/` | hardware detection and the per-stage plan (built in a separate thread) |
| `reaper_pipeline/` | the any-MoE REAP pipeline with the teacher on the local GPU (built in a separate thread) |

`reaper_tui/backend.py` holds the contract between them. Until the other two
packages exist, the TUI reads `nvidia-smi` itself and refuses to start a run
(`--demo` still works).

- **Hardware:** `detect()` returns the GPUs (`name`, `mem_gb`, `cc`), `cpus`,
  `ram_gb` and `disk_free_gb`. `plan(hardware, model_id, prune)` returns `ok`,
  `notes` (what was adapted, shown under Machine) and `problems`.
- **Pipeline:** `python -m reaper_pipeline --job DIR` with `DIR/config.json` =
  `{"model", "purpose", "prune", "calib", "calib_samples", "quant"}`. It prints one JSON
  event per line (`{"stage", "status", "pct", "msg"}`, LobBot's protocol) and
  ends with `{"stage": "pipeline", "status": "done", "model": "<output dir>"}`.
  A module-level `STAGES` list names the stages, and SIGTERM stops a run.

Jobs live in `$REAPER_HOME/jobs/<model>-prune<pct>/` (default `~/.reaper`),
and each job's output is in `events.log`.
