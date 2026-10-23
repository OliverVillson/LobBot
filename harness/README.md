# blackhole_tui

The BlackHole harness TUI: run it on the GPU VM over SSH, pick a source model and a
schema, and watch the job collapse the model down to something that fits your own
machine. Layout and job handling follow REAPer's `reaper_tui`.

```bash
python3 -m venv ~/bh-venv && . ~/bh-venv/bin/activate
pip install -e .
blackhole --demo                     # simulated B200 + pretend pipeline, no GPU needed
blackhole "qwen3 coder" "Debugging Python" --schema pocket-coder --target mac32 --demo
```

| key / command | what it does |
| --- | --- |
| F3 · `/model <name, hf id or folder>` | any model: type an `org/name` Hugging Face id or a folder on the VM and it is auto-detected (family, dense or MoE, experts, size, fits here, fits your target) before you confirm; the marketplace list is a shortcut |
| F4 · `/schema <id>` | schema marketplace with a step-by-step view; Tab to describe your own |
| `/describe <idea>` | drafts a schema from words (offline keyword draft now; Claude/GPT later) |
| `/target <machine>` | rtx3060 rtx4070 rtx4090 rtx5090 mac16 mac32 mac64 cpu32 |
| `/purpose <text>` | what the model is for; steers calibration and synthetic data |
| F5 · `/run` | start the job, detached: close the TUI or lose SSH and it keeps going |
| F6 · `/logs` | full-screen log; `/stop`, `/quit` |

Running `blackhole` again reattaches to the newest job under `~/.blackhole/jobs`
(`$BLACKHOLE_HOME` overrides). The logo switches to the large version on terminals of
at least 150 × 66.

## The real pipeline (`blackhole_pipeline`)

Without `--demo`, F5 starts `python -m blackhole_pipeline --job <job>` detached. It drives
REAPer's stages (`reaper_pipeline`, in `../reaper`) plus two of our own on the REAP-pruned model,
using the GPTQ and per-expert table code in `../reaper/gpu_perf`:

| stage | what |
| --- | --- |
| detect | config.json + weight index: family, MoE or dense, experts, size; free disk check |
| plan · weights · calibrate · score · prune · eval | REAPer: teacher-written calibration, REAP scores, prune, KL/top-1 vs teacher |
| gptq | GPTQ int4 g128 on routed experts (sequential, 12-layer chunks, 64 calibration sequences), measured vs the teacher, written as a compressed-tensors W4A16 checkpoint: the shippable model |
| reaq | REAQ per-expert codebooks (lab): top half of each layer's experts by REAP score on 16-level tables, the rest 8-level, block 32, GPTQ rounding; measured vs the teacher, packed indices/tables/scales saved (`REAQ_LAYOUT.txt`), no kernel yet |
| quant · package | REAPer's fp8-experts folder, model cards, `receipt.json` and `RESULTS.md` in the job dir |

The schema is `reaq-codebook` (lab). A REAPer job on the same model and purpose under
`~/.reaper/jobs` (e.g. `flint-prune50`) donates its calibration set, REAP scores and teacher
reference, so the vLLM teacher and the scoring pass are skipped (`BLACKHOLE_NO_REUSE=1` turns that
off). Install into the REAPer venv on the VM:

```bash
pip install --no-deps -e ~/LobBot/reaper -e ~/LobBot/harness
blackhole flint --schema reaq-codebook       # starts straight away; F6 for the log
```

`python -m pytest tests` includes a full CPU run on a tiny random Flint.

## Layout

- `art.py` the black hole (generated from a small lensing model onto half-block pixels) and the banner
- `theme.py` palette and Textual CSS: void, ash, crimson, redshift, ember, flare, photon
- `catalog.py` source models, target machines, schemas (draft data)
- `jobs.py` job dirs, detached start, tail, reattach, stop; REAPer's JSON-lines events
- `fake_pipeline.py` the `--demo` pipeline, run detached like a real one
- `../blackhole_pipeline/` the real pipeline (see above)
- `detect.py` auto-detection from config.json + the safetensors index (no weights downloaded); stand-in for `blackhole.hw.model.detect`
- `architect.py` `/describe` (frontier-model hook, offline draft for now)
- `app.py` the app and its screens: home, models, schemas, logs

`python -m pytest tests` runs the tests; `python shots.py` re-renders `shots/`.

## Event contract

Same as REAPer: one JSON object per line in `<job>/events.log`,
`{"stage", "status": running|done|error|skipped, "pct", "msg"}`; anything else is a log
line. Stage names come from the schema. The final `pipeline` event carries `outcome`
(done|stopped|error), `gates_passed` and `model`; any event with `gate: "failed"` is shown
as a failed quality gate.
