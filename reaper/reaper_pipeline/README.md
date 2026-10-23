# reaper_pipeline

REAP expert pruning for any Hugging Face MoE model, run on the GPU machine
itself, with the unpruned model as teacher on the same GPUs. No provider API
keys: everything runs locally.

```bash
pip install -e 'reaper[pipeline]'      # + `pip install vllm` for a fast teacher
python -m reaper_pipeline --model Qwen/Qwen3-30B-A3B-Instruct-2507 --purpose "Debugging only" --job ~/.reaper/jobs/x
python -m reaper_pipeline --job DIR    # what the TUI runs; DIR/config.json holds the settings
python -m reaper_pipeline --job DIR --from score    # redo from a stage; finished stages are otherwise skipped
```

## Contract with the TUI

- **Start:** `python -m reaper_pipeline --job DIR`. `DIR/config.json` is
  `{"model", "purpose", "prune", "calib", "calib_samples"}` plus optional knobs
  (all fields of `job.Config`; unknown keys are ignored). `purpose` is `""` when
  none is given.
- **Events:** one JSON line per event on stdout, LobBot's protocol:
  `{"stage", "status": "running|done|error|skipped", "pct", "msg", "ts"}`.
  Every other line is a log line. Stage names are `reaper_pipeline.STAGES`:
  `plan, weights, calibrate, score, prune, eval, quant, package`.
  The `eval` done event also carries `top1_agreement`, `kl`, `ppl_teacher` and `ppl_pruned`.
- **End:** `{"stage": "pipeline", "status": "done", "model": "<output dir>"}` and
  exit code 0, or `{"stage": "pipeline", "status": "error", "msg": ...}` and exit 1.
- **Stop:** SIGTERM (or SIGINT) stops the running stage process, which frees the GPU.
- **Hardware:** the `plan` stage asks `reaper_hw` (`detect()`, `plan(hw, model, prune, calib_seq)`)
  and falls back to a small built-in planner when it is not installed (or `REAPER_HW=builtin`).
  The flattened result is in `DIR/plan.json`.

## Stages

| stage | process | what |
|---|---|---|
| plan | plan | read the model config (refuse non-MoE), detect hardware, plan dtype, placement, batch sizes and the teacher |
| weights | weights | download to `$REAPER_HOME/models/<org>--<name>` (skipped for a local dir) |
| calibrate | calibrate | the teacher (vLLM, else transformers) writes prompts for the purpose and answers them |
| score | reap | forward the calibration set through the unpruned model with REAP hooks on every router |
| prune | reap | keep the top-n experts per layer, rewrite the checkpoint shard by shard in its own format |
| eval | eval | pruned model vs the teacher's held-out answers: top-1 agreement, KL, perplexity, samples |
| quant | quant | routed experts to fp8 with 128x128 block scales (`quant: "fp8-experts"`, the default), attention, routers, shared experts and the head stay bf16; written to `out/<name>-FP8/`. `quant: "none"` skips it |
| package | package | model card, `reap_saliency.json`, `reap_eval.json` next to the weights |

Each process frees the GPU when it exits, so the teacher and the REAP pass each
get the whole machine. Set `REAPER_VLLM_PY` to run the calibrate stage in a
separate vLLM venv.

## Purpose → calibration

REAP keeps the experts that fire on the calibration tokens, so the purpose
decides what survives. With a purpose ("Debugging only"):

1. the teacher lists ~24 sub-tasks of the purpose (stack traces, flaky tests, ...),
2. it writes varied, self-contained user requests for each sub-task (code, logs and data inline),
3. 25% (`purpose_mix`) comes from the closest public preset (keywords: debugging → `code`),
4. the teacher answers every prompt, and REAP is scored on prompt + answer.

Without a purpose, `calib` picks a preset (`mix` = chat + code + math). Offline,
a built-in prompt set is used and the run says so.

## Flint

transformers has no Flint (RecursiveSI/flint) yet, so `models/flint.py` is a
plain PyTorch version of it (sliding/full attention with sinks, sigmoid top-6
routing, shared expert), registered for `model_type: flint` when transformers
does not know it. It has no KV cache, so the teacher should be vLLM (nightly
has Flint): with the HF teacher, generation recomputes the whole sequence per
token. Eval's `ppl_teacher` is this port's perplexity on vLLM's answers, so a
value near 2-5 also checks the port.

## Any MoE

Nothing is written per architecture (`moe.py`). A MoE block is any module with
an `experts` child (ModuleList, or fused `[E, ...]` tensors) next to a router
(`gate`/`router`) with an `[E, hidden]` weight. Routing is read from what the
router returns when it returns its choice (DeepSeek-V3, GLM-4.5, Kimi K2,
gpt-oss and every transformers 5 router), so grouped top-k, correction biases
and scaling come out as the model applies them; plain linear gates are
recomputed from the config (sigmoid + bias for MiniMax-M2, softmax top-k for
Qwen/Mixtral/OLMoE). Expert outputs come from the model's own experts module
with the routing forced to one expert, checked once per block for linearity.

Pruning rewrites the safetensors files rather than calling `save_pretrained`,
so the output keeps the original's layout and quantization (fp8 blocks, MXFP4)
and loads wherever the original does. DeepSeek-style expert groups keep the
same number per group; multi-token-prediction layers (not scored) are dropped.

Tested on CPU with tiny random checkpoints of Qwen3-MoE, Qwen3-Next, Mixtral,
OLMoE, gpt-oss, DeepSeek-V3, GLM-4.5 and MiniMax-M2 in transformers 5, plus a
Kimi-style remote-code block: the routing and expert outputs rebuild the MoE
output exactly, and a pruned checkpoint gives identical logits wherever the
routing is unchanged (`tests/test_pipeline_moe.py`).
