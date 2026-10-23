"""REAPer's pipeline: REAP expert pruning for any Hugging Face MoE model, with
the unpruned model as teacher on the same machine.

    python -m reaper_pipeline --job DIR                     # DIR/config.json holds the settings
    python -m reaper_pipeline --model ID --purpose "Debugging only" [--prune 0.5]

See README.md in this directory for stages, files and events.
"""

STAGES = ["plan", "weights", "calibrate", "score", "prune", "eval", "quant", "package"]

# Stages that share one process (score and prune: the scored model's keep lists
# go straight into the checkpoint rewrite). Each process frees its GPU memory on exit.
PROCESSES = [("plan", ["plan"]), ("weights", ["weights"]), ("calibrate", ["calibrate"]),
             ("reap", ["score", "prune"]), ("eval", ["eval"]), ("quant", ["quant"]),
             ("package", ["package"])]
