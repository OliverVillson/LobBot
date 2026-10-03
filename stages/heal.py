"""Stage 3: heal by distillation.

LoRA SFT of the pruned MoE on the unpruned teacher's answers, then merge.
If dense_fallback is on, also SFTs the dense student on the same data so the
eval stage has a second candidate. Writes:
  work/healed/   pruned + healed MoE (merged HF model)
  work/dense/    dense student (merged HF model), if enabled
"""

from __future__ import annotations

import argparse
import shutil

from common.progress import emit
from stages._util import DRY_RUN, Job

STAGE = "heal"


def sft(job: Job, src: str, out_name: str, targets, r: int, pct_lo: float, pct_hi: float) -> None:
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    from trl import SFTConfig, SFTTrainer

    cfg = job.config
    tok = AutoTokenizer.from_pretrained(src)
    model = AutoModelForCausalLM.from_pretrained(src, torch_dtype=torch.bfloat16, device_map="cuda")
    ds = load_dataset("json", data_files=str(job.path("data", "train.jsonl")), split="train")

    class Progress(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kw):
            if state.max_steps:
                frac = state.global_step / state.max_steps
                loss = (logs or {}).get("loss")
                emit(STAGE, pct=pct_lo + (pct_hi - pct_lo) * frac,
                     msg=f"{out_name}: step {state.global_step}/{state.max_steps}" + (f" loss {loss:.3f}" if loss else ""))

    trainer = SFTTrainer(
        model=model,
        processing_class=tok,
        train_dataset=ds,
        peft_config=LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.0, target_modules=targets, task_type="CAUSAL_LM"),
        args=SFTConfig(
            output_dir=str(job.path("work", f"{out_name}-ckpt")),
            num_train_epochs=cfg.heal_epochs,
            learning_rate=cfg.heal_lr,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            gradient_checkpointing=True,
            bf16=True,
            max_length=2048,
            logging_steps=5,
            save_strategy="no",
            report_to=[],
        ),
        callbacks=[Progress()],
    )
    trainer.train()
    merged = trainer.model.merge_and_unload()
    out = job.path("work", out_name)
    if out.exists():
        shutil.rmtree(out)
    merged.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    del trainer, model, merged
    torch.cuda.empty_cache()


def run_stage(job: Job) -> None:
    cfg = job.config
    if DRY_RUN:
        shutil.copytree(job.path("work", "reaped"), job.path("work", "healed"), dirs_exist_ok=True)
        if cfg.dense_fallback:
            job.path("work", "dense").mkdir(exist_ok=True)
        emit(STAGE, pct=90, msg="dry run: copied pruned model")
    else:
        emit(STAGE, pct=1, msg="healing pruned MoE")
        sft(job, str(job.path("work", "reaped")), "healed", cfg.heal_targets, cfg.heal_lora_r, 2, 60 if cfg.dense_fallback else 98)
        if cfg.dense_fallback:
            emit(STAGE, pct=60, msg=f"distilling dense student {cfg.student}")
            sft(job, job.model_path(cfg.student), "dense", "all-linear", 32, 60, 98)

    job.mark_done(STAGE, {"dense": cfg.dense_fallback})
    emit(STAGE, "done", 100, "healed" + (" + dense student" if cfg.dense_fallback else ""))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    run_stage(Job(ap.parse_args().job))
