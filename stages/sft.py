"""LoRA SFT on teacher answers (sequence-level distillation), with optional
logit distillation from the full teacher. Used by the heal stage for both the
REAP-pruned MoE and the dense Qwen3-4B student. The LoRA is merged before
saving so later stages see a plain HF model."""
from __future__ import annotations

import math
import random
import shutil
import time
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from stages import moe_utils as mu
from stages.taskdata import tokenize_example

ATTN = ["q_proj", "k_proj", "v_proj", "o_proj"]
MLP = ["gate_proj", "up_proj", "down_proj"]


def log(msg: str) -> None:
    print(msg, flush=True)


def load_causal_lm(path: str):
    """bf16 on GPU, fp32 on CPU; works with transformers 4.x and 5.x."""
    import transformers
    from transformers import AutoModelForCausalLM
    kw = {"device_map": "auto"} if torch.cuda.is_available() else {}
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    key = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
    return AutoModelForCausalLM.from_pretrained(path, **{key: dtype}, **kw)


def _lora_model(model, r: int, alpha: int, targets: list[str], train_experts: bool, train_router: bool):
    from peft import LoraConfig, get_peft_model

    names = {n.rsplit(".", 1)[-1] for n, m in model.named_modules() if isinstance(m, nn.Linear)}
    blocks = mu.find_moe_blocks(model)
    if train_experts or not blocks:
        targets = list(targets) + MLP  # expert MLPs (4.x ModuleList) or dense MLPs
    targets = [t for t in dict.fromkeys(targets) if t in names]
    attn_only = [t for t in targets if t not in MLP]
    if hasattr(getattr(model, "model", None), "language_model"):
        # multimodal checkpoint (Gemma 4): LoRA the text decoder only, not the
        # vision/audio towers that reuse the same projection names
        targets = rf".*language_model\..*\.({'|'.join(targets)})"
        attn_only = rf".*language_model\..*\.({'|'.join(attn_only)})"
    extra = {}
    if blocks and train_experts and not isinstance(blocks[0][1].experts, nn.ModuleList):
        # fused experts (transformers 5.x): LoRA on the stacked expert weights
        extra["target_parameters"] = ["experts.gate_up_proj", "experts.down_proj"]
    modules_to_save = ["gate"] if (blocks and train_router) else None
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=0.0, target_modules=targets,
                     modules_to_save=modules_to_save, task_type="CAUSAL_LM", **extra)
    try:
        pm = get_peft_model(model, cfg)
    except (TypeError, ValueError) as e:
        if not extra:
            raise
        log(f"heal: LoRA on fused experts unsupported here ({e}); training attention and router only")
        cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=0.0,
                         target_modules=attn_only,
                         modules_to_save=modules_to_save, task_type="CAUSAL_LM")
        pm = get_peft_model(model, cfg)
    for p in pm.parameters():  # keep trainable weights in fp32 for stable updates
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()
    return pm


def _batches(data: list[dict], max_tokens: int, max_bs: int, seed: int):
    order = sorted(range(len(data)), key=lambda i: len(data[i]["input_ids"]))
    batches, cur, cur_max = [], [], 0
    for i in order:
        L = len(data[i]["input_ids"])
        if cur and (max(cur_max, L) * (len(cur) + 1) > max_tokens or len(cur) >= max_bs):
            batches.append(cur)
            cur, cur_max = [], 0
        cur.append(i)
        cur_max = max(cur_max, L)
    if cur:
        batches.append(cur)
    random.Random(seed).shuffle(batches)
    return batches


def _collate(data, idxs, pad_id, device):
    L = max(len(data[i]["input_ids"]) for i in idxs)
    ids = torch.full((len(idxs), L), pad_id, dtype=torch.long)
    lab = torch.full((len(idxs), L), -100, dtype=torch.long)
    att = torch.zeros((len(idxs), L), dtype=torch.long)
    for r, i in enumerate(idxs):
        x, y = data[i]["input_ids"], data[i]["labels"]
        ids[r, : len(x)] = torch.tensor(x)
        lab[r, : len(y)] = torch.tensor(y)
        att[r, : len(x)] = 1
    return ids.to(device), lab.to(device), att.to(device)


def train_sft(base: str, examples: list[dict], out_dir: Path, *,
              epochs: float = 1.0, lr: float = 2e-4, r: int = 16, alpha: int = 32,
              max_len: int = 2048, max_tokens: int = 16384, max_bs: int = 16,
              grad_accum: int = 1, kd_teacher: str | None = None, kd_weight: float = 0.0,
              kd_temp: float = 1.0, targets: list[str] = ATTN, train_experts: bool = True,
              train_router: bool = True, seed: int = 0, max_minutes: float = 0.0,
              progress: Callable[[float, str], None] = lambda pct, msg: None) -> Path:
    from transformers import AutoTokenizer

    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(base)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    data = [tokenize_example(tok, ex["messages"], max_len) for ex in examples]
    n_cut = sum(len(d["input_ids"]) >= max_len for d in data)
    data = [d for d in data if any(l != -100 for l in d["labels"][1:])]
    if not data:
        raise ValueError("no training examples with answer tokens")
    if n_cut:
        log(f"sft: {n_cut}/{len(examples)} examples hit max_len={max_len} and were cut short "
            f"(raise LOBBOT_HEAL_MAX_LEN for long answers)")

    progress(1, f"loading {base}")
    model = load_causal_lm(base)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = _lora_model(model, r, alpha, targets, train_experts, train_router)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"sft: {len(data)} examples, {n_train / 1e6:.1f}M trainable params, base {base}")

    teacher = None
    if kd_teacher and kd_weight > 0:
        progress(2, f"loading KD teacher {kd_teacher}")
        teacher = load_causal_lm(kd_teacher).eval()

    device = next(model.parameters()).device
    batches_per_epoch = _batches(data, max_tokens, max_bs, seed)
    total = max(1, math.ceil(len(batches_per_epoch) * epochs / grad_accum))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0, betas=(0.9, 0.99))
    warm = max(1, int(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else torch.autocast("cpu", enabled=False)

    model.train()
    step, micro, t0, ema = 0, 0, time.time(), None
    epoch = 0
    while step < total:
        batches = batches_per_epoch if epoch == 0 else _batches(data, max_tokens, max_bs, seed + epoch)
        for idxs in batches:
            ids, lab, att = _collate(data, idxs, pad_id, device)
            with amp:
                out = model(input_ids=ids, attention_mask=att)
                tgt = lab[:, 1:]
                mask = tgt != -100
                # fp32 only for the answer positions: a full fp32 copy of the
                # logits is tokens x vocab x 4 bytes (17 GB for 16k tokens of Gemma 4)
                logits = out.logits[:, :-1][mask].float()
                ce = F.cross_entropy(logits, tgt[mask])
                loss = ce
                if teacher is not None:
                    with torch.no_grad():
                        t_logits = teacher(input_ids=ids.to(teacher.device), attention_mask=att.to(teacher.device)
                                           ).logits[:, :-1][mask.to(teacher.device)].float().to(device)
                    s = F.log_softmax(logits / kd_temp, -1)
                    t = F.log_softmax(t_logits / kd_temp, -1)
                    kl = F.kl_div(s, t, log_target=True, reduction="batchmean") * kd_temp ** 2
                    loss = (1 - kd_weight) * ce + kd_weight * kl
            (loss / grad_accum).backward()
            micro += 1
            if micro % grad_accum:
                continue
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            ema = loss.item() if ema is None else 0.9 * ema + 0.1 * loss.item()
            if step % max(1, total // 100) == 0 or step == total:
                el = time.time() - t0
                eta = el / step * (total - step)
                progress(5 + 85 * step / total,
                         f"step {step}/{total} loss {ema:.3f} lr {sched.get_last_lr()[0]:.1e} eta {eta / 60:.0f}m")
            if step >= total:
                break
            if max_minutes and time.time() - t0 > 60 * max_minutes:
                log(f"sft: time cap of {max_minutes:g} min reached at step {step}/{total}; stopping early")
                total = step
                break
        epoch += 1

    progress(92, "merging LoRA and saving")
    del teacher, opt
    model = model.merge_and_unload()
    if device.type == "cuda":
        model.to(torch.bfloat16)
    model.config.use_cache = True
    tmp = out_dir.parent / (out_dir.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    model.save_pretrained(tmp, safe_serialization=True, max_shard_size="5GB")
    tok.save_pretrained(tmp)
    mu.write_expert_count_alias(tmp)
    shutil.rmtree(out_dir, ignore_errors=True)
    tmp.rename(out_dir)
    log(f"sft: saved merged model to {out_dir} (final loss {ema:.3f})")
    return out_dir
