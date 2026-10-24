"""Stage calibrate: the calibration set, with the teacher (the unpruned model
on this machine's GPUs) doing the writing.

With a purpose, the teacher writes the prompts for it (see calib.py); without
one, a public preset is used. Then the teacher answers every prompt, so REAP
is scored on the tokens the model itself produces for the task, the
generations it has to keep getting right. Writes data/calib.jsonl,
data/heldout.jsonl (kept out of scoring, for eval) and data/topics.json."""

from __future__ import annotations

from reaper_pipeline import calib
from reaper_pipeline.events import emit, log
from reaper_pipeline.job import Job
from reaper_pipeline.stages._common import profile, stage_main, write_jsonl
from reaper_pipeline.teacher import HfTeacher, make_teacher, pick_backend

STAGE = "calibrate"
CHUNK = 64  # HF generate batch; vLLM gets far bigger submissions, see answer_chunk()


def answer_chunk(prof: dict, backend: str, n: int) -> int:
    """Prompts per teacher.chat call. vLLM schedules continuously inside one call, but every
    call ends with its slowest answer: 64-prompt calls leave most of max_num_seqs idle and
    pay a full 1024-token straggler tail per call. Submit ~4x the batch the engine can hold
    (still >= 10 progress updates on big sets), which keeps the GPU full until the end."""
    if backend != "vllm":
        return max(CHUNK, min(n, 128))  # the hf teacher batches internally (128 with cached generation)
    seqs = int((prof.get("teacher") or {}).get("max_num_seqs") or 256)
    return max(CHUNK, min(n, max(4 * seqs, -(-n // 10))))


def run(job: Job) -> None:
    cfg = job.config
    prof = profile(job)
    need_teacher = bool(cfg.purpose.strip()) or cfg.teacher_answers
    teacher, backend = None, "hf"
    if need_teacher:
        backend = pick_backend(prof, cfg.teacher_backend)
        emit(STAGE, pct=1, msg=f"starting the teacher ({backend})")
        teacher = make_teacher(str(job.model_dir), prof, cfg.teacher_backend, cfg.trust_remote_code, cfg.seed)

    n = cfg.calib_samples + cfg.heldout
    what = f'prompts for "{cfg.purpose}"' if cfg.purpose.strip() else f"{cfg.calib} prompts"
    emit(STAGE, pct=5, msg=f"teacher writing {what}" if cfg.purpose.strip() else f"loading {what}")
    rows, info = calib.build_prompts(
        teacher, cfg.purpose, cfg.calib, n, cfg.purpose_mix, cfg.seed,
        on_progress=lambda got, of: emit(STAGE, pct=5 + 35 * got / of, msg=f"{got}/{of} prompts written"),
        max_tokens=cfg.writer_max_tokens)
    log(f"calibration: {info['note']}")
    if info.get("topics"):
        log("sub-tasks: " + "; ".join(info["topics"]))
    job.write("data/topics.json", info)

    if cfg.teacher_answers:
        chunk = answer_chunk(prof, backend, len(rows))
        for i in range(0, len(rows), chunk):
            part = rows[i:i + chunk]
            msgs = [[{"role": "user", "content": r["prompt"]}] for r in part]
            if isinstance(teacher, HfTeacher):
                def prog(got, of, note, i=i):
                    emit(STAGE, pct=40 + 58 * (i + got * len(part) / of) / len(rows),
                         msg=f"teacher answering {i + 1}–{i + len(part)} of {len(rows)}: {note}")
                answers = teacher.chat(msgs, cfg.answer_max_tokens, 0.6, on_progress=prog)
            else:
                answers = teacher.chat(msgs, cfg.answer_max_tokens, 0.6)
            for r, a in zip(part, answers):
                r["answer"] = a
            done = min(i + chunk, len(rows))
            emit(STAGE, pct=40 + 58 * done / len(rows), msg=f"teacher answered {done}/{len(rows)}")
        rows = [r for r in rows if r.get("answer", "").strip()]
    if teacher is not None:
        teacher.close()

    if len(rows) < 2:
        raise RuntimeError("calibration set is empty")
    held = min(cfg.heldout, len(rows) // 4) if cfg.teacher_answers else 0
    write_jsonl(job.path("data", "calib.jsonl"), rows[held:])
    write_jsonl(job.path("data", "heldout.jsonl"), rows[:held])
    job.mark_done(STAGE, {"calib": len(rows) - held, "heldout": held})
    emit(STAGE, "done", 100, f"{len(rows) - held} calibration + {held} held-out samples · {info['note']}")


if __name__ == "__main__":
    stage_main(run, STAGE)
