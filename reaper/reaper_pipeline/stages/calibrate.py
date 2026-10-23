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
from reaper_pipeline.teacher import make_teacher, pick_backend

STAGE = "calibrate"
CHUNK = 64


def run(job: Job) -> None:
    cfg = job.config
    prof = profile(job)
    need_teacher = bool(cfg.purpose.strip()) or cfg.teacher_answers
    teacher = None
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
        for i in range(0, len(rows), CHUNK):
            part = rows[i:i + CHUNK]
            answers = teacher.chat([[{"role": "user", "content": r["prompt"]}] for r in part],
                                   cfg.answer_max_tokens, 0.6)
            for r, a in zip(part, answers):
                r["answer"] = a
            done = min(i + CHUNK, len(rows))
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
