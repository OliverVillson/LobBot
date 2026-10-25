"""Run a TUI job: python -m blackhole_pipeline --job DIR [--from STAGE]

Each stage group runs as its own process (REAPer's or ours), its output is relayed to
stdout (DIR/events.log when the TUI started us) and teed to DIR/reaper/logs/<process>.log.
Finished stages are skipped on a rerun. SIGTERM stops the running stage and frees the GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from blackhole_pipeline import PROCESSES, STAGES
from blackhole_pipeline.job import BHJob

_child: subprocess.Popen | None = None


def emit(stage: str, status: str = "running", pct: float | None = None, msg: str = "", **extra) -> None:
    ev = {"stage": stage, "status": status, "ts": round(time.time(), 3)}
    if pct is not None:
        ev["pct"] = round(max(0.0, min(100.0, pct)), 1)
    if msg:
        ev["msg"] = msg
    ev.update(extra)
    print(json.dumps(ev), flush=True)


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def parse(line: str) -> dict | None:
    from reaper_pipeline.events import parse as p
    return p(line)


def translate(ev: dict) -> dict:
    """REAPer's eval event in the names the TUI panel reads."""
    if ev["stage"] == "eval" and ev["status"] == "done" and "top1_agreement" in ev:
        ev = {**ev, "top1": ev["top1_agreement"], "kl": round(ev["kl"], 4),
              "ppl_teacher": round(ev["ppl_teacher"], 3), "ppl_student": round(ev["ppl_pruned"], 3)}
    return ev


def run_process(name: str, who: str, bh: BHJob, rj) -> tuple[int, list[str], dict | None]:
    global _child
    if who == "reaper":
        py = os.environ.get("REAPER_VLLM_PY") if name == "calibrate" else None
        cmd = [py or sys.executable, "-u", "-m", f"reaper_pipeline.stages.{name}", "--job", str(rj.root)]
    else:
        cmd = [sys.executable, "-u", "-m", f"blackhole_pipeline.stages.{name}", "--job", str(bh.root)]
    env = {k: str(v) for k, v in ((rj.read("plan.json") or {}).get("env") or {}).items()}
    tail: list[str] = []
    last = None
    with open(rj.path("logs", f"{name}.log"), "a") as logf:
        logf.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(cmd)}\n")
        _child = subprocess.Popen(cmd, env={**os.environ, **env, "PYTHONUNBUFFERED": "1"}, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, bufsize=1, errors="replace")
        assert _child.stdout
        for line in _child.stdout:
            logf.write(line)
            logf.flush()
            ev = parse(line)
            if ev is None:
                sys.stdout.write(line)
                if line.strip():
                    tail = (tail + [line.rstrip()])[-30:]
            else:
                ev = translate(ev)
                print(json.dumps(ev), flush=True)
                last = ev
        rc = _child.wait()
        _child = None
    return rc, tail, last


def error_message(rc: int, tail: list[str]) -> str:
    for line in reversed(tail):
        if line and not line.startswith(" ") and ("Error" in line or "Exception" in line):
            return line[:400]
    if rc < 0:
        return f"stage killed by signal {-rc}"
    return tail[-1][:400] if tail else f"stage exited with {rc}"


def _same_model(c: dict, rj) -> bool:
    if c.get("model") == rj.config.model:
        return True
    try:  # a local path vs the hub id of the same download
        p = Path(str(c.get("model", ""))).expanduser()
        return p.is_dir() and p.resolve() == rj.model_dir.resolve()
    except OSError:
        return False


def seed_from_previous(bh: BHJob, rj) -> None:
    """Reuse what an earlier REAPer job on the same model and purpose (e.g. ~/.reaper/jobs/flint-prune50)
    already paid for. Its calibration set and teacher answers skip the teacher (vLLM) entirely; its REAP
    scores and teacher reference, when complete, also skip the scoring pass (REAP saliency does not
    depend on the prune ratio, so only the kept lists are recomputed). Every candidate looked at is
    logged with the reason it was or was not used."""
    if rj.is_done("calibrate") or os.environ.get("BLACKHOLE_NO_REUSE"):
        return
    from reaper_pipeline.job import home
    cfg = rj.config
    jobs = home() / "jobs"
    data_ok, score_ok = [], []
    cands = sorted(jobs.glob("*")) if jobs.is_dir() else []
    cands += sorted(bh.root.parent.glob("*/reaper"))  # other BlackHole jobs' REAPer parts
    for d in cands:
        if d.resolve() == rj.root.resolve() or not d.is_dir():
            continue
        try:
            c = json.loads((d / "config.json").read_text())
        except (OSError, ValueError):
            log(f"reuse: {d.name}: no readable config.json")
            continue
        if not _same_model(c, rj):
            log(f"reuse: {d.name}: other model ({c.get('model')})")
            continue
        if c.get("purpose", "") != cfg.purpose:
            log(f"reuse: {d.name}: other purpose ({c.get('purpose')!r} vs {cfg.purpose!r})")
            continue
        if not cfg.purpose and c.get("calib", "mix") != cfg.calib:
            log(f"reuse: {d.name}: other calibration set ({c.get('calib', 'mix')!r} vs {cfg.calib!r})")
            continue
        missing = [n for n in ("data/calib.jsonl", "data/heldout.jsonl") if not (d / n).exists()]
        if missing:
            log(f"reuse: {d.name}: missing {', '.join(missing)}")
            continue
        data_ok.append(d)
        smiss = [n for n in ("work/saliency.json", "work/reap.json", "work/teacher_ref.pt", ".done/score")
                 if not (d / n).exists()]
        if smiss:
            log(f"reuse: {d.name}: calibration usable, REAP scores not ({', '.join(smiss)} missing)")
        else:
            score_ok.append(d)
    if not data_ok:
        if jobs.is_dir():
            log(f"reuse: nothing reusable in {jobs} for {cfg.model} (purpose {cfg.purpose!r})")
        return
    newest = lambda d: (d / "data" / "calib.jsonl").stat().st_mtime  # noqa: E731
    src = max(score_ok or data_ok, key=newest)
    for p in (src / "data").iterdir():
        if p.is_file():
            shutil.copy2(p, rj.path("data", p.name))
    rj.mark_done("calibrate", {"reused_from": str(src)})
    if src in score_ok:
        for f in ("saliency.json", "reap.json", "teacher_ref.pt"):
            shutil.copy2(src / "work" / f, rj.path("work", f))
        rekeep(rj)
        rj.mark_done("score", {"reused_from": str(src)})
        log(f"reusing calibration, teacher answers, REAP scores and teacher reference from {src}")
    else:
        log(f"reusing calibration and teacher answers from {src}; REAP scoring runs again")


def teacher_check(rj) -> None:
    """When calibrate must run: if the installed vLLM has no model class for this architecture (current
    vLLM releases have no Flint), use REAPer's PyTorch model as the teacher instead of failing, with
    shorter answers and fewer samples so it finishes in reasonable time (Flint's PyTorch port generates
    with a KV cache in batches of 128; other models fall back to transformers generate)."""
    if rj.config.teacher_backend != "auto":
        return
    mc = json.loads((rj.model_dir / "config.json").read_text()) if (rj.model_dir / "config.json").exists() else {}
    arch = (mc.get("architectures") or [None])[0]
    if not arch:
        return
    py = os.environ.get("REAPER_VLLM_PY") or sys.executable
    code = ("import sys\ntry:\n from vllm import ModelRegistry\nexcept Exception:\n print('novllm'); sys.exit()\n"
            f"print('yes' if {arch!r} in ModelRegistry.get_supported_archs() else 'no')")
    try:
        out = subprocess.run([py, "-c", code], capture_output=True, text=True, timeout=300).stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError):
        return
    if not out or out[-1] != "no":
        return
    raw = json.loads(rj.path("config.json").read_text())
    raw.update(teacher_backend="hf", answer_max_tokens=min(raw.get("answer_max_tokens", 1024), 512),
               calib_samples=min(raw.get("calib_samples", 256), 160))
    rj.path("config.json").write_text(json.dumps(raw, indent=2) + "\n")
    log(f"teacher: this vLLM has no {arch}; using REAPer's PyTorch model as the teacher "
        f"({raw['calib_samples']} samples, answers up to {raw['answer_max_tokens']} tokens)")


def rekeep(rj) -> None:
    """Kept experts per layer for this job's prune ratio, from the saved REAP scores."""
    from reaper_pipeline import moe
    s, r = rj.read("work/saliency.json"), rj.read("work/reap.json")
    mc = json.loads((rj.model_dir / "config.json").read_text()) if (rj.model_dir / "config.json").exists() else {}
    mc = mc.get("text_config") or mc
    groups = int(r.get("groups") or 1)
    n_keep = moe.experts_to_keep(s["experts"], rj.config.prune, int(r["top_k"]), groups, mc.get("topk_group"))
    for L in s["layers"]:
        L["kept"] = moe.choose_keep(L["saliency"], L["freq"], n_keep, groups)
    s["kept"] = n_keep
    r = {k: v for k, v in r.items() if k not in ("bytes_before", "bytes_after", "out")}
    r["experts_kept"] = n_keep
    rj.write("work/saliency.json", s)
    rj.write("work/reap.json", r)


def schema_stages(schema: str | None) -> set[str]:
    """The stages a schema runs (its stage_list); empty = every stage."""
    try:
        from blackhole_tui.catalog import SCHEMAS
    except ImportError:
        return set()
    s = next((s for s in SCHEMAS if s.id == schema), None)
    return set(s.stage_list) if s is not None and s.stage_list else set()


def run(bh: BHJob, start: str | None = None) -> int:
    rj = bh.reaper()
    if start:
        rj.clear_from(STAGES, start)
    p = bh.params
    emit("pipeline", "running", 0, f"started {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"job {bh.root.name}: {bh.config['model']} · schema {bh.config.get('schema')} · REAP {p['prune']:.0%}")
    wanted = schema_stages(bh.config.get("schema"))
    for name, stages, who in PROCESSES:
        if wanted and not any(s in wanted for s in stages):
            continue
        if name == "calibrate":
            seed_from_previous(bh, rj)
            if not rj.is_done("calibrate"):
                teacher_check(rj)
        if all(rj.is_done(s) for s in stages):
            for s in stages:
                info = json.loads(rj.path(".done", s).read_text() or "{}")
                why = "reused from " + Path(info["reused_from"]).name if info.get("reused_from") else "cached"
                emit(s, "skipped", 100, why)
            continue
        if name == "reap" and rj.is_done("score"):
            rekeep(rj)  # a saliency.json from another prune ratio
        emit(next(s for s in stages if not rj.is_done(s)), "running", 0, "starting")
        rc, tail, last = run_process(name, who, bh, rj)
        failed = next((s for s in stages if not rj.is_done(s)), None)
        if rc != 0 or failed:
            stage = failed or (last or {}).get("stage") or stages[-1]
            if not (last and last["status"] == "error"):
                emit(stage, "error", msg=error_message(rc, tail))
            emit("pipeline", "error", msg=f"{stage} failed", outcome="error")
            return 1
    rec = bh.read("receipt.json") or {}
    final = Path(rec.get("model") or rj.final_dir)
    emit("pipeline", "done", 100, str(final), model=str(final), size_gb=rec.get("size_gb"), outcome="done",
         gates_passed=rec.get("gates_passed", True))
    return 0


def _stop(signum, _frame):
    if _child and _child.poll() is None:
        _child.terminate()
        try:
            _child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            _child.kill()
    emit("pipeline", "error", msg="stopped", outcome="stopped")
    sys.exit(128 + signum)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--job", required=True, help="TUI job directory (holds config.json)")
    ap.add_argument("--from", dest="start", choices=STAGES, help="rerun from this stage onward")
    a = ap.parse_args(argv)
    venv = Path.home() / "vllm-venv" / "bin" / "python"   # setup-vllm.sh: vLLM in its own venv
    if venv.exists() and not os.environ.get("BLACKHOLE_NO_VLLM"):
        os.environ.setdefault("REAPER_VLLM_PY", str(venv))
        os.environ.setdefault("BLACKHOLE_VLLM_PY", str(venv))
    bh = BHJob(a.job)
    (bh.root / "pid").write_text(str(os.getpid()))
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    try:
        return run(bh, a.start)
    except Exception as e:  # orchestration bug: say so in the TUI instead of dying silently
        import traceback
        traceback.print_exc()
        emit("pipeline", "error", msg=f"{type(e).__name__}: {e}"[:400], outcome="error")
        return 1


if __name__ == "__main__":
    sys.exit(main())
