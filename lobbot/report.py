"""The results report for a finished job: who won, by how much, how fast, and examples."""

from __future__ import annotations

import json
import re
import shutil
from collections import Counter

STAGES = ["data", "reap", "heal", "quantize", "eval", "package"]

# Runs on the VM: everything the report needs from the job dir, as one JSON object.
FACTS_PY = r'''
import json, os, sys, time
d = os.path.expanduser(sys.argv[1])
def rj(p):
    try:
        with open(os.path.join(d, p)) as f:
            return json.load(f)
    except Exception:
        return None
def lines(p):
    try:
        with open(os.path.join(d, p)) as f:
            return sum(1 for l in f if l.strip())
    except Exception:
        return None
times = {}
for s in ["data", "reap", "heal", "quantize", "eval", "package"]:
    done, start = os.path.join(d, ".done", s), None
    try:
        with open(os.path.join(d, "logs", s + ".log"), errors="replace") as f:
            heads = [l for l in f if l.startswith("===== ")]
        if heads:
            start = time.mktime(time.strptime(heads[-1][6:25], "%Y-%m-%d %H:%M:%S"))
    except Exception:
        pass
    end = os.path.getmtime(done) if os.path.exists(done) else None
    times[s] = [start, end]
gguf = os.path.join(d, "out", "model.gguf")
print(json.dumps({
    "eval": rj("out/eval.json"), "taskspec": rj("taskspec.json"), "stats": rj("data/stats.json"),
    "config": rj("config.json") or {}, "effective": rj("work/config.effective.json") or {},
    "heldout": lines("data/heldout.jsonl"), "train": lines("data/train.jsonl"),
    "gguf_bytes": os.path.getsize(gguf) if os.path.exists(gguf) else None, "times": times,
}))
'''


def job_facts(r, job: str) -> dict:
    """Read the job dir on the VM (one SSH call)."""
    from lobbot.remote import rpath

    path = rpath(r.s.jobs.rstrip("/") + "/" + job)
    out = r.sh(f"python3 - {path} <<'PY'\n{FACTS_PY}\nPY", timeout=60)
    return json.loads(out.strip().splitlines()[-1])


def effective_config(facts: dict) -> dict:
    """Config defaults, overlaid with the job's config.json and the recorded effective config."""
    try:
        from dataclasses import asdict

        from stages._util import Config

        cfg = asdict(Config())
    except Exception:
        cfg = {}
    cfg.update(facts.get("config") or {})
    cfg.update(facts.get("effective") or {})
    return cfg


# ---------------------------------------------------------------- formatting

def _c(text: str, code: str, color: bool) -> str:
    return f"\x1b[{code}m{text}\x1b[0m" if color else text


def _vis(s: str) -> int:
    return len(re.sub(r"\x1b\[[0-9;]*m", "", s))


def _pad(s: str, w: int, right: bool = False) -> str:
    gap = " " * max(0, w - _vis(s))
    return gap + s if right else s + gap


def _dur(s: float) -> str:
    s = int(round(s))
    if s >= 3600:
        return f"{s // 3600}h{s % 3600 // 60:02d}m"
    return f"{s // 60}m{s % 60:02d}s" if s >= 60 else f"{s}s"


def _score(v, method: str) -> str:
    if v is None:
        return "–"
    return f"{v * 10:.1f}/10" if method == "judge" else f"{v:.0%}"


def _same(a: str, b: str) -> bool:
    try:
        return json.loads(a) == json.loads(b)
    except Exception:
        return " ".join(a.split()) == " ".join(b.split())


def field_diff(mine: str, theirs: str) -> list[tuple] | None:
    """For JSON object answers: (key, same, my value, teacher value) for each short field
    (labels like category or priority; free text like a summary always differs). None if not JSON."""
    try:
        a, b = json.loads(mine), json.loads(theirs)
    except Exception:
        return None
    if not isinstance(a, dict) or not isinstance(b, dict):
        return None
    rows = []
    for k in b:
        va, vb = a.get(k), b[k]
        if isinstance(vb, (str, int, float, bool)) and len(str(vb)) <= 24:
            rows.append((k, va == vb, va if va is not None else "missing", vb))
    return rows or None


def _one_line(text: str, width: int) -> str:
    t = " ".join(str(text).split())
    return t if len(t) <= width else t[: width - 1] + "…"


def _wrapped(label: str, text: str, width: int, max_lines: int = 4) -> list[str]:
    import textwrap

    lines = textwrap.wrap(" ".join(str(text).split()), max(20, width)) or [""]
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1][: max(0, width - 1)] + "…"
    return [f"  {label if i == 0 else ' ' * 7}  {line}" for i, line in enumerate(lines)]


def why(rep: dict, target: dict) -> str:
    """Plain-English reason the winner won, following stages/eval.py's rule."""
    cands = rep["candidates"]
    win = next(c for c in cands if c["name"] == rep["winner"])
    eligible = [c for c in cands if c["meets_target"]]
    fit = f"{target.get('max_size_gb', 7):g} GB / {target.get('min_tok_s', 40):g} tok/s target"
    what = "judge score" if rep["score_method"] == "judge" else "agreement with the teacher"
    if not eligible:
        return f"no candidate met the {fit}, so the best {what} won"
    best = max(eligible, key=lambda c: c["score"])
    others = [c for c in cands if c is not win]
    if win is best:
        if len(cands) == 1:
            return f"the only candidate; it meets the {fit}"
        missed = [c for c in others if not c["meets_target"]]
        notes = []
        for m in missed:
            reason = target_cell(m, target)[2:]
            notes.append(f"{m['name']} scores {_score(m['score'], rep['score_method'])} but misses it ({reason})")
        return f"best {what} among candidates that meet the {fit}" + (f"; {'; '.join(notes)}" if notes else "")
    return (f"within 2 points of {best['name']} ({_score(best['score'], rep['score_method'])}), "
            "and the compressed MoE is preferred because it is the fast one on a laptop")


def target_cell(c: dict, target: dict) -> str:
    if c["meets_target"]:
        return "✓"
    why_not = []
    if c["size_gb"] > target.get("max_size_gb", 7):
        why_not.append(f"{c['size_gb']:.1f} GB")
    if (c.get("tok_s_est") or 0) < target.get("min_tok_s", 40):
        why_not.append(f"{c.get('tok_s_est') or 0:.0f} tok/s")
    return "✗ " + ", ".join(why_not)


def render(job: str, facts: dict, color: bool = True, width: int | None = None) -> list[str]:
    rep = facts.get("eval")
    if not rep:
        return [f"No eval report yet for job {job} (out/eval.json is missing)."]
    width = width or min(110, shutil.get_terminal_size((100, 20)).columns)
    spec = facts.get("taskspec") or {}
    target = spec.get("target") or {}
    cfg = effective_config(facts)
    method = rep["score_method"]
    teacher = rep["teacher"]
    t_score = teacher.get("score")
    win = next(c for c in rep["candidates"] if c["name"] == rep["winner"])
    bold = lambda s: _c(s, "1", color)
    dim = lambda s: _c(s, "2", color)
    green = lambda s: _c(s, "32", color)

    out = []
    title = f" LobBot results · {spec.get('task_name', '?')} · job {job} "
    out.append(bold("━━" + title + "━" * max(0, width - len(title) - 2)))
    if spec.get("description"):
        out.append(dim("  " + _one_line(spec["description"], width - 2)))
    out.append("")

    rel = (f" ({win['score'] / t_score:.0%} of the teacher)"
           if method == "judge" and t_score and win.get("score") is not None else "")
    tok = f"~{win['tok_s_est']:.0f} tok/s on a {target.get('laptop_ram_gb', 16)} GB laptop" if win.get("tok_s_est") else ""
    out.append(f"  {bold('Winner')}   {green(bold(win['name']))}  {win['size_gb']:.2f} GB  {tok}  "
               f"score {bold(_score(win['score'], method))}{rel}")
    import textwrap

    for i, line in enumerate(textwrap.wrap(why(rep, target), max(30, width - 11))):
        out.append(f"  {bold('Why') if i == 0 else '   '}      {line}")
    out.append("")

    # Scoreboard
    student = str(cfg.get("student", "")).split("/")[-1]
    label = {"lobbot-moe": "lobbot-moe (pruned MoE)", "dense": f"dense ({student})" if student else "dense"}
    head = ["Model", "Size", "Laptop tok/s", "VM tok/s", "Judge" if method == "judge" else "Agreement",
            "vs teacher", "Target"]
    rows = [[f"{re.sub(r'-Instruct.*$', '', teacher['name'])} (teacher)", f"{teacher['size_gb']:.1f} GB", "–", "–", _score(t_score, method),
             "100%", ""]]
    for cand in rep["candidates"]:
        name = label.get(cand["name"], cand["name"])
        if cand["name"] == rep["winner"]:
            name = green(name + " ★")
        rows.append([name, f"{cand['size_gb']:.2f} GB", f"{cand['tok_s_est']:.0f}" if cand.get("tok_s_est") else "–",
                     f"{cand['tok_s_vm']:.0f}" if cand.get("tok_s_vm") else "–", _score(cand["score"], method),
                     f"{cand['score'] / t_score:.0%}" if t_score and cand.get("score") is not None else "–",
                     target_cell(cand, target)])
    widths = [max(_vis(r[i]) for r in [head] + rows) for i in range(len(head))]
    right = {1, 2, 3, 4, 5}
    out.append(("  " + "  ".join(_pad(dim(h), widths[i], i in right) for i, h in enumerate(head))).rstrip())
    for row in rows:
        out.append(("  " + "  ".join(_pad(v, widths[i], i in right) for i, v in enumerate(row))).rstrip())
    out.append("")

    # Facts
    def fact(label: str, text: str) -> None:
        for i, line in enumerate(textwrap.wrap(text, max(30, width - 11))):
            out.append(f"  {_pad(bold(label), 9) if i == 0 else ' ' * 9}{line}")

    stats = facts.get("stats") or {}
    n_held = facts.get("heldout") or stats.get("heldout")
    source = stats.get("heldout_source")
    src = f", written by {source}" if source and source != "teacher" else (", written by the teacher" if source else "")
    fact("Tests", f"{n_held or '?'} held-out examples{src}, never seen in training")
    fact("Scoring", (f"{cfg.get('judge_model')} grades each answer 0-10 against the task's criteria"
                                          if method == "judge" else
                                          "agreement with the teacher's answers (no judge key was set on the VM)"))
    n_train = facts.get("train") or stats.get("train")
    if n_train:
        fact("Data", f"{n_train} training examples written by {re.sub(r'-Instruct.*$', '', str(cfg.get('teacher', 'the teacher')).split('/')[-1])}"
                   + (f" from {stats['seeds']} seeds" if stats.get("seeds") else ""))
    times = facts.get("times") or {}
    parts, total = [], 0.0
    for s in STAGES:
        start, end = (times.get(s) or [None, None])
        if start and end and end >= start:
            parts.append(f"{s} {_dur(end - start)}")
            total += end - start
    if parts:
        fact("Time", f"{_dur(total)} on the GPU: " + " · ".join(parts))
    bits = [b for b in rep.get("bit_widths") or [] if isinstance(b, dict) and b.get("type")]
    if bits:
        cnt = Counter(b["type"] for b in bits)
        mix = ", ".join(f"{t} {n / len(bits):.0%}" for t, n in cnt.most_common(4))
        avg = sum(b.get("bits") or 0 for b in bits) / len(bits)
        fact("Bits", f"expert weights: {mix} (avg {avg:.2f} bits/weight)")
    if facts.get("gguf_bytes"):
        fact("File", f"out/model.gguf, {facts['gguf_bytes'] / 1e9:.2f} GB")
    out.append("")

    # Examples
    samples = (rep.get("samples") or {}).get(rep["winner"]) or []
    if samples:
        out.append(bold(f"  Held-out examples, answered by {rep['winner']}"))
        w = width - 12
        for i, sm in enumerate(samples[:2], 1):
            fields = field_diff(sm.get("output", ""), sm.get("reference", ""))
            if fields is not None:  # JSON answers: compare the short fields with the teacher's
                marks = [green(f"{k} ✓") if same else _c(f"{k} {mine}≠{theirs}", "33", color)
                         for k, same, mine, theirs in fields]
                out.append(dim(f"  ── {i}  vs teacher: ") + dim(" · ").join(marks))
            else:
                ok = _same(sm.get("output", ""), sm.get("reference", ""))
                out.append(dim(f"  ── {i}  ") + (green("same as the teacher") if ok
                                                 else _c("differs from the teacher", "33", color)))
            out += _wrapped(dim("input  "), sm.get("input", ""), w, max_lines=3)
            out += _wrapped(dim("model  "), sm.get("output", ""), w)
            if fields is None and not _same(sm.get("output", ""), sm.get("reference", "")):
                out += _wrapped(dim("teacher"), sm.get("reference", ""), w)
        out.append("")
    return out
