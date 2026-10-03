"""The results report, rendered from real eval reports of the B200 runs."""

import json
from pathlib import Path

from lobbot.report import field_diff, render, why

DATA = Path(__file__).parent / "data"


def facts(job):
    return json.loads((DATA / f"vm_{job}_facts.json").read_text())


def test_quality_report():
    text = "\n".join(render("quality", facts("quality"), color=False, width=100))
    flat = " ".join(text.split())
    assert "Winner   lobbot-moe  6.52 GB  ~49 tok/s" in text
    assert "9.7/10 (99% of the teacher)" in text
    assert "dense scores 9.8/10 but misses it (26 tok/s)" in " ".join(text.split())
    assert "100 held-out examples, written by gemini-3.8-flash" in text
    assert "2030 training examples" in text
    assert "25m26s on the GPU: data 2m29s · reap 8m39s" in flat
    assert "q2_k 77%" in text
    assert "vs teacher: category account≠other" in text
    assert all(len(l) <= 100 for l in text.splitlines() if "─" not in l and "━" not in l)


def test_fast_report_single_candidate():
    text = "\n".join(render("fast", facts("fast"), color=False, width=100))
    assert "the only candidate; it meets the 7 GB / 40 tok/s target" in text
    assert "30 held-out examples, never seen in training" in text


def test_agreement_mode_and_missing_report():
    f = facts("fast")
    f["eval"]["score_method"] = "agreement"
    f["eval"]["teacher"]["score"] = 1.0
    f["eval"]["candidates"][0]["score"] = 0.762
    text = "\n".join(render("fast", f, color=False))
    assert "score 76%" in text and "of the teacher" not in text.split("\n")[3]
    assert "agreement with the teacher's answers" in text
    assert render("x", {"eval": None}, color=False)[0].startswith("No eval report yet")


def test_why_prefers_moe_within_two_points():
    rep = {"score_method": "judge", "winner": "lobbot-moe", "candidates": [
        {"name": "lobbot-moe", "score": 0.95, "size_gb": 6.5, "tok_s_est": 49, "meets_target": True},
        {"name": "dense", "score": 0.96, "size_gb": 5.3, "tok_s_est": 45, "meets_target": True}]}
    assert why(rep, {}).startswith("within 2 points of dense")


def test_field_diff():
    a = '{"category": "bug", "priority": "high", "summary": "a long free-text summary of the email"}'
    b = '{"category": "bug", "priority": "medium", "summary": "another long free-text summary"}'
    assert field_diff(a, b) == [("category", True, "bug", "bug"), ("priority", False, "high", "medium")]
    assert field_diff("plain text", b) is None
