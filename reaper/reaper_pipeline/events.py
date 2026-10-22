"""Progress protocol, the same as LobBot's common/progress.py: one JSON object
per stdout line,

    {"stage": "score", "status": "running", "pct": 42, "msg": "calibrating 12/512"}

status is running, done, error or skipped. Every other line is a log line. The
orchestrator also appends each event to <job>/events.log, which the TUI tails.
"""

from __future__ import annotations

import json
import sys
import time

STATUSES = ("running", "done", "error", "skipped")


def emit(stage: str, status: str = "running", pct: float | None = None, msg: str = "", **extra) -> dict:
    if status not in STATUSES:
        raise ValueError(f"bad status {status!r}")
    event = {"stage": stage, "status": status, "ts": round(time.time(), 3)}
    if pct is not None:
        event["pct"] = round(max(0.0, min(100.0, pct)), 1)
    if msg:
        event["msg"] = msg
    event.update(extra)
    sys.stdout.write(json.dumps(event) + "\n")
    sys.stdout.flush()
    return event


def parse(line: str) -> dict | None:
    """The event for a protocol line, or None for a log line."""
    line = line.strip()
    if not line.startswith("{"):
        return None
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        return None
    if isinstance(event, dict) and "stage" in event and event.get("status") in STATUSES:
        return event
    return None


def log(msg: str) -> None:
    print(msg, flush=True)


def size(n_bytes: float) -> str:
    return f"{n_bytes / 1e9:.1f} GB" if n_bytes >= 1e9 else f"{n_bytes / 1e6:.1f} MB"
