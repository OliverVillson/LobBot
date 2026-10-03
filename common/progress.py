"""Progress protocol: one JSON object per stdout line.

    {"stage": "reap", "status": "running", "pct": 42, "msg": "calibrating"}

status is one of: running, done, error, skipped. Anything that is not a JSON
line is a log line. The agent relays these lines to the TUI unchanged.
"""

from __future__ import annotations

import json
import sys
import time

STATUSES = ("running", "done", "error", "skipped")


def emit(stage: str, status: str = "running", pct: float | None = None, msg: str = "", **extra) -> None:
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


def parse(line: str) -> dict | None:
    """Return the event for a protocol line, or None for a log line."""
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
