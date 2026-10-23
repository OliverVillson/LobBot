"""Jobs: one directory per run under ~/.blackhole/jobs (or $BLACKHOLE_HOME/jobs).

    <job>/config.json   model, schema, target, purpose
    <job>/events.log    the pipeline's stdout: progress events and log lines
    <job>/pid           the detached pipeline process
    <job>/out/          the finished model

The pipeline runs in its own session, so closing the TUI or losing SSH leaves
it running; the next `blackhole` reattaches by tailing events.log. Progress
events are REAPer's protocol, one JSON object per line:

    {"stage": "reap", "status": "running", "pct": 42, "msg": "layer 12/48"}
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

STATUSES = ("running", "done", "error", "skipped")


def home() -> Path:
    return Path(os.environ.get("BLACKHOLE_HOME", Path.home() / ".blackhole"))


def tilde(path: Path | str) -> str:
    """A path for display, with the home directory as ~."""
    p, h = str(path), str(Path.home())
    return "~" + p[len(h):] if p == h or p.startswith(h + "/") else p


def parse_event(line: str) -> dict | None:
    """The event for a protocol line, or None for a log line."""
    line = line.strip()
    if not line.startswith("{"):
        return None
    try:
        ev = json.loads(line)
    except json.JSONDecodeError:
        return None
    if isinstance(ev, dict) and "stage" in ev and ev.get("status") in STATUSES:
        return ev
    return None


def pipeline_module(demo: bool) -> str | None:
    """The module that runs a job. The real one is not written yet."""
    if demo:
        return "blackhole_tui.fake_pipeline"
    try:
        import blackhole_pipeline  # noqa: F401
        return "blackhole_pipeline"
    except ImportError:
        return None


def job_name(settings: dict) -> str:
    model = settings["model"].rstrip("/").split("/")[-1].lower()
    return f"{model}-{settings['schema']}-{time.strftime('%m%d-%H%M%S')}"


class Run:
    """One pipeline run in a job dir: start, tail, stop."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.log = self.root / "events.log"
        self._pos = 0

    @property
    def config(self) -> dict:
        p = self.root / "config.json"
        return json.loads(p.read_text()) if p.exists() else {}

    @classmethod
    def start(cls, settings: dict, demo: bool = False) -> "Run":
        mod = pipeline_module(demo)
        if mod is None:
            raise RuntimeError("the pipeline (blackhole_pipeline) is not installed yet; `blackhole --demo` simulates one")
        root = home() / "jobs" / job_name(settings)
        root.mkdir(parents=True, exist_ok=True)
        (root / "config.json").write_text(json.dumps(settings, indent=2) + "\n")
        run = cls(root)
        with open(run.log, "a") as out:
            proc = subprocess.Popen([sys.executable, "-u", "-m", mod, "--job", str(root)],
                                    stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                    start_new_session=True, env={**os.environ, "PYTHONUNBUFFERED": "1"})
        (root / "pid").write_text(str(proc.pid))
        return run

    @classmethod
    def latest(cls) -> "Run | None":
        jobs = home() / "jobs"
        if not jobs.is_dir():
            return None
        dirs = [d for d in jobs.iterdir() if (d / "events.log").exists()]
        return cls(max(dirs, key=lambda d: (d / "events.log").stat().st_mtime)) if dirs else None

    @property
    def pid(self) -> int | None:
        try:
            return int((self.root / "pid").read_text())
        except (OSError, ValueError):
            return None

    def running(self) -> bool:
        pid = self.pid
        if pid is None:
            return False
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        # a zombie (our own child that exited) still answers kill(0)
        try:
            return Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"
        except OSError:
            return True

    def read_new(self) -> list[tuple[dict | None, str]]:
        if not self.log.exists():
            return []
        # bytes, not text: the offset must stay a byte offset when lines hold "→" and friends
        with open(self.log, "rb") as f:
            f.seek(self._pos)
            data = f.read()
        cut = data.rfind(b"\n") + 1  # leave a half-written line for next time
        self._pos += cut
        text = data[:cut].decode("utf-8", errors="replace")
        return [(parse_event(l), l) for l in text.splitlines() if l.strip()]

    def replay(self) -> list[tuple[dict | None, str]]:
        self._pos = 0
        return self.read_new()

    def stop(self) -> bool:
        pid = self.pid
        if pid is None or not self.running():
            return False
        try:
            os.killpg(pid, signal.SIGTERM)
        except OSError:
            return False
        return True
