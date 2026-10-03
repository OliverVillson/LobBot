"""Client for agent/server.py (reached through the SSH tunnel)."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Iterator

from lobbot.remote import NO_PROXY


class ApiError(RuntimeError):
    def __init__(self, status: int, msg: str):
        super().__init__(f"API {status}: {msg}")
        self.status = status


class Api:
    def __init__(self, base: str, token: str):
        self.base = base.rstrip("/")
        self.token = token

    def _req(self, method: str, path: str, body: dict | None = None, headers: dict | None = None):
        data = json.dumps(body).encode() if body is not None else None
        h = {"Authorization": f"Bearer {self.token}", **(headers or {})}
        if data is not None:
            h["Content-Type"] = "application/json"
        return urllib.request.Request(self.base + path, data=data, method=method, headers=h)

    def _open(self, req, timeout: float = 30):
        try:
            return NO_PROXY.open(req, timeout=timeout)
        except urllib.error.HTTPError as e:
            try:
                detail = json.loads(e.read()).get("detail", "")
            except Exception:
                detail = e.reason
            raise ApiError(e.code, str(detail)) from None
        except (urllib.error.URLError, ConnectionError) as e:
            reason = getattr(e, "reason", e)
            raise ApiError(0, f"cannot reach the API at {self.base} ({reason}); run: lobbot up") from None

    def get(self, path: str):
        with self._open(self._req("GET", path)) as r:
            body = r.read()
            return json.loads(body) if r.headers.get_content_type() == "application/json" else body.decode()

    def post(self, path: str, body: dict | None = None):
        with self._open(self._req("POST", path, body if body is not None else {})) as r:
            return json.loads(r.read())

    def health(self) -> dict:
        with self._open(urllib.request.Request(self.base + "/health"), timeout=5) as r:
            return json.loads(r.read())

    def events(self, job_id: str) -> Iterator[dict]:
        """Progress events of the job's latest run, live, until the run ends ({} for a keepalive)."""
        with self._open(self._req("GET", f"/jobs/{job_id}/events"), timeout=120) as r:
            for raw in r:
                line = raw.decode().strip()
                if line.startswith("data: "):
                    yield json.loads(line[6:])
                elif line.startswith(":"):
                    yield {}  # keepalive: lets a live view refresh its clock

    def download(self, job_id: str, dest: Path, progress: Callable[[int, int], None] | None = None) -> Path:
        """Download the GGUF to dest, resuming a partial dest.part if there is one."""
        part = dest.with_name(dest.name + ".part")
        have = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            r = self._open(self._req("GET", f"/jobs/{job_id}/model", headers=headers), timeout=60)
        except ApiError as e:
            if e.status == 416:  # part file is already complete
                part.replace(dest)
                return dest
            raise
        with r:
            if have and r.status != 206:
                have = 0  # server ignored the range; start over
            total = have + int(r.headers.get("Content-Length") or 0)
            with part.open("ab" if have else "wb") as f:
                done = have
                while chunk := r.read(1 << 20):
                    f.write(chunk)
                    done += len(chunk)
                    if progress:
                        progress(done, total)
        if total and part.stat().st_size != total:
            raise ApiError(0, f"download stopped at {part.stat().st_size} of {total} bytes; run it again to resume")
        os.replace(part, dest)
        return dest
