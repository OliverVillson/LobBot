"""Running things on the VM: shell commands over SSH, the API server, the tunnel."""

from __future__ import annotations

import contextlib
import shlex
import socket
import subprocess
import time
import urllib.request
from pathlib import Path

from lobbot.config import Settings


class RemoteError(RuntimeError):
    pass


def rpath(p: str) -> str:
    """Quote a remote path for bash, keeping a leading ~ expandable."""
    if p == "~":
        return '"$HOME"'
    if p.startswith("~/"):
        return '"$HOME"/' + shlex.quote(p[2:])
    return shlex.quote(p)


class Remote:
    def __init__(self, s: Settings):
        self.s = s

    def _ssh_base(self, multiplex: bool = True) -> list[str]:
        cmd = ["ssh", "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=15"]
        if multiplex:
            # One SSH connection shared by every command for 10 minutes: much faster.
            cm = Path.home() / ".cache" / "lobbot"
            cm.mkdir(parents=True, exist_ok=True)
            cmd += ["-o", "ControlMaster=auto", "-o", f"ControlPath={cm}/cm-%C", "-o", "ControlPersist=10m"]
        else:
            cmd += ["-o", "ControlMaster=no", "-o", "ControlPath=none"]
        if self.s.ssh_key:
            cmd += ["-i", str(Path(self.s.ssh_key).expanduser())]
        return cmd + list(self.s.ssh_opts)

    def argv(self, script: str, tty: bool = False) -> list[str]:
        if self.s.local:
            return ["bash", "-c", script]
        return self._ssh_base() + (["-t"] if tty else []) + [self.s.host, "bash -c " + shlex.quote(script)]

    def sh(self, script: str, input: str | None = None, check: bool = True, timeout: float | None = 120) -> str:
        """Run a bash script on the VM and return its stdout."""
        try:
            p = subprocess.run(self.argv(script), input=input, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            raise RemoteError(f"command on {self.s.host} timed out after {timeout:.0f}s")
        if check and p.returncode != 0:
            err = (p.stderr.strip() or p.stdout.strip()).splitlines()[-8:]
            if p.returncode == 255 and not self.s.local:
                raise RemoteError(f"cannot reach {self.s.host} over SSH:\n  " + "\n  ".join(err)
                                  + "\n  (SSH is only allowed from your own IP; check the evroc security group.)")
            raise RemoteError(f"remote command failed ({p.returncode}):\n  " + "\n  ".join(err))
        return p.stdout

    def stream(self, script: str) -> int:
        """Run a script with output going straight to this terminal (for tail -f and the like)."""
        try:
            return subprocess.call(self.argv(script))
        except KeyboardInterrupt:
            return 130

    # ----- API server on the VM -----

    def env_prelude(self) -> str:
        s = self.s
        return (f"cd {rpath(s.repo)} || {{ echo 'no LobBot checkout at {s.repo} (set repo with: lobbot init)' >&2; exit 3; }}\n"
                "[ -f .env.vm ] && . ./.env.vm\n"
                "[ -f \"$HOME/.lobbot-env\" ] && . \"$HOME/.lobbot-env\"\n"
                f"export LOBBOT_JOBS={rpath(s.jobs)}\n")

    def token(self) -> str:
        if self.s.token:
            return self.s.token
        tok = self.sh('f="$HOME/.lobbot-token"; [ -s "$f" ] && cat "$f" || true').strip()
        if not tok:
            raise RemoteError("no API token yet on the VM; run: lobbot up")
        return tok

    def start_server(self, restart: bool = False) -> str:
        """Start agent/server.py on the VM unless it already answers. Returns 'running' or 'started'."""
        s = self.s
        log = rpath(s.jobs.rstrip("/") + "/../lobbot-api.log")
        script = f"""set -e
f="$HOME/.lobbot-token"
if [ ! -s "$f" ]; then (umask 077; python3 -c 'import secrets; print(secrets.token_hex(24))' > "$f"); fi
health() {{ python3 - <<'PY'
import sys, urllib.request
try:
    urllib.request.urlopen("http://127.0.0.1:{s.remote_port}/health", timeout=3)
except Exception:
    sys.exit(1)
PY
}}
if [ "{int(restart)}" = 1 ]; then
  pkill -f "[u]vicorn agent.server:app --host 127.0.0.1 --port {s.remote_port}" || true
  for i in $(seq 20); do health || break; sleep 0.5; done
fi
if health; then echo running; exit 0; fi
{self.env_prelude()}
export LOBBOT_TOKEN="$(cat "$f")"
mkdir -p "$LOBBOT_JOBS"
# "agent.server:app" is quoted so this script's own text never matches the pkill pattern above.
nohup setsid python3 -m uvicorn "agent.server:app" --host 127.0.0.1 --port {s.remote_port} >> {log} 2>&1 < /dev/null &
for i in $(seq 40); do if health; then echo started; exit 0; fi; sleep 0.5; done
echo "the API did not come up; last lines of {s.jobs}/../lobbot-api.log:" >&2
tail -n 25 {log} >&2
exit 1
"""
        return self.sh(script, timeout=90).strip().splitlines()[-1]

    # ----- tunnel -----

    @contextlib.contextmanager
    def tunnel(self, port: int | None = None):
        """Yield the local base URL of the API, opening an SSH tunnel if needed.

        Reuses a tunnel already answering on the configured local port; otherwise
        opens one on `port`, or on a free port for the duration of the command."""
        s = self.s
        if s.local:
            yield f"http://127.0.0.1:{s.remote_port}"
            return
        if _healthy(port or s.local_port):
            yield f"http://127.0.0.1:{port or s.local_port}"  # a tunnel (or `lobbot tunnel`) is already up
            return
        port = port or _free_port()
        cmd = self._ssh_base(multiplex=False) + [
            "-N", "-o", "ExitOnForwardFailure=yes", "-L", f"127.0.0.1:{port}:127.0.0.1:{s.remote_port}", s.host]
        proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        try:
            deadline = time.time() + 20
            while not _healthy(port):
                if proc.poll() is not None:
                    raise RemoteError("SSH tunnel failed: " + (proc.stderr.read() if proc.stderr else "").strip()[-500:])
                if time.time() > deadline:
                    raise RemoteError(f"the API on the VM is not answering on port {s.remote_port}; run: lobbot up")
                time.sleep(0.3)
            yield f"http://127.0.0.1:{port}"
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


# Never send localhost traffic through an HTTP(S) proxy from the environment or macOS settings.
NO_PROXY = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _healthy(port: int) -> bool:
    try:
        with NO_PROXY.open(f"http://127.0.0.1:{port}/health", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False
