"""CLI settings, stored in ~/.config/lobbot/config.json (override with LOBBOT_CONFIG)."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path


@dataclass
class Settings:
    # SSH target. "local" runs the remote side on this machine (dev and tests).
    host: str = "evroc-user@194.14.81.33"
    ssh_key: str = ""  # passed as ssh -i when set
    ssh_opts: list[str] = field(default_factory=list)  # extra ssh arguments
    repo: str = "~/LobBot"  # LobBot checkout on the VM
    jobs: str = "/mnt/nvme/jobs"  # LOBBOT_JOBS on the VM
    remote_port: int = 8700  # agent/server.py port on the VM
    local_port: int = 8700  # reuse a tunnel already listening here, else pick a free port
    token: str = ""  # API token; empty reads ~/.lobbot-token on the VM
    models_dir: str = "~/lobbot-models"  # where downloads land on this machine

    @property
    def local(self) -> bool:
        return self.host == "local"


def config_path() -> Path:
    return Path(os.environ.get("LOBBOT_CONFIG", Path.home() / ".config" / "lobbot" / "config.json"))


def load() -> Settings:
    p = config_path()
    data = json.loads(p.read_text()) if p.exists() else {}
    known = {f.name for f in fields(Settings)}
    s = Settings(**{k: v for k, v in data.items() if k in known})
    if os.environ.get("LOBBOT_HOST"):
        s.host = os.environ["LOBBOT_HOST"]
    return s


def save(s: Settings) -> Path:
    p = config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(asdict(s), indent=2) + "\n")
    p.chmod(0o600)  # may hold the API token
    return p
