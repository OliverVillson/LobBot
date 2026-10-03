"""RobotSpec: what the robot pipeline (robot_pipeline.py) is asked to build.

The map page or `lobbot robot run` writes <job>/robotspec.json; every robot
stage reads it. See docs/robot-mvp.md for the contract.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

SOURCES = ("auto", "lantmateriet", "copernicus", "synthetic")


@dataclass
class SitePick:
    """A square on the map: centre (WGS84), side length, data source."""

    lat: float
    lon: float
    size_m: float = 300.0
    name: str = ""
    source: str = "auto"


@dataclass
class RobotTarget:
    max_size_gb: float = 8.0  # the onboard computer's budget for the policy
    min_hz: float = 10.0  # control rate the policy must keep (the sim steps at 10 Hz)


@dataclass
class RobotSpec:
    task_name: str
    description: str
    site: SitePick | None = None  # build a site from a map pick ...
    site_id: str | None = None  # ... or reuse one already built (farmsim/site.py ids)
    n_demos: int = 50
    finetune_steps: int = 2000
    eval_episodes: int = 20
    max_steps: int = 600  # per episode, at 10 Hz
    target: RobotTarget = field(default_factory=RobotTarget)
    version: int = 1

    def validate(self) -> None:
        if self.version != 1:
            raise ValueError(f"unsupported RobotSpec version {self.version}")
        if not self.task_name or not self.description:
            raise ValueError("task_name and description are required")
        if (self.site is None) == (not self.site_id):
            raise ValueError("give exactly one of site {lat, lon, ...} or site_id")
        if self.site_id and not self.site_id.isalnum():
            raise ValueError("site_id is letters and digits only")
        if self.site:
            s = self.site
            if not (-90 <= s.lat <= 90 and -180 <= s.lon <= 180):
                raise ValueError("site lat/lon out of range")
            if not 100 <= s.size_m <= 5000:
                raise ValueError("site.size_m must be 100 to 5000 m")
            if s.source not in SOURCES:
                raise ValueError(f"site.source must be one of {', '.join(SOURCES)}")
        if not 1 <= self.n_demos <= 5000:
            raise ValueError("n_demos must be 1 to 5000")
        if not 0 <= self.finetune_steps <= 200_000:
            raise ValueError("finetune_steps must be 0 to 200000")
        if not 1 <= self.eval_episodes <= 1000:
            raise ValueError("eval_episodes must be 1 to 1000")
        if not 10 <= self.max_steps <= 20_000:
            raise ValueError("max_steps must be 10 to 20000")
        if not 0.1 <= self.target.max_size_gb <= 64:
            raise ValueError("target.max_size_gb out of range")
        if not 0.1 <= self.target.min_hz <= 1000:
            raise ValueError("target.min_hz out of range")

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict) -> "RobotSpec":
        site = d.get("site")
        spec = cls(
            task_name=d["task_name"],
            description=d["description"],
            site=SitePick(**site) if site else None,
            site_id=d.get("site_id") or None,
            n_demos=int(d.get("n_demos", 50)),
            finetune_steps=int(d.get("finetune_steps", 2000)),
            eval_episodes=int(d.get("eval_episodes", 20)),
            max_steps=int(d.get("max_steps", 600)),
            target=RobotTarget(**d.get("target", {})),
            version=d.get("version", 1),
        )
        spec.validate()
        return spec

    @classmethod
    def load(cls, path: str | Path) -> "RobotSpec":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def save(self, path: str | Path) -> None:
        self.validate()
        Path(path).write_text(self.to_json())
