"""TaskSpec: the contract between the planner (frontend) and the pipeline (backend).

The planner writes <job>/taskspec.json; the data stage reads it.
Change this file only with both halves of the team agreeing.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Example:
    input: str
    output: str


@dataclass
class Target:
    max_size_gb: float = 7.0
    min_tok_s: float = 40.0
    laptop_ram_gb: int = 16


@dataclass
class TaskSpec:
    task_name: str
    description: str
    input_format: str
    output_format: str
    seed_examples: list[Example]
    eval_criteria: str
    target: Target = field(default_factory=Target)
    version: int = 1

    def validate(self) -> None:
        if self.version != 1:
            raise ValueError(f"unsupported TaskSpec version {self.version}")
        if not self.task_name or not self.description:
            raise ValueError("task_name and description are required")
        if not 3 <= len(self.seed_examples) <= 50:
            raise ValueError("seed_examples must have 3 to 50 entries")
        if not 1.0 <= self.target.max_size_gb <= 64:
            raise ValueError("target.max_size_gb out of range")

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict) -> "TaskSpec":
        spec = cls(
            task_name=d["task_name"],
            description=d["description"],
            input_format=d.get("input_format", ""),
            output_format=d.get("output_format", ""),
            seed_examples=[Example(**e) for e in d["seed_examples"]],
            eval_criteria=d.get("eval_criteria", ""),
            target=Target(**d.get("target", {})),
            version=d.get("version", 1),
        )
        spec.validate()
        return spec

    @classmethod
    def load(cls, path: str | Path) -> "TaskSpec":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def save(self, path: str | Path) -> None:
        self.validate()
        Path(path).write_text(self.to_json())
