"""This machine: GPUs from nvidia-smi, CPUs, RAM and free disk. A small copy of
REAPer's detection; the planner (reaper_hw) plugs in later."""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field

from blackhole_tui.jobs import home


@dataclass
class Gpu:
    name: str
    mem_gb: float


@dataclass
class Hardware:
    gpus: list[Gpu]
    cpus: int
    ram_gb: float
    disk_gb: float
    source: str = "nvidia-smi"
    warnings: list[str] = field(default_factory=list)

    @property
    def gpu_gb(self) -> float:
        return sum(g.mem_gb for g in self.gpus)

    def gpu_line(self) -> str:
        if not self.gpus:
            return "no NVIDIA GPU"
        n = len(self.gpus)
        return f"{n}× {self.gpus[0].name} · {self.gpu_gb:.0f} GB" if n > 1 else f"{self.gpus[0].name} · {self.gpu_gb:.0f} GB"

    def host_line(self) -> str:
        return f"{self.cpus} CPUs · {self.ram_gb:.0f} GB RAM · {self.disk_gb:.0f} GB free"


DEMO = Hardware([Gpu("NVIDIA B200", 179.1)], 32, 256.0, 1800.0, source="demo")


def detect() -> Hardware:
    gpus, warnings = [], []
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=10).stdout
            for line in out.strip().splitlines():
                name, mem = (p.strip() for p in line.split(","))
                gpus.append(Gpu(name, float(mem) / 1024))
        except (OSError, ValueError, subprocess.SubprocessError) as e:
            warnings.append(f"nvidia-smi failed: {e}")
    try:
        ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except (ValueError, OSError):
        ram = 0.0
    root = home()
    while not root.exists():
        root = root.parent
    disk = shutil.disk_usage(root).free / 1e9
    return Hardware(gpus, os.cpu_count() or 0, ram, disk, warnings=warnings)


def source_fits(hw: Hardware | None, total_b: float) -> str:
    """Can this machine hold the bf16 source model for the teacher pass?"""
    if hw is None or not total_b:
        return "?"
    need = total_b * 2
    if hw.gpu_gb and need <= hw.gpu_gb * 0.9:
        return "✓"
    if hw.gpu_gb and need / 2 <= hw.gpu_gb * 0.9:
        return "fp8"
    return "✗"
