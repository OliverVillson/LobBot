"""What machine are we on: GPUs, CPU, RAM, disk and the CUDA / driver / torch stack.

Reads nvidia-smi first (no torch needed), then fills in from torch when it is
installed. Never raises: a laptop or a CPU-only VM gets a Hardware with no GPUs.
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

GB = 1024**3

# (minimum compute capability, architecture). Checked top down.
ARCHES = [(12.0, "Blackwell (RTX)"), (10.0, "Blackwell"), (9.0, "Hopper"), (8.9, "Ada"),
          (8.0, "Ampere"), (7.5, "Turing"), (7.0, "Volta"), (0.0, "pre-Volta")]

# Fallback when nvidia-smi is too old to report compute_cap. Checked in order.
CC_BY_NAME = [("GB300", 10.3), ("B300", 10.3), ("GB200", 10.0), ("B200", 10.0), ("B100", 10.0),
              ("RTX PRO 6000", 12.0), ("RTX 50", 12.0), ("H200", 9.0), ("H100", 9.0), ("H800", 9.0),
              ("GH200", 9.0), ("H20", 9.0), ("L40", 8.9), ("L4", 8.9), ("RTX 40", 8.9), ("RTX 6000 ADA", 8.9),
              ("A100", 8.0), ("A800", 8.0), ("A30", 8.0), ("A10", 8.6), ("A40", 8.6), ("RTX 30", 8.6),
              ("A6000", 8.6), ("T4", 7.5), ("V100", 7.0)]

FAMILIES = ["GB300", "B300", "GB200", "B200", "B100", "GH200", "H200", "H100", "H800", "H20", "A100",
            "A800", "L40S", "L40", "L4", "A10G", "A10", "A40", "A30", "T4", "V100"]


def arch_of(cc: float) -> str:
    return next(name for low, name in ARCHES if cc >= low) if cc else "unknown"


@dataclass
class Gpu:
    index: int
    name: str
    mem_gb: float
    free_gb: float
    cc: float = 0.0  # compute capability, e.g. 10.0 for B200, 9.0 for H100
    uuid: str = ""

    @property
    def family(self) -> str:
        n = self.name.upper()
        for key in FAMILIES:
            if key in n:
                return key
        return self.name.replace("NVIDIA ", "") or "GPU"

    @property
    def arch(self) -> str:
        return arch_of(self.cc)

    @property
    def sm(self) -> str:
        return f"sm_{round(self.cc * 10)}" if self.cc else "sm_?"


@dataclass
class Hardware:
    gpus: list[Gpu] = field(default_factory=list)
    driver: str = ""             # NVIDIA driver version
    cuda: str = ""               # highest CUDA version the driver supports
    torch: str = ""              # torch version, "" when not installed
    torch_cuda: str = ""         # CUDA version torch was built against
    torch_arches: list[str] = field(default_factory=list)  # sm_XX kernels torch ships
    nvlink: bool = False
    cpu_model: str = ""
    cpus: int = 1
    ram_gb: float = 0.0
    ram_free_gb: float = 0.0
    disk_path: str = ""
    disk_free_gb: float = 0.0
    warnings: list[str] = field(default_factory=list)  # anything detection could not read

    @property
    def n_gpus(self) -> int:
        return len(self.gpus)

    @property
    def gpu_mem_gb(self) -> float:
        """Smallest GPU's memory: plans assume every GPU is the weakest one."""
        return min((g.mem_gb for g in self.gpus), default=0.0)

    @property
    def cc(self) -> float:
        return min((g.cc for g in self.gpus), default=0.0)

    @property
    def arch(self) -> str:
        return arch_of(self.cc) if self.gpus else "none"

    @property
    def blackwell(self) -> bool:
        return self.cc >= 10.0

    @property
    def bf16(self) -> bool:
        return self.cc >= 8.0

    @property
    def fp8(self) -> bool:
        """FP8 tensor cores: Ada, Hopper, Blackwell."""
        return self.cc >= 8.9

    @property
    def fp4(self) -> bool:
        """NVFP4 / MXFP4 tensor cores: Blackwell."""
        return self.cc >= 10.0

    def summary(self) -> str:
        host = f"{self.cpus} CPUs · {self.ram_gb:.0f} GB RAM · {self.disk_free_gb:.0f} GB free disk"
        if not self.gpus:
            return f"no GPU · {host}"
        g = self.gpus[0]
        names = {x.family for x in self.gpus}
        gpu = f"{self.n_gpus}x {g.family} {g.mem_gb:.0f} GB" if len(names) == 1 else \
            " + ".join(f"{x.family} {x.mem_gb:.0f} GB" for x in self.gpus)
        link = " NVLink" if self.nvlink and self.n_gpus > 1 else ""
        return f"{gpu} ({self.arch} {g.sm}{link}) · {host}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(arch=self.arch, bf16=self.bf16, fp8=self.fp8, fp4=self.fp4, summary=self.summary())
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


def _run(cmd: list[str], timeout: float = 20) -> str | None:
    """stdout of a command, None when it is missing or fails. Tests replace this."""
    if not shutil.which(cmd[0]):
        return None
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=True).stdout
    except (subprocess.SubprocessError, OSError):
        return None


def _num(s: str) -> float | None:
    return float(s) if re.fullmatch(r"\d+(\.\d+)?", s.strip()) else None


def _cc_from_name(name: str) -> float:
    n = name.upper()
    return next((cc for key, cc in CC_BY_NAME if key in n), 0.0)


def _nvidia_smi(hw: Hardware) -> list[Gpu]:
    q = "index,name,memory.total,memory.free,compute_cap,uuid,driver_version"
    out = _run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"])
    if out is None:
        # Older drivers reject compute_cap; ask again without it.
        q = "index,name,memory.total,memory.free,uuid,driver_version"
        out = _run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"])
        if out is None:
            return []
    has_cc = "compute_cap" in q
    gpus = []
    for line in out.strip().splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) < (7 if has_cc else 6) or _num(p[0]) is None:
            continue
        total, free = _num(p[2]) or 0.0, _num(p[3]) or 0.0  # MiB
        cc = (_num(p[4]) if has_cc else None) or _cc_from_name(p[1])
        uuid, hw.driver = (p[5], p[6]) if has_cc else (p[4], p[5])
        gpus.append(Gpu(int(p[0]), p[1], round(total / 1024, 1), round(free / 1024, 1), cc, uuid))
    head = _run(["nvidia-smi"]) or ""
    m = re.search(r"CUDA Version:\s*([\d.]+)", head)
    hw.cuda = m.group(1) if m else ""
    if len(gpus) > 1:
        topo = _run(["nvidia-smi", "topo", "-m"]) or ""
        hw.nvlink = bool(re.search(r"\bNV\d+\b", topo))
    return gpus


def _torch(hw: Hardware, probe: bool) -> list[Gpu]:
    """Fills the torch fields. With probe, also returns the GPUs torch sees (used when
    nvidia-smi is absent); without it CUDA is left uninitialised, so a process that
    forks vLLM workers afterwards is unaffected."""
    try:
        import torch
    except Exception:
        return []
    hw.torch = torch.__version__.split("+")[0]
    hw.torch_cuda = torch.version.cuda or ""
    try:  # build-time kernel list, does not initialise CUDA
        hw.torch_arches = sorted(set((torch._C._cuda_getArchFlags() or "").split()))
    except Exception:
        hw.torch_arches = []
    gpus = []
    if not probe:
        return gpus
    try:
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                free, total = torch.cuda.mem_get_info(i)
                gpus.append(Gpu(i, p.name, round(total / GB, 1), round(free / GB, 1), p.major + p.minor / 10))
    except Exception as e:  # driver/runtime mismatch shows up here
        hw.warnings.append(f"torch cannot use CUDA: {e}")
    return gpus


def _visible(gpus: list[Gpu]) -> list[Gpu]:
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env is None:
        return gpus
    wanted = [v.strip() for v in env.split(",") if v.strip()]
    keep = [g for g in gpus if str(g.index) in wanted or (g.uuid and any(g.uuid.startswith(w) for w in wanted))]
    return keep


def _meminfo() -> tuple[float, float]:
    try:
        info = dict(l.split(":", 1) for l in Path("/proc/meminfo").read_text().splitlines() if ":" in l)
        kb = lambda k: float(info[k].split()[0]) / 1024**2  # noqa: E731
        return kb("MemTotal"), kb("MemAvailable")
    except (OSError, KeyError, ValueError):
        return 0.0, 0.0


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _cpus() -> int:
    try:
        return len(os.sched_getaffinity(0))  # what this process may use, e.g. inside a container
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def home() -> str:
    """Where models and jobs live: $REAPER_HOME, else ~/.reaper."""
    return os.environ.get("REAPER_HOME") or str(Path.home() / ".reaper")


def _disk_free(path: str) -> float:
    p = Path(path)
    while not p.exists() and p != p.parent:  # measure the filesystem it will be created on
        p = p.parent
    try:
        return shutil.disk_usage(p).free / GB
    except OSError:
        return 0.0


def detect(work_dir: str | None = None) -> Hardware:
    """Everything the planner needs about this machine. Safe on machines without a GPU."""
    hw = Hardware()
    smi = _nvidia_smi(hw)
    seen = _torch(hw, probe=not smi)
    hw.gpus = _visible(smi) if smi else seen  # torch already applies CUDA_VISIBLE_DEVICES
    if smi and not hw.gpus:
        hw.warnings.append(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} hides every GPU")
    hw.cpu_model, hw.cpus = _cpu_model(), _cpus()
    hw.ram_gb, hw.ram_free_gb = (round(x, 1) for x in _meminfo())
    hw.disk_path = work_dir or home()
    hw.disk_free_gb = round(_disk_free(hw.disk_path), 1)
    return hw
