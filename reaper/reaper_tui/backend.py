"""Glue between the TUI and the two backend modules, which live beside it:

    hardware   reaper_hw (or hw):  detect() -> hardware, plan(hardware, model_id, prune) -> plan
    pipeline   reaper_pipeline (or pipeline): `python -m <pkg> --job DIR`, JSON progress lines on stdout

Either may be missing (they are built in parallel). Without the hardware module
the TUI reads nvidia-smi itself and plans nothing; without the pipeline module,
or with --demo, runs are simulated by mock.py.

A run is started detached (its own session), with stdout going to
DIR/events.log, so it keeps going when the SSH connection drops. The TUI
tails that file, and on start reattaches to the newest job.
"""

from __future__ import annotations

import importlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_STAGES = ["plan", "weights", "calibrate", "score", "prune", "eval", "quant", "package"]


def _module(*names: str):
    for n in names:
        try:
            return importlib.import_module(n)
        except ImportError:
            continue
    return None


def _get(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


# --------------------------------------------------------------------------- hardware

@dataclass
class Gpu:
    name: str
    mem_gb: float
    cc: float = 0.0


@dataclass
class Hardware:
    gpus: list[Gpu]
    cpus: int
    ram_gb: float
    disk_free_gb: float
    raw: object = None  # the hardware module's own object, passed back to its plan()
    source: str = ""
    warnings: list[str] = field(default_factory=list)

    @property
    def total_gpu_gb(self) -> float:
        return sum(g.mem_gb for g in self.gpus)

    def gpu_line(self) -> str:
        if not self.gpus:
            return "no GPU found"
        g = self.gpus[0]
        name = re.sub(r"^NVIDIA\s+", "", g.name)
        cc = f"  sm_{int(round(g.cc * 10))}" if g.cc else ""
        return f"{len(self.gpus)}x {name}  {g.mem_gb:.0f} GB{cc}"

    def host_line(self) -> str:
        return f"{self.cpus} CPUs · {self.ram_gb:.0f} GB RAM · {self.disk_free_gb:.0f} GB free disk"


@dataclass
class Plan:
    ok: bool
    notes: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)


def reaper_home() -> Path:
    return Path(os.environ.get("REAPER_HOME") or Path.home() / ".reaper")


def _nvidia_smi() -> list[Gpu]:
    if not shutil.which("nvidia-smi"):
        return []
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,compute_cap",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True,
                             timeout=20, check=True).stdout
    except (subprocess.SubprocessError, OSError):
        return []
    gpus = []
    for line in out.strip().splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) >= 2 and p[1].replace(".", "").isdigit():
            cc = float(p[2]) if len(p) > 2 and re.fullmatch(r"\d+(\.\d+)?", p[2]) else 0.0
            gpus.append(Gpu(p[0], float(p[1]) / 1024, cc))
    return gpus


def _host() -> tuple[int, float, float]:
    ram = 0.0
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                ram = float(line.split()[1]) / 1024**2
    except OSError:
        pass
    home = reaper_home()
    home.mkdir(parents=True, exist_ok=True)
    return os.cpu_count() or 1, ram, shutil.disk_usage(home).free / 1024**3


def detect_hardware() -> Hardware:
    mod = _module("reaper_hw", "hw")
    if mod is not None and hasattr(mod, "detect"):
        try:
            raw = mod.detect()
            gpus = [Gpu(str(_get(g, "name", "GPU")), float(_get(g, "mem_gb", 0) or 0), float(_get(g, "cc", 0) or 0))
                    for g in _get(raw, "gpus", []) or []]
            cpus, ram, disk = _host()
            return Hardware(gpus, int(_get(raw, "cpus", cpus) or cpus), float(_get(raw, "ram_gb", ram) or ram),
                            float(_get(raw, "disk_free_gb", disk) or disk), raw, mod.__name__,
                            [str(w) for w in _get(raw, "warnings", []) or []])
        except Exception as e:  # a broken backend must not take the TUI down
            hw = Hardware(_nvidia_smi(), *_host(), source="nvidia-smi")
            hw.source = f"nvidia-smi ({mod.__name__} failed: {e})"
            return hw
    return Hardware(_nvidia_smi(), *_host(), source="nvidia-smi")


def _planner():
    mod = _module("reaper_hw", "hw")
    fn = (getattr(mod, "plan", None) or getattr(mod, "make_plan", None)) if mod else None
    return mod, fn


def _sizes(mod, model_id: str):
    """The marketplace's headline sizes as the planner's ModelInfo (no network), or None."""
    from reaper_tui.catalog import MARKET
    m = next((m for m in MARKET if m.id == model_id), None)
    info = getattr(mod, "ModelInfo", None)
    if m is None or not m.total_b or info is None or not hasattr(info, "from_sizes"):
        return None
    return info.from_sizes(m.id, m.total_b, m.active_b, m.experts, m.top_k)


def _to_plan(p) -> Plan:
    return Plan(bool(_get(p, "ok", True)), list(_get(p, "notes", []) or []), list(_get(p, "problems", []) or []))


def plan_for(hw: Hardware, model_id: str, prune: float) -> Plan | None:
    """The hardware module's plan for this model, or None when there is no planner.
    Reads the model's config from the Hub; when that fails (no network, gated
    repo) a marketplace model is planned from its headline sizes instead."""
    mod, fn = _planner()
    if fn is None or hw.raw is None:
        return None
    try:
        p = _to_plan(fn(hw.raw, model_id, prune))
    except Exception as e:
        return Plan(False, [], [f"planner failed: {e}"])
    if not p.ok and (sizes := _sizes(mod, model_id)) is not None:
        try:
            q = _to_plan(fn(hw.raw, sizes, prune))
        except Exception:
            return p
        if q.ok or len(q.problems) < len(p.problems):
            q.notes.insert(0, "planned from marketplace sizes (could not read the model's config.json)")
            return q
    return p


def fits(hw: Hardware, total_b: float, model_id: str = "") -> str:
    """Whether a marketplace model runs here: the hardware planner's verdict from
    the model's headline sizes, or a rough bf16 estimate without the planner."""
    if not hw.gpus:
        return "no GPU"
    mod, fn = _planner()
    sizes = _sizes(mod, model_id) if fn and hw.raw is not None and model_id else None
    if sizes is not None:
        try:
            p = fn(hw.raw, sizes, 0.5)
        except Exception:
            p = None
        if p is not None:
            if not _get(p, "ok", False):
                return "✗"
            reap = _get(p, "reap", {}) or {}
            teacher = _get(p, "teacher", {}) or {}
            if str(reap.get("placement", "")).endswith("offload"):
                return "✓ offload"
            if teacher.get("quantization") == "fp8":
                return "✓ fp8"
            n = max(len(reap.get("gpus") or []), int(teacher.get("tensor_parallel_size") or 1))
            return f"✓ {n} GPU" + ("s" if n > 1 else "")
    return _rough_fit(hw, total_b)


def _rough_fit(hw: Hardware, total_b: float) -> str:
    if not hw.gpus:
        return "no GPU"
    need = total_b * 2 * 1.1
    per = hw.gpus[0].mem_gb * 0.9
    for n in (1, 2, 4, 8):
        if n <= len(hw.gpus) and n * per >= need:
            return f"✓ {n} GPU" + ("s" if n > 1 else "")
    if hw.gpus[0].cc >= 8.9 and len(hw.gpus) * per >= need / 2:
        return "✓ fp8"
    return f"✗ {need:.0f} GB"


# --------------------------------------------------------------------------- pipeline

def pipeline_module() -> str | None:
    if _module("reaper_pipeline"):
        return "reaper_pipeline"
    # Accept a package named `pipeline`, not LobBot's single-file pipeline.py.
    mod = _module("pipeline")
    if mod is not None and Path(getattr(mod, "__file__", "") or "").name == "__init__.py":
        return "pipeline"
    return None


def pipeline_stages() -> list[str]:
    name = pipeline_module()
    mod = _module(name) if name else None
    stages = getattr(mod, "STAGES", None)
    return list(stages) if stages else list(DEFAULT_STAGES)


def eval_line(ev: dict | None) -> str:
    """The eval stage's comparison of the pruned model with the teacher, or ""."""
    if not ev or ev.get("top1_agreement") is None:
        return ""
    out = f"top-1 agreement {float(ev['top1_agreement']):.0%}"
    if ev.get("kl") is not None:
        out += f" · KL {float(ev['kl']):.3f}"
    if ev.get("ppl_teacher") is not None and ev.get("ppl_pruned") is not None:
        out += f" · ppl {float(ev['ppl_teacher']):.2f} → {float(ev['ppl_pruned']):.2f}"
    return out


def parse_event(line: str) -> dict | None:
    line = line.strip()
    if not line.startswith("{"):
        return None
    try:
        ev = json.loads(line)
    except json.JSONDecodeError:
        return None
    if isinstance(ev, dict) and "stage" in ev and ev.get("status") in ("running", "done", "error", "skipped"):
        return ev
    return None


def job_dir(model: str, prune: float) -> Path:
    base = re.sub(r"[^A-Za-z0-9._-]+", "-", model.rstrip("/").split("/")[-1])
    return reaper_home() / "jobs" / f"{base}-prune{round(prune * 100)}"


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
    def start(cls, settings: dict) -> "Run":
        mod = pipeline_module()
        if mod is None:
            raise RuntimeError("the pipeline module is not installed yet (reaper_pipeline); use --demo to simulate")
        root = job_dir(settings["model"], settings["prune"])
        root.mkdir(parents=True, exist_ok=True)
        (root / "config.json").write_text(json.dumps(settings, indent=2) + "\n")
        run = cls(root)
        with open(run.log, "a") as out:
            out.write(json.dumps({"stage": "pipeline", "status": "running", "pct": 0,
                                  "msg": f"started {time.strftime('%Y-%m-%d %H:%M:%S')}"}) + "\n")
            out.flush()
            run._pos = run.log.stat().st_size
            proc = subprocess.Popen([sys.executable, "-u", "-m", mod, "--job", str(root)],
                                    stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                    start_new_session=True, env={**os.environ, "PYTHONUNBUFFERED": "1"})
        (root / "pid").write_text(str(proc.pid))
        return run

    @classmethod
    def latest(cls) -> "Run | None":
        jobs = reaper_home() / "jobs"
        logs = sorted(jobs.glob("*/events.log"), key=lambda p: p.stat().st_mtime) if jobs.exists() else []
        return cls(logs[-1].parent) if logs else None

    def pid(self) -> int | None:
        try:
            pid = int((self.root / "pid").read_text())
            os.kill(pid, 0)
            return pid
        except (OSError, ValueError):
            return None

    def running(self) -> bool:
        return self.pid() is not None

    def read_new(self) -> list[tuple[dict | None, str]]:
        """(event or None, raw line) for every complete line written since the last call."""
        if not self.log.exists():
            return []
        with open(self.log, "rb") as f:
            f.seek(self._pos)
            data = f.read()
        cut = data.rfind(b"\n") + 1  # a partial last line waits for the next call
        self._pos += cut
        lines = data[:cut].decode(errors="replace").splitlines()
        return [(parse_event(l), l) for l in lines if l.strip()]

    def replay(self) -> list[tuple[dict | None, str]]:
        self._pos = 0
        return self.read_new()

    def stop(self) -> bool:
        pid = self.pid()
        if pid is None:
            return False
        os.killpg(pid, signal.SIGTERM)
        return True


# --------------------------------------------------------------------------- model names

# Model makers whose own repos win over fine-tunes and quantized re-uploads.
OFFICIAL = ("qwen", "deepseek-ai", "zai-org", "thudm", "openai", "mistralai", "meta-llama", "moonshotai",
            "minimaxai", "baidu", "tencent", "allenai", "google", "microsoft", "ibm-granite", "inclusionai",
            "xiaomimimo", "stepfun-ai", "nvidia", "lgai-exaone", "cohere", "arcee-ai", "meituan-longcat")
REUPLOAD = re.compile(r"gguf|awq|gptq|mlx|fp8|int4|int8|bnb|4bit|8bit|exl2|nvfp4|quant|reap|pruned|abliterated",
                      re.I)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9.]+", "", s.lower())


def rank(query: str, ids: list[tuple[str, int]]) -> list[str]:
    """Repo ids (with download counts) best match first for a fuzzy query."""
    nq = _norm(query.split("/")[-1])

    def score(item) -> tuple:
        rid, downloads = item
        org, name = rid.split("/", 1) if "/" in rid else ("", rid)
        n = _norm(name)
        return (
            nq in n,                                  # the whole query appears in the name
            org.lower() in OFFICIAL,                  # the maker's own repo
            not REUPLOAD.search(rid),                 # not a quantized or pruned re-upload
            "instruct" in n or "chat" in n,           # the assistant version, not the base model
            -abs(len(n) - len(nq)),                   # fewest extra words
            downloads or 0,
        )

    return [rid for rid, _ in sorted(ids, key=score, reverse=True)]


def resolve_model(query: str, api=None) -> tuple[str | None, list[str]]:
    """A fuzzy name ("qwen3.8-27B-A3B") to a Hugging Face repo id, plus up to five
    other candidates. Local dirs and exact ids are returned as given; the
    marketplace is tried before searching the Hub."""
    from reaper_tui.catalog import MARKET

    q = query.strip()
    if Path(q).expanduser().is_dir():
        return str(Path(q).expanduser().resolve()), []
    nq = _norm(q.split("/")[-1])
    local = [(m.id, 0) for m in MARKET if m.id.lower() == q.lower() or nq in _norm(m.id.split("/")[-1])]
    if local:
        ranked = rank(q, local)
        return ranked[0], ranked[1:6]
    if api is None:
        try:
            from huggingface_hub import HfApi
            api = HfApi()
        except ImportError:
            return (q if "/" in q else None), []
    if "/" in q:
        try:
            return api.model_info(q).id, []
        except Exception:
            q = q.split("/", 1)[1]
    # The full name first, then drop trailing parts ("-A3B", "-27B") until the Hub finds something.
    parts = q.replace(" ", "-").split("-")
    terms = [q, "-".join(parts)] + ["-".join(parts[:n]) for n in range(len(parts) - 1, 0, -1)]
    found: list = []
    for term in dict.fromkeys(terms):
        try:
            found = list(api.list_models(search=term, sort="downloads", limit=50))
        except Exception:
            return None, []
        if found:
            break
    if not found:
        return None, []
    ranked = rank(q, [(m.id, getattr(m, "downloads", 0)) for m in found])
    return ranked[0], ranked[1:6]
