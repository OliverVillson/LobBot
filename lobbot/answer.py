"""Readable model answers: JSON answers shown as labelled fields and real code.

A task model often answers with JSON, e.g. {"pattern_name": ..., "header": "...",
"implementation": "..."}. Printed raw, its code is one escaped line full of \\n
and \\". This module parses such answers (tolerating code fences and text around
the object), shows short fields as labelled lines and multi-line strings as code
under a heading, guesses a file name for each code field, writes the files, and
checks that C code compiles. Anything that is not JSON is shown as it is.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass
class Code:
    field: str
    text: str
    name: str  # guessed file name


def parse(text: str):
    """The JSON object or array in an answer, or None if there isn't one."""
    t = text.strip()
    m = re.match(r"^```[\w-]*\n(.*?)\n?```$", t, re.S)
    if m:
        t = m.group(1).strip()
    for candidate in (t, t[t.find("{"):t.rfind("}") + 1] if "{" in t else ""):
        if not candidate:
            continue
        try:
            obj = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(obj, (dict, list)):
            return obj
    return None


def looks_like_json(text: str) -> bool:
    t = text.lstrip()
    return t.startswith("{") or t.startswith("[{") or t.startswith("```json")


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:40]


def _ext(field: str, text: str) -> str:
    if re.search(r"^\s*#\s*(ifndef\s+\w*_H\w*|pragma\s+once)\s*$", text, re.M) or "header" in field.lower():
        return ".h"
    if re.search(r"^\s*#\s*include\b", text, re.M) or re.search(r"\bint\s+main\s*\(", text):
        return ".c"
    if re.search(r"^\s*(def|class|import|from)\s", text, re.M):
        return ".py"
    if re.search(r"^\s*(SELECT|CREATE|INSERT|UPDATE)\b", text, re.M | re.I):
        return ".sql"
    if text.startswith("#!") or re.search(r"^\s*(echo|export|set -e)\b", text, re.M):
        return ".sh"
    if re.search(r"^\s*(function|const|let|export)\s", text, re.M):
        return ".js"
    if re.match(r"\s*[\[{]", text) and parse(text) is not None:
        return ".json"
    if re.search(r"^\w[\w\s\*]*\b\w+\s*\([^;{)]*\)\s*\{", text, re.M) or re.search(r"^\s*typedef\b", text, re.M):
        return ".c"  # C-like function definitions or typedefs without any #include
    return ".txt"


def code_fields(obj, base: str = "") -> list[Code]:
    """Multi-line string fields, each with a guessed file name.

    C headers are named from their include guard (OBSERVER_H -> observer.h) and
    sources from the header they include; otherwise from a short name-like field
    of the answer (e.g. pattern_name) or the field itself."""
    if not isinstance(obj, dict):
        return []
    found = [(k, v) for k, v in obj.items() if isinstance(v, str) and "\n" in v.strip()]
    if not base:
        short = [v for k, v in obj.items() if isinstance(v, str) and "\n" not in v and 0 < len(v) <= 60
                 and re.search(r"name|title|module|pattern|file", k, re.I)]
        base = _slug(short[0]) if short else ""
    out: list[Code] = []
    for k, v in found:
        ext = _ext(k, v)
        name = ""
        if ext == ".h":
            g = re.search(r"^\s*#\s*ifndef\s+([A-Za-z_]\w*?)_H(?:_|EADER)?_*\s*$", v, re.M)
            if g:
                name = g.group(1).lower() + ".h"
        elif ext == ".c":
            inc = re.findall(r'^\s*#\s*include\s+"([\w./-]+\.h)"', v, re.M)
            if inc:
                name = Path(inc[0]).stem + ".c"
        out.append(Code(k, v, name or f"{base or _slug(k)}{ext}"))
    # A header named by its guard and a source still on the fallback name: pair them.
    hs = [c for c in out if c.name.endswith(".h")]
    for c in out:
        if c.name.endswith(".c") and not re.search(r'#\s*include\s+"', c.text) and len(hs) == 1:
            c.name = Path(hs[0].name).stem + ".c"
    seen: dict[str, int] = {}
    for c in out:  # two fields with one name: number the later ones
        n = seen.get(c.name, 0)
        seen[c.name] = n + 1
        if n:
            p = Path(c.name)
            c.name = f"{p.stem}_{n + 1}{p.suffix}"
    return out


def _scalar(v) -> str:
    if isinstance(v, bool):
        return "yes" if v else "no"
    if v is None:
        return "-"
    if isinstance(v, (list, tuple)) and all(not isinstance(x, (dict, list)) for x in v):
        return ", ".join(str(x) for x in v)
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False)
    return str(v)


def render(text: str, color=lambda s, code: s, width: int = 80) -> tuple[str, list[Code]]:
    """The answer as readable text, and its code fields."""
    obj = parse(text)
    if obj is None:
        note = color("(not valid JSON, so shown as it came)", "2") + "\n" if looks_like_json(text) else ""
        return note + text.strip(), []
    if isinstance(obj, list):
        return json.dumps(obj, indent=2, ensure_ascii=False), []
    codes = code_fields(obj)
    by_field = {c.field: c for c in codes}
    short = [(k, v) for k, v in obj.items() if k not in by_field]
    pad = max((len(k) for k, _ in short), default=0)
    lines = []
    for k, v in short:
        if isinstance(v, dict) and v and all(not isinstance(x, (dict, list)) for x in v.values()):
            lines.append(color(k, "1"))
            sub = max(len(str(x)) for x in v)
            lines += [f"  {color(str(sk).ljust(sub), '2')}  {_scalar(sv)}" for sk, sv in v.items()]
        else:
            lines.append(f"{color(k.ljust(pad), '1')}  {_scalar(v)}")
    for c in codes:
        head = f"── {c.field} · {c.name} "
        lines += ["", color(head + "─" * max(3, width - len(head)), "36"), c.text.rstrip("\n")]
    return "\n".join(lines), codes


def write(codes: list[Code], out: Path) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for c in codes:
        p = out / c.name
        p.write_text(c.text if c.text.endswith("\n") else c.text + "\n")
        paths.append(p)
    return paths


def c_compiler() -> str | None:
    return next((shutil.which(x) for x in ("cc", "clang", "gcc") if shutil.which(x)), None)


def compile_check(codes: list[Code], cc: str | None = None) -> tuple[bool, str, str] | None:
    """Compile the C files of an answer (syntax and types only, nothing is linked).

    Returns (ok, command, compiler output), or None if the answer has no C code
    or no compiler is installed."""
    cfiles = [c for c in codes if c.name.endswith((".c", ".h"))]
    cc = cc or c_compiler()
    if not cfiles or not cc:
        return None
    with tempfile.TemporaryDirectory(prefix="lobbot-cc-") as d:
        write(cfiles, Path(d))
        srcs = [c.name for c in cfiles if c.name.endswith(".c")]
        if not srcs:  # header only: check it from a source that includes it
            Path(d, "_check.c").write_text("".join(f'#include "{c.name}"\n' for c in cfiles))
            srcs = ["_check.c"]
        cmd = [Path(cc).name, "-std=c11", "-Wall", "-fsyntax-only", "-I.", *srcs]
        try:
            p = subprocess.run([cc, *cmd[1:]], cwd=d, capture_output=True, text=True, timeout=60)
        except subprocess.TimeoutExpired:
            return False, " ".join(cmd), "the compiler took over 60 s"
        return p.returncode == 0, " ".join(cmd), (p.stdout + p.stderr).strip()


# ----------------------------------------------------------------- Ollama

def ollama_base() -> str:
    import os

    host = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434").rstrip("/")
    if "://" not in host:
        host = "http://" + host
    return host.replace("0.0.0.0", "127.0.0.1")


class OllamaError(RuntimeError):
    pass


def ollama_chat(model: str, messages: list[dict], on_token: Callable[[str], None] | None = None) -> tuple[str, dict]:
    """Stream one answer from the local Ollama; returns (text, final stats)."""
    import urllib.error
    import urllib.request

    from lobbot.remote import NO_PROXY

    base = ollama_base()
    req = urllib.request.Request(base + "/api/chat", method="POST", headers={"Content-Type": "application/json"},
                                 data=json.dumps({"model": model, "messages": messages, "stream": True}).encode())
    parts: list[str] = []
    try:
        with NO_PROXY.open(req, timeout=600) as r:
            for raw in r:
                if not raw.strip():
                    continue
                ev = json.loads(raw)
                if ev.get("error"):
                    raise OllamaError(ev["error"])
                tok = (ev.get("message") or {}).get("content", "")
                if tok:
                    parts.append(tok)
                    if on_token:
                        on_token(tok)
                if ev.get("done"):
                    return "".join(parts), ev
    except urllib.error.HTTPError as e:
        try:
            msg = json.loads(e.read()).get("error", e.reason)
        except ValueError:
            msg = e.reason
        if "not found" in str(msg):
            msg = f"{msg}; see what is installed with: ollama list"
        raise OllamaError(str(msg)) from None
    except (urllib.error.URLError, ConnectionError) as e:
        raise OllamaError(f"cannot reach Ollama at {base} ({getattr(e, 'reason', e)}); "
                          "start the Ollama app or run: ollama serve") from None
    return "".join(parts), {}


def speed(stats: dict) -> str:
    n, ns = stats.get("eval_count"), stats.get("eval_duration")
    if not n or not ns:
        return ""
    return f"{n} tokens · {n / (ns / 1e9):.1f} tok/s"
