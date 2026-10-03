"""Stage 6: package the winner for download.

Writes:
  out/model.gguf   the model the TUI downloads
  out/Modelfile    for `ollama create <name> -f Modelfile`

The Modelfile spells out the ChatML template and stop tokens for Qwen-family
models instead of relying on Ollama to recognise the GGUF's Jinja template,
so a REAP-pruned model still chats correctly in any Ollama version.
"""

from __future__ import annotations

import argparse
import json
import os

from common.progress import emit
from stages._util import Job

STAGE = "package"

CHATML_TEMPLATE = """{{- if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{- range .Messages }}{{- if ne .Role "system" }}<|im_start|>{{ .Role }}
{{ .Content }}<|im_end|>
{{ end }}{{- end }}<|im_start|>assistant
"""


def chat_template(job: Job, gguf_path) -> str:
    """The chat template embedded in the GGUF, or "" if unreadable (dry run)."""
    import sys
    from pathlib import Path

    try:
        sys.path.insert(0, str(Path(job.config.llama_cpp) / "gguf-py"))
        import gguf

        f = gguf.GGUFReader(str(gguf_path)).fields.get("tokenizer.chat_template")
        return str(bytes(f.parts[-1]), "utf-8") if f else ""
    except Exception:
        return ""


def modelfile(system: str, template: str) -> str:
    lines = ["FROM ./model.gguf"]
    if "<|im_start|>" in template or not template:  # Qwen family (also the dry-run default)
        lines += [f'TEMPLATE """{CHATML_TEMPLATE}"""',
                  'PARAMETER stop "<|im_end|>"', 'PARAMETER stop "<|im_start|>"']
    lines += ["PARAMETER temperature 0.3", "PARAMETER num_ctx 8192", f'SYSTEM """{system}"""']
    return "\n".join(lines) + "\n"


def run_stage(job: Job) -> None:
    from stages.data import system_prompt

    spec = job.spec
    report = json.loads(job.path("out", "eval.json").read_text())
    alloc = json.loads(job.path("work", "allocation.json").read_text())
    src = alloc["candidates"][report["winner"]]["path"]

    dst = job.path("out", "model.gguf")
    dst.unlink(missing_ok=True)
    os.link(src, dst)  # same filesystem; avoids copying several GB

    system = system_prompt(spec).replace('"""', "'''")
    job.path("out", "Modelfile").write_text(modelfile(system, chat_template(job, dst)))
    job.mark_done(STAGE, {"winner": report["winner"]})
    emit(STAGE, "done", 100, f"{report['winner']} ready at out/model.gguf")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    run_stage(Job(ap.parse_args().job))
