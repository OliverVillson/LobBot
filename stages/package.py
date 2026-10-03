"""Stage 6: package the winner for download.

Writes:
  out/model.gguf   the model the TUI downloads
  out/Modelfile    for `ollama create <name> -f Modelfile`

The Modelfile spells out the chat template and stop tokens (ChatML for Qwen,
Gemma 4's <|turn> format, Gemma 2/3's <start_of_turn>) instead of relying on
Ollama to recognise the GGUF's Jinja template, so the model still chats
correctly in an Ollama version that does not know the family yet.
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


# Gemma 4: system is its own turn; the assistant role is "model". The
# generation prompt ends with an empty thought channel, Gemma 4's non-thinking
# prompt and the exact prefix the student is trained on (stages/taskdata.py).
GEMMA4_TEMPLATE = """{{- if .System }}<|turn>system
{{ .System }}<turn|>
{{ end }}{{- range .Messages }}{{- if ne .Role "system" }}<|turn>{{ if eq .Role "assistant" }}model{{ else }}{{ .Role }}{{ end }}
{{ .Content }}<turn|>
{{ end }}{{- end }}<|turn>model
<|channel>thought
<channel|>"""

# Gemma 2/3 have no system role: the system prompt is folded into the first user turn.
GEMMA3_TEMPLATE = """{{- $sys := .System }}{{- $first := true }}{{- range .Messages }}{{- if eq .Role "user" }}<start_of_turn>user
{{ if and $first $sys }}{{ $sys }}

{{ end }}{{ .Content }}<end_of_turn>
{{ $first = false }}{{- else if eq .Role "assistant" }}<start_of_turn>model
{{ .Content }}<end_of_turn>
{{ end }}{{- end }}<start_of_turn>model
"""

FAMILIES = {
    "chatml": (CHATML_TEMPLATE, ["<|im_end|>", "<|im_start|>"]),
    "gemma4": (GEMMA4_TEMPLATE, ["<turn|>", "<|turn>"]),
    "gemma3": (GEMMA3_TEMPLATE, ["<end_of_turn>", "<start_of_turn>"]),
}


def template_family(template: str, model_id: str = "") -> str | None:
    """Which explicit template to write: from the GGUF's Jinja template, else
    (unreadable, e.g. dry run) from the HF model id. None leaves it to Ollama."""
    if "<|turn>" in template:
        return "gemma4"
    if "<start_of_turn>" in template:
        return "gemma3"
    if "<|im_start|>" in template:
        return "chatml"
    if template:
        return None
    m = model_id.lower()
    if "gemma-4" in m or "gemma4" in m:
        return "gemma4"
    if "gemma" in m:
        return "gemma3"
    return "chatml"  # Qwen family (also the dry-run default)


def modelfile(system: str, template: str, model_id: str = "") -> str:
    lines = ["FROM ./model.gguf"]
    family = template_family(template, model_id)
    if family:
        tmpl, stops = FAMILIES[family]
        lines.append(f'TEMPLATE """{tmpl}"""')
        lines += [f'PARAMETER stop "{s}"' for s in stops]
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
    model_id = job.config.student if report["winner"] == "dense" else job.config.teacher
    job.path("out", "Modelfile").write_text(modelfile(system, chat_template(job, dst), model_id))
    job.mark_done(STAGE, {"winner": report["winner"]})
    emit(STAGE, "done", 100, f"{report['winner']} ready at out/model.gguf")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    run_stage(Job(ap.parse_args().job))
