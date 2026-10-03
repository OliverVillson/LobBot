"""Stage 6: package the winner for download.

Writes:
  out/model.gguf   the model the TUI downloads
  out/Modelfile    for `ollama create <name> -f Modelfile`
"""

from __future__ import annotations

import argparse
import json
import os

from common.progress import emit
from stages._util import Job

STAGE = "package"


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
    job.path("out", "Modelfile").write_text(
        f'FROM ./model.gguf\nPARAMETER temperature 0.3\nSYSTEM """{system}"""\n'
    )
    job.mark_done(STAGE, {"winner": report["winner"]})
    emit(STAGE, "done", 100, f"{report['winner']} ready at out/model.gguf")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    run_stage(Job(ap.parse_args().job))
