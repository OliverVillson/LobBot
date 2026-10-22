"""Stage weights: download the model to <models_dir>/<org>--<name> (skipped
for a local directory). Only what loading needs: configs, tokenizer,
safetensors and remote code, not the duplicate original/ or GGUF copies."""

from __future__ import annotations

import threading
from pathlib import Path

from reaper_pipeline.events import emit
from reaper_pipeline.job import Job
from reaper_pipeline.stages._common import stage_main

STAGE = "weights"
ALLOW = ["*.json", "*.safetensors", "*.py", "*.model", "*.tiktoken", "*.txt", "*.jinja", "tokenizer*"]
IGNORE = ["original/*", "metal/*", "*.gguf", "*.bin", "*.pth"]


def _size(d: Path) -> int:
    return sum(p.stat().st_size for p in d.rglob("*") if p.is_file()) if d.exists() else 0


def run(job: Job) -> None:
    cfg = job.config
    dest = job.model_dir
    if Path(cfg.model).expanduser().is_dir():
        job.mark_done(STAGE, {"path": str(dest)})
        emit(STAGE, "done", 100, f"local model {dest}")
        return
    from huggingface_hub import HfApi, snapshot_download

    total = 0
    try:
        info = HfApi().model_info(cfg.model, files_metadata=True)
        total = sum((s.size or 0) for s in info.siblings if s.rfilename.endswith(".safetensors")
                    and not s.rfilename.startswith(("original/", "metal/")))
    except Exception:
        pass
    err: list[BaseException] = []

    def fetch():
        try:
            snapshot_download(cfg.model, local_dir=str(dest), allow_patterns=ALLOW, ignore_patterns=IGNORE)
        except BaseException as e:  # noqa: BLE001 - reported below
            err.append(e)

    t = threading.Thread(target=fetch, daemon=True)
    t.start()
    while t.is_alive():
        t.join(5)
        if total:
            got = _size(dest)
            emit(STAGE, pct=99 * min(1.0, got / total), msg=f"{got / 1e9:.1f} / {total / 1e9:.1f} GB")
    if err:
        raise err[0]
    if not (dest / "config.json").exists():
        raise RuntimeError(f"download finished but {dest}/config.json is missing")
    job.mark_done(STAGE, {"path": str(dest)})
    emit(STAGE, "done", 100, f"{_size(dest) / 1e9:.1f} GB in {dest}")


if __name__ == "__main__":
    stage_main(run, STAGE)
