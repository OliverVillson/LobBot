"""End-to-end test of the `lobbot` CLI against the dry-run API.

Uses host "local", so the "VM" side (server start script, job files, logs)
runs on this machine through the same bash scripts the CLI sends over SSH.
"""

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture()
def cli(tmp_path):
    port = free_port()
    home = tmp_path / "home"
    home.mkdir()
    (home / ".lobbot-env").write_text("export LOBBOT_DRY_RUN=1\n")  # the API inherits this, so jobs dry-run
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "ollama").write_text('#!/bin/sh\necho "ollama $*" >> "$HOME/ollama.calls"\n')
    (bin_dir / "ollama").chmod(0o755)
    conf = tmp_path / "config.json"
    conf.write_text(json.dumps({"host": "local", "repo": str(ROOT), "jobs": str(tmp_path / "jobs"),
                                "remote_port": port, "models_dir": str(tmp_path / "models")}))
    env = {**os.environ, "HOME": str(home), "LOBBOT_CONFIG": str(conf),
           "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}", "PYTHONPATH": str(ROOT)}
    env.pop("LOBBOT_HOST", None)

    def run(*args, ok=True):
        p = subprocess.run([sys.executable, "-m", "lobbot.app", *args], cwd=tmp_path, env=env,
                           capture_output=True, text=True, timeout=120)
        if ok:
            assert p.returncode == 0, p.stdout + p.stderr
        return p

    run.tmp = tmp_path
    run.home = home
    yield run
    subprocess.run(["pkill", "-f", f"[u]vicorn agent.server:app --host 127.0.0.1 --port {port}"])


def test_full_flow(cli):
    assert "API started" in cli("up").stdout
    assert "API running" in cli("up").stdout
    assert len(cli("token").stdout.strip()) == 48

    p = cli("run", "--example", "--fast", "--name", "demo", "--set", "heal_lr=0.0002")
    assert "pipeline finished" in p.stdout
    conf = json.loads((cli.tmp / "jobs/demo/config.json").read_text())
    assert conf["n_generate"] == 400 and conf["dense_fallback"] is False and conf["heal_lr"] == 0.0002

    assert "demo" in cli("status").stdout
    st = cli("status", "demo").stdout
    assert "done" in st and "package" in st
    assert "lobbot-moe" in cli("eval", "demo").stdout
    assert json.loads(cli("eval", "demo", "--json").stdout)["winner"]
    assert '"stage": "package"' in cli("logs", "demo").stdout
    assert "healed" in cli("logs", "demo", "-s", "heal").stdout

    # Download, then resume a partial download.
    cli("download", "demo")
    d = cli.tmp / "models/support-email-to-ticket-demo"
    model = (cli.tmp / "jobs/demo/out/model.gguf").read_bytes()
    assert (d / "model.gguf").read_bytes() == model
    assert "FROM ./model.gguf" in (d / "Modelfile").read_text()
    (d / "model.gguf").unlink()
    (d / "model.gguf.part").write_bytes(model[:3])
    cli("download", "demo")
    assert (d / "model.gguf").read_bytes() == model and not (d / "model.gguf.part").exists()

    cli("install", "demo")
    assert "ollama create lobbot-support-email-to-ticket -f Modelfile" in (cli.home / "ollama.calls").read_text()

    p = cli("resume", "demo", "--from", "eval")
    assert "pipeline finished" in p.stdout
    assert "quantize skipped" in p.stdout.replace("  ", " ").replace("  ", " ")

    assert "already exists" in cli("run", "--example", "--name", "demo", ok=False).stderr


def test_bad_inputs(cli, tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"task_name": "x", "description": "y", "seed_examples": []}))
    assert "seed_examples" in cli("run", str(bad), ok=False).stderr
    assert "unknown config key" in cli("run", "--example", "--set", "nope=1", ok=False).stderr
    assert "letters and digits" in cli("status", "../etc", ok=False).stderr


def test_secret_and_restart(cli):
    cli("up")
    env_file = cli.home / ".lobbot-env"
    p = subprocess.run([sys.executable, "-m", "lobbot.app", "secret", "GEMINI_API_KEY"], cwd=cli.tmp,
                       env={**os.environ, "HOME": str(cli.home), "LOBBOT_CONFIG": str(cli.tmp / "config.json"),
                            "PYTHONPATH": str(ROOT), "GEMINI_API_KEY": "k'ey"},
                       capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr
    assert "export GEMINI_API_KEY='k'\"'\"'ey'" in env_file.read_text()
    assert "LOBBOT_DRY_RUN=1" in env_file.read_text()
    assert oct(env_file.stat().st_mode & 0o777) == "0o600"
    assert "API restarted" in p.stdout
    assert "API started" in cli("up", "--restart").stdout
