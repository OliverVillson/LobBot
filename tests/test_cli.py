"""End-to-end test of the `lobbot` CLI against the dry-run API.

Uses host "local", so the "VM" side (server start script, job files, logs)
runs on this machine through the same bash scripts the CLI sends over SSH.
"""

import json
import os
import shutil
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
    (bin_dir / "ollama").write_text(
        '#!/bin/sh\n'
        'if [ "$1" = list ]; then echo "NAME ID SIZE MODIFIED"; '
        'grep "^ollama create" "$HOME/ollama.calls" 2>/dev/null | awk \'{print $3 ":latest x 6GB now"}\'; exit 0; fi\n'
        'echo "ollama $*" >> "$HOME/ollama.calls"\n')
    (bin_dir / "ollama").chmod(0o755)
    conf = tmp_path / "config.json"
    conf.write_text(json.dumps({"host": "local", "repo": str(ROOT), "jobs": str(tmp_path / "jobs"),
                                "remote_port": port, "models_dir": str(tmp_path / "models")}))
    env = {**os.environ, "HOME": str(home), "LOBBOT_CONFIG": str(conf),
           "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}", "PYTHONPATH": str(ROOT)}
    for k in ("LOBBOT_HOST", "GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY"):
        env.pop(k, None)

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
    assert conf["heal_max_minutes"] == 20
    assert "lobbot save demo --chat" in p.stdout and "wipes /mnt/nvme" in p.stdout
    assert "LobBot results" in p.stdout and "Winner   lobbot-moe" in p.stdout

    assert "demo" in cli("status").stdout
    st = cli("status", "demo").stdout
    assert "done" in st and "package" in st
    assert "lobbot-moe" in cli("eval", "demo").stdout
    assert "Held-out" in cli("results", "demo").stdout or "Winner" in cli("results", "demo").stdout
    assert "LobBot results" in st
    assert json.loads(cli("results", "demo", "--json").stdout)["eval"]["winner"]
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

    # A corrupted local copy is caught by the sha256 check and moved aside.
    (d / "model.gguf").write_bytes(b"garbage")
    assert "checksum mismatch" in cli("download", "demo", ok=False).stderr
    assert (d / "model.gguf.bad").exists() and not (d / "model.gguf").exists()

    p = cli("save", "demo")
    assert "sha256" in p.stdout and "verified" in p.stdout
    assert "ollama create lobbot-support-email-to-ticket -f Modelfile" in (cli.home / "ollama.calls").read_text()
    assert (d / "model.gguf").read_bytes() == model
    assert (d / "model.gguf.sha256").read_text().split()[0] == json.loads((d / "lobbot.json").read_text())["sha256"]
    assert json.loads((d / "eval.json").read_text())["winner"] == "lobbot-moe"
    saved = cli.home / "lobbot-saved/support-email-to-ticket-demo"
    assert (saved / "model.gguf").read_bytes() == model
    assert {"taskspec.json", "config.json", "eval.json", "Modelfile"} <= {f.name for f in saved.iterdir()}
    cli("install", "demo", "--no-ollama")  # alias, and a second save reuses the backup and download
    p = cli("results", "demo")
    assert "Saved" in p.stdout and "lobbot chat lobbot-support-email-to-ticket" in p.stdout
    assert "lobbot save demo" not in p.stdout

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


def test_run_save_chain(cli):
    p = cli("run", "--example", "--fast", "--name", "chained", "--save")
    assert "imported into Ollama" in p.stdout
    assert (cli.tmp / "models/support-email-to-ticket-chained/model.gguf").exists()


def test_init_keeps_remote_tilde(cli):
    home = str(cli.home)
    p = cli("init", "--host", "local", "--repo", f"{home}/LobBot", "--jobs", "/mnt/nvme/jobs")
    conf = json.loads((cli.tmp / "config.json").read_text())
    assert conf["repo"] == "~/LobBot" and conf["jobs"] == "/mnt/nvme/jobs"
    assert "~/LobBot" in p.stdout


def test_pull_dirty_checkout(cli, tmp_path):
    def git(*args, cwd):
        subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", *args], cwd=cwd, check=True,
                       capture_output=True)

    origin, vm, dev = tmp_path / "origin.git", tmp_path / "vm", tmp_path / "dev"
    git("init", "-q", "--bare", "-b", "main", str(origin), cwd=tmp_path)
    git("clone", "-q", str(origin), str(dev), cwd=tmp_path)
    (dev / "data.py").write_text("v1\n")
    git("add", ".", cwd=dev)
    git("commit", "-qm", "v1", cwd=dev)
    git("push", "-q", "origin", "HEAD:main", cwd=dev)
    git("clone", "-q", str(origin), str(vm), cwd=tmp_path)
    (dev / "data.py").write_text("v2 upstream\n")
    git("commit", "-qam", "v2", cwd=dev)
    git("push", "-q", "origin", "HEAD:main", cwd=dev)
    (vm / "data.py").write_text("hotfix on the vm\n")

    conf = json.loads((cli.tmp / "config.json").read_text())
    (cli.tmp / "config.json").write_text(json.dumps({**conf, "repo": str(vm)}))
    p = cli("pull", ok=False)  # origin is not GitHub (on the VM it was a bundle file)
    assert "not GitHub" in p.stderr and "--set-origin" in p.stderr
    p = cli("pull", "--set-origin", str(origin), ok=False)
    assert "local changes" in p.stderr and "data.py" in p.stderr and "--stash" in p.stderr
    assert (vm / "data.py").read_text() == "hotfix on the vm\n"  # nothing touched

    p = cli("pull", "--set-origin", str(origin), "--stash", "-y")
    assert "stashed 1 file" in p.stdout and "updated to" in p.stdout and "v2" in p.stdout
    assert (vm / "data.py").read_text() == "v2 upstream\n"
    assert "lobbot pull" in subprocess.run(["git", "stash", "list"], cwd=vm, capture_output=True, text=True).stdout
    assert "already up to date" in cli("pull", "--set-origin", str(origin)).stdout


def test_job_run_directly_with_pipeline(cli):
    """Jobs started with pipeline.py (not the API) show their real state and can be saved."""
    cli("up")
    job = cli.tmp / "jobs/direct"
    job.mkdir(parents=True)
    (job / "taskspec.json").write_text((ROOT / "examples/support-tickets.taskspec.json").read_text())
    env = {**os.environ, "LOBBOT_DRY_RUN": "1"}
    subprocess.run([sys.executable, "pipeline.py", "--job", str(job), "--only", "data"], cwd=ROOT, env=env, check=True,
                   capture_output=True)
    # Newer APIs read the job dir themselves; older ones say "queued" and the CLI reads it.
    assert any(w in cli("status").stdout for w in ("stopped*:data", "partial"))
    assert "no packaged model" in cli("save", "direct", ok=False).stderr
    subprocess.run([sys.executable, "pipeline.py", "--job", str(job)], cwd=ROOT, env=env, check=True,
                   capture_output=True)
    assert " done" in cli("status").stdout
    assert "queued" not in cli("status", "direct").stdout
    p = cli("save", "direct", "--no-ollama")
    assert "verified" in p.stdout
    assert (cli.tmp / "models/support-email-to-ticket-direct/model.gguf").exists()


def test_run_long(cli, tmp_path):
    """--long sets the long-answer limits, --set wins over it, and a long spec without it is warned about."""
    p = cli("run", "--example", "--long", "--set", "heal_max_len=6144", "--set", "data_max_len=5000",
            "--name", "longjob", "-d")
    conf = json.loads((cli.tmp / "jobs/longjob/config.json").read_text())
    assert conf == {"data_answer_max_tokens": 4096, "data_max_len": 5000, "heal_max_len": 6144}
    assert "answers up to 4096 tokens" in p.stdout
    assert "use data_max_len=6144 or more" in p.stderr and "without --long" not in p.stderr
    cli("watch", "longjob")

    spec = json.loads((ROOT / "examples/support-tickets.taskspec.json").read_text())
    spec["seed_examples"][0]["output"] = "x" * 4000
    (tmp_path / "long.json").write_text(json.dumps(spec))
    p = cli("run", "long.json", "--fast", "--name", "longwarn", "-d")
    assert "~1000 tokens" in p.stderr and "without --long" in p.stderr
    cli("watch", "longwarn")


def test_run_refuses_keys_the_vm_lacks(cli, tmp_path):
    """The Mac's Config can be newer than the VM checkout's; the VM's decides."""
    vm = tmp_path / "oldvm"
    shutil.copytree(ROOT, vm, ignore=shutil.ignore_patterns(".git", "tests", "__pycache__"))
    util = vm / "stages/_util.py"
    util.write_text(util.read_text().replace("    data_max_len: int | None", "    _gone: int | None"))
    conf = json.loads((cli.tmp / "config.json").read_text())
    (cli.tmp / "config.json").write_text(json.dumps({**conf, "repo": str(vm)}))
    p = cli("run", "--example", "--long", "--name", "old", "-d", ok=False)
    assert "does not know config key(s) data_max_len" in p.stderr and "lobbot pull" in p.stderr
    assert not (cli.tmp / "jobs/old").exists()


def test_new_drafts_on_vm_with_its_key(cli, tmp_path):
    """No key on the laptop: `new` runs the drafting on the VM, where ~/.lobbot-env has the key."""
    fake = tmp_path / "fakegemini"
    fake.mkdir()
    spec = json.loads((ROOT / "examples/support-tickets.taskspec.json").read_text())
    (fake / "spec.json").write_text(json.dumps(spec))
    # Stands in for Gemini on the "VM": answers only when the VM's key is sent.
    (fake / "sitecustomize.py").write_text('''
import io, json, os, urllib.request
_real = urllib.request.urlopen
def _fake(req, *a, **k):
    url = getattr(req, "full_url", req)
    if "generativelanguage" in str(url):
        assert req.get_header("X-goog-api-key") == "vm-key", "wrong key"
        text = open(os.path.join(os.path.dirname(__file__), "spec.json")).read()
        body = json.dumps({"candidates": [{"content": {"parts": [{"text": text}]}}]}).encode()
        class R(io.BytesIO):
            def __enter__(self): return self
            def __exit__(self, *e): return False
        return R(body)
    return _real(req, *a, **k)
urllib.request.urlopen = _fake
''')
    with (cli.home / ".lobbot-env").open("a") as f:
        f.write(f"export GEMINI_API_KEY=vm-key\nexport PYTHONPATH={fake}:{ROOT}\n")
    p = cli("new", "turn support emails into JSON tickets", "-o", "drafted.json")
    assert "drafting on the VM" in p.stdout
    drafted = json.loads((cli.tmp / "drafted.json").read_text())
    assert drafted["task_name"] == "support-email-to-ticket" and len(drafted["seed_examples"]) == 15
    assert "vm-key" not in p.stdout + p.stderr
    assert "--fast\n" in p.stdout and "--long" not in p.stdout

    spec["seed_examples"][1]["output"] = "y" * 6000  # ~1500 tokens: suggest --long
    (fake / "spec.json").write_text(json.dumps(spec))
    p = cli("new", "write C code", "-o", "long.json")
    assert "up to ~1500 tokens" in p.stdout and "lobbot run long.json --fast --long" in p.stdout
