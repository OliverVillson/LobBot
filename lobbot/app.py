"""`lobbot`: Oliver's developer CLI for running the compression pipeline on the GPU VM.

    lobbot init                      where the VM is (saved in ~/.config/lobbot/config.json)
    lobbot doctor                    check SSH, checkout, venvs, weights, GPU, API
    lobbot up                        start the API on the VM (if needed) and check it
    lobbot new "what it should do"   draft a TaskSpec with Gemini
    lobbot run spec.json --fast      start a job and watch its stages live
    lobbot status [job]              all jobs, or one job's stages
    lobbot watch / logs / resume / stop / eval <job>
    lobbot download <job>            fetch the GGUF (resumable) and Modelfile
    lobbot install <job> --chat      import into Ollama and chat with it

Everything talks to agent/server.py on the VM through an SSH tunnel the CLI
opens itself; logs and job files are read over the same SSH connection.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import secrets
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import fields
from pathlib import Path

from lobbot import config as cfgmod
from lobbot.api import Api, ApiError
from lobbot.remote import Remote, RemoteError, rpath

STAGES = ["data", "reap", "heal", "quantize", "eval", "package"]

# Per-job overrides of stages/_util.py:Config. "fast" gets a real GGUF quickly
# (small data set, short calibration, no dense student); "full" is the defaults.
PRESETS: dict[str, dict] = {
    "fast": {"n_generate": 400, "n_heldout": 30, "reap_calib_samples": 128, "heal_epochs": 1.0,
             "dense_fallback": False},
    "full": {},
}


class CliError(RuntimeError):
    pass


# ----------------------------------------------------------------- output helpers

TTY = sys.stdout.isatty()


def c(text: str, code: str) -> str:
    return f"\x1b[{code}m{text}\x1b[0m" if TTY else text


def ok(msg: str) -> None:
    print(c("✓ ", "32") + msg)


def warn(msg: str) -> None:
    print(c("! ", "33") + msg, file=sys.stderr)


def ago(ts: float) -> str:
    s = max(0, time.time() - ts)
    if s < 90:
        return f"{s:.0f}s ago"
    if s < 5400:
        return f"{s / 60:.0f}m ago"
    return f"{s / 3600:.1f}h ago"


def dur(s: float) -> str:
    s = int(s)
    return f"{s // 3600}h{s % 3600 // 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"


ICON = {"pending": c("·", "2"), "running": c("▸", "36"), "done": c("✓", "32"),
        "skipped": c("↷", "2"), "error": c("✗", "31")}
STATE_COLOR = {"done": "32", "running": "36", "error": "31", "queued": "2"}


class Board:
    """Live per-stage progress table, redrawn in place on a terminal."""

    def __init__(self, stages: dict | None = None):
        self.stages = {s: {"status": "pending", "pct": 0, "msg": ""} for s in STAGES}
        for k, v in (stages or {}).items():
            if k in self.stages:
                self.stages[k].update(v)
        self.started: dict[str, float] = {}
        self.ended: dict[str, float] = {}
        self.drawn = 0

    def update(self, e: dict) -> None:
        st = e.get("stage")
        if st not in self.stages:
            return
        row = self.stages[st]
        ts = e.get("ts", time.time())
        if e["status"] == "running" and row["status"] not in ("running",):
            self.started[st] = ts
        if e["status"] in ("done", "error", "skipped"):
            self.ended[st] = ts
        row["status"] = e["status"]
        if "pct" in e:
            row["pct"] = e["pct"]
        if e.get("msg"):
            row["msg"] = e["msg"]
        if not TTY:  # plain log lines when piped
            pct = f" {e['pct']:>5.1f}%" if "pct" in e else ""
            print(f"[{time.strftime('%H:%M:%S')}] {st:<8} {e['status']:<7}{pct} {e.get('msg', '')}", flush=True)

    def lines(self) -> list[str]:
        width = shutil.get_terminal_size((100, 20)).columns
        out = []
        for st, row in self.stages.items():
            pct = float(row.get("pct") or 0)
            if row["status"] in ("done", "skipped"):
                pct = 100.0
            n = int(pct / 5)
            bar = c("█" * n, "36") + c("░" * (20 - n), "2")
            t = ""
            if st in self.started:
                t = dur(self.ended.get(st, time.time()) - self.started[st]) if row["status"] != "skipped" else ""
            msg = row["msg"].replace("\n", " ")[: max(10, width - 52)]  # 52 = visible width before msg
            out.append(f" {ICON.get(row['status'], '?')} {st:<9}{bar} {pct:5.1f}%  {t:>7}  {msg}")
        return out

    def draw(self) -> None:
        if not TTY:
            return
        if self.drawn:
            sys.stdout.write(f"\x1b[{self.drawn}F")
        ls = self.lines()
        for line in ls:
            sys.stdout.write("\x1b[2K" + line + "\n")
        sys.stdout.flush()
        self.drawn = len(ls)


# ----------------------------------------------------------------- shared steps

def settings(a) -> cfgmod.Settings:
    s = cfgmod.load()
    if getattr(a, "host", None):
        s.host = a.host
    return s


def connect(a):
    """(remote, settings) with the API running on the VM."""
    s = settings(a)
    r = Remote(s)
    r.start_server()
    return r, s


def check_job_id(job: str, loose: bool = False) -> str:
    """API job ids are letters and digits. Commands that work over plain SSH (logs,
    stop) also accept - and _, for job dirs made by hand with pipeline.py."""
    if not (job.replace("-", "").replace("_", "") if loose else job).isalnum():
        raise CliError(f"job ids are letters and digits{', - and _' if loose else ''} only, got {job!r}")
    return job


def watch(r: Remote, job_id: str) -> str:
    """Stream a job's progress until its run ends, reconnecting if the tunnel drops.

    Returns the final state: done, error, detached or unknown."""
    import http.client

    board = Board()
    print(c(f"job {job_id}", "1") + c("  (Ctrl-C stops watching; the job keeps running)", "2"))
    final, seen, failures = None, 0, 0
    try:
        while final is None:
            try:
                with r.tunnel() as base:
                    n = 0
                    for e in Api(base, r.token()).events(job_id):
                        n += 1
                        if n <= seen:
                            continue  # the stream replays the run from its start after a reconnect
                        seen, failures = n, 0
                        if e.get("stage") == "pipeline":
                            if e["status"] != "running":
                                final = e
                            continue
                        board.update(e)
                        board.draw()
                if final is None:
                    break  # stream closed cleanly without a final event
            except (OSError, http.client.HTTPException, ApiError, RemoteError) as e:
                if isinstance(e, ApiError) and e.status in (401, 404):
                    raise
                failures += 1
                if failures > 20:
                    raise CliError(f"lost the connection to the VM ({e}); the job keeps running. "
                                   f"Try again with: lobbot watch {job_id}")
                if not TTY:
                    print(f"connection lost ({e}); reconnecting...", file=sys.stderr)
                time.sleep(min(30, 3 * failures))
    except KeyboardInterrupt:
        print(f"\nStopped watching. Pick it up again with: lobbot watch {job_id}")
        return "detached"
    board.draw()
    if final is None:
        warn("the progress stream ended early; check: lobbot status " + job_id)
        return "unknown"
    if final["status"] == "done":
        ok(f"pipeline finished. Next: lobbot eval {job_id}  then  lobbot install {job_id} --chat")
        return "done"
    print(c("✗ pipeline failed: ", "31") + str(final.get("msg", "")))
    print(f"  logs: lobbot logs {job_id}    retry: lobbot resume {job_id}")
    return "error"


def gpu_processes(r: Remote) -> list[str]:
    out = r.sh("command -v nvidia-smi >/dev/null || exit 0; "
               "nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader", check=False)
    return [l.strip() for l in out.splitlines() if l.strip()]


def example_text(r: Remote) -> str:
    """The bundled support-ticket TaskSpec, from this checkout or else the VM's."""
    local = Path(__file__).resolve().parents[1] / "examples" / "support-tickets.taskspec.json"
    if local.exists():
        return local.read_text()
    return r.sh(f"cat {rpath(r.s.repo)}/examples/support-tickets.taskspec.json")


def load_spec(a, r: Remote) -> dict:
    from common.taskspec import TaskSpec

    if a.example:
        text = example_text(r)
    elif a.spec:
        text = Path(a.spec).expanduser().read_text()
    else:
        raise CliError("give a TaskSpec file (make one with: lobbot new \"...\") or --example")
    d = json.loads(text)
    TaskSpec.from_dict(d)  # validate here rather than after the upload
    return d


def job_config(a) -> dict:
    from stages._util import Config

    known = {f.name: f for f in fields(Config)}
    out = dict(PRESETS[a.preset])
    for kv in a.set or []:
        if "=" not in kv:
            raise CliError(f"--set expects key=value, got {kv!r}")
        k, v = kv.split("=", 1)
        if k not in known:
            raise CliError(f"unknown config key {k!r}; known: {', '.join(sorted(known))}")
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


# ----------------------------------------------------------------- commands

def cmd_init(a) -> None:
    s = cfgmod.load()
    for name, label in [("host", "SSH target (user@host or an ~/.ssh/config alias)"),
                        ("ssh_key", "SSH key file (blank uses your ssh defaults)"),
                        ("repo", "LobBot checkout on the VM"),
                        ("jobs", "jobs directory on the VM")]:
        flag = getattr(a, name.replace("ssh_", "") if name == "ssh_key" else name, None)
        if flag is not None:
            setattr(s, name, flag)
        elif sys.stdin.isatty():
            val = input(f"{label} [{getattr(s, name)}]: ").strip()
            if val:
                setattr(s, name, val)
    p = cfgmod.save(s)
    ok(f"saved {p}")
    print("Next: lobbot doctor")


def cmd_doctor(a) -> None:
    s = settings(a)
    r = Remote(s)
    print(f"VM: {s.host}   repo: {s.repo}   jobs: {s.jobs}")
    script = f"""
echo "ssh=ok $(hostname)"
if cd {rpath(s.repo)} 2>/dev/null; then
  echo "repo=ok $(git log -1 --format='%h %s' 2>/dev/null | cut -c1-70) [$(git rev-parse --abbrev-ref HEAD 2>/dev/null)]"
else echo "repo=missing no checkout at {s.repo}"; fi
[ -f .env.vm ] && echo "envvm=ok .env.vm present" || echo "envvm=missing run scripts/setup_vm.sh"
[ -f .env.vm ] && . ./.env.vm
if python3 -c 'import fastapi, uvicorn' 2>/dev/null; then echo "venv=ok $(command -v python3)"; else echo "venv=missing fastapi/uvicorn not importable by $(command -v python3)"; fi
if [ -n "$LOBBOT_VLLM_PY" ] && [ -x "$LOBBOT_VLLM_PY" ]; then echo "vllm=ok $LOBBOT_VLLM_PY"; else echo "vllm=missing vLLM venv"; fi
m="${{LOBBOT_MODELS:-/mnt/nvme/models}}"
if ls "$m" >/dev/null 2>&1; then echo "models=ok $(ls "$m" | tr '\\n' ' ')"; else echo "models=missing $m"; fi
q="${{LOBBOT_LLAMA_CPP:-/mnt/nvme/llama.cpp}}/build/bin/llama-quantize"
[ -x "$q" ] && echo "llama=ok llama.cpp built" || echo "llama=missing $q"
if grep -qs '^export GEMINI_API_KEY=.\\+' "$HOME/.lobbot-env" || [ -n "$GEMINI_API_KEY" ]; then echo "gemini=ok GEMINI_API_KEY set"; else echo "gemini=warn GEMINI_API_KEY not set (lobbot secret GEMINI_API_KEY); eval falls back to teacher agreement"; fi
if command -v nvidia-smi >/dev/null; then
  echo "gpu=ok $(nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader | head -1)"
  n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c . || true)
  [ "$n" -gt 0 ] && echo "gpubusy=warn $n process(es) on the GPU (a job is probably running)" || echo "gpubusy=ok GPU idle"
else echo "gpu=missing nvidia-smi not found"; fi
df -h {rpath(s.jobs.rstrip('/') + '/..')} 2>/dev/null | awk 'NR==2 {{print "disk=ok " $4 " free of " $2 " on " $6}}'
python3 -c 'import urllib.request; urllib.request.urlopen("http://127.0.0.1:{s.remote_port}/health", timeout=3)' 2>/dev/null \
  && echo "api=ok API answering on :{s.remote_port}" || echo "api=warn API not running (lobbot up starts it)"
"""
    try:
        out = r.sh(script, check=False, timeout=60)
    except RemoteError as e:
        raise CliError(str(e))
    if "ssh=ok" not in out:
        r.sh("true")  # raises the SSH error with its message
    bad = 0
    for line in out.splitlines():
        if "=" not in line:
            continue
        key, rest = line.split("=", 1)
        status, _, detail = rest.partition(" ")
        mark = {"ok": c("✓", "32"), "warn": c("!", "33")}.get(status, c("✗", "31"))
        bad += status == "missing"
        print(f" {mark} {key:<8} {detail}")
    if bad:
        raise CliError(f"{bad} check(s) failed")


def cmd_up(a) -> None:
    s = settings(a)
    r = Remote(s)
    if a.restart and not a.force:
        try:
            with r.tunnel() as base:
                running = [j["job_id"] for j in Api(base, r.token()).get("/jobs") if j["state"] == "running"]
        except (RemoteError, ApiError):
            running = []
        if running:
            raise CliError(f"job(s) {', '.join(running)} are running under this API; restarting it would kill them. "
                           "Wait for them, or add --force.")
    state = r.start_server(restart=a.restart)
    with r.tunnel() as base:
        api = Api(base, r.token())
        h = api.health()
        try:
            jobs = api.get("/jobs")
        except ApiError as e:
            if e.status == 401:
                raise CliError("the API on the VM uses a different token than ~/.lobbot-token. "
                               "Restart it with: lobbot up --restart")
            raise
    ok(f"API {state} on {s.host}  (gpu: {h.get('gpu') or 'none'}{', DRY RUN' if h.get('dry_run') else ''})")
    running = [j for j in jobs if j["state"] == "running"]
    print(f"  {len(jobs)} job(s), {len(running)} running" + (": " + ", ".join(j["job_id"] for j in running) if running else ""))


def cmd_tunnel(a) -> None:
    r, s = connect(a)
    with r.tunnel(port=a.port or s.local_port) as base:
        ok(f"API at {base}  (token: lobbot token)")
        print("Leave this running; Ctrl-C closes the tunnel.")
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            pass


def cmd_token(a) -> None:
    print(Remote(settings(a)).token())


def cmd_secret(a) -> None:
    name = a.name
    if not name.replace("_", "").isalnum():
        raise CliError("secret names are letters, digits and underscores")
    value = os.environ.get(name) if not a.prompt else None
    if value:
        print(f"Using {name} from your local environment.")
    else:
        try:
            value = getpass.getpass(f"{name}: ")
        except EOFError:
            value = ""
    if not value:
        raise CliError("empty value")
    s = settings(a)
    r = Remote(s)
    line = f"export {name}={shlex.quote(value)}"
    r.sh(f"""f="$HOME/.lobbot-env"; umask 077; touch "$f"; chmod 600 "$f"
grep -v '^export {name}=' "$f" > "$f.tmp" || true
cat >> "$f.tmp"; mv "$f.tmp" "$f" """, input=line + "\n")
    ok(f"{name} saved in ~/.lobbot-env on the VM (mode 600)")
    # The pipeline inherits the API's environment, so the API must restart to see it.
    try:
        with r.tunnel() as base:
            running = [j for j in Api(base, r.token()).get("/jobs") if j["state"] == "running"]
    except (RemoteError, ApiError):
        return  # API not running; it reads the file when it starts
    if running:
        warn("jobs are running, so the API was not restarted. Jobs started now won't see the new value; "
             "run `lobbot up --restart` once they finish.")
    else:
        r.start_server(restart=True)
        ok("API restarted so new jobs see it")


def cmd_new(a) -> None:
    if a.example:
        spec = json.loads(example_text(Remote(settings(a))))
    else:
        if not a.description:
            raise CliError('describe the task, e.g. lobbot new "turn support emails into JSON tickets"')
        from lobbot.taskgen import TaskgenError, draft_taskspec

        print("Drafting a TaskSpec (this calls Gemini, or Claude if only ANTHROPIC_API_KEY is set)...")
        try:
            spec = draft_taskspec(" ".join(a.description), n_seeds=a.seeds, provider=a.provider)
        except TaskgenError as e:
            raise CliError(str(e))
    out = Path(a.out or f"{spec['task_name']}.taskspec.json")
    out.write_text(json.dumps(spec, indent=2, ensure_ascii=False) + "\n")
    ok(f"wrote {out}  ({spec['task_name']}, {len(spec['seed_examples'])} seed examples)")
    first = spec["seed_examples"][0]
    print(c("  e.g. ", "2") + first["input"][:100].replace("\n", " ") + c("  →  ", "2") + first["output"][:100].replace("\n", " "))
    print(f"Edit it if you like, then: lobbot run {out} --fast")


def cmd_run(a) -> None:
    s = settings(a)
    r = Remote(s)
    spec = load_spec(a, r)
    conf = job_config(a)
    if not a.force:
        busy = gpu_processes(r)
        if busy:
            raise CliError("the GPU is in use, so another job is probably running:\n  " + "\n  ".join(busy)
                           + "\nWait for it (lobbot status), or add --force to start anyway.")
    job_id = check_job_id(a.name) if a.name else secrets.token_hex(5)
    bundle = {"dir": s.jobs.rstrip("/") + "/" + job_id, "files": {"taskspec.json": spec, "config.json": conf}}
    r.sh("""python3 -c '
import json, os, sys
b = json.load(sys.stdin)
d = os.path.expanduser(b["dir"])
if os.path.exists(os.path.join(d, "taskspec.json")):
    sys.exit("job " + d + " already exists")
os.makedirs(d, exist_ok=True)
for name, body in b["files"].items():
    with open(os.path.join(d, name), "w") as f:
        json.dump(body, f, indent=2, ensure_ascii=False)
'""", input=json.dumps(bundle))
    r.start_server()
    with r.tunnel() as base:
        api = Api(base, r.token())
        api.post(f"/jobs/{job_id}/resume", {})  # the API launches pipeline.py on the prepared job dir
        ok(f"started job {job_id}: {spec['task_name']}, preset {a.preset}"
           + (f", overrides {json.dumps(conf)}" if conf else ""))
    if a.detach:
        print(f"Watch it with: lobbot watch {job_id}")
        return
    if watch(r, job_id) == "error":
        sys.exit(1)


def cmd_status(a) -> None:
    r, s = connect(a)
    with r.tunnel() as base:
        api = Api(base, r.token())
        if a.job:
            st = api.get(f"/jobs/{check_job_id(a.job)}")
            print(c(f"job {st['job_id']}", "1") + f"  {st['task_name']}  "
                  + c(st["state"], STATE_COLOR.get(st["state"], "0")) + c(f"  created {ago(st['created'])}", "2"))
            for line in Board(st["stages"]).lines():
                print(line)
            if st.get("error"):
                print(c("error: ", "31") + st["error"])
            return
        jobs = api.get("/jobs")
    if not jobs:
        print("No jobs yet. Start one with: lobbot run --example --fast")
    for j in jobs:
        print(f" {j['job_id']:<12} {c(j['state'], STATE_COLOR.get(j['state'], '0')):<18} "
              f"{(j['task_name'] or '')[:36]:<36} {ago(j['created'])}")
    busy = gpu_processes(r)
    print(c(f"GPU: {len(busy)} process(es) running" if busy else "GPU: idle", "2"))


def cmd_watch(a) -> None:
    r, s = connect(a)
    if watch(r, check_job_id(a.job)) == "error":
        sys.exit(1)


def cmd_logs(a) -> None:
    s = settings(a)
    r = Remote(s)
    d = rpath(s.jobs.rstrip("/") + "/" + check_job_id(a.job, loose=True))
    if a.stage:
        pick = f'f={d}/logs/{a.stage}.log'
    else:  # the API's combined log, else the newest stage log (jobs run without the API)
        pick = f'f={d}/pipeline.log; [ -f "$f" ] || f=$(ls -t {d}/logs/*.log 2>/dev/null | head -1)'
    follow = "-F" if a.follow else ""
    script = (f'{pick}\n[ -n "$f" ] && [ -f "$f" ] || {{ echo "no logs yet for job {a.job}" >&2; exit 1; }}\n'
              f'echo "==> $f" >&2\nexec tail -n {int(a.lines)} {follow} "$f"')
    if a.follow and not s.local and sys.stdout.isatty():
        sys.exit(subprocess.call(r.argv(script, tty=True)))  # a tty so Ctrl-C stops the remote tail
    sys.exit(r.stream(script))


def cmd_resume(a) -> None:
    r, s = connect(a)
    with r.tunnel() as base:
        api = Api(base, r.token())
        job = check_job_id(a.job)
        api.post(f"/jobs/{job}/resume", {"from": a.start} if a.start else {})
        ok(f"resumed {job}" + (f" from {a.start}" if a.start else " (finished stages are skipped)"))
    if not a.detach and watch(r, job) == "error":
        sys.exit(1)


def cmd_stop(a) -> None:
    s = settings(a)
    r = Remote(s)
    job = check_job_id(a.job, loose=True)
    if not a.yes and sys.stdin.isatty():
        if input(f"Stop job {job}? Its current stage is lost; finished stages stay cached. [y/N] ").lower() != "y":
            return
    out = r.sh(f'pkill -TERM -f "[p]ipeline.py --job [^ ]*/{job}( |$)" && echo stopped || echo none', check=False)
    if "stopped" in out:
        ok(f"sent stop to job {job}; it frees the GPU within ~30s. Restart with: lobbot resume {job}")
    else:
        warn(f"no running pipeline found for job {job}")


def cmd_eval(a) -> None:
    r, s = connect(a)
    with r.tunnel() as base:
        rep = Api(base, r.token()).get(f"/jobs/{check_job_id(a.job)}/eval")
    if a.json:
        print(json.dumps(rep, indent=2))
        return
    t = rep["teacher"]
    print(f"teacher {t['name']}: {t['size_gb']} GB, score {t['score']}   (scored by {rep['score_method']})")
    print(f" {'candidate':<16}{'size GB':>8}{'score':>8}{'tok/s est':>11}{'tok/s VM':>10}  target")
    for cand in rep["candidates"]:
        win = c(" ← winner", "32") if cand["name"] == rep["winner"] else ""
        print(f" {cand['name']:<16}{cand['size_gb']:>8}{cand['score']:>8}{cand['tok_s_est']:>11}"
              f"{str(cand['tok_s_vm'] or '-'):>10}  {'yes' if cand['meets_target'] else 'no'}{win}")
    for name, samples in (rep.get("samples") or {}).items():
        if name == rep["winner"] and samples:
            sm = samples[0]
            print(c("\nsample input:  ", "2") + sm["input"][:300])
            print(c("winner output: ", "2") + sm["output"][:300])
            print(c("teacher:       ", "2") + sm["reference"][:300])


def model_dir(s: cfgmod.Settings, api: Api, job: str, out: str | None) -> Path:
    if out:
        return Path(out).expanduser()
    task = api.get(f"/jobs/{job}").get("task_name") or "model"
    return Path(s.models_dir).expanduser() / f"{task}-{job}"


def do_download(a, r: Remote, s: cfgmod.Settings) -> Path:
    job = check_job_id(a.job)
    with r.tunnel() as base:
        api = Api(base, r.token())
        d = model_dir(s, api, job, a.out)
        d.mkdir(parents=True, exist_ok=True)
        dest = d / "model.gguf"
        (d / "Modelfile").write_text(api.get(f"/jobs/{job}/modelfile"))
        if dest.exists() and not a.force:
            ok(f"already downloaded: {dest}  (--force to fetch again)")
            return d
        if a.force:
            dest.unlink(missing_ok=True)
        t0, last = time.time(), [0.0]

        def progress(done: int, total: int) -> None:
            now = time.time()
            if now - last[0] < 0.25 and done != total:
                return
            last[0] = now
            rate = done / max(now - t0, 1e-6) / 1e6
            pct = 100 * done / total if total else 0
            msg = f"\r  {done / 1e9:6.2f} / {total / 1e9:.2f} GB  {pct:5.1f}%  {rate:6.1f} MB/s"
            sys.stdout.write(msg if TTY else "")
            sys.stdout.flush()

        api.download(job, dest, progress)
        if TTY:
            print()
    size = dest.stat().st_size / 1e9
    ok(f"{dest}  ({size:.2f} GB) and Modelfile")
    return d


def cmd_download(a) -> None:
    r, s = connect(a)
    d = do_download(a, r, s)
    print(f"Next: lobbot install {a.job} --chat   (or: cd {d} && ollama create mymodel -f Modelfile)")


def cmd_install(a) -> None:
    if not shutil.which("ollama"):
        raise CliError("ollama is not installed here; get it from https://ollama.com/download")
    r, s = connect(a)
    d = do_download(a, r, s)
    name = a.name or "lobbot-" + d.name.rsplit("-", 1)[0]
    print(f"Importing into Ollama as {name} ...")
    if subprocess.call(["ollama", "create", name, "-f", "Modelfile"], cwd=d) != 0:
        raise CliError("ollama create failed (is the Ollama app running?)")
    ok(f"installed. Chat with: lobbot chat {name}   (or: ollama run {name})")
    if a.chat:
        os.execvp("ollama", ["ollama", "run", name])


def cmd_chat(a) -> None:
    if not shutil.which("ollama"):
        raise CliError("ollama is not installed here; get it from https://ollama.com/download")
    os.execvp("ollama", ["ollama", "run", "--verbose", a.model, *a.prompt] if a.verbose
              else ["ollama", "run", a.model, *a.prompt])


# ----------------------------------------------------------------- argument parsing

def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="lobbot", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", help="SSH target for this command (default from lobbot init); 'local' for this machine")
    sub = p.add_subparsers(dest="cmd", required=True, metavar="command")

    def add(name, fn, help, aliases=()):
        sp = sub.add_parser(name, help=help, description=help, aliases=list(aliases))
        sp.set_defaults(fn=fn)
        return sp

    sp = add("init", cmd_init, "set where the VM is")
    sp.add_argument("--host", dest="host")
    sp.add_argument("--key", help="SSH key file")
    sp.add_argument("--repo", help="LobBot checkout on the VM")
    sp.add_argument("--jobs", help="jobs directory on the VM")

    add("doctor", cmd_doctor, "check the VM is ready to run the pipeline")

    sp = add("up", cmd_up, "start the API on the VM if needed and check it", aliases=["connect"])
    sp.add_argument("--restart", action="store_true", help="restart the API (refuses while jobs run)")
    sp.add_argument("--force", action="store_true", help="restart even with running jobs (kills them)")

    sp = add("tunnel", cmd_tunnel, "hold an SSH tunnel to the API open (for curl or the desktop app)")
    sp.add_argument("--port", type=int, help="local port (default 8700)")

    add("token", cmd_token, "print the API token")

    sp = add("secret", cmd_secret, "store an API key (e.g. GEMINI_API_KEY) in ~/.lobbot-env on the VM")
    sp.add_argument("name")
    sp.add_argument("--prompt", action="store_true", help="always ask, even if it is set locally")

    sp = add("new", cmd_new, "draft a TaskSpec from a plain-English description")
    sp.add_argument("description", nargs="*")
    sp.add_argument("-o", "--out", help="output file (default <task_name>.taskspec.json)")
    sp.add_argument("--seeds", type=int, default=12, help="seed examples to draft (3-50)")
    sp.add_argument("--provider", choices=["gemini", "anthropic"])
    sp.add_argument("--example", action="store_true", help="write the bundled support-ticket example instead")

    sp = add("run", cmd_run, "start a pipeline job on the VM and watch it")
    sp.add_argument("spec", nargs="?", help="TaskSpec JSON file")
    sp.add_argument("--example", action="store_true", help="use the bundled support-ticket TaskSpec")
    g = sp.add_mutually_exclusive_group()
    g.add_argument("--preset", choices=sorted(PRESETS), default="full")
    g.add_argument("--fast", dest="preset", action="store_const", const="fast", help="same as --preset fast")
    sp.add_argument("--set", action="append", metavar="KEY=VALUE", help="override a stages/_util.py Config field")
    sp.add_argument("--name", help="job id (letters and digits; default random)")
    sp.add_argument("--force", action="store_true", help="start even if the GPU is busy")
    sp.add_argument("-d", "--detach", action="store_true", help="start and return without watching")

    sp = add("status", cmd_status, "list jobs, or show one job's stages", aliases=["jobs", "ls"])
    sp.add_argument("job", nargs="?")

    sp = add("watch", cmd_watch, "follow a job's stage progress live")
    sp.add_argument("job")

    sp = add("logs", cmd_logs, "show a job's log (the pipeline log, or one stage's)")
    sp.add_argument("job")
    sp.add_argument("-s", "--stage", choices=STAGES)
    sp.add_argument("-n", "--lines", type=int, default=80)
    sp.add_argument("-f", "--follow", action="store_true")

    sp = add("resume", cmd_resume, "rerun a job, skipping finished stages")
    sp.add_argument("job")
    sp.add_argument("--from", dest="start", choices=STAGES, help="rerun from this stage onward")
    sp.add_argument("-d", "--detach", action="store_true")

    sp = add("stop", cmd_stop, "stop a running job (finished stages stay cached)")
    sp.add_argument("job")
    sp.add_argument("-y", "--yes", action="store_true")

    sp = add("eval", cmd_eval, "show a job's eval report")
    sp.add_argument("job")
    sp.add_argument("--json", action="store_true")

    for name, fn, help in [("download", cmd_download, "download a job's GGUF (resumable) and Modelfile"),
                           ("install", cmd_install, "download a job's model and import it into Ollama")]:
        sp = add(name, fn, help)
        sp.add_argument("job")
        sp.add_argument("-o", "--out", help="directory (default ~/lobbot-models/<task>-<job>)")
        sp.add_argument("--force", action="store_true", help="download again even if present")
        if name == "install":
            sp.add_argument("--name", help="Ollama model name (default lobbot-<task>)")
            sp.add_argument("--chat", action="store_true", help="start chatting right after")

    sp = add("chat", cmd_chat, "chat with an installed model in Ollama")
    sp.add_argument("model")
    sp.add_argument("prompt", nargs="*", help="one-shot prompt (default interactive)")
    sp.add_argument("-v", "--verbose", action="store_true", help="show tok/s after each answer")
    return p


def main(argv: list[str] | None = None) -> None:
    a = parser().parse_args(argv)
    try:
        a.fn(a)
    except (CliError, RemoteError, ApiError) as e:
        print(c("error: ", "31") + str(e), file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
