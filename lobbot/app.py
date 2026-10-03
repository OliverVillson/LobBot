"""`lobbot`: Oliver's developer CLI for running the compression pipeline on the GPU VM.

    lobbot init                      where the VM is (saved in ~/.config/lobbot/config.json)
    lobbot doctor                    check SSH, checkout, venvs, weights, GPU, API
    lobbot up                        start the API on the VM (if needed) and check it
    lobbot new "what it should do"   draft a TaskSpec with Gemini
    lobbot run spec.json --fast      start a job and watch its stages live (--long for code)
    lobbot status [job]              all jobs, or one job's stages
    lobbot results <job>             scores vs the teacher, size, speed, examples
    lobbot watch / logs / resume / stop <job>
    lobbot save <job> --chat         back it up on the VM, download it (resumable,
                                     sha256-checked), import into Ollama, chat
    lobbot pull                      update the VM checkout

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
             "heal_max_minutes": 20, "dense_fallback": False},
    "full": {},
}
# --long: room for long answers (code, documents). Without it the teacher's answers
# are cut at 1536 tokens and dropped as truncated, and heal trains on 2048 tokens.
LONG: dict[str, int] = {"data_answer_max_tokens": 4096, "data_max_len": 16384, "heal_max_len": 8192}
LONG_SEED_TOKENS = 800  # `new` and `run` suggest --long when a seed answer is longer (chars/4)


GITHUB_URL = "https://github.com/OliverVillson/LobBot"


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
STATE_COLOR = {"done": "32", "running": "36", "error": "31", "queued": "2", "partial": "33", "stopped": "33"}


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

    def footer(self) -> str:
        """Elapsed time and the stage being worked on."""
        if not self.started:
            return c("  waiting for the first stage...", "2")
        t0 = min(self.started.values())
        end = max(self.ended.values()) if all(r["status"] in ("done", "skipped", "error") for r in self.stages.values()) \
            and self.ended else time.time()
        cur = next((st for st, r in self.stages.items() if r["status"] == "running"), None)
        return c(f"  elapsed {dur(end - t0)}" + (f" · now: {cur}" if cur else ""), "2")

    def draw(self) -> None:
        if not TTY:
            return
        if self.drawn:
            sys.stdout.write(f"\x1b[{self.drawn}F")
        ls = self.lines() + [self.footer()]
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
                        if not e:
                            board.draw()  # keepalive: refresh the elapsed clock
                            continue
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
        if "model" in final and not final["model"]:
            ok(f"pipeline run finished: {final.get('msg', '')}")
            return "done"
        ok("pipeline finished")
        print()
        show_results(r, job_id)
        return "done"
    if final["status"] == "stopped":
        warn(f"job {job_id} was stopped; finished stages stay cached. Continue with: lobbot resume {job_id}")
        return "stopped"
    print(c("✗ pipeline failed: ", "31") + str(final.get("msg", "")))
    print(f"  logs: lobbot logs {job_id}    retry: lobbot resume {job_id}")
    return "error"


def gpu_processes(r: Remote) -> list[str]:
    out = r.sh("command -v nvidia-smi >/dev/null || exit 0; "
               "nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader", check=False)
    return [l.strip() for l in out.splitlines() if l.strip()]


def disk_state(r: Remote, ids: list[str]) -> dict[str, dict]:
    """What the job dirs say, for jobs the API has no events for (run directly with pipeline.py)."""
    if not ids:
        return {}
    jobs = rpath(r.s.jobs.rstrip("/"))
    script = "".join(
        f'd={jobs}/{i}; done=$(ls "$d/.done" 2>/dev/null | tr "\\n" ,); run=0; model=0\n'
        f'pgrep -f "[p]ipeline.py --job [^ ]*/{i}( |$)" >/dev/null && run=1\n'
        f'[ -f "$d/.done/package" ] && [ -f "$d/out/model.gguf" ] && model=1\n'
        f'echo "{i}|$done|$run|$model"\n' for i in ids if i.replace("-", "").replace("_", "").isalnum())
    out = {}
    for line in r.sh(script, check=False).splitlines():
        i, done, run, model = (line.split("|") + ["", "", "", ""])[:4]
        out[i] = {"done": [x for x in done.split(",") if x in STAGES], "running": run == "1", "model": model == "1"}
    return out


def disk_label(d: dict) -> str:
    if d["running"]:
        return "running*"
    if len(d["done"]) == len(STAGES):
        return "done*"
    if d["done"]:
        last = [st for st in STAGES if st in d["done"]][-1]
        return f"stopped*:{last}"
    return "queued"


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
    if getattr(a, "long", False):
        out.update(LONG)  # checked against the VM's Config when the job is written
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


def longest_seed_tokens(spec: dict) -> int:
    """Rough token count (chars/4) of the longest seed answer."""
    return max((len(e.get("output", "")) // 4 for e in spec.get("seed_examples", [])), default=0)


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
    home = str(Path.home())
    for name in ("repo", "jobs"):  # an unquoted ~/x is expanded by the local shell to this Mac's home
        v = getattr(s, name)
        if v == home or v.startswith(home + "/"):
            setattr(s, name, "~" + v[len(home):])
    p = cfgmod.save(s)
    ok(f"saved {p}")
    for name in ("host", "ssh_key", "repo", "jobs"):
        print(f"  {name:<8} {getattr(s, name) or '(ssh default)'}")
    print("Next: lobbot doctor")


def cmd_doctor(a) -> None:
    s = settings(a)
    r = Remote(s)
    print(f"VM: {s.host}   repo: {s.repo}   jobs: {s.jobs}")
    script = f"""
echo "ssh=ok $(hostname)"
if cd {rpath(s.repo)} 2>/dev/null; then
  echo "repo=ok $(git log -1 --format='%h %s' 2>/dev/null | cut -c1-70) [$(git rev-parse --abbrev-ref HEAD 2>/dev/null)]"
  timeout 15 git fetch -q origin 2>/dev/null
  behind=$(git rev-list --count HEAD..@{{u}} 2>/dev/null || echo "?")
  dirty=$(git status --porcelain --untracked-files=no 2>/dev/null | grep -c . || true)
  if [ "$behind" = "?" ]; then echo "git=warn no upstream branch to compare with"
  elif [ "$behind" != 0 ] || [ "$dirty" != 0 ]; then echo "git=warn $behind commit(s) behind $(git rev-parse --abbrev-ref @{{u}}), $dirty changed file(s) (lobbot pull updates it)"
  else echo "git=ok up to date with $(git rev-parse --abbrev-ref @{{u}})"; fi
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
    print(c("  note: " + PAUSE_WARNING, "2"))
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


def draft_on_vm(r: Remote, desc: str, seeds: int, provider: str | None) -> dict:
    """Run lobbot.taskgen on the VM with its ~/.lobbot-env keys; the key never leaves the VM."""
    from common.taskspec import TaskSpec

    print(f"No API key on this machine, so drafting on the VM ({r.s.host}) with the key in ~/.lobbot-env...")
    args = f"{shlex.quote(desc)} --seeds {int(seeds)}" + (f" --provider {shlex.quote(provider)}" if provider else "")
    try:
        out = r.sh(r.env_prelude() + f"exec python3 -m lobbot.taskgen {args}", timeout=360)
    except RemoteError as e:
        msg = str(e)
        if "No module named lobbot.taskgen" in msg:
            msg += "\nThe VM checkout is too old for this; update it with: lobbot pull"
        elif "no LLM API key" in msg or "needs" in msg:
            msg += "\nAdd a key on the VM with: lobbot secret GEMINI_API_KEY"
        raise CliError("drafting on the VM failed: " + msg)
    try:
        spec = json.loads(out[out.index("{"):])
        TaskSpec.from_dict(spec)
    except (ValueError, KeyError, TypeError) as e:
        raise CliError(f"the VM returned something that is not a valid TaskSpec: {e}")
    return spec


def cmd_new(a) -> None:
    if a.example:
        spec = json.loads(example_text(Remote(settings(a))))
    else:
        if not a.description:
            raise CliError('describe the task, e.g. lobbot new "turn support emails into JSON tickets"')
        from lobbot.taskgen import TaskgenError, _resolve_provider, draft_taskspec

        desc = " ".join(a.description)
        try:
            local = not a.on_vm and bool(_resolve_provider(a.provider))
        except TaskgenError:
            local = False  # no key on this Mac: use the one in ~/.lobbot-env on the VM
        if local:
            print("Drafting a TaskSpec with your local API key (Gemini, or Claude if only ANTHROPIC_API_KEY is set)...")
            try:
                spec = draft_taskspec(desc, n_seeds=a.seeds, provider=a.provider)
            except TaskgenError as e:
                raise CliError(str(e))
        else:
            spec = draft_on_vm(Remote(settings(a)), desc, a.seeds, a.provider)
    out = Path(a.out or f"{spec['task_name']}.taskspec.json")
    out.write_text(json.dumps(spec, indent=2, ensure_ascii=False) + "\n")
    ok(f"wrote {out}  ({spec['task_name']}, {len(spec['seed_examples'])} seed examples)")
    first = spec["seed_examples"][0]
    print(c("  e.g. ", "2") + first["input"][:100].replace("\n", " ") + c("  →  ", "2") + first["output"][:100].replace("\n", " "))
    longest = longest_seed_tokens(spec)
    if longest > LONG_SEED_TOKENS:
        print(f"Its answers are long (up to ~{longest} tokens), so --long gives the teacher and heal room for them.")
        print(f"Edit it if you like, then: lobbot run {out} --fast --long")
    else:
        print(f"Edit it if you like, then: lobbot run {out} --fast")


def describe_config(conf: dict) -> str:
    """Config overrides in plain words, e.g. '400 examples, 30 tests, heal at most 20 min'."""
    words = {"n_generate": "{} training examples", "n_heldout": "{} held-out tests",
             "reap_calib_samples": "{} calibration samples", "heal_max_minutes": "heal at most {} min",
             "heal_epochs": "{} heal epoch(s)", "reap_sparsity": "{:.0%} of experts pruned",
             "data_answer_max_tokens": "answers up to {} tokens", "data_max_len": "{}-token data context",
             "heal_max_len": "heal on up to {} tokens"}
    out = []
    for k, v in conf.items():
        if k == "dense_fallback":
            out.append("with the dense fallback" if v else "no dense fallback")
        elif k in words:
            out.append(words[k].format(v))
        else:
            out.append(f"{k}={v}")
    return ", ".join(out)


def cmd_run(a) -> None:
    s = settings(a)
    r = Remote(s)
    spec = load_spec(a, r)
    conf = job_config(a)
    if "data_answer_max_tokens" in conf or "data_max_len" in conf:
        # Defaults as in stages/data.py when only one of the two is set (env vars on the VM aside).
        cap, ctx = int(conf.get("data_answer_max_tokens") or 1536), int(conf.get("data_max_len") or 8192)
        if ctx < cap + 2048:
            warn(f"data_max_len={ctx} leaves under 2048 tokens of prompt room above data_answer_max_tokens={cap}; "
                 f"use data_max_len={cap + 2048} or more")
    longest = longest_seed_tokens(spec)
    if longest > LONG_SEED_TOKENS and "data_answer_max_tokens" not in conf:
        warn(f"seed answers run to ~{longest} tokens; without --long, teacher answers over 1536 tokens "
             "are dropped as truncated")
    if not a.force:
        busy = gpu_processes(r)
        if busy:
            raise CliError("the GPU is in use, so another job is probably running:\n  " + "\n  ".join(busy)
                           + "\nWait for it (lobbot status), or add --force to start anyway.")
    job_id = check_job_id(a.name) if a.name else secrets.token_hex(5)
    bundle = {"dir": s.jobs.rstrip("/") + "/" + job_id, "files": {"taskspec.json": spec, "config.json": conf}}
    r.sh(r.env_prelude() + """python3 -c '
import json, os, sys
from dataclasses import fields
from stages._util import Config
b = json.load(sys.stdin)
unknown = sorted(set(b["files"]["config.json"]) - {f.name for f in fields(Config)})
if unknown:
    sys.exit("the VM checkout does not know config key(s) " + ", ".join(unknown)
             + "; update it with: lobbot pull (once no job is running)")
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
        try:
            api.post(f"/jobs/{job_id}/resume", {})  # the API launches pipeline.py on the prepared job dir
        except ApiError as e:
            if e.status != 409:
                raise
            raise CliError(f"{e.msg}\nJob {job_id} is set up but not started; start it once the GPU is free with: "
                           f"lobbot resume {job_id}")
        ok(f"started job {c(job_id, '1')} · {spec['task_name']} · preset {a.preset}"
           + (f" ({describe_config(conf)})" if conf else " (default settings)"))
    if a.detach:
        print(f"Watch it with: lobbot watch {job_id}")
        return
    state = watch(r, job_id)
    if state == "error":
        sys.exit(1)
    if state == "done" and a.save:
        cmd_save(argparse.Namespace(host=a.host, job=job_id, out=None, force=False, name=None, chat=False,
                                    no_ollama=False, no_backup=False))


def cmd_status(a) -> None:
    r, s = connect(a)
    with r.tunnel() as base:
        api = Api(base, r.token())
        if a.job:
            st = api.get(f"/jobs/{check_job_id(a.job)}")
        else:
            jobs = api.get("/jobs")
    direct = "  * run directly with pipeline.py; read from the job dir's .done markers"
    if a.job:
        state, stages = st["state"], st["stages"]
        if state == "queued":  # no API events: fall back to the job dir
            d = disk_state(r, [st["job_id"]]).get(st["job_id"])
            if d and (d["done"] or d["running"]):
                state = disk_label(d)
                for name in d["done"]:
                    stages[name] = {"status": "done", "pct": 100, "msg": "done (.done marker)"}
                if d["running"]:
                    nxt = next((x for x in STAGES if x not in d["done"]), None)
                    if nxt:
                        stages[nxt] = {"status": "running", "pct": 0, "msg": f"see: lobbot logs {st['job_id']}"}
        color = STATE_COLOR.get(state.rstrip("*").split("*")[0], "0")
        print(c(f"job {st['job_id']}", "1") + f"  {st['task_name']}  " + c(state, color)
              + c(f"  created {ago(st['created'])}", "2"))
        for line in Board(stages).lines():
            print(line)
        if st.get("error"):
            print(c("error: ", "31") + st["error"])
        if "*" in state:
            print(c(direct, "2"))
        if state.startswith("done"):
            print()
            show_results(r, st["job_id"])
        return
    if not jobs:
        print("No jobs yet. Start one with: lobbot run --example --fast")
    disk = disk_state(r, [j["job_id"] for j in jobs if j["state"] == "queued"])
    starred = False
    for j in jobs:
        state = disk_label(disk[j["job_id"]]) if j["job_id"] in disk else j["state"]
        starred |= "*" in state
        color = STATE_COLOR.get(state.split("*")[0], "0")
        print(f" {j['job_id']:<12} {c(f'{state:<16}', color)} {(j['task_name'] or '')[:36]:<36} {ago(j['created'])}")
    if starred:
        print(c(direct, "2"))
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
        try:
            api.post(f"/jobs/{job}/resume", {"from": a.start} if a.start else {})
        except ApiError as e:
            if e.status == 409:
                raise CliError(e.msg)
            raise
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
    try:
        r.start_server()
        with r.tunnel() as base:
            Api(base, r.token()).post(f"/jobs/{job}/stop", {}, timeout=90)  # returns once the GPU is free
        ok(f"stopped job {job}; finished stages stay cached. Continue with: lobbot resume {job}")
        return
    except ApiError as e:
        if e.status == 409:
            warn(f"no running pipeline found for job {job}")
            return
        if e.status not in (404, 405):
            raise
    # Older VM checkout without POST /stop, or a job dir the API does not know: signal it directly.
    out = r.sh(f'pkill -TERM -f "[p]ipeline.py --job [^ ]*/{job}( |$)" && echo stopped || echo none', check=False)
    if "stopped" in out:
        ok(f"sent stop to job {job}; it frees the GPU within ~30s. Restart with: lobbot resume {job}")
    else:
        warn(f"no running pipeline found for job {job}")


def show_results(r: Remote, job: str, facts: dict | None = None) -> None:
    """Print the results report and what to do next."""
    from lobbot.report import job_facts, render

    try:
        facts = facts or job_facts(r, job)
    except (RemoteError, ValueError) as e:
        warn(f"could not read the results for job {job}: {e}")
        return
    for line in render(job, facts, color=TTY):
        print(line)
    if not (facts.get("eval") and facts.get("gguf_bytes")):
        return
    saved = local_copy(r.s, job, facts)
    if saved:
        d, name, in_ollama = saved
        print(c("  Saved ", "1") + f"{d}" + c("  (sha256 matches the VM)", "2"))
        if in_ollama:
            print(c("  Next  ", "1") + f"lobbot chat {name}" + c(f"   already in Ollama as {name}", "2"))
        else:
            print(c("  Next  ", "1") + f"lobbot save {job} --chat" + c("   imports the saved copy into Ollama", "2"))
        return
    print(c("  Next  ", "1") + f"lobbot save {job} --chat" + c("   download, verify, import into Ollama, chat", "2"))
    print(c("        " + PAUSE_WARNING.split(". ")[0] + ".", "2"))


def local_copy(s: cfgmod.Settings, job: str, facts: dict) -> tuple[Path, str, bool] | None:
    """(dir, Ollama name, already in Ollama) if this Mac has a verified copy of the job's model."""
    task = (facts.get("taskspec") or {}).get("task_name") or "model"
    d = Path(s.models_dir).expanduser() / f"{task}-{job}"
    try:
        meta = json.loads((d / "lobbot.json").read_text())
        local_sha = (d / "model.gguf.sha256").read_text().split()[0]
    except (OSError, ValueError, IndexError):
        return None
    if not (d / "model.gguf").exists() or local_sha != meta.get("sha256"):
        return None
    if facts.get("gguf_sha256") and facts["gguf_sha256"] != local_sha:
        return None  # the VM has a newer model (e.g. a rerun) than the saved one
    name = meta.get("ollama_name") or f"lobbot-{task}"
    in_ollama = False
    if shutil.which("ollama"):
        try:
            listed = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=5).stdout
            in_ollama = any(l.split()[0].split(":")[0] == name for l in listed.splitlines()[1:] if l.strip())
        except (subprocess.SubprocessError, OSError, IndexError):
            pass
    return d, name, in_ollama


def cmd_results(a) -> None:
    s = settings(a)
    r = Remote(s)
    job = check_job_id(a.job, loose=True)
    from lobbot.report import job_facts

    facts = job_facts(r, job)
    if a.json:
        print(json.dumps(facts, indent=2))
        return
    if not facts.get("eval"):
        raise CliError(f"job {job} has no eval report yet; see: lobbot status {job}")
    show_results(r, job, facts)


def model_dir(s: cfgmod.Settings, api: Api, job: str, out: str | None) -> Path:
    if out:
        return Path(out).expanduser()
    task = api.get(f"/jobs/{job}").get("task_name") or "model"
    return Path(s.models_dir).expanduser() / f"{task}-{job}"


PAUSE_WARNING = ("Pausing or stopping the VM wipes /mnt/nvme, including every job and model on it. "
                 "Keep models with: lobbot save <job>")


def sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


def remote_sha256(r: Remote, job: str) -> str:
    """sha256 of the job's GGUF on the VM, cached next to it in out/model.gguf.sha256."""
    f = rpath(r.s.jobs.rstrip("/") + f"/{job}/out/model.gguf")
    out = r.sh(f'f={f}; h="$f.sha256"\n[ -f "$f" ] || {{ echo "no model.gguf for job {job}" >&2; exit 1; }}\n'
               'if [ -s "$h" ] && [ "$h" -nt "$f" ]; then cat "$h"; else sha256sum "$f" | cut -d" " -f1 | tee "$h"; fi',
               timeout=600)
    return out.strip().split()[0]


def require_packaged(r: Remote, api: Api, job: str) -> dict:
    """The job's API state, or an error if it has no packaged model. The job dir
    (.done/package plus out/model.gguf) wins over the API, which has no state for
    jobs run directly with pipeline.py."""
    st = api.get(f"/jobs/{job}")
    if st["stages"]["package"]["status"] in ("done", "skipped"):
        return st
    if disk_state(r, [job]).get(job, {}).get("model"):
        return st
    raise CliError(f"job {job} has no packaged model yet (no .done/package and out/model.gguf on the VM); "
                   f"see: lobbot status {job}")


def do_download(a, r: Remote, s: cfgmod.Settings) -> Path:
    """Download the job's GGUF, Modelfile and eval report, and verify the GGUF's sha256."""
    job = check_job_id(a.job)
    with r.tunnel() as base:
        api = Api(base, r.token())
        require_packaged(r, api, job)
        d = model_dir(s, api, job, a.out)
        d.mkdir(parents=True, exist_ok=True)
        dest = d / "model.gguf"
        try:
            (d / "Modelfile").write_text(api.get(f"/jobs/{job}/modelfile"))
            (d / "eval.json").write_text(json.dumps(api.get(f"/jobs/{job}/eval"), indent=2))
        except ApiError as e:
            if e.status != 404:
                raise
        if a.force:
            dest.unlink(missing_ok=True)
        if dest.exists():
            print(f"Already downloaded: {dest}")
        else:
            t0, last = time.time(), [0.0]

            def progress(done: int, total: int) -> None:
                now = time.time()
                if now - last[0] < 0.25 and done != total:
                    return
                last[0] = now
                rate = done / max(now - t0, 1e-6) / 1e6
                pct = 100 * done / total if total else 0
                if TTY:
                    sys.stdout.write(f"\r  {done / 1e9:6.2f} / {total / 1e9:.2f} GB  {pct:5.1f}%  {rate:6.1f} MB/s")
                    sys.stdout.flush()

            try:
                api.download(job, dest, progress)
            except ApiError as e:
                if e.status == 404:
                    raise CliError(f"job {job} has no model.gguf on the VM: {e}")
                raise
            if TTY:
                print()
    print("Verifying sha256 ...")
    want, got = remote_sha256(r, job), sha256_file(dest)
    if want != got:
        bad = dest.with_name("model.gguf.bad")
        dest.replace(bad)
        raise CliError(f"checksum mismatch (VM {want[:12]}, local {got[:12]}); moved the file to {bad}. "
                       f"Run lobbot save {job} again to re-download.")
    (d / "model.gguf.sha256").write_text(f"{got}  model.gguf\n")
    ok(f"{dest}  ({dest.stat().st_size / 1e9:.2f} GB, sha256 {got[:12]} verified)")
    return d


def backup_on_vm(r: Remote, job: str, name: str) -> None:
    """Copy the job's final artifacts to ~/lobbot-saved/<name> on the VM's system disk,
    which survives a pause. The GGUF is copied only if that small disk has room for it."""
    s = r.s
    src = rpath(s.jobs.rstrip("/") + "/" + job)
    out = r.sh(f"""src={src}; dst="$HOME/lobbot-saved/{name}"; mkdir -p "$dst"
for f in taskspec.json config.json out/eval.json out/Modelfile out/model.gguf.sha256; do
  [ -f "$src/$f" ] && cp "$src/$f" "$dst/"
done
size=$(stat -c %s "$src/out/model.gguf")
free=$(df --output=avail -B1 "$dst" | tail -1)
if [ -f "$dst/model.gguf" ] && [ "$(stat -c %s "$dst/model.gguf")" = "$size" ]; then echo "gguf=present $dst"
elif [ $((free - size)) -gt $((2 * 1024 * 1024 * 1024)) ]; then cp "$src/out/model.gguf" "$dst/model.gguf.tmp" && mv "$dst/model.gguf.tmp" "$dst/model.gguf" && echo "gguf=copied $dst"
else echo "gguf=skipped $dst $((size / 1000000)) $((free / 1000000))"; fi""", timeout=900).strip()
    kind, _, rest = out.splitlines()[-1].partition("=")
    parts = rest.split()
    if parts[0] in ("present", "copied"):
        ok(f"VM backup: {parts[1]} (system disk, survives a pause)")
    else:
        ok(f"VM backup: report, spec and Modelfile in {parts[1]}")
        warn(f"the model ({int(parts[2]) / 1000:.1f} GB) does not fit on the VM's system disk "
             f"({int(parts[3]) / 1000:.1f} GB free), so it was not backed up there. Its only VM copy is on "
             "/mnt/nvme, which a pause or stop wipes. The download to this Mac is the real save.")


def cmd_download(a) -> None:
    r, s = connect(a)
    d = do_download(a, r, s)
    print(f"Next: lobbot save {a.job} --chat   (or: cd {d} && ollama create mymodel -f Modelfile)")


def cmd_save(a) -> None:
    """Keep a finished model: VM backup, verified download to this Mac, Ollama import."""
    r, s = connect(a)
    job = check_job_id(a.job)
    if not a.no_backup:
        with r.tunnel() as base:
            task = require_packaged(r, Api(base, r.token()), job).get("task_name") or "model"
        backup_on_vm(r, job, f"{task}-{job}")
    d = do_download(a, r, s)
    name = a.name or "lobbot-" + d.name.rsplit("-", 1)[0]
    manifest = {"job": job, "host": s.host, "saved": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "sha256": (d / "model.gguf.sha256").read_text().split()[0], "ollama_name": name}
    if (d / "eval.json").exists():
        rep = json.loads((d / "eval.json").read_text())
        manifest["winner"] = next((c for c in rep["candidates"] if c["name"] == rep["winner"]), None)
    (d / "lobbot.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if a.no_ollama:
        print(f"Saved in {d}")
        return
    if not shutil.which("ollama"):
        warn(f"ollama is not installed, so the model is saved in {d} but not imported. "
             "Get it from https://ollama.com/download, then run this again.")
        return
    print(f"Importing into Ollama as {name} ...")
    if subprocess.call(["ollama", "create", name, "-f", "Modelfile"], cwd=d) != 0:
        raise CliError("ollama create failed (is the Ollama app running?)")
    ok(f"saved in {d} and imported into Ollama. Chat with: lobbot chat {name}")
    if a.chat:
        os.execvp("ollama", ["ollama", "run", name])


def pull_script(r: Remote, repo: str) -> str:
    return r.sh(f"""set -e
export GIT_TERMINAL_PROMPT=0
cd {repo}
before=$(git rev-parse HEAD)
git fetch -q origin
git pull -q --ff-only
[ "$before" = "$(git rev-parse HEAD)" ] && echo "moved=no" || echo "moved=yes"
echo "head=$(git log -1 --format='%h %s')"
git diff --quiet "$before" HEAD -- agent/ common/ || echo "api=changed" """, timeout=180)


def cmd_pull(a) -> None:
    """Update the VM checkout (git pull --ff-only)."""
    s = settings(a)
    r = Remote(s)
    if not a.force and gpu_processes(r):
        raise CliError("a job is using the GPU; its next stage would pick up the new code. "
                       "Wait for it, or add --force.")
    repo = rpath(s.repo)
    url = r.sh(f"cd {repo} && git remote get-url origin", check=False).strip()
    if a.set_origin:
        r.sh(f"cd {repo} && git remote set-url origin {shlex.quote(a.set_origin)}")
        ok(f"VM checkout origin: {url or '(none)'} -> {a.set_origin}")
        url = a.set_origin
    elif "github.com" not in url:
        raise CliError(f"the VM checkout's origin is {url or '(none)'}, not GitHub, so pulling can't bring new code.\n"
                       f"Point it at GitHub with: lobbot pull --set-origin {GITHUB_URL}")
    dirty = [l for l in r.sh(f"cd {repo} && git status --porcelain --untracked-files=no").splitlines() if l.strip()]
    if dirty:
        files = "\n  ".join(l[3:] for l in dirty)
        if not a.stash:
            raise CliError(f"the VM checkout has local changes, so git pull would refuse:\n  {files}\n"
                           "If they are hotfixes that are now upstream, run: lobbot pull --stash "
                           "(stashes them with git stash, so nothing is lost).")
        if not a.yes and sys.stdin.isatty():
            if input(f"Stash these changes on the VM and pull?\n  {files}\n[y/N] ").lower() != "y":
                return
        label = "lobbot pull " + time.strftime("%Y-%m-%d %H:%M:%S")
        r.sh(f"cd {repo} && git stash push -q -m {shlex.quote(label)}")
        ok(f'stashed {len(dirty)} file(s) on the VM as "{label}" (get them back with: git stash pop)')
    try:
        out = pull_script(r, repo)
    except RemoteError as e:
        if "Username" in str(e) or "Authentication" in str(e) or "terminal prompts" in str(e):
            raise CliError(f"the VM can't log in to {url}: {e}\nThe VM needs read access to the repo "
                           "(for example a deploy key or a token in the URL).")
        raise
    info = dict(l.split("=", 1) for l in out.splitlines() if "=" in l)
    if info.get("moved") == "no":
        ok(f"VM checkout already up to date with {url} at {info.get('head', '?')}")
    else:
        ok(f"VM checkout updated to {info.get('head', '?')}")
    if "api" in info:
        print("The API code changed; restart it when no job is running: lobbot up --restart")


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
    sp.add_argument("--on-vm", action="store_true",
                    help="draft on the VM with its ~/.lobbot-env key even if this machine has one")

    sp = add("run", cmd_run, "start a pipeline job on the VM and watch it")
    sp.add_argument("spec", nargs="?", help="TaskSpec JSON file")
    sp.add_argument("--example", action="store_true", help="use the bundled support-ticket TaskSpec")
    g = sp.add_mutually_exclusive_group()
    g.add_argument("--preset", choices=sorted(PRESETS), default="full")
    g.add_argument("--fast", dest="preset", action="store_const", const="fast", help="same as --preset fast")
    sp.add_argument("--long", action="store_true",
                    help="room for long answers such as code: " + ", ".join(f"{k}={v}" for k, v in LONG.items()))
    sp.add_argument("--set", action="append", metavar="KEY=VALUE",
                    help="override a stages/_util.py Config field (wins over --fast and --long)")
    sp.add_argument("--name", help="job id (letters and digits; default random)")
    sp.add_argument("--force", action="store_true", help="start even if the GPU is busy")
    sp.add_argument("-d", "--detach", action="store_true", help="start and return without watching")
    sp.add_argument("--save", action="store_true", help="when it finishes, save the model (lobbot save)")

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

    sp = add("results", cmd_results, "show a finished job's results: scores, speed, size, examples",
             aliases=["eval", "report"])
    sp.add_argument("job")
    sp.add_argument("--json", action="store_true", help="raw eval report and job facts")

    sp = add("download", cmd_download, "download a job's GGUF (resumable, sha256-checked) and Modelfile")
    sp.add_argument("job")
    sp.add_argument("-o", "--out", help="directory (default ~/lobbot-models/<task>-<job>)")
    sp.add_argument("--force", action="store_true", help="download again even if present")

    sp = add("save", cmd_save, "keep a finished model: VM backup, verified download, Ollama import",
             aliases=["install"])
    sp.add_argument("job")
    sp.add_argument("-o", "--out", help="directory (default ~/lobbot-models/<task>-<job>)")
    sp.add_argument("--force", action="store_true", help="download again even if present")
    sp.add_argument("--name", help="Ollama model name (default lobbot-<task>)")
    sp.add_argument("--chat", action="store_true", help="start chatting right after")
    sp.add_argument("--no-ollama", action="store_true", help="only download")
    sp.add_argument("--no-backup", action="store_true", help="skip the copy to ~/lobbot-saved on the VM")

    sp = add("pull", cmd_pull, "update the LobBot checkout on the VM (git pull --ff-only)")
    sp.add_argument("--force", action="store_true", help="pull even while the GPU is busy")
    sp.add_argument("--stash", action="store_true", help="git stash local changes on the VM first")
    sp.add_argument("--set-origin", nargs="?", const=GITHUB_URL, metavar="URL",
                    help=f"re-point the VM checkout's origin first (default {GITHUB_URL})")
    sp.add_argument("-y", "--yes", action="store_true", help="don't ask before stashing")

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
