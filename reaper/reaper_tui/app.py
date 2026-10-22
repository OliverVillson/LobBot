"""`reaper`: TUI for REAP (router-weighted expert activation pruning) of MoE models.
Run it on the GPU machine itself, over SSH:

    reaper                                    open the TUI
    reaper "qwen3.8-27B-A3B" "Debugging only"  find the model, prune it for that purpose, show progress
    reaper --demo                             simulate a B200 (no GPU needed)

    F2 / /setup              model, share of experts to prune, calibration set
    F3 / /market             pick a model from the MoE marketplace
    F5 / /reap               start the pipeline on this machine
    /model <name or hf id>   e.g. /model qwen3 next, /model org/any-moe-model
    /prune <percent>         share of routed experts to drop per layer (default 50)
    /purpose <text>          what the pruned model is for; becomes its calibration set
    /calib <set or path>     without a purpose: mix, chat, code, math, or a .jsonl/.txt of prompts
    /hw                      detect the hardware again and show the plan
    /stop · /logs · /quit    stop the run · last log lines · leave (a run keeps going)

The run is detached from this terminal: close the TUI or lose SSH and it keeps
going; `reaper` again picks it back up.
"""

from __future__ import annotations

import argparse
import secrets
import socket
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

from rich.console import Group
from rich.table import Table
from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.theme import Theme
from textual.widgets import Button, DataTable, Input, Label, RichLog, Select, Static

from reaper_tui import backend, mock
from reaper_tui.art import BANNER, REAPER, REAPER_LARGE
from reaper_tui.backend import Hardware, Plan, Run
from reaper_tui.catalog import MARKET, MoeModel, custom_model, find_model

VERSION = "v0.2.0 (2026.10.22)"
CALIB_SETS = ["mix", "chat", "code", "math"]

MONO = Theme(
    name="mono",
    primary="#ffffff",
    secondary="#ffffff",
    accent="#ffffff",
    foreground="#ffffff",
    background="#000000",
    surface="#000000",
    panel="#000000",
    boost="#000000",
    success="#ffffff",
    warning="#ffffff",
    error="#ffffff",
    dark=True,
)


@dataclass
class State:
    demo: bool = False
    hw: Hardware | None = None
    plan: Plan | None = None
    model: MoeModel | None = None
    ratio: float = 0.5
    purpose: str = ""
    calib: str = "mix"
    samples: int = 512
    stage_names: list[str] = field(default_factory=lambda: list(backend.DEFAULT_STAGES))
    stages: dict[str, dict] = field(default_factory=dict)
    run: Run | None = None
    running: bool = False
    started: float = 0.0
    result: str = ""
    tail: deque = field(default_factory=lambda: deque(maxlen=200))

    def settings(self) -> dict:
        return {"model": self.model.id, "purpose": self.purpose, "prune": self.ratio, "calib": self.calib,
                "calib_samples": self.samples}


def bar(pct: float, width: int = 12) -> str:
    n = round(width * pct / 100)
    return "█" * n + "░" * (width - n)


def row(label: str, value: str | Text, mark: str = "") -> Text:
    t = Text()
    t.append(f"{label}: ", style="bold")
    t.append(value if isinstance(value, Text) else Text(value))
    if mark:
        t.append(f"  {mark}", style="bold")
    return t


def parse_pct(text: str) -> float | None:
    t = text.strip().rstrip("%")
    try:
        v = float(t)
    except ValueError:
        return None
    v = v / 100 if v > 1 else v
    return v if 0.05 <= v <= 0.9 else None


class SetupScreen(ModalScreen[dict | None]):
    """Run settings. There are no keys or providers: REAPer runs on this machine."""

    BINDINGS = [Binding("escape", "dismiss(None)", "cancel")]

    def __init__(self, state: State):
        super().__init__()
        self.state = state

    def compose(self) -> ComposeResult:
        s = self.state
        calib_opts = [(c, c) for c in CALIB_SETS]
        custom = s.calib not in CALIB_SETS
        with Vertical(id="dialog") as d:
            d.border_title = "Run settings"
            yield Label("Model (Hugging Face id or local path; any MoE)", classes="h")
            yield Input(s.model.id if s.model else "", placeholder="Qwen/Qwen3-30B-A3B-Instruct-2507", id="model")
            yield Label("Purpose (what the pruned model is for; empty = general)", classes="h")
            yield Input(s.purpose, placeholder="Debugging only", id="purpose")
            yield Label("Experts to prune per layer (%)", classes="h")
            yield Input(f"{s.ratio * 100:g}", id="prune")
            yield Label("Calibration prompts when there is no purpose", classes="h")
            yield Select(calib_opts + [("my own file…", "file")], value="file" if custom else s.calib,
                         allow_blank=False, id="calib")
            yield Input(s.calib if custom else "", placeholder="/path/to/prompts.jsonl", id="calib_file")
            yield Label("Calibration samples", classes="h")
            yield Input(str(s.samples), id="samples")
            yield Static("", id="check")
            with Horizontal(id="buttons"):
                yield Button("Save", id="save")
                yield Button("Cancel", id="cancel")

    @on(Button.Pressed, "#cancel")
    def cancel(self) -> None:
        self.dismiss(None)

    @on(Button.Pressed, "#save")
    @on(Input.Submitted)
    def save(self) -> None:
        out = self.query_one("#check", Static)
        model = self.query_one("#model", Input).value.strip()
        ratio = parse_pct(self.query_one("#prune", Input).value)
        calib = str(self.query_one("#calib", Select).value)
        if calib == "file":
            calib = self.query_one("#calib_file", Input).value.strip()
        samples = self.query_one("#samples", Input).value.strip()
        problems = []
        if not model:
            problems.append("choose a model (or F3 for the marketplace)")
        if ratio is None:
            problems.append("prune must be between 5 and 90 %")
        if not calib:
            problems.append("give the path of your prompts file")
        if not samples.isdigit() or not 16 <= int(samples) <= 100_000:
            problems.append("samples must be a number from 16 to 100000")
        if problems:
            out.update("✗ " + "\n✗ ".join(problems))
            return
        self.dismiss({"model": model, "purpose": self.query_one("#purpose", Input).value.strip(), "prune": ratio, "calib": calib, "samples": int(samples)})


class MarketScreen(ModalScreen[MoeModel | None]):
    BINDINGS = [Binding("escape", "dismiss(None)", "close")]

    def __init__(self, hw: Hardware | None):
        super().__init__()
        self.hw = hw

    def compose(self) -> ComposeResult:
        with Vertical(id="market") as v:
            v.border_title = "MoE Marketplace"
            v.border_subtitle = "enter: choose · esc: close · any other MoE: /model <hf id>"
            yield DataTable(cursor_type="row", zebra_stripes=False)

    def on_mount(self) -> None:
        t = self.query_one(DataTable)
        t.add_columns("Model", "Org", "Size", "Experts", "Released", "Fits here", "Repo")
        for m in MARKET:
            fit = backend.fits(self.hw, m.total_b) if self.hw else "?"
            t.add_row(m.name, m.org, m.size, f"{m.experts} top-{m.top_k}", m.released, fit, m.id, key=m.id)
        t.focus()

    @on(DataTable.RowSelected)
    def chosen(self, e: DataTable.RowSelected) -> None:
        self.dismiss(next(m for m in MARKET if m.id == e.row_key.value))


class Reaper(App):
    TITLE = "REAPer"
    CSS = """
    Screen { background: #000000; color: #ffffff; }
    #banner { color: #ffffff; text-style: bold; padding: 0 1; height: auto; }
    #main { border: round #ffffff; border-title-align: right; border-title-style: bold;
            border-subtitle-align: left; height: auto; padding: 0 1; }
    #left { width: 52; height: auto; }
    #art { height: auto; }
    #who { height: auto; }
    #right { width: 1fr; height: auto; padding-left: 2; }
    #right Static { height: auto; margin-bottom: 1; }
    #log { height: 1fr; min-height: 3; background: #000000; padding: 0 1; scrollbar-size: 0 0; }
    #status { height: 1; padding: 0 1; background: #ffffff; color: #000000; text-style: bold; }
    #prompt { border: none; height: 1; padding: 0 1; background: #000000; color: #ffffff; }
    #prompt:focus { border: none; background: #000000; }
    #promptrow { height: 3; border-top: solid #ffffff; border-bottom: solid #ffffff; }
    #caret { width: 3; padding-left: 1; text-style: bold; }

    ModalScreen { background: #000000 80%; align: center middle; }
    #dialog { width: 70; height: auto; border: round #ffffff; border-title-style: bold;
              padding: 1 2; background: #000000; }
    #dialog .h { text-style: bold; margin-top: 1; }
    #dialog Input, #dialog Select { margin-top: 1; }
    #check { height: auto; margin-top: 1; }
    #buttons { height: auto; margin-top: 1; align-horizontal: right; }
    #buttons Button { margin-left: 2; min-width: 16; }
    Button { background: #000000; color: #ffffff; border: tall #ffffff; }
    Button:hover, Button:focus { background: #ffffff; color: #000000; text-style: bold; }
    Input { background: #000000; border: tall #ffffff 50%; }
    Input:focus { border: tall #ffffff; }
    Select > SelectCurrent { background: #000000; border: tall #ffffff 50%; }
    Select:focus > SelectCurrent { border: tall #ffffff; }
    #market { width: 120; height: auto; max-height: 90%; border: round #ffffff;
              border-title-style: bold; background: #000000; padding: 0 1; }
    DataTable { background: #000000; height: auto; }
    DataTable > .datatable--header { background: #000000; color: #ffffff; text-style: bold underline; }
    DataTable > .datatable--cursor { background: #ffffff; color: #000000; text-style: bold; }
    """
    BINDINGS = [
        Binding("f2", "setup", "setup"),
        Binding("f3", "market", "market"),
        Binding("f5", "reap", "reap"),
        Binding("ctrl+c", "quit", "quit", priority=True),
    ]

    def __init__(self, demo: bool = False, model: str | None = None, purpose: str = "",
                 prune: float | None = None, samples: int | None = None):
        super().__init__()
        self.state = State(demo=demo, purpose=purpose)
        if prune is not None:
            self.state.ratio = prune
        if samples is not None:
            self.state.samples = samples
        self.launch_query = model  # `reaper "<model>" "<purpose>"`: resolve, then start
        self.launch_ready = {"hw": demo, "model": False}
        self.session = datetime.now().strftime("%Y%m%d_%H%M%S_") + secrets.token_hex(3)

    def compose(self) -> ComposeResult:
        yield Static(BANNER, id="banner")
        with Horizontal(id="main") as main:
            main.border_title = f"REAPer {VERSION}"
            with Vertical(id="left"):
                yield Static(REAPER, id="art")
                yield Static(id="who")
            with Vertical(id="right"):
                yield Static(id="machine")
                yield Static(id="setup")
                yield Static(id="pipeline")
                yield Static(id="marketplace")
        yield RichLog(id="log", markup=False, wrap=True)
        yield Static(id="status")
        with Horizontal(id="promptrow"):
            yield Static("❯", id="caret")
            yield Input(placeholder="/model <name> · /purpose <text> · F5 reap · /help", id="prompt")

    def on_mount(self) -> None:
        self.register_theme(MONO)
        self.theme = "mono"
        self.fit_art()
        s = self.state
        if s.demo:
            s.hw = mock.DEMO_HW
            self.say("Welcome to REAPer! Demo mode: a simulated B200, nothing runs. /help for commands.")
        else:
            s.stage_names = backend.pipeline_stages()
            self.say("Welcome to REAPer! Detecting this machine… /setup to choose a model, /help for commands.")
            if not self.launch_query:
                self.reattach()
            self.detect()
        if self.launch_query:
            self.say(f"looking up “{self.launch_query}”…")
            self.resolve(self.launch_query)
        self.refresh_panels()
        self.set_interval(1, self.refresh_status)
        self.set_interval(0.5, self.poll)
        self.query_one("#prompt", Input).focus()

    # hardware ----------------------------------------------------------------

    @work(thread=True, exclusive=True, group="hw")
    def detect(self) -> None:
        hw = backend.detect_hardware()
        self.call_from_thread(self._detected, hw)

    def _detected(self, hw: Hardware) -> None:
        self.state.hw = hw
        if hw.gpus:
            self.say(f"✓ {hw.gpu_line()} · {hw.host_line()}")
        else:
            self.say("✗ no NVIDIA GPU found here. Run REAPer on the GPU VM (ssh -t user@vm reaper), "
                     "or `reaper --demo` to look around.")
        self.replan()
        self.launch_ready["hw"] = True
        self.maybe_launch()

    @work(thread=True, exclusive=True, group="resolve")
    def resolve(self, query: str) -> None:
        if self.state.demo:
            m = find_model(query)
            rid, others = (m.id if m else None), []
        else:
            rid, others = backend.resolve_model(query)
        self.call_from_thread(self._resolved, query, rid, others)

    def _resolved(self, query: str, rid: str | None, others: list[str]) -> None:
        if rid is None:
            if self.state.demo:
                rid = query
            else:
                self.launch_query = None
                self.say(f"✗ no model found for “{query}”. Try a fuller name or the exact id (org/model), "
                         "or F3 for the marketplace.", bold=True)
                return
        self.choose(find_model(rid) or custom_model(rid))
        if others:
            self.say("  other matches: " + ", ".join(others[:4]) + "  (/model <id> to switch)")
        self.launch_ready["model"] = True
        self.maybe_launch()

    def maybe_launch(self) -> None:
        if self.launch_query and all(self.launch_ready.values()):
            self.launch_query = None
            self.action_reap()

    @work(thread=True, exclusive=True, group="plan")
    def replan(self) -> None:
        s = self.state
        if s.demo:
            plan = mock.demo_plan(s.model)
        elif s.hw and s.model:
            plan = backend.plan_for(s.hw, s.model.id, s.ratio)
        else:
            plan = None
        self.call_from_thread(self._planned, plan)

    def _planned(self, plan: Plan | None) -> None:
        self.state.plan = plan
        self.refresh_panels()

    # rendering ---------------------------------------------------------------

    def on_resize(self) -> None:
        self.fit_art()

    def fit_art(self) -> None:
        large = self.size.height >= 66 and self.size.width >= 150
        self.query_one("#art", Static).update(REAPER_LARGE if large else REAPER)
        self.query_one("#left").styles.width = 64 if large else 52

    def say(self, msg: str, bold: bool = False) -> None:
        self.query_one("#log", RichLog).write(Text(msg, style="bold" if bold else ""))

    def refresh_panels(self) -> None:
        s = self.state
        who = Text(justify="center")
        who.append("REAPer", style="bold")
        who.append(" · expert pruning for MoE\n")
        who.append(f"{'demo' if s.demo else socket.gethostname()}:{backend.reaper_home()}/jobs\n")
        who.append(f"Session: {self.session}")
        self.query_one("#who", Static).update(who)

        mach = Text()
        mach.append("Machine\n", style="bold underline")
        if s.hw is None:
            mach.append_text(row("gpu", "detecting…"))
        else:
            mach.append_text(row("gpu", s.hw.gpu_line(), "✓" if s.hw.gpus else "✗"))
            mach.append("\n")
            mach.append_text(row("host", s.hw.host_line()))
        if s.plan:
            for n in s.plan.notes[:4]:
                mach.append(f"\n↳ {n}", style="dim")
            for p in s.plan.problems[:3]:
                mach.append(f"\n✗ {p}", style="bold")
        elif s.hw and s.model and not s.demo:
            mach.append("\n↳ hardware planner not installed yet; the pipeline adapts on its own", style="dim")
        self.query_one("#machine", Static).update(mach)

        setup = Text()
        setup.append("Run\n", style="bold underline")
        if s.model:
            m = s.model
            shape = f"{m.size}, {m.experts} experts top-{m.top_k}" if m.total_b else "shape read at plan stage"
            setup.append_text(row("model", f"{m.name}  {shape}", "✓"))
        else:
            setup.append_text(row("model", "not chosen  (F3 marketplace or /model)", "○"))
        setup.append("\n")
        keep = f" → {round(s.model.experts * (1 - s.ratio))} experts / layer" if s.model and s.model.experts else ""
        setup.append_text(row("prune", f"{s.ratio:.0%} of routed experts{keep}"))
        setup.append("\n")
        if s.purpose:
            setup.append_text(row("purpose", s.purpose))
            setup.append("\n")
        calib = "built from the purpose" if s.purpose else s.calib
        setup.append_text(row("calibration", f"{calib} · {s.samples} samples · teacher on this GPU"))
        self.query_one("#setup", Static).update(setup)

        pipe = Text()
        pipe.append("Pipeline", style="bold underline")
        if s.run and not s.demo:
            pipe.append(f"  {s.run.root.name}", style="dim")
        width = max(9, max(len(n) for n in s.stage_names) + 1)
        for name in s.stage_names:
            ev = s.stages.get(name)
            pct = float(ev.get("pct") or 0) if ev else 0
            status = ev["status"] if ev else "waiting"
            if status in ("done", "skipped"):
                pct = 100
            mark = {"done": "✓", "skipped": "✓", "running": "▸", "error": "✗"}.get(status, " ")
            pipe.append(f"\n{mark} {name:<{width}}", style="bold" if status in ("running", "error") else "")
            pipe.append(f"{bar(pct)} {pct:>3.0f}%  ")
            msg = (ev or {}).get("msg") or ("cached" if status == "skipped" else "")
            pipe.append(msg[:60], style="" if status in ("running", "error") else "dim")
        if s.result:
            pipe.append(f"\n☠ {s.result}", style="bold")
        self.query_one("#pipeline", Static).update(pipe)

        mk = Table.grid(padding=(0, 2))
        for _ in range(5):
            mk.add_column()
        for m in MARKET[:5]:
            chosen = s.model is not None and s.model.id == m.id
            fit = backend.fits(s.hw, m.total_b) if s.hw else ""
            mk.add_row("●" if chosen else "·", Text(m.name, style="bold" if chosen else ""),
                       m.size, f"{m.experts}x{m.top_k}", fit)
        head = Text("Marketplace", style="bold underline")
        foot = Text(f"(and {len(MARKET) - 5} more… F3 · any MoE: /model <hf id>)", style="dim")
        self.query_one("#marketplace", Static).update(Group(head, mk, foot))

        main = self.query_one("#main")
        main.border_subtitle = ((" demo · " if s.demo else " ") +
                                "F2 settings · F3 market · F5 reap · /help ")
        self.refresh_status()

    def refresh_status(self) -> None:
        s = self.state
        done = sum(1 for n in s.stage_names if s.stages.get(n, {}).get("status") in ("done", "skipped"))
        cur = next((s.stages[n] for n in s.stage_names if s.stages.get(n, {}).get("status") == "running"), None)
        pct = 100 * (done + (float(cur.get("pct") or 0) / 100 if cur else 0)) / max(len(s.stage_names), 1)
        el = int(time.time() - s.started) if s.started and s.running else 0
        model = s.model.id.split("/")[-1].lower() if s.model else "no model"
        gpu = s.hw.gpu_line() if s.hw else "detecting"
        stage = cur["stage"] if cur else ("running" if s.running else "idle")
        self.query_one("#status", Static).update(
            f" ☠ {model} │ {gpu} │ {stage} │ [{bar(pct, 10)}] {pct:.0f}% │ {el // 60}m{el % 60:02d}s ")

    # runs --------------------------------------------------------------------

    def reattach(self) -> None:
        run = Run.latest()
        if run is None:
            return
        s = self.state
        s.run = run
        cfg = run.config
        if cfg.get("model"):
            s.model = find_model(cfg["model"]) or custom_model(cfg["model"])
            s.ratio = float(cfg.get("prune", s.ratio))
            s.calib = cfg.get("calib", s.calib)
            s.purpose = cfg.get("purpose", s.purpose) or ""
            s.samples = int(cfg.get("calib_samples", s.samples))
        for ev, line in run.replay():
            self.apply(ev, line, quiet=True)
        s.running = run.running()
        if s.running:
            s.started = (run.root / "pid").stat().st_mtime
            self.say(f"↻ reattached to the running job {run.root.name}", bold=True)
        else:
            self.say(f"last job: {run.root.name} ({self.last_status()})")

    def last_status(self) -> str:
        s = self.state
        if s.result:
            return "done"
        err = next((e for e in s.stages.values() if e["status"] == "error"), None)
        return f"failed in {err['stage']}" if err else "stopped"

    def apply(self, ev: dict | None, line: str, quiet: bool = False) -> None:
        s = self.state
        if ev is None:
            s.tail.append(line)
            if not quiet and ("Traceback" in line or "Error" in line):
                self.say(line[:300])
            return
        stage = ev["stage"]
        if stage == "pipeline":
            if ev["status"] == "done":
                s.result = str(ev.get("model") or ev.get("msg") or "done")
                if not quiet:
                    self.say(f"☠ done: {s.result}", bold=True)
            elif ev["status"] == "running" and str(ev.get("msg", "")).startswith("started"):
                s.stages, s.result = {}, ""
            elif ev["status"] == "error" and not quiet:
                self.say(f"✗ {ev.get('msg', 'pipeline failed')}", bold=True)
            return
        if stage not in s.stage_names:
            s.stage_names.append(stage)
        prev = s.stages.get(stage, {})
        s.stages[stage] = {**prev, **ev} if ev["status"] == "running" else ev
        if not quiet and ev["status"] in ("done", "error", "skipped"):
            mark = {"done": "✓", "skipped": "✓", "error": "✗"}[ev["status"]]
            self.say(f"{mark} {stage}{': ' + ev['msg'] if ev.get('msg') else ''}",
                     bold=ev["status"] == "error")

    def poll(self) -> None:
        s = self.state
        if s.demo or s.run is None:
            return
        lines = s.run.read_new()
        for ev, line in lines:
            self.apply(ev, line)
        was = s.running
        s.running = s.run.running()
        if was and not s.running:
            for ev, line in s.run.read_new():
                self.apply(ev, line)
            if not s.result:
                self.say(f"■ the run ended: {self.last_status()}. /logs shows the last output.", bold=True)
        if lines or was != s.running:
            self.refresh_panels()

    # actions -----------------------------------------------------------------

    def action_setup(self) -> None:
        def done(cfg: dict | None) -> None:
            if not cfg:
                return
            s = self.state
            s.model = find_model(cfg["model"]) or custom_model(cfg["model"])
            s.ratio, s.calib, s.samples, s.purpose = cfg["prune"], cfg["calib"], cfg["samples"], cfg["purpose"]
            self.say(f"✓ {s.model.id} · prune {s.ratio:.0%} · {s.purpose or s.calib} × {s.samples}")
            self.replan()
            self.refresh_panels()
        self.push_screen(SetupScreen(self.state), done)

    def action_market(self) -> None:
        def done(m: MoeModel | None) -> None:
            if m:
                self.choose(m)
        self.push_screen(MarketScreen(self.state.hw), done)

    def choose(self, m: MoeModel) -> None:
        self.state.model = m
        shape = f"{m.size}, {m.experts} experts top-{m.top_k}" if m.total_b else "not in the marketplace; shape read at plan stage"
        fit = f", here: {backend.fits(self.state.hw, m.total_b)}" if self.state.hw and m.total_b else ""
        self.say(f"✓ model: {m.id}  ({shape}{fit})")
        self.replan()
        self.refresh_panels()

    def action_reap(self, force: bool = False) -> None:
        s = self.state
        if s.running:
            self.say("already reaping; /stop first")
            return
        if s.model is None:
            self.say("✗ choose a model first: F3 or /model <name or hf id>")
            return
        if s.demo:
            self.demo_reap()
            return
        if s.hw is not None and not s.hw.gpus and not force:
            self.say("✗ no GPU on this machine. Run REAPer on the GPU VM, or /reap --force to try anyway.")
            return
        if s.plan and not s.plan.ok and not force:
            self.say("✗ the plan says this machine can't run it: " + "; ".join(s.plan.problems) +
                     ". /reap --force to try anyway.")
            return
        try:
            s.run = Run.start(s.settings())
        except Exception as e:
            self.say(f"✗ {e}")
            return
        s.stages, s.result, s.running, s.started = {}, "", True, time.time()
        why = f"for “{s.purpose}”" if s.purpose else f"calibration {s.calib}"
        self.say(f"☠ reaping {s.model.id} {why}: {s.ratio:.0%} of experts, {s.samples} samples", bold=True)
        self.say(f"  job {s.run.root} · keeps running if you close this or SSH drops", )
        self.refresh_panels()

    @work(exclusive=True, group="reap")
    async def demo_reap(self) -> None:
        s = self.state
        s.running, s.stages, s.result, s.started = True, {}, "", time.time()
        why = f" for “{s.purpose}”" if s.purpose else ""
        self.say(f"☠ reaping {s.model.name}{why}: {s.ratio:.0%} of experts on {s.hw.gpu_line()} (demo)", bold=True)
        async for ev in mock.run_pipeline(s.model, s.ratio):
            self.apply(ev, "")
            self.refresh_panels()
        s.running = False
        self.refresh_panels()

    def stop(self) -> None:
        s = self.state
        if s.run and s.run.stop():
            self.say("■ stop sent; the current stage frees the GPU and exits")
        else:
            self.say("nothing is running")

    # commands ----------------------------------------------------------------

    @on(Input.Submitted, "#prompt")
    def command(self, e: Input.Submitted) -> None:
        line = e.value.strip()
        e.input.value = ""
        if not line:
            return
        self.say(f"❯ {line}")
        cmd, *args = line.split()
        s = self.state
        match cmd.lower():
            case "/help":
                for h in __doc__.splitlines():
                    if h.startswith("    "):
                        self.say(h)
            case "/setup" | "/settings":
                self.action_setup()
            case "/market":
                self.action_market()
            case "/reap":
                self.action_reap(force="--force" in args)
            case "/stop":
                self.stop()
            case "/hw":
                self.detect()
            case "/logs":
                for l in list(s.tail)[-20:] or ["(no log lines yet)"]:
                    self.say(l[:300])
            case "/model" if args:
                q = " ".join(args)
                m = find_model(q)
                if m is None and ("/" in q or q.startswith((".", "~"))):
                    m = custom_model(q)
                if m:
                    self.choose(m)
                else:
                    self.say("✗ no single match; F3 for the marketplace, or give a full hf id like org/model")
            case "/prune" if args and parse_pct(args[0]) is not None:
                s.ratio = parse_pct(args[0])
                self.say(f"✓ prune {s.ratio:.0%} of experts")
                self.replan()
                self.refresh_panels()
            case "/purpose":
                s.purpose = " ".join(args).strip().strip('"')
                self.say(f"✓ purpose: {s.purpose}" if s.purpose else "✓ no purpose: general calibration")
                self.refresh_panels()
            case "/calib" if args:
                s.calib = args[0] if args[0] in CALIB_SETS else " ".join(args)
                self.say(f"✓ calibration: {s.calib}")
                self.refresh_panels()
            case "/samples" if args and args[0].isdigit():
                s.samples = int(args[0])
                self.refresh_panels()
            case "/quit" | "/exit":
                self.exit()
            case _:
                self.say("✗ unknown command or missing arguments; /help")


def main() -> None:
    ap = argparse.ArgumentParser(prog="reaper", description="REAP expert pruning for MoE models, on this machine.")
    ap.add_argument("model", nargs="?", help='model to prune, fuzzy is fine: "qwen3-30b-a3b" or org/name')
    ap.add_argument("purpose", nargs="?", default="", help='what the pruned model is for: "Debugging only"')
    ap.add_argument("--prune", type=parse_pct, help="share of experts to drop, e.g. 50 or 0.5 (default 50)")
    ap.add_argument("--samples", type=int, help="calibration samples (default 512)")
    ap.add_argument("--demo", action="store_true", help="simulate a B200 and a run (no GPU needed)")
    a = ap.parse_args()
    Reaper(demo=a.demo, model=a.model, purpose=a.purpose, prune=a.prune, samples=a.samples).run()


if __name__ == "__main__":
    main()
