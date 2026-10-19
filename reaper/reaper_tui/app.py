"""`reaper`: TUI for REAP (router-weighted expert activation pruning) of MoE models.

    F2 / /setup                       inference key, compute provider, GPU
    F3 / /market                      pick a model from the MoE marketplace
    F5 / /reap                        run the pipeline on the chosen model
    /key <provider> <key>             e.g. /key nebius sk-...
    /compute <provider> <token> [n]   e.g. /compute evroc tok-... 2  (n = GPU preset)
    /model <name>                     e.g. /model qwen3 next
    /prune <percent>                  share of experts to drop (default 50)

First draft: every provider call is mocked (reaper_tui/mock.py) and keys only
live in memory for the session.
"""

from __future__ import annotations

import secrets
import time
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

from reaper_tui import mock
from reaper_tui.art import BANNER, REAPER, REAPER_LARGE
from reaper_tui.catalog import COMPUTE, GPUS, INFERENCE, MARKET, MoeModel, Provider, find_model, find_provider

VERSION = "v0.1.0 (2026.10.19)"

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
    inference: Provider | None = None
    inference_key: str = ""
    compute: Provider | None = None
    compute_key: str = ""
    gpu: str = GPUS[0]
    model: MoeModel | None = None
    ratio: float = 0.5
    stages: dict[str, dict] = field(default_factory=dict)
    running: bool = False
    started: float = 0.0


def mask(key: str) -> str:
    return "••••" + key[-4:] if key else "not set"


def bar(pct: int, width: int = 12) -> str:
    n = round(width * pct / 100)
    return "█" * n + "░" * (width - n)


def row(label: str, value: str | Text, mark: str = "") -> Text:
    t = Text()
    t.append(f"{label}: ", style="bold")
    t.append(value if isinstance(value, Text) else Text(value))
    if mark:
        t.append(f"  {mark}", style="bold")
    return t


class SetupScreen(ModalScreen[bool]):
    BINDINGS = [Binding("escape", "dismiss(False)", "cancel")]

    def __init__(self, state: State):
        super().__init__()
        self.state = state

    def compose(self) -> ComposeResult:
        s = self.state
        with Vertical(id="dialog") as d:
            d.border_title = "Setup"
            yield Label("Inference provider (calibration data + judge)", classes="h")
            yield Select([(p.name, p.id) for p in INFERENCE], value=(s.inference or INFERENCE[0]).id,
                         allow_blank=False, id="inf")
            yield Input(s.inference_key, placeholder="API key", password=True, id="inf_key")
            yield Label("Compute provider (GPU VM that runs the prune)", classes="h")
            yield Select([(p.name, p.id) for p in COMPUTE], value=(s.compute or COMPUTE[0]).id,
                         allow_blank=False, id="cmp")
            yield Input(s.compute_key, placeholder="API token", password=True, id="cmp_key")
            yield Select([(g, g) for g in GPUS], value=s.gpu, allow_blank=False, id="gpu")
            yield Static("", id="check")
            with Horizontal(id="buttons"):
                yield Button("Check & save", id="save")
                yield Button("Cancel", id="cancel")

    @on(Button.Pressed, "#cancel")
    def cancel(self) -> None:
        self.dismiss(False)

    @on(Button.Pressed, "#save")
    def save(self) -> None:
        self.check()

    @work(exclusive=True)
    async def check(self) -> None:
        out = self.query_one("#check", Static)
        inf = next(p for p in INFERENCE if p.id == self.query_one("#inf", Select).value)
        cmp = next(p for p in COMPUTE if p.id == self.query_one("#cmp", Select).value)
        gpu = str(self.query_one("#gpu", Select).value)
        ik = self.query_one("#inf_key", Input).value.strip()
        ck = self.query_one("#cmp_key", Input).value.strip()
        out.update("checking keys…")
        ok1, m1 = await mock.check_inference_key(inf.id, ik)
        ok2, m2 = await mock.check_compute(cmp.id, ck, gpu)
        out.update(f"{'✓' if ok1 else '✗'} {m1}\n{'✓' if ok2 else '✗'} {m2}")
        if ok1 and ok2:
            s = self.state
            s.inference, s.inference_key, s.compute, s.compute_key, s.gpu = inf, ik, cmp, ck, gpu
            self.dismiss(True)


class MarketScreen(ModalScreen[MoeModel | None]):
    BINDINGS = [Binding("escape", "dismiss(None)", "close")]

    def compose(self) -> ComposeResult:
        with Vertical(id="market") as v:
            v.border_title = "MoE Marketplace"
            v.border_subtitle = "enter: choose · esc: close · ✓ = REAPer adapter ready"
            yield DataTable(cursor_type="row", zebra_stripes=False)

    def on_mount(self) -> None:
        t = self.query_one(DataTable)
        t.add_columns(" ", "Model", "Org", "Size", "Experts", "Released", "Needs", "Repo")
        for m in MARKET:
            t.add_row("✓" if m.supported else "·", m.name, m.org, m.size, f"{m.experts} top-{m.top_k}",
                      m.released, m.gpus, m.id, key=m.id)
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
    #left { width: 48; height: auto; }
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

    def __init__(self):
        super().__init__()
        self.state = State()
        self.session = datetime.now().strftime("%Y%m%d_%H%M%S_") + secrets.token_hex(3)

    def compose(self) -> ComposeResult:
        yield Static(BANNER, id="banner")
        with Horizontal(id="main") as main:
            main.border_title = f"REAPer {VERSION}"
            with Vertical(id="left"):
                yield Static(REAPER, id="art")
                yield Static(id="who")
            with Vertical(id="right"):
                yield Static(id="setup")
                yield Static(id="pipeline")
                yield Static(id="marketplace")
        yield RichLog(id="log", markup=False, wrap=True)
        yield Static(id="status")
        with Horizontal(id="promptrow"):
            yield Static("❯", id="caret")
            yield Input(placeholder="/setup to begin · /help for commands", id="prompt")

    def on_mount(self) -> None:
        self.register_theme(MONO)
        self.theme = "mono"
        self.fit_art()
        self.say("Welcome to REAPer! Type /setup to begin or /help for commands. (provider calls are mocked)")
        self.refresh_panels()
        self.set_interval(1, self.refresh_status)
        self.query_one("#prompt", Input).focus()

    # rendering ---------------------------------------------------------------

    def on_resize(self) -> None:
        self.fit_art()

    def fit_art(self) -> None:
        large = self.size.height >= 64 and self.size.width >= 140
        self.query_one("#art", Static).update(REAPER_LARGE if large else REAPER)
        self.query_one("#left").styles.width = 58 if large else 48

    def say(self, msg: str, bold: bool = False) -> None:
        self.query_one("#log", RichLog).write(Text(msg, style="bold" if bold else ""))

    def refresh_panels(self) -> None:
        s = self.state
        who = Text(justify="center")
        who.append("REAPer", style="bold")
        who.append(" · expert pruning for MoE\n")
        who.append("~/.reaper/jobs\n")
        who.append(f"Session: {self.session}")
        self.query_one("#who", Static).update(who)

        setup = Text()
        setup.append("Setup\n", style="bold underline")
        setup.append_text(row("inference", f"{s.inference.name}  {mask(s.inference_key)}" if s.inference
                              else "not set  (F2)", "✓" if s.inference else "○"))
        setup.append("\n")
        setup.append_text(row("compute", f"{s.compute.name}  {s.gpu}" if s.compute else "not set  (F2)",
                              "✓" if s.compute else "○"))
        setup.append("\n")
        if s.model:
            m = s.model
            setup.append_text(row("model", f"{m.name}  {m.size}, {m.experts} experts top-{m.top_k}", "✓"))
        else:
            setup.append_text(row("model", "not chosen  (F3 marketplace)", "○"))
        setup.append("\n")
        keep = f" → {round(s.model.experts * (1 - s.ratio))} experts / layer" if s.model else ""
        setup.append_text(row("prune", f"{s.ratio:.0%} of experts{keep}"))
        self.query_one("#setup", Static).update(setup)

        pipe = Text()
        pipe.append("Pipeline\n", style="bold underline")
        for i, (name, what) in enumerate(mock.STAGES):
            ev = s.stages.get(name)
            pct = ev["pct"] if ev else 0
            status = ev["status"] if ev else "waiting"
            mark = {"done": "✓", "running": "▸", "error": "✗"}.get(status, " ")
            pipe.append(f"{mark} {name:<10}", style="bold" if status == "running" else "")
            pipe.append(f"{bar(pct)} {pct:>3}%  ")
            pipe.append(ev["msg"] if ev and ev["msg"] else what, style="" if status == "running" else "dim")
            if i < len(mock.STAGES) - 1:
                pipe.append("\n")
        self.query_one("#pipeline", Static).update(pipe)

        mk = Table.grid(padding=(0, 2))
        for _ in range(5):
            mk.add_column()
        for m in MARKET[:6]:
            chosen = s.model is not None and s.model.id == m.id
            mk.add_row("●" if chosen else ("✓" if m.supported else "·"), Text(m.name, style="bold" if chosen else ""),
                       m.size, f"{m.experts}x{m.top_k}", m.released)
        head = Text()
        head.append("Marketplace", style="bold underline")
        foot = Text(f"(and {len(MARKET) - 6} more… F3 / /market)", style="dim")
        self.query_one("#marketplace", Static).update(Group(head, mk, foot))

        main = self.query_one("#main")
        ready = sum(x is not None for x in (s.inference, s.compute, s.model))
        main.border_subtitle = (f"Profile: draft · {ready}/3 configured · {len(MARKET)} models · "
                                "F2 setup · F3 market · F5 reap · /help")
        self.refresh_status()

    def refresh_status(self) -> None:
        s = self.state
        done = sum(1 for e in s.stages.values() if e["status"] == "done")
        pct = round(100 * done / len(mock.STAGES))
        el = int(time.monotonic() - s.started) if s.started else 0
        model = s.model.id.split("/")[-1].lower() if s.model else "no model"
        inf = s.inference.id if s.inference else "no key"
        cmp = s.compute.id if s.compute else "no vm"
        self.query_one("#status", Static).update(
            f" ☠ {model} │ {inf} │ {cmp} │ [{bar(pct, 10)}] {pct}% │ {el}s ")

    # actions -----------------------------------------------------------------

    def action_setup(self) -> None:
        def done(ok: bool | None) -> None:
            if ok:
                self.say(f"✓ setup saved: {self.state.inference.name} + {self.state.compute.name} ({self.state.gpu})")
            self.refresh_panels()
        self.push_screen(SetupScreen(self.state), done)

    def action_market(self) -> None:
        def done(m: MoeModel | None) -> None:
            if m:
                self.choose(m)
        self.push_screen(MarketScreen(), done)

    def choose(self, m: MoeModel) -> None:
        self.state.model = m
        self.say(f"✓ model: {m.id}  ({m.size}, {m.experts} experts top-{m.top_k}, needs {m.gpus})")
        if not m.supported:
            self.say("  no REAPer router adapter for this one yet; the run is mocked anyway")
        self.refresh_panels()

    def action_reap(self) -> None:
        s = self.state
        missing = [n for n, v in (("inference key", s.inference), ("compute provider", s.compute),
                                  ("model", s.model)) if v is None]
        if missing:
            self.say("✗ can't reap yet, missing: " + ", ".join(missing))
            return
        if s.running:
            self.say("already reaping")
            return
        self.reap()

    @work(exclusive=True, group="reap")
    async def reap(self) -> None:
        s = self.state
        s.running, s.stages, s.started = True, {}, time.monotonic()
        self.say(f"☠ reaping {s.model.name}: {s.ratio:.0%} of experts on {s.compute.name} {s.gpu}", bold=True)
        async for ev in mock.run_pipeline(s.model, s.ratio):
            s.stages[ev["stage"]] = ev
            if ev["status"] == "done":
                self.say(f"✓ {ev['stage']}")
            self.refresh_panels()
        s.running = False
        keep = round(s.model.experts * (1 - s.ratio))
        self.say(f"☠ done: ~/.reaper/jobs/{s.model.id.split('/')[-1]}-REAP-{keep}e (mock)", bold=True)

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
                for h in __doc__.splitlines()[2:10]:
                    self.say(h)
            case "/setup":
                self.action_setup()
            case "/market":
                self.action_market()
            case "/reap":
                self.action_reap()
            case "/key" if len(args) == 2:
                p = find_provider(args[0], INFERENCE)
                if not p:
                    self.say("✗ inference providers: " + ", ".join(x.id for x in INFERENCE))
                    return
                self.check_key(p, args[1])
            case "/compute" if len(args) >= 2:
                p = find_provider(args[0], COMPUTE)
                if not p:
                    self.say("✗ compute providers: " + ", ".join(x.id for x in COMPUTE))
                    return
                gpu = GPUS[int(args[2]) - 1] if len(args) > 2 and args[2].isdigit() and 0 < int(args[2]) <= len(GPUS) else s.gpu
                self.check_compute(p, args[1], gpu)
            case "/model" if args:
                m = find_model(" ".join(args))
                if m:
                    self.choose(m)
                else:
                    self.say("✗ no single match; F3 opens the marketplace")
            case "/prune" if args and args[0].rstrip("%").isdigit() and 5 <= int(args[0].rstrip("%")) <= 90:
                s.ratio = int(args[0].rstrip("%")) / 100
                self.say(f"✓ prune {s.ratio:.0%} of experts")
                self.refresh_panels()
            case "/quit" | "/exit":
                self.exit()
            case _:
                self.say("✗ unknown command or missing arguments; /help")

    @work
    async def check_key(self, p: Provider, key: str) -> None:
        ok, msg = await mock.check_inference_key(p.id, key)
        self.say(("✓ " if ok else "✗ ") + msg)
        if ok:
            self.state.inference, self.state.inference_key = p, key
            self.refresh_panels()

    @work
    async def check_compute(self, p: Provider, key: str, gpu: str) -> None:
        ok, msg = await mock.check_compute(p.id, key, gpu)
        self.say(("✓ " if ok else "✗ ") + msg)
        if ok:
            self.state.compute, self.state.compute_key, self.state.gpu = p, key, gpu
            self.refresh_panels()


def main() -> None:
    Reaper().run()


if __name__ == "__main__":
    main()
