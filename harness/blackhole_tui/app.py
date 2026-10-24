"""`blackhole`: pull a big open model past the event horizon and get back one that
fits your own machine. Run it on the GPU VM, over SSH:

    blackhole                                 open the TUI (reattaches to a running job)
    blackhole "qwen3 coder" --schema pocket-coder --target rtx4090
    blackhole --demo                          simulated B200 and pipeline, no GPU needed

    F3 / /model              pick a source model
    F4 / /schema             pick a schema from the marketplace, or describe your own
    F5 / /run                start the job on this machine
    F6 / /logs               full log view
    /model <name, hf id, dir> any model: e.g. /model glm air, /model org/any-model, /model ~/models/mine
    /schema <id>             e.g. /schema laptop-chat
    /target <machine>        what it must run on: rtx3060 rtx4070 rtx4090 rtx5090 mac16 mac32 mac64 cpu32
    /purpose <text>          what the model is for; steers calibration and synthetic data
    /describe <idea>         a frontier model drafts a schema from your words
    /stop · /quit            stop the job · leave (a job keeps running)

Jobs are detached from this terminal: close the TUI or lose SSH and they keep
going; `blackhole` again picks them back up.
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
from textual.screen import ModalScreen, Screen
from textual.widgets import DataTable, Input, Label, RichLog, Static

from blackhole_tui import architect, detect, hardware, theme
from blackhole_tui.art import HOLE, HOLE_LARGE, banner
from blackhole_tui.catalog import (MODELS, SCHEMAS, TARGETS, Model, Schema, Target, find_model, find_schema,
                                   find_target, fit_mark, model_from_config, out_gb)
from blackhole_tui.hardware import Hardware
from blackhole_tui.jobs import Run, home, tilde
from blackhole_tui.theme import FLARE, CRIMSON, DIM, ASH, EMBER, REDSHIFT, PHOTON

VERSION = "v0.1.0 (2026.10.23)"


@dataclass
class State:
    demo: bool = False
    hw: Hardware | None = None
    model: Model | None = None
    schema: Schema | None = None
    target: Target = TARGETS[2]
    purpose: str = ""
    detected: detect.Detected | None = None
    schemas: list[Schema] = field(default_factory=lambda: list(SCHEMAS))
    stage_names: list[str] = field(default_factory=list)
    stages: dict[str, dict] = field(default_factory=dict)
    run: Run | None = None
    running: bool = False
    started: float = 0.0
    result: str = ""
    tail: deque = field(default_factory=lambda: deque(maxlen=2000))

    def settings(self) -> dict:
        m = self.model
        shape = {"name": m.name, "org": m.org, "total_b": m.total_b, "active_b": m.active_b,
                 "experts": m.experts, "top_k": m.top_k}
        return {"model": m.id, "model_shape": shape, "schema": self.schema.id, "target": self.target.key,
                "purpose": self.purpose, "steps": [s.__dict__ for s in self.schema.steps], "quant": self.schema.quant}


def bar(pct: float, width: int = 14, color: str = EMBER) -> Text:
    n = round(width * pct / 100)
    return Text.assemble(("█" * n, color), ("░" * (width - n), ASH))


def row(label: str, value: str | Text, mark: str = "", mark_color: str = FLARE) -> Text:
    t = Text()
    t.append(f"{label}: ", style=f"bold {DIM}")
    t.append(value if isinstance(value, Text) else Text(value))
    if mark:
        t.append(f"  {mark}", style=f"bold {mark_color}")
    return t


def heading(text: str) -> Text:
    return Text(text, style=f"bold {EMBER}")


FIT_COLOR = {"✓": FLARE, "~": EMBER, "✗": REDSHIFT, "fp8": EMBER, "?": DIM}


def schema_flow(s: Schema) -> Text:
    """A schema's steps as a vertical flow, for the detail pane."""
    t = Text()
    t.append(f"{s.name}\n", style=f"bold {PHOTON}")
    t.append(f"{s.id} · by {s.author}", style=DIM)
    if s.proven:
        t.append(f" · proven on {s.runs} runs", style=FLARE)
    t.append(f"\n\n{s.summary}\n\n")
    fixed = Text.assemble(("  ○ ", ASH), ("detect → plan → weights → calibrate", DIM), "\n")
    t.append_text(fixed)
    for st in s.steps:
        t.append("  │\n", style=ASH)
        t.append("  ◉ ", style=EMBER)
        t.append(f"{st.kind:<7}", style=f"bold {FLARE}")
        t.append(f" {st.what}\n")
        if st.params:
            t.append("          " + "  ".join(f"{k}={v}" for k, v in st.params.items()) + "\n", style=DIM)
    t.append("  │\n", style=ASH)
    if s.stage_list:
        skip = {"detect", "plan", "weights", "calibrate"} | {st.kind for st in s.steps}
        tail = " → ".join(x for x in s.stage_list if x not in skip)
    else:
        tail = f"eval → gguf {s.quant} → package (Ollama)"
    t.append_text(Text.assemble(("  ○ ", ASH), (tail, DIM)))
    if s.tags:
        t.append("\n\n" + "  ".join(f"#{x}" for x in s.tags), style=CRIMSON)
    return t


def detected_text(d: detect.Detected, state: State) -> Text:
    """What detection found, and what it means on this VM and the target."""
    t = Text()
    where = {"hub": "Hugging Face", "local": "local folder", "demo": "bundled demo config"}[d.where]
    t.append(f"{d.source}\n", style=f"bold {PHOTON}")
    t.append(f"{where} · {d.family} · {d.arch}\n\n", style=DIM)
    if d.moe:
        shape = f"MoE · {d.experts} experts, top-{d.top_k or '?'}" + (f", {d.shared_experts} shared" if d.shared_experts else "")
        t.append_text(row("shape", shape, "REAP ✓", FLARE))
    else:
        t.append_text(row("shape", "dense", "no experts to REAP: heal/sft/quant only", DIM))
    t.append("\n")
    if d.total_b:
        size = f"{d.total_b:.1f}B" + (f" total, {d.active_b:.1f}B active" if d.moe else "")
        t.append_text(row("size", f"{size} · {d.dtype}" + ("" if d.size_exact else " · estimated from config")))
    else:
        t.append_text(row("size", "unknown until the weights stage"))
    t.append("\n")
    t.append_text(row("layers", f"{d.layers} × hidden {d.hidden}"))
    m = d.to_model()
    if m.total_b:
        src = hardware.source_fits(state.hw, m.total_b)
        t.append("\n")
        t.append_text(row("fits here", {"✓": "teacher in bf16 on this GPU", "fp8": "teacher only in fp8",
                                        "✗": "needs a bigger VM"}.get(src, "?"), src, FIT_COLOR.get(src, DIM)))
        fit = fit_mark(m, state.schema, state.target)
        sc = state.schema.name if state.schema else "no schema"
        t.append("\n")
        t.append_text(row(f"on {state.target.name}", f"≈ {out_gb(m, state.schema):.0f} GB with {sc}", fit,
                          FIT_COLOR.get(fit, DIM)))
    for n in d.notes:
        t.append(f"\n! {n}", style=FLARE)
    t.append("\n\nenter: use this model · edit the field to detect another", style=DIM)
    return t


class ModelScreen(ModalScreen[Model | None]):
    """Any model: type an HF id or a local folder and it is detected; the
    marketplace below is a shortcut."""

    BINDINGS = [Binding("escape", "dismiss(None)", "close")]

    def __init__(self, state: State):
        super().__init__()
        self.state = state
        self.found: tuple[str, detect.Detected] | None = None

    def compose(self) -> ComposeResult:
        with Vertical(id="models", classes="box") as v:
            v.border_title = "Source model"
            v.border_subtitle = "type any model + enter to detect · ↓ marketplace · esc: close"
            with Horizontal(id="any_row"):
                yield Static("❯", id="caret")
                yield Input(placeholder="any model: org/name on Hugging Face, or a folder on this VM (~/models/x)",
                            id="any_model")
            yield Static(id="detected")
            yield Label("Marketplace", id="market_label")
            yield DataTable(cursor_type="row")
            with Horizontal(id="target_row"):
                yield Label(self.target_note())

    def target_note(self) -> str:
        s = self.state
        sc = f"schema “{s.schema.name}” ({s.schema.quant})" if s.schema else "no schema (bf16, nothing pruned)"
        return (f"Out size and “On {s.target.name}” assume {sc}: ✓ fits · ~ tight · ✗ too big.\n"
                "Fits here: this VM holds the bf16 teacher (fp8: only quantized; ✗: needs a bigger VM).")

    def on_mount(self) -> None:
        s = self.state
        self.query_one("#detected").display = False
        t = self.query_one(DataTable)
        t.add_columns("Model", "Org", "Size", "Experts", "Released", "Fits here", "Out size", f"On {s.target.name}")
        for m in MODELS:
            src = hardware.source_fits(s.hw, m.total_b)
            fit = fit_mark(m, s.schema, s.target)
            t.add_row(Text(m.name, style=f"bold {PHOTON}"), m.org, m.size, m.shape, m.released,
                      Text(src, style=FIT_COLOR.get(src, "")), f"{out_gb(m, s.schema):.0f} GB",
                      Text(fit, style=FIT_COLOR.get(fit, "")), key=m.id)
        if s.model and s.model in MODELS:
            t.move_cursor(row=MODELS.index(s.model))
        self.query_one("#any_model", Input).focus()

    def key_down(self) -> None:
        if self.focused is self.query_one("#any_model"):
            self.query_one(DataTable).focus()

    @on(Input.Submitted, "#any_model")
    def submitted(self, e: Input.Submitted) -> None:
        q = e.value.strip()
        if not q:
            self.query_one(DataTable).focus()
            return
        if self.found and self.found[0] == q:
            self.state.detected = self.found[1]
            self.dismiss(self.found[1].to_model())
            return
        listed = find_model(q)
        if listed and "/" not in q:
            self.state.detected = None
            self.dismiss(listed)
            return
        box = self.query_one("#detected", Static)
        box.display = True
        box.update(Text(f"detecting {q}… (config.json only, no weights)", style=DIM))
        self.run_detect(q)

    @work(thread=True, exclusive=True, group="detect")
    def run_detect(self, q: str) -> None:
        try:
            d = (detect.detect_demo if self.state.demo else detect.detect)(q)
            self.app.call_from_thread(self._detected, q, d, None)
        except detect.DetectError as err:
            self.app.call_from_thread(self._detected, q, None, str(err))

    def _detected(self, q: str, d: detect.Detected | None, err: str | None) -> None:
        box = self.query_one("#detected", Static)
        if d is None:
            self.found = None
            box.update(Text(f"✗ {err}", style=REDSHIFT))
            return
        self.found = (q, d)
        box.update(detected_text(d, self.state))

    @on(DataTable.RowSelected)
    def chosen(self, e: DataTable.RowSelected) -> None:
        self.state.detected = None
        self.dismiss(next(m for m in MODELS if m.id == e.row_key.value))


class SchemaScreen(ModalScreen[Schema | None]):
    BINDINGS = [Binding("escape", "dismiss(None)", "close")]

    def __init__(self, state: State):
        super().__init__()
        self.state = state

    def compose(self) -> ComposeResult:
        with Vertical(id="schemas", classes="box") as v:
            v.border_title = "Schema marketplace"
            v.border_subtitle = "enter: use · tab: describe your own · esc: close"
            with Horizontal(id="schema_body"):
                yield DataTable(cursor_type="row", id="schema_table")
                yield Static(id="schema_detail")
            with Horizontal(id="describe_row"):
                yield Static("✦", id="caret")
                yield Input(placeholder="describe your idea: “a tiny Rust tutor that runs on my 16 GB Mac”",
                            id="describe")

    def on_mount(self) -> None:
        t = self.query_one(DataTable)
        t.add_columns("Schema", "By", "Steps", "Quant", "Runs", "")
        self.fill()
        t.focus()

    def fill(self, select: str | None = None) -> None:
        t = self.query_one(DataTable)
        t.clear()
        for s in self.state.schemas:
            t.add_row(Text(s.name, style=f"bold {PHOTON}"), s.author, " › ".join(st.kind for st in s.steps),
                      s.quant, str(s.runs or "–"), Text("proven", style=FLARE) if s.proven else "", key=s.id)
        ids = [s.id for s in self.state.schemas]
        pick = select or (self.state.schema.id if self.state.schema else None)
        if pick in ids:
            t.move_cursor(row=ids.index(pick))
        self.show(pick or ids[0])

    def show(self, sid: str) -> None:
        s = next((x for x in self.state.schemas if x.id == sid), None)
        if s:
            self.query_one("#schema_detail", Static).update(schema_flow(s))

    @on(DataTable.RowHighlighted)
    def highlighted(self, e: DataTable.RowHighlighted) -> None:
        if e.row_key and e.row_key.value:
            self.show(e.row_key.value)

    @on(DataTable.RowSelected)
    def chosen(self, e: DataTable.RowSelected) -> None:
        self.dismiss(next(s for s in self.state.schemas if s.id == e.row_key.value))

    @on(Input.Submitted, "#describe")
    def describe(self, e: Input.Submitted) -> None:
        idea = e.value.strip()
        if not idea:
            return
        s = self.app.draft(idea)
        e.input.value = ""
        self.fill(select=s.id)
        self.query_one(DataTable).focus()


class LogsScreen(Screen):
    BINDINGS = [Binding("escape", "app.pop_screen", "back"), Binding("f6", "app.pop_screen", "back")]

    def __init__(self, state: State):
        super().__init__()
        self.state = state
        self.shown = 0

    def compose(self) -> ComposeResult:
        with Vertical(id="logs") as v:
            run = self.state.run
            v.border_title = f"Logs · {run.root.name}" if run else "Logs"
            v.border_subtitle = "esc / F6: back · the job keeps running"
            yield RichLog(markup=False, wrap=True, highlight=False)

    def on_mount(self) -> None:
        self.pump()
        self.set_interval(0.3, self.pump)

    def pump(self) -> None:
        lines = list(self.state.tail)
        log = self.query_one(RichLog)
        for line in lines[self.shown:] if self.shown <= len(lines) else lines:
            style = (REDSHIFT if ("Error" in line or "Traceback" in line or line.startswith("✗")) else
                     EMBER if "──" in line else FLARE if line.startswith("✓") else "")
            log.write(Text(line, style=style))
        self.shown = len(lines)
        log.scroll_end(animate=False)


class BlackHole(App):
    TITLE = "BlackHole"
    CSS = theme.CSS
    BINDINGS = [
        Binding("f3", "models", "model"),
        Binding("f4", "schemas", "schema"),
        Binding("f5", "run", "run"),
        Binding("f6", "logs", "logs"),
        Binding("ctrl+c", "quit", "quit", priority=True),
    ]

    def __init__(self, demo: bool = False, model: str | None = None, schema: str | None = None,
                 target: str | None = None, purpose: str = ""):
        super().__init__()
        self.state = State(demo=demo, purpose=purpose)
        self.pending_model = None
        if model:
            self.state.model = find_model(model)
            self.pending_model = None if self.state.model else model
        if schema:
            self.state.schema = find_schema(schema)
        if target:
            self.state.target = find_target(target) or self.state.target
        self.autostart = bool(model and schema)
        self.session = datetime.now().strftime("%Y%m%d_%H%M%S_") + secrets.token_hex(3)

    def compose(self) -> ComposeResult:
        yield Static(banner(), id="banner")
        with Horizontal(id="main") as main:
            main.border_title = f"BlackHole harness {VERSION}"
            with Vertical(id="left"):
                yield Static(HOLE, id="art")
                yield Static(id="who")
            with Vertical(id="right"):
                yield Static(id="machine")
                yield Static(id="recipe")
                yield Static(id="pipeline")
                yield Static(id="market")
        yield RichLog(id="log", markup=False, wrap=True)
        yield Static(id="status")
        with Horizontal(id="promptrow"):
            yield Static("❯", id="caret")
            yield Input(placeholder="/model <name> · /schema <id> · /describe <idea> · F5 run · /help", id="prompt")

    def on_mount(self) -> None:
        self.register_theme(theme.THEME)
        self.theme = "blackhole"
        self.fit_art()
        s = self.state
        if s.demo:
            s.hw = hardware.DEMO
            self.say("Welcome to BlackHole. Demo mode: a simulated B200 and a pretend pipeline. /help for commands.")
        else:
            self.say("Welcome to BlackHole. Detecting this machine… F3 model, F4 schema, /help for commands.")
            self.detect()
        self.reattach()
        if self.pending_model:
            self.detect_model(self.pending_model, start=self.autostart and not s.running)
        elif self.autostart and not s.running:
            self.action_run()
        self.refresh_panels()
        self.set_interval(1, self.refresh_status)
        self.set_interval(0.25, self.poll)
        self.query_one("#prompt", Input).focus()

    @work(thread=True, exclusive=True, group="hw")
    def detect(self) -> None:
        hw = hardware.detect()
        self.call_from_thread(self._detected, hw)

    def _detected(self, hw: Hardware) -> None:
        self.state.hw = hw
        if hw.gpus:
            self.say(f"✓ {hw.gpu_line()} · {hw.host_line()}")
        else:
            self.say("✗ no NVIDIA GPU here. Run BlackHole on the GPU VM (ssh -t user@vm blackhole), "
                     "or `blackhole --demo` to look around.", color=REDSHIFT)
        self.refresh_panels()

    # rendering ---------------------------------------------------------------

    def on_resize(self) -> None:
        self.fit_art()

    def fit_art(self) -> None:
        large = self.size.height >= 66 and self.size.width >= 150
        self.query_one("#art", Static).update(HOLE_LARGE if large else HOLE)
        self.query_one("#left").styles.width = 66 if large else 52

    def say(self, msg: str, bold: bool = False, color: str = "") -> None:
        self.query_one("#log", RichLog).write(Text(msg, style=" ".join(x for x in ("bold" if bold else "", color) if x)))

    def refresh_panels(self) -> None:
        s = self.state
        who = Text(justify="center")
        who.append("BlackHole", style=f"bold {FLARE}")
        who.append(" · shrink big models to fit your machine\n", style=DIM)
        who.append(f"{'demo' if s.demo else socket.gethostname()}:{tilde(home())}/jobs\n", style=DIM)
        who.append(f"Session: {self.session}", style=ASH)
        self.query_one("#who", Static).update(who)

        mach = Text()
        mach.append_text(heading("Machine\n"))
        if s.hw is None:
            mach.append_text(row("gpu", "detecting…"))
        else:
            mach.append_text(row("gpu", s.hw.gpu_line(), "✓" if s.hw.gpus else "✗", FLARE if s.hw.gpus else REDSHIFT))
            mach.append("\n")
            mach.append_text(row("host", s.hw.host_line()))
            for w in s.hw.warnings[:2]:
                mach.append(f"\n! {w}", style=REDSHIFT)
        self.query_one("#machine", Static).update(mach)

        rec = Text()
        rec.append_text(heading("Recipe\n"))
        if s.model:
            m = s.model
            shape = f"{m.size}, {m.shape}" if m.total_b else "shape read at plan stage"
            if s.detected and s.detected.source == m.id:
                shape += f" · {s.detected.family}, detected"
            src = hardware.source_fits(s.hw, m.total_b)
            rec.append_text(row("model", f"{m.name}  {shape}", "✓" if src == "✓" else src, FIT_COLOR.get(src, FLARE)))
        else:
            rec.append_text(row("model", "not chosen  (F3 or /model)", "○", DIM))
        rec.append("\n")
        if s.schema:
            rec.append_text(row("schema", f"{s.schema.name}  " + " › ".join(st.kind for st in s.schema.steps) +
                                f" › {s.schema.quant}", "✓"))
        else:
            rec.append_text(row("schema", "not chosen  (F4, /schema or /describe)", "○", DIM))
        rec.append("\n")
        if s.purpose:
            rec.append_text(row("purpose", f"“{s.purpose}”"))
            rec.append("\n")
        if s.model and s.model.total_b:
            fit = fit_mark(s.model, s.schema, s.target)
            gb = out_gb(s.model, s.schema)
            note = {"✓": "fits", "~": "tight", "✗": "too big"}.get(fit, "")
            rec.append_text(row("target", f"{s.target.name} · out ≈ {gb:.0f} GB, {note}", fit, FIT_COLOR.get(fit, FLARE)))
        else:
            rec.append_text(row("target", s.target.name))
        self.query_one("#recipe", Static).update(rec)

        pipe = Text()
        pipe.append_text(heading("Pipeline"))
        if s.run:
            pipe.append(f"  {s.run.root.name}", style=DIM)
        names = s.stage_names or (s.schema.stages() if s.schema else [])
        if not names:
            pipe.append("\n  choose a model and a schema, then F5", style=DIM)
        width = max([10] + [len(n) + 1 for n in names])
        for name in names:
            ev = s.stages.get(name)
            status = ev["status"] if ev else "waiting"
            pct = 100.0 if status in ("done", "skipped") else float((ev or {}).get("pct") or 0)
            mark, mcol = {"done": ("✓", FLARE), "skipped": ("✓", DIM), "running": ("▸", PHOTON),
                          "error": ("✗", REDSHIFT)}.get(status, ("·", ASH))
            pipe.append(f"\n{mark} ", style=f"bold {mcol}")
            pipe.append(f"{name:<{width}}", style=f"bold {PHOTON}" if status == "running" else
                        (REDSHIFT if status == "error" else ("" if ev else DIM)))
            pipe.append_text(bar(pct, color=FLARE if status == "running" else REDSHIFT if status == "error" else EMBER))
            pipe.append(f" {pct:>3.0f}%  ", style=DIM)
            msg = (ev or {}).get("msg") or ""
            pipe.append(msg[:56], style="" if status in ("running", "error") else DIM)
        ev = s.stages.get("eval")
        if ev and ev.get("status") == "done" and "top1" in ev:
            pipe.append(f"\n  eval vs teacher: top-1 {ev['top1']:.0%} · ppl {ev['ppl_teacher']} → {ev['ppl_student']}"
                        f" · kl {ev['kl']:.3f}", style=FLARE)
        g, q = s.stages.get("gptq") or {}, s.stages.get("reaq") or {}
        if g.get("status") == "done" and "kl" in g:
            pipe.append(f"\n  int4 (shipped): kl {g['kl']:.3f} · experts {g.get('expert_gb', 0):.1f} GB", style=FLARE)
            if q.get("status") == "done" and "kl" in q:
                d = (q["kl"] / g["kl"] - 1) if g["kl"] else 0.0
                pipe.append(f"  ·  REAQ codebook (lab): kl {q['kl']:.3f} · {q.get('expert_gb', 0):.1f} GB · "
                            f"{d:+.0%} kl", style=f"bold {EMBER}")
        b = s.stages.get("bench") or {}
        if b.get("status") == "done" and "lcb_int4" in b:
            pipe.append(f"\n  LiveCodeBench pass@1: original {b['lcb_original']:.1%} → int4 {b['lcb_int4']:.1%}"
                        f" ({b.get('problems', '?')} problems)", style=f"bold {FLARE}")
        if s.result:
            pipe.append(f"\n◉ {s.result}", style=f"bold {PHOTON}")
        self.query_one("#pipeline", Static).update(pipe)

        mk = Table.grid(padding=(0, 2))
        for _ in range(4):
            mk.add_column()
        for sc in s.schemas[:5]:
            chosen = s.schema is not None and s.schema.id == sc.id
            mk.add_row(Text("●" if chosen else "·", style=EMBER if chosen else ASH),
                       Text(sc.name, style=f"bold {PHOTON}" if chosen else ""),
                       Text(" › ".join(st.kind for st in sc.steps), style=DIM),
                       Text("proven" if sc.proven else sc.author, style=FLARE if sc.proven else DIM))
        foot = Text(f"({len(s.schemas) - 5} more… F4 · or /describe <your idea>)", style=DIM)
        self.query_one("#market", Static).update(Group(heading("Schemas"), mk, foot))

        self.query_one("#main").border_subtitle = ((" demo · " if s.demo else " ") +
                                                   "F3 model · F4 schema · F5 run · F6 logs · /help ")
        self.refresh_status()

    def refresh_status(self) -> None:
        s = self.state
        names = s.stage_names or (s.schema.stages() if s.schema else [])
        done = sum(1 for n in names if s.stages.get(n, {}).get("status") in ("done", "skipped"))
        cur = next((s.stages[n] for n in names if s.stages.get(n, {}).get("status") == "running"), None)
        pct = 100 * (done + (float(cur.get("pct") or 0) / 100 if cur else 0)) / max(len(names), 1)
        el = int(time.time() - s.started) if s.started and s.running else 0
        model = s.model.id.split("/")[-1].lower() if s.model else "no model"
        schema = s.schema.id if s.schema else "no schema"
        stage = cur["stage"] if cur else ("running" if s.running else ("done" if s.result else "idle"))
        n = round(10 * pct / 100)
        self.query_one("#status", Static).update(
            f" ◉ {model} │ {schema} │ {s.target.key} │ {stage} │ [{'█' * n}{'░' * (10 - n)}] {pct:.0f}% │ "
            f"{el // 60}m{el % 60:02d}s ")

    # jobs --------------------------------------------------------------------

    def reattach(self) -> None:
        run = Run.latest()
        if run is None:
            return
        s = self.state
        cfg = run.config
        if not s.model and cfg.get("model"):
            s.model = find_model(cfg["model"]) or model_from_config(cfg)
        if not s.schema and cfg.get("schema"):
            s.schema = find_schema(cfg["schema"])
        if cfg.get("target"):
            s.target = find_target(cfg["target"]) or s.target
        s.purpose = s.purpose or cfg.get("purpose", "")
        s.run = run
        s.stage_names = s.schema.stages() if s.schema else []
        for ev, line in run.replay():
            self.apply(ev, line, quiet=True)
        s.running = run.running()
        if s.running:
            s.started = (run.root / "pid").stat().st_mtime
            self.say(f"↻ reattached to the running job {run.root.name}", bold=True, color=FLARE)
        else:
            self.say(f"last job: {run.root.name} ({'done' if s.result else 'stopped'})", color=DIM)

    def apply(self, ev: dict | None, line: str, quiet: bool = False) -> None:
        s = self.state
        if ev is None:
            s.tail.append(line)
            if not quiet and ("Traceback" in line or "Error" in line):
                self.say(line[:300], color=REDSHIFT)
            return
        stage = ev["stage"]
        if ev.get("gate") == "failed":
            s.tail.append(f"✗ gate failed in {stage}: {ev.get('msg', '')}")
            if not quiet:
                self.say(f"✗ quality gate failed in {stage}: {ev.get('msg', '')}", bold=True, color=REDSHIFT)
        if stage == "pipeline":
            if ev["status"] == "done" and ev.get("outcome", "done") == "done":
                s.result = f"{tilde(ev.get('model', 'done'))}  (~{ev.get('size_gb', '?')} GB)"
                if not quiet:
                    self.say(f"◉ done: {s.result}", bold=True, color=FLARE)
            elif ev["status"] == "running" and str(ev.get("msg", "")).startswith("started"):
                s.stages, s.result = {}, ""
            elif (ev["status"] == "error" or ev.get("outcome") in ("stopped", "error")) and not quiet:
                self.say(f"✗ {ev.get('msg') or ev.get('outcome') or 'pipeline failed'}", bold=True, color=REDSHIFT)
            return
        if stage not in s.stage_names:
            s.stage_names.append(stage)
        prev = s.stages.get(stage, {})
        s.stages[stage] = {**prev, **ev} if ev["status"] == "running" else {**prev, **ev, "msg": ev.get("msg", "")}
        if ev["status"] in ("done", "error", "skipped"):
            s.tail.append(f"{'✗' if ev['status'] == 'error' else '✓'} {stage} {ev['status']}"
                          f"{': ' + ev['msg'] if ev.get('msg') else ''}")
        if not quiet and ev["status"] in ("done", "error", "skipped"):
            mark = {"done": "✓", "skipped": "✓", "error": "✗"}[ev["status"]]
            self.say(f"{mark} {stage}{': ' + ev['msg'] if ev.get('msg') else ''}",
                     color=REDSHIFT if ev["status"] == "error" else "")

    def poll(self) -> None:
        s = self.state
        if s.run is None:
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
                self.say("■ the job ended without a model. F6 shows the log.", bold=True, color=REDSHIFT)
        if lines or was != s.running:
            self.refresh_panels()

    # actions -----------------------------------------------------------------

    def action_models(self) -> None:
        def done(m: Model | None) -> None:
            if m:
                self.choose_model(m)
        self.push_screen(ModelScreen(self.state), done)

    def action_schemas(self) -> None:
        def done(sc: Schema | None) -> None:
            if sc:
                self.choose_schema(sc)
        self.push_screen(SchemaScreen(self.state), done)

    def action_logs(self) -> None:
        if isinstance(self.screen, LogsScreen):
            self.pop_screen()
        else:
            self.push_screen(LogsScreen(self.state))

    @work(thread=True, exclusive=True, group="detect")
    def detect_model(self, q: str, start: bool = False) -> None:
        try:
            d = (detect.detect_demo if self.state.demo else detect.detect)(q)
            self.call_from_thread(self._model_detected, d, start)
        except detect.DetectError as e:
            self.call_from_thread(self.say, f"✗ {e}", False, REDSHIFT)

    def _model_detected(self, d: detect.Detected, start: bool) -> None:
        self.state.detected = d
        shape = f"MoE {d.experts}×top-{d.top_k}" if d.moe else "dense"
        size = f"{d.total_b:.1f}B{'' if d.size_exact else ' (est.)'}" if d.total_b else "size unknown"
        self.say(f"⌕ detected {d.source}: {d.family}, {shape}, {size}", color=FLARE)
        for n in d.notes:
            self.say(f"  ! {n}", color=FLARE)
        self.choose_model(d.to_model())
        if start:
            self.action_run()

    def choose_model(self, m: Model) -> None:
        self.state.model = m
        shape = f"{m.size}, {m.shape}" if m.total_b else "not in the list; shape read at plan stage"
        self.say(f"✓ model: {m.id}  ({shape})")
        self.refresh_panels()

    def choose_schema(self, sc: Schema) -> None:
        s = self.state
        s.schema = sc
        if not s.running:
            s.stage_names, s.stages, s.result = [], {}, ""
        self.say(f"✓ schema: {sc.name} — " + " › ".join(st.kind for st in sc.steps) + f" › {sc.quant}")
        self.refresh_panels()

    def draft(self, idea: str) -> Schema:
        """/describe: a schema from words. Offline keyword draft until a frontier model is wired in."""
        who = architect.provider()
        sc = architect.draft_offline(idea)
        s = self.state
        s.schemas = [x for x in s.schemas if x.id != sc.id] + [sc]
        via = f"{who} key found, but the frontier call is not wired yet; " if who else ""
        self.say(f"✦ drafted “{sc.name}”: " + " › ".join(st.kind for st in sc.steps) +
                 f" › {sc.quant}  ({via}offline draft; F4 to review)", color=FLARE)
        self.refresh_panels()
        return sc

    def action_run(self) -> None:
        s = self.state
        if s.running:
            self.say("a job is already running; /stop first")
            return
        if s.model is None or s.schema is None:
            self.say("✗ choose a model (F3) and a schema (F4) first", color=REDSHIFT)
            return
        try:
            s.run = Run.start(s.settings(), demo=s.demo)
        except Exception as e:
            self.say(f"✗ {e}", color=REDSHIFT)
            return
        s.stage_names, s.stages, s.result = s.schema.stages(), {}, ""
        s.running, s.started = True, time.time()
        s.tail.clear()
        why = f" for “{s.purpose}”" if s.purpose else ""
        self.say(f"◉ collapsing {s.model.name} with {s.schema.name}{why} → {s.target.name}", bold=True, color=EMBER)
        self.say(f"  job {tilde(s.run.root)} · keeps running if you close this or SSH drops", color=DIM)
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
        self.say(f"❯ {line}", color=DIM)
        cmd, *args = line.split()
        arg = " ".join(args).strip().strip('"')
        s = self.state
        match cmd.lower():
            case "/help":
                for h in (__doc__ or "").splitlines():
                    if h.startswith("    "):
                        self.say(h)
            case "/model" if arg:
                m = find_model(arg) if "/" not in arg else None
                if m:
                    s.detected = None
                    self.choose_model(m)
                elif "/" in arg or arg.startswith((".", "~")):
                    self.say(f"⌕ detecting {arg}…", color=DIM)
                    self.detect_model(arg)
                else:
                    self.say("✗ not in the list; give an org/name Hugging Face id or a folder, or F3")
            case "/model":
                self.action_models()
            case "/schema" if arg:
                sc = next((x for x in s.schemas if x.id == arg), None) or find_schema(arg)
                self.choose_schema(sc) if sc else self.say("✗ no such schema; F4 for the marketplace")
            case "/schema" | "/market":
                self.action_schemas()
            case "/target" if arg:
                t = find_target(arg)
                if t:
                    s.target = t
                    self.say(f"✓ target: {t.name}")
                    self.refresh_panels()
                else:
                    self.say("✗ targets: " + " ".join(t.key for t in TARGETS))
            case "/purpose":
                s.purpose = arg
                self.say(f"✓ purpose: {arg}" if arg else "✓ no purpose: general calibration")
                self.refresh_panels()
            case "/describe" if arg:
                self.choose_schema(self.draft(arg))
            case "/run":
                self.action_run()
            case "/stop":
                self.stop()
            case "/logs":
                self.action_logs()
            case "/quit" | "/exit":
                self.exit()
            case _:
                self.say("✗ unknown command or missing argument; /help", color=REDSHIFT)


def main() -> None:
    ap = argparse.ArgumentParser(prog="blackhole", description="Shrink a big open model to fit your own machine.")
    ap.add_argument("model", nargs="?", help='source model, fuzzy is fine: "qwen3 coder" or org/name')
    ap.add_argument("purpose", nargs="?", default="", help='what the model is for: "Rust tutoring"')
    ap.add_argument("--schema", help="schema id from the marketplace, e.g. pocket-coder")
    ap.add_argument("--target", help="machine it must run on, e.g. rtx4090, mac32")
    ap.add_argument("--demo", action="store_true", help="simulate a B200 and a pipeline (no GPU needed)")
    a = ap.parse_args()
    BlackHole(demo=a.demo, model=a.model, schema=a.schema, target=a.target, purpose=a.purpose).run()


if __name__ == "__main__":
    main()
