"""Logo art. The black hole is drawn, not traced: a small ray-free model of a
Gargantua-style hole (shadow, photon ring, a tilted accretion disk in front and
its lensed far side bent over the top), sampled onto half-block pixels so each
terminal cell holds two square-ish pixels. Brightness is snapped to a six-step
black-hole palette, so the terminal's own black is the event horizon.

The art is deterministic; `python -m blackhole_tui.art` prints both sizes.
"""

from __future__ import annotations

import math

from rich.text import Text

from blackhole_tui.theme import BANNER_GRADIENT, HEAT, SHADOW_EDGE

# "BLACKHOLE" in figlet ansi_shadow. Solid strokes get the heat gradient, the
# box-drawing shadow strokes stay dark red.
BANNER = """\
██████╗ ██╗      █████╗  ██████╗██╗  ██╗██╗  ██╗ ██████╗ ██╗     ███████╗
██╔══██╗██║     ██╔══██╗██╔════╝██║ ██╔╝██║  ██║██╔═══██╗██║     ██╔════╝
██████╔╝██║     ███████║██║     █████╔╝ ███████║██║   ██║██║     █████╗
██╔══██╗██║     ██╔══██║██║     ██╔═██╗ ██╔══██║██║   ██║██║     ██╔══╝
██████╔╝███████╗██║  ██║╚██████╗██║  ██╗██║  ██║╚██████╔╝███████╗███████╗
╚═════╝ ╚══════╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚══════╝"""


def banner(tag: str = "harness") -> Text:
    lines = BANNER.splitlines()
    width = max(len(l) for l in lines)
    out = Text(no_wrap=True)
    for y, line in enumerate(lines):
        for x, ch in enumerate(line):
            if ch == "█":
                g = BANNER_GRADIENT[min(len(BANNER_GRADIENT) - 1, x * len(BANNER_GRADIENT) // width)]
                out.append(ch, style=g)
            elif ch != " ":
                out.append(ch, style=SHADOW_EDGE)
            else:
                out.append(" ")
        if y == len(lines) - 2 and tag:
            out.append("  " + tag, style=f"bold {HEAT[-1]}")
        out.append("\n")
    out.rstrip()
    return out


DISK_IN, DISK_OUT = 1.4, 3.1


def _intensity(x: float, y: float) -> float:
    """Brightness 0..1 at (x, y), in units of the shadow radius; +y is down."""
    r = math.hypot(x, y)
    tilt = 0.17  # sin of the disk's inclination: how flat it looks

    # the near side of the disk passes in front of the shadow
    if y >= 0:
        rho = math.hypot(x, y / tilt)
        if DISK_IN <= rho <= DISK_OUT and abs(y) <= 0.62:
            phi = math.atan2(y / tilt, x)
            fall = (DISK_OUT - rho) / (DISK_OUT - DISK_IN)
            beam = 1.0 + 0.45 * (-x / rho)  # approaching side (left) is brighter
            streak = 0.82 + 0.18 * math.sin(phi * 9 + rho * 6)
            v = (0.25 + 0.75 * fall ** 1.4) * beam * streak
            if r < 1.0:
                v *= 1.05
            return min(v, 1.0)

    if r < 1.0:
        return 0.0  # event horizon

    # photon ring hugging the shadow
    ring = math.exp(-((r - 1.07) / 0.055) ** 2)

    # far side of the disk, lensed into a halo; thick on top, thin underneath
    halo = 0.0
    if 1.08 < r < 2.25:
        thick = 0.95 if y < 0 else 0.45
        edge = 1.12 + thick
        if r < edge:
            t = (r - 1.08) / (edge - 1.08)
            halo = (1 - t) ** 1.6 * (0.85 if y < 0 else 0.5)
            halo *= 1.0 + 0.35 * (-x / r)
            halo *= 0.85 + 0.15 * math.sin(math.atan2(y, x) * 11 + r * 5)

    # far side of the disk, behind the hole, visible past the shadow's edge
    back = 0.0
    if y < 0:
        rho = math.hypot(x, y / tilt)
        if DISK_IN <= rho <= DISK_OUT and abs(y) <= 0.62:
            back = 0.55 * (DISK_OUT - rho) / (DISK_OUT - DISK_IN) * (1.0 + 0.4 * (-x / rho))

    # faint glow around everything
    glow = 0.10 * math.exp(-max(0.0, r - 1.1) * 1.6) if r < DISK_OUT + 0.3 else 0.0
    return min(1.0, max(ring * 0.95, halo, back, glow))


def _palette_index(v: float) -> int:
    # cut points tuned so the disk shows every step from ember to white-hot
    cuts = (0.06, 0.16, 0.30, 0.46, 0.64, 0.82)
    for i, c in enumerate(cuts):
        if v < c:
            return i  # 0 is "not lit": terminal black
    return len(cuts)


def render(cols: int, rows: int, scale: float = 1.0) -> Text:
    """The hole in cols cells wide and at most rows tall, two pixels per cell
    (half blocks). Unlit rows at the top and bottom are dropped."""
    px_h = rows * 2
    # fit the disk (radius DISK_OUT) to the width, keeping pixels square
    unit = (cols / 2) / ((DISK_OUT + 0.1) * scale)
    grid = []
    for py in range(px_h):
        line = []
        for px in range(cols):
            x = (px + 0.5 - cols / 2) / unit
            y = (py + 0.5 - px_h / 2) / unit
            line.append(_palette_index(_intensity(x, y)))
        grid.append(line)

    lit = [i for i, l in enumerate(grid) if any(l)]
    grid = grid[lit[0] // 2 * 2: (lit[-1] // 2 + 1) * 2] if lit else grid
    rows = len(grid) // 2
    palette = [None, *HEAT]  # 0 = unlit
    out = Text(no_wrap=True)
    for cy in range(rows):
        top, bot = grid[2 * cy], grid[2 * cy + 1]
        for t, b in zip(top, bot):
            ct, cb = palette[t], palette[b]
            if ct is None and cb is None:
                out.append(" ")
            elif ct == cb:
                out.append("█", style=ct)
            elif cb is None:
                out.append("▀", style=ct)
            elif ct is None:
                out.append("▄", style=cb)
            else:
                out.append("▀", style=f"{ct} on {cb}")
        if cy < rows - 1:
            out.append("\n")
    return out


# 50 x 18: the default left column, as REAPer's small reaper
HOLE = render(50, 18)
# 64 x 26: when the terminal is at least 150 x 66
HOLE_LARGE = render(64, 26)


if __name__ == "__main__":
    from rich.console import Console

    c = Console()
    c.print(banner())
    c.print(HOLE)
    c.print(HOLE_LARGE)
