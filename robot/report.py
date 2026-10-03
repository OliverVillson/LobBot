"""Robot report: the footprint-vs-performance graph, the training video, a summary.

    out/footprint_vs_performance.png   model size per candidate, and sim success vs size
    out/training.mp4                   site, expert demo, teacher, each compressed candidate, graph
    out/report.json                    what is in them, and where

Reads out/eval.json (robot/simeval.py) and work/site.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Neutral palette (dataviz reference instance, light mode): teacher in blue,
# compressed candidates in one recessive ink, the expert as a dashed reference.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
TEACHER = "#2a78d6"
CANDIDATE = "#8a8984"
OVER_BUDGET = "#e34948"


def _font(size: int):
    from PIL import ImageFont

    for name in ("DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


# ----------------------------------------------------------------- graph


def plot(report: dict, path: Path, site_name: str = "") -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cands = report["candidates"]
    dry = report.get("dry_run")
    budget = (report.get("target") or {}).get("max_size_gb")
    names = [c["name"] for c in cands]
    sizes = [c["size_gb"] for c in cands]
    colors = [TEACHER if n == "teacher" else CANDIDATE for n in names]

    plt.rcParams.update({"font.size": 12, "axes.edgecolor": INK_2, "axes.labelcolor": INK, "xtick.color": INK_2,
                         "ytick.color": INK_2, "text.color": INK, "axes.titlesize": 14})
    fig, (left, right) = plt.subplots(1, 2, figsize=(16, 8), dpi=100, facecolor=SURFACE,
                                      gridspec_kw={"width_ratios": [1, 1.25]})
    for ax in (left, right):
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=1)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    # Left: footprint per candidate.
    y = np.arange(len(cands))[::-1]
    left.barh(y, sizes, color=colors, height=0.6, edgecolor=SURFACE, linewidth=2)
    left.set_yticks(y, names)
    left.grid(False, axis="y")
    for yi, c in zip(y, cands):
        bits = f"  {c['bits_per_weight']:.1f} b/w" if c.get("bits_per_weight") else ""
        left.text(c["size_gb"], yi, f"  {c['size_gb']:.2f} GB{bits}", va="center", color=INK, fontsize=11)
    if budget:
        left.axvline(budget, color=INK_2, linestyle=":", linewidth=1.5)
        left.text(budget, y.min() - 0.45, f" budget {budget:g} GB", color=INK_2, fontsize=10, va="center")
    left.set_xlim(0, max(sizes + [budget or 0]) * 1.35)
    left.set_xlabel("model footprint (GB)")
    left.set_title("Footprint per candidate", loc="left")

    # Right: success vs footprint; marker area shows mean cross-track error.
    right.set_xlim(0, max(sizes) * 1.3)
    right.set_ylim(-5, 115)
    exp = report.get("expert") or {}
    if "success_rate" in exp:
        right.axhline(exp["success_rate"] * 100, color=INK_2, linestyle="--", linewidth=1.5, zorder=1)
        right.text(0.99, 0.03, f"- - -  scripted expert (reference): {exp['success_rate']:.0%}, "
                   f"CTE {exp.get('cte_mean_m', 0):.2f} m", transform=right.transAxes, ha="right", va="bottom",
                   color=INK_2, fontsize=10)
    for c, col in zip(cands, colors):
        area = 60 + 900 * min(c.get("cte_mean_m", 0.0), 1.5)
        right.scatter(c["size_gb"], c["success_rate"] * 100, s=area, color=col, alpha=0.9,
                      edgecolor=SURFACE, linewidth=2, zorder=4 if c["name"] == "teacher" else 3)
    _label_points(fig, right, cands)
    right.set_xlabel("model footprint (GB)")
    right.set_ylabel(f"sim success rate (%, {report.get('episodes', '?')} paired episodes)")
    right.set_title("Sim performance vs footprint (marker size = mean cross-track error)", loc="left")

    title = "GR00T N1.7 compression: footprint vs farm-sim performance"
    if site_name:
        title += f"  ·  {site_name}"
    if dry:
        title = "DRY RUN (expert + noise, not model results)  ·  " + title
    fig.suptitle(title, x=0.06, ha="left", fontsize=16, color=OVER_BUDGET if dry else INK)
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def _label_points(fig, ax, cands) -> None:
    """Direct labels that do not collide: for each point try a few offsets around
    it (in pixels) and take the first whose box overlaps no label placed so far."""
    fig.canvas.draw()
    to_px = ax.transData.transform
    taken: list[tuple[float, float, float, float]] = []
    offsets = [(12, 8), (12, -40), (-12, 8), (-12, -40), (12, 44), (12, -76), (-12, 44), (-12, -76)]
    for c in cands:
        text = f"{c['name']}\n{c['success_rate']:.0%} · CTE {c.get('cte_mean_m', 0):.2f} m"
        w, h = 8.5 * max(len(l) for l in text.splitlines()) + 6, 36.0
        px, py = to_px((c["size_gb"], c["success_rate"] * 100))
        for dx, dy in offsets:
            x0 = px + dx if dx > 0 else px + dx - w
            box = (x0, py + dy, x0 + w, py + dy + h)
            if not any(box[0] < t[2] and t[0] < box[2] and box[1] < t[3] and t[1] < box[3] for t in taken):
                break
        taken.append(box)
        ax.annotate(text, (c["size_gb"], c["success_rate"] * 100), textcoords="offset pixels",
                    xytext=(dx, dy), ha="left" if dx > 0 else "right", va="bottom", fontsize=10, color=INK,
                    fontweight="bold" if c["name"] == "teacher" else "normal",
                    arrowprops={"arrowstyle": "-", "color": GRID, "linewidth": 1} if abs(dy) > 20 else None)


# ----------------------------------------------------------------- video


def _fit(img: np.ndarray, w: int, h: int, bg=(20, 20, 20)) -> np.ndarray:
    """Letterbox an RGB image into w x h."""
    from PIL import Image

    im = Image.fromarray(np.asarray(img)[..., :3].astype(np.uint8))
    scale = min(w / im.width, h / im.height)
    im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))), Image.LANCZOS)
    canvas = Image.new("RGB", (w, h), bg)
    canvas.paste(im, ((w - im.width) // 2, (h - im.height) // 2))
    return np.asarray(canvas)


def _caption(frame: np.ndarray, text: str) -> np.ndarray:
    """Burn a one-line caption bar along the bottom of a frame."""
    from PIL import Image, ImageDraw

    im = Image.fromarray(frame)
    d = ImageDraw.Draw(im, "RGBA")
    h = max(22, im.height // 14)
    d.rectangle([0, im.height - h, im.width, im.height], fill=(0, 0, 0, 160))
    d.text((10, im.height - h + h // 6), text, fill=(255, 255, 255), font=_font(int(h * 0.6)))
    return np.asarray(im)


def _read_clip(path: Path, max_frames: int) -> list[np.ndarray]:
    """Frames of an MP4, evenly subsampled to at most max_frames."""
    import imageio.v2 as imageio

    if not path.exists():
        return []
    frames = [np.asarray(f) for f in imageio.get_reader(str(path))]
    if len(frames) > max_frames:
        idx = np.linspace(0, len(frames) - 1, max_frames).round().astype(int)
        frames = [frames[i] for i in idx]
    return frames


def _card(text: str, w: int, h: int, seconds: float, fps: int) -> list[np.ndarray]:
    from farmsim.video import title_card

    return [np.asarray(f) for f in title_card(text, width=w, height=h, seconds=seconds, fps=fps)]


def stitch(job, report: dict, site: dict, graph: Path, path: Path) -> tuple[Path, float]:
    from farmsim.video import write_mp4

    cfg = job.config
    w, h, fps = cfg.clip_width, cfg.clip_height, cfg.fps
    cands = report["candidates"]
    dry = " (DRY RUN)" if report.get("dry_run") else ""
    # Budget: fixed cards plus clips sharing what is left of video_seconds.
    fixed = 2.0 + 2.0 + 4.0 + 1.5 * (1 + len(cands))
    per_clip = max(2.0, (cfg.video_seconds - fixed) / (1 + len(cands)))
    max_frames = int(per_clip * fps)

    frames: list[np.ndarray] = []
    title = f"{site.get('name') or site.get('id')}:\n{site['lat']:.4f}, {site['lon']:.4f}, {site['source']}"
    if dry:
        title += "\nDRY RUN: candidates are the expert plus noise, not GR00T"
    frames += _card(title, w, h, 2.5, fps)
    preview = Path(site["dir"]) / "preview.png"
    if preview.exists():
        from PIL import Image

        still = _caption(_fit(np.asarray(Image.open(preview).convert("RGB")), w, h),
                         f"{site.get('size_m', 0):.0f} m square · {site['source']}")
        frames += [still] * int(2.0 * fps)
    frames += _card("Scripted expert: the demonstrations", w, h, 1.5, fps)
    frames += [_caption(f, "expert (pure pursuit on the true row)")
               for f in _read_clip(job.path("out", "clips", "expert.mp4"), max_frames)]
    for c in cands:
        label = "teacher (fine-tuned, uncompressed)" if c["name"] == "teacher" else c["name"]
        line = f"{c['size_gb']:.2f} GB · success {c['success_rate']:.0%} · CTE {c.get('cte_mean_m', 0):.2f} m"
        if c.get("latency_ms"):
            line += f" · {c['latency_ms']:.0f} ms"
        frames += _card(f"{label}\n{line}{dry}", w, h, 1.5, fps)
        frames += [_caption(f, f"{c['name']} · {line}") for f in _read_clip(job.path(c["video"]), max_frames)]
    if graph.exists():
        from PIL import Image

        frames += [_fit(np.asarray(Image.open(graph).convert("RGB")), w, h, bg=(252, 252, 251))] * int(4.0 * fps)
    frames = [_fit(f, w, h) if f.shape[:2] != (h, w) else f[..., :3] for f in frames]
    write_mp4(frames, path, fps=fps)
    return path, len(frames) / fps


def build_report(job) -> dict:
    report = json.loads(job.path("out", "eval.json").read_text())
    site = json.loads(job.path("work", "site.json").read_text())
    graph = plot(report, job.path("out", "footprint_vs_performance.png"), site.get("name", ""))
    video, seconds = stitch(job, report, site, graph, job.path("out", "training.mp4"))
    fits = [c for c in report["candidates"] if c.get("fits")]
    best = min(fits, key=lambda c: c["size_gb"]) if fits else None
    summary = {
        "kind": "robot",
        "dry_run": report.get("dry_run", False),
        "task_name": job.spec.task_name,
        "site": {k: site.get(k) for k in ("id", "name", "lat", "lon", "size_m", "source")},
        "graph": str(graph.relative_to(job.root)),
        "video": str(video.relative_to(job.root)),
        "video_seconds": round(seconds, 1),
        "candidates": [{k: c.get(k) for k in ("name", "size_gb", "success_rate", "cte_mean_m", "latency_ms", "fits")}
                       for c in report["candidates"]],
        "smallest_fitting": best["name"] if best else None,
        "expert": report.get("expert"),
    }
    job.path("out", "report.json").write_text(json.dumps(summary, indent=2))
    return summary
