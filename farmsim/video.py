"""Record policy rollouts in FarmEnv as video: chase view, front-cam inset, HUD.

rollout() drives one seeded episode; write_mp4() encodes H.264/yuv420p so the
file plays in browsers and QuickTime; title_card() makes text slates.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _font(size: int) -> ImageFont.ImageFont:
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def compose_frame(chase: np.ndarray, front: np.ndarray | None, hud: list[str]) -> np.ndarray:
    """Chase frame with the front cam as a top-right inset and HUD lines top-left."""
    img = Image.fromarray(chase)
    w, h = img.size
    draw = ImageDraw.Draw(img, "RGBA")
    if front is not None and front.size:
        side = max(48, int(h * 0.42))
        inset = Image.fromarray(front).resize((side, side), Image.BILINEAR)
        x0, y0 = w - side - 10, 10
        draw.rectangle([x0 - 2, y0 - 2, x0 + side + 1, y0 + side + 1], fill=(255, 255, 255, 230))
        img.paste(inset, (x0, y0))
        draw.text((x0 + 4, y0 + side - 16), "cab cam", font=_font(12), fill=(255, 255, 255, 255))
    font = _font(max(11, h // 26))
    lh = int(font.size * 1.3) if hasattr(font, "size") else 14
    if hud:
        tw = max(int(draw.textlength(t, font=font)) for t in hud)
        draw.rectangle([6, 6, 18 + tw, 12 + lh * len(hud)], fill=(0, 0, 0, 140))
        for i, t in enumerate(hud):
            draw.text((12, 9 + lh * i), t, font=font, fill=(255, 255, 255, 255))
    return np.asarray(img, dtype=np.uint8)


def rollout(env, policy, seed, max_steps: int = 600, frames: bool = True,
            width: int = 640, height: int = 360) -> tuple[dict, list[np.ndarray]]:
    """Drive one episode; returns (env.metrics(), chase frames with inset and HUD)."""
    if hasattr(policy, "bind"):
        policy.bind(env)
    obs = env.reset(seed)
    policy.reset()
    name = getattr(policy, "name", type(policy).__name__)
    size = getattr(policy, "size_label", None)
    out: list[np.ndarray] = []

    def snap(step: int, info: dict | None) -> None:
        st = obs["state"]
        hud = [f"{name}" + (f"  [{size}]" if size else ""),
               f"step {step}   {float(st[3]):.1f} m/s",
               f"CTE {abs(env.cte):.2f} m   row {100 * float(st[5]):.0f}%",
               env.instruction]
        if env.weather.get("rain_mm_h", 0) > 0:
            hud.append(f"rain {env.weather['rain_mm_h']:.1f} mm/h")
        if info and info.get("reason"):
            hud.append("done: " + ("SUCCESS" if env.metrics()["success"] else info["reason"]))
        front = obs.get("front")
        if front is None or not front.size:
            front = env.render_front(160)
        out.append(compose_frame(env.render_chase(width, height), front, hud))

    if frames:
        snap(0, None)
    for step in range(1, max_steps + 1):
        action = np.asarray(policy.act(obs), dtype=np.float32)
        obs, done, info = env.step(action)
        if frames:
            snap(step, info if done else None)
        if done:
            break
    if frames and out:  # hold the last frame for a second
        out.extend([out[-1]] * int(round(1.0 / env.DT)))
    return env.metrics(), out


def write_mp4(frames, path, fps: int = 10) -> Path:
    """H.264 yuv420p mp4 (plays everywhere). Odd frame sizes are padded to even."""
    import imageio.v2 as imageio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(path), fps=fps, codec="libx264", pixelformat="yuv420p",
                                macro_block_size=1, quality=8,
                                ffmpeg_params=["-movflags", "+faststart"])
    try:
        for f in frames:
            f = np.asarray(f, dtype=np.uint8)
            if f.ndim == 2:
                f = np.stack([f] * 3, axis=-1)
            h, w = f.shape[:2]
            if h % 2 or w % 2:
                f = np.pad(f, ((0, h % 2), (0, w % 2), (0, 0)), mode="edge")
            writer.append_data(f[..., :3])
    finally:
        writer.close()
    return path


def title_card(text, width: int = 640, height: int = 360, seconds: float = 1.5, fps: int = 10) -> list[np.ndarray]:
    """A dark slate with centred text (newlines split lines); first line is larger."""
    img = Image.new("RGB", (width, height), (18, 32, 22))
    draw = ImageDraw.Draw(img)
    lines = str(text).split("\n")
    fonts = [_font(max(14, height // 12))] + [_font(max(11, height // 22))] * (len(lines) - 1)
    heights = [int(getattr(f, "size", 14) * 1.35) for f in fonts]
    y = (height - sum(heights)) // 2
    for line, font, lh in zip(lines, fonts, heights):
        tw = draw.textlength(line, font=font)
        draw.text(((width - tw) / 2, y), line, font=font, fill=(235, 240, 230))
        y += lh
    draw.rectangle([0, height - 6, width, height], fill=(60, 160, 70))
    frame = np.asarray(img, dtype=np.uint8)
    return [frame] * max(1, int(round(seconds * fps)))
