"""Synthetic site: deterministic rolling terrain and a field-like orthophoto.

Needs no network. The seed comes from the rounded centre lat/lon, so the same
pick always gives the same site. Terrain is a few sine waves of world position
(a few metres of relief over a few hundred metres) with a drainage ditch. The
ortho shows north-south crop rows inside a grass headland, a dirt track along
one edge and some trees.
"""

from __future__ import annotations

import hashlib

import numpy as np

from farmsim import geo

NAME = "synthetic"


def _rng(box: geo.Box) -> np.random.Generator:
    lat, lon = geo.to_wgs84((box.west + box.east) / 2, (box.north + box.south) / 2)
    key = f"{round(float(lat), 4):.4f},{round(float(lon), 4):.4f}".encode()
    return np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:8], "little"))


def _layout(box: geo.Box) -> dict:
    """Random but deterministic site layout shared by the DEM and the ortho."""
    rng = _rng(box)
    return {
        "base": float(rng.uniform(15.0, 60.0)),
        "waves": [(float(rng.uniform(200.0, 700.0)), float(rng.uniform(0, 2 * np.pi)),
                   float(rng.uniform(0.3, 1.2)), float(rng.uniform(0, 2 * np.pi))) for _ in range(4)],
        "tilt": (float(rng.normal(0, 0.006)), float(rng.normal(0, 0.006))),
        "ditch_angle": float(rng.uniform(-0.02, 0.02)),  # rad from east-west
        "ditch_offset": float(rng.uniform(0.01, 0.03)),  # fraction of size from the south edge
        "track_side": int(rng.integers(0, 2)),  # 0 west edge, 1 east edge
        "trees": rng.uniform(0, 1, size=(int(rng.integers(12, 30)), 3)),
        "row_spacing": float(rng.choice([0.75, 1.5, 3.0])),
    }


def _ditch_dist(e: np.ndarray, n: np.ndarray, box: geo.Box, lay: dict) -> np.ndarray:
    """Distance in metres from each point to the ditch line."""
    x = e - box.west
    y = n - box.south
    y_line = lay["ditch_offset"] * box.size_m + np.tan(lay["ditch_angle"]) * (x - box.size_m / 2)
    return np.abs(y - y_line) * np.cos(lay["ditch_angle"])


def dem(box: geo.Box, n: int) -> np.ndarray:
    e, nn = box.cell_centres(n)
    lay = _layout(box)
    z = np.full(e.shape, lay["base"], dtype=np.float64)
    for wl, ang, amp, ph in lay["waves"]:
        z += amp * np.sin(2 * np.pi * (np.cos(ang) * e + np.sin(ang) * nn) / wl + ph)
    z += lay["tilt"][0] * (e - box.west) + lay["tilt"][1] * (nn - box.south)
    z -= 1.0 * np.exp(-(_ditch_dist(e, nn, box, lay) / 3.5) ** 2)
    return z.astype(np.float32)


def ortho(box: geo.Box, n: int) -> np.ndarray:
    lay = _layout(box)
    rng = _rng(box)
    e, nn = box.cell_centres(n)
    x = e - box.west
    y = nn - box.south
    s = box.size_m
    px = s / n

    img = np.empty((n, n, 3), dtype=np.float32)
    img[:] = (92, 128, 60)  # grass

    margin = max(6.0, 0.06 * s)
    field = (x > margin) & (x < s - margin) & (y > margin) & (y < s - margin)
    spacing = max(lay["row_spacing"], 4 * px)
    phase = (x % spacing) / spacing
    crop = np.abs(phase - 0.5) < 0.25
    tram = np.abs((x - margin) % 24.0 - 12.0) < max(0.25, 0.6 * px)  # sprayer tramlines
    soil = np.array((112, 86, 58), np.float32)
    green = np.array((70, 120, 45), np.float32)
    img[field & crop] = green
    img[field & ~crop] = soil
    img[field & tram] = soil * 1.12

    track_x = margin / 2 if lay["track_side"] == 0 else s - margin / 2
    track = np.abs(x - track_x) < max(1.6, 1.5 * px)
    img[track] = (150, 128, 96)

    ditch = _ditch_dist(e, nn, box, lay) < max(1.0, 1.2 * px)
    img[ditch & ~track] = (58, 72, 52)

    for tx, ty, tr in lay["trees"]:
        # trees sit in the headland ring, not on the crop
        cx, cy = _to_headland(tx, ty, s, margin)
        r = 2.0 + 3.5 * tr
        d2 = (x - cx) ** 2 + (y - cy) ** 2
        canopy = d2 < r * r
        img[canopy] = np.array((38, 74, 34), np.float32) * (0.8 + 0.4 * np.sqrt(d2[canopy]) / r)[:, None]

    noise = rng.normal(0, 6.0, size=(n, n, 1)).astype(np.float32)
    patch = _smooth_noise(rng, n, 12) * 7.0
    img += noise + patch[..., None]
    return np.clip(img, 0, 255).astype(np.uint8)


def _to_headland(u: float, v: float, s: float, margin: float) -> tuple[float, float]:
    """Map a unit-square sample onto the headland ring around the field."""
    side = int(u * 4) % 4
    t = (u * 4) % 1.0 * s
    off = margin * (0.2 + 0.6 * v)
    return [(t, off), (t, s - off), (off, t), (s - off, t)][side]


def _smooth_noise(rng: np.random.Generator, n: int, k: int) -> np.ndarray:
    """Low-frequency noise: a k x k random grid upsampled bilinearly to n x n."""
    from farmsim.sources.common import bilinear

    g = rng.normal(0, 1, size=(k, k)).astype(np.float32)
    idx = np.linspace(0, k - 1, n)
    cc, rr = np.meshgrid(idx, idx)
    return bilinear(g, cc, rr)


def fetch(box: geo.Box, n_dem: int, n_ortho: int) -> tuple[np.ndarray, np.ndarray]:
    return dem(box, n_dem), ortho(box, n_ortho)
