"""Sites: a map pick turned into a square of elevation and orthophoto on disk.

    site = build_site(59.8581, 17.6389, size_m=300, name="Ultuna")
    z = site.dem()            # float32 [H, W] metres, row 0 north, col 0 west

A site lives in <sites_dir>/<id>/ as site.json, dem.npy, ortho.png and
preview.png. The square is north-up in SWEREF 99 TM (EPSG:3006); origin_e and
origin_n are the coordinates of its north-west corner. See docs/robot-mvp.md.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from farmsim import geo

SITES = Path(os.environ.get("LOBBOT_SITES", "/mnt/nvme/sites"))
MAX_DEM_CELLS = 512   # DEM grid side cap; 1 m cells up to 512 m, coarser above
ORTHO_PX = 1024       # ortho image side
PREVIEW_PX = 512
AUTO_ORDER = ("lantmateriet", "copernicus", "synthetic")


@dataclass
class Site:
    id: str
    name: str
    lat: float
    lon: float
    size_m: float
    crs: str
    origin_e: float
    origin_n: float
    res_m: float
    source: str
    summary: dict = field(default_factory=dict)
    dir: Path = Path(".")

    def dem(self) -> np.ndarray:
        """Elevation in metres, float32 [H, W]; row 0 = north, col 0 = west."""
        return np.load(self.dir / "dem.npy").astype(np.float32)

    @property
    def ortho_path(self) -> Path:
        return self.dir / "ortho.png"

    @property
    def preview_path(self) -> Path:
        return self.dir / "preview.png"

    def to_json(self) -> dict:
        d = asdict(self)
        d["dir"] = str(self.dir)
        return d


def _sites_dir(sites_dir) -> Path:
    return Path(sites_dir) if sites_dir is not None else SITES


def site_id(lat: float, lon: float, size_m: float) -> str:
    """Letters-and-digits id, e.g. s59p8581n17p6389z300 (lat, lon to 4 decimals, size in m)."""
    def enc(v: float) -> str:
        return f"{v:.4f}".replace("-", "m").replace(".", "p")
    return f"s{enc(lat)}n{enc(lon)}z{int(round(size_m))}"


def grid_cells(size_m: float) -> int:
    """DEM side in cells: about 1 m per cell, capped at MAX_DEM_CELLS."""
    return int(max(8, min(MAX_DEM_CELLS, math.ceil(size_m))))


def _source_module(name: str):
    if name == "lantmateriet":
        from farmsim.sources import lantmateriet as mod
    elif name == "copernicus":
        from farmsim.sources import copernicus as mod
    elif name == "synthetic":
        from farmsim.sources import synthetic as mod
    else:
        raise ValueError(f"unknown source {name!r}; use auto, {', '.join(AUTO_ORDER)}")
    return mod


def _fetch(source: str, box: geo.Box, n_dem: int) -> tuple[str, np.ndarray, np.ndarray, dict]:
    """Run one source, or the auto chain. Returns (winner, dem, ortho, errors of the losers)."""
    order = AUTO_ORDER if source == "auto" else (source,)
    errors: dict[str, str] = {}
    for name in order:
        mod = _source_module(name)
        if name == "lantmateriet" and source == "auto" and not mod.has_credentials():
            errors[name] = "skipped: no credentials in the environment"
            continue
        try:
            dem, ortho = mod.fetch(box, n_dem, ORTHO_PX)
        except Exception as e:  # noqa: BLE001 - every failure moves on to the next source
            if source != "auto":
                raise
            errors[name] = f"{type(e).__name__}: {e}"
            continue
        _check(dem, ortho, n_dem)
        return name, dem, ortho, errors
    raise RuntimeError(f"every source failed: {errors}")


def _check(dem: np.ndarray, ortho: np.ndarray, n_dem: int) -> None:
    if dem.shape != (n_dem, n_dem) or not np.isfinite(dem).all():
        raise ValueError(f"bad DEM: shape {dem.shape}, finite={np.isfinite(dem).all()}")
    if ortho.ndim != 3 or ortho.shape[2] != 3 or ortho.dtype != np.uint8:
        raise ValueError(f"bad ortho: shape {ortho.shape}, dtype {ortho.dtype}")


def slope_deg(dem: np.ndarray, res_m: float) -> np.ndarray:
    """Slope in degrees from the DEM gradient."""
    gy, gx = np.gradient(dem.astype(np.float64), res_m)
    return np.degrees(np.arctan(np.hypot(gx, gy)))


def hillshade(dem: np.ndarray, res_m: float, azim_deg: float = 315.0, elev_deg: float = 45.0) -> np.ndarray:
    """Lambertian hillshade in 0..1. Row 0 north, so +row is south."""
    gy, gx = np.gradient(dem.astype(np.float64), res_m)
    # surface normal with x east, y north: dz/dnorth = -gy (rows increase southward)
    nx, ny, nz = -gx, gy, np.ones_like(gx)
    norm = np.sqrt(nx * nx + ny * ny + nz * nz)
    az, el = math.radians(azim_deg), math.radians(elev_deg)
    sx, sy, sz = math.sin(az) * math.cos(el), math.cos(az) * math.cos(el), math.sin(el)
    return np.clip((nx * sx + ny * sy + nz * sz) / norm, 0.0, 1.0)


def summarize(dem: np.ndarray, res_m: float, lat: float, lon: float, size_m: float,
              source: str, today: datetime | None = None) -> dict:
    today = today or datetime.now(timezone.utc)
    slope = slope_deg(dem, res_m)
    jun21 = geo.noon_elevation(lat, lon, datetime(today.year, 6, 21, tzinfo=timezone.utc))
    return {
        "elev_min_m": round(float(dem.min()), 2),
        "elev_max_m": round(float(dem.max()), 2),
        "elev_mean_m": round(float(dem.mean()), 2),
        "relief_m": round(float(dem.max() - dem.min()), 2),
        "slope_mean_deg": round(float(slope.mean()), 2),
        "slope_max_deg": round(float(slope.max()), 2),
        "sun_noon_elev_deg": round(jun21, 1),
        "sun_noon_elev_jun21_deg": round(jun21, 1),
        "sun_noon_elev_today_deg": round(geo.noon_elevation(lat, lon, today), 1),
        "area_ha": round(size_m * size_m / 1e4, 2),
        "source": source,
        "res_m": round(res_m, 4),
    }


def _write_preview(dem: np.ndarray, ortho: np.ndarray, res_m: float, path: Path) -> None:
    from PIL import Image

    rgb = Image.fromarray(ortho).resize((PREVIEW_PX, PREVIEW_PX), Image.BILINEAR)
    hs = (hillshade(dem, res_m) * 255).astype(np.uint8)
    shade = Image.fromarray(hs).resize((PREVIEW_PX, PREVIEW_PX), Image.BILINEAR)
    a = np.asarray(rgb).astype(np.float32)
    s = np.asarray(shade).astype(np.float32)[..., None] / 255.0
    out = a * (0.45 + 0.75 * s)
    Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).save(path)


def build_site(lat: float, lon: float, size_m: float = 300.0, name: str = "", source: str = "auto",
               sites_dir=None) -> Site:
    """Fetch (or synthesize) a square site centred on lat/lon and write it to disk."""
    if not (100.0 <= size_m <= 5000.0):
        raise ValueError(f"size_m must be within 100..5000 m, got {size_m}")
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        raise ValueError(f"bad lat/lon {lat}, {lon}")
    box = geo.square_box(lat, lon, size_m)
    n = grid_cells(size_m)
    res = size_m / n
    winner, dem, ortho, errors = _fetch(source, box, n)

    sid = site_id(lat, lon, size_m)
    d = _sites_dir(sites_dir) / sid
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "dem.npy", dem.astype(np.float32))
    from PIL import Image

    Image.fromarray(ortho).save(d / "ortho.png")
    _write_preview(dem, ortho, res, d / "preview.png")

    site = Site(id=sid, name=name or sid, lat=float(lat), lon=float(lon), size_m=float(size_m),
                crs=geo.SWEREF99TM, origin_e=float(box.west), origin_n=float(box.north), res_m=float(res),
                source=winner, summary=summarize(dem, res, lat, lon, size_m, winner), dir=d)
    meta = site.to_json()
    meta.pop("dir")
    meta["source_requested"] = source
    meta["source_errors"] = errors
    meta["dem_shape"] = list(dem.shape)
    meta["ortho_shape"] = list(ortho.shape)
    meta["created"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    tmp = d / "site.json.tmp"
    tmp.write_text(json.dumps(meta, indent=2))
    tmp.replace(d / "site.json")
    return site


def load_site(id_or_dir, sites_dir=None) -> Site:
    """Load a site by id (under sites_dir) or by its directory path."""
    p = Path(id_or_dir)
    d = p if (p / "site.json").exists() else _sites_dir(sites_dir) / str(id_or_dir)
    meta_path = d / "site.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"no site at {d}")
    meta = json.loads(meta_path.read_text())
    keys = {f.name for f in fields(Site)} - {"dir"}
    return Site(**{k: v for k, v in meta.items() if k in keys}, dir=d)


def read_meta(site: Site) -> dict:
    """The full site.json dict (includes source_errors, created and shapes)."""
    meta = json.loads((site.dir / "site.json").read_text())
    meta["dir"] = str(site.dir)
    return meta


def list_sites(sites_dir=None) -> list[Site]:
    """Every readable site under sites_dir, newest first."""
    root = _sites_dir(sites_dir)
    if not root.is_dir():
        return []
    out = []
    for d in root.iterdir():
        if (d / "site.json").exists():
            try:
                out.append(load_site(d))
            except (ValueError, TypeError, OSError):
                continue
    out.sort(key=lambda s: (s.dir / "site.json").stat().st_mtime, reverse=True)
    return out
