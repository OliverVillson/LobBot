"""Copernicus fallback: GLO-30 DEM from the public AWS bucket, Sentinel-2 cloudless ortho.

UNTESTED against the live services (the build sandbox blocks them); the code
path is covered only up to the network call.

DEM: Copernicus DEM GLO-30 (30 m, a surface model, so it includes trees and
buildings) as 1x1 degree Cloud Optimized GeoTIFFs, no auth:

    https://copernicus-dem-30m.s3.amazonaws.com/
      Copernicus_DSM_COG_10_N59_00_E017_00_DEM/Copernicus_DSM_COG_10_N59_00_E017_00_DEM.tif

The tile name is the floor of the SW corner's latitude and longitude ("N59",
"E017"). Tiles are EPSG:4326 float32, DEFLATE with a floating-point predictor
(reading them needs the `imagecodecs` package). Whole tiles (about 30-50 MB)
are cached under $LOBBOT_CACHE (default ~/.cache/lobbot).

Ortho: EOX Sentinel-2 cloudless (10 m, CC BY-NC-SA 4.0 for 2018 and later,
"Sentinel-2 cloudless - https://s2maps.eu by EOX IT Services GmbH") via WMS
1.1.1 GetMap in EPSG:4326, then warped onto the SWEREF square.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np

from farmsim import geo
from farmsim.sources.common import (SourceError, bilinear, decode_image, fill_nan, http_get,
                                    read_geotiff)

NAME = "copernicus"
DEM_BUCKET = os.environ.get("COPERNICUS_DEM_URL", "https://copernicus-dem-30m.s3.amazonaws.com")
EOX_WMS = os.environ.get("EOX_WMS_URL", "https://tiles.maps.eox.at/wms")
EOX_LAYER = os.environ.get("EOX_LAYER", "s2cloudless-2023")
WMS_MAX_PX = 2048


def cache_dir() -> Path:
    d = Path(os.environ.get("LOBBOT_CACHE", Path.home() / ".cache" / "lobbot")) / "copernicus"
    d.mkdir(parents=True, exist_ok=True)
    return d


def tile_name(lat: float, lon: float) -> str:
    """GLO-30 tile name for the 1x1 degree tile that contains (lat, lon)."""
    la, lo = math.floor(lat), math.floor(lon)
    ns = f"N{la:02d}" if la >= 0 else f"S{-la:02d}"
    ew = f"E{lo:03d}" if lo >= 0 else f"W{-lo:03d}"
    return f"Copernicus_DSM_COG_10_{ns}_00_{ew}_00_DEM"


def tile_url(name: str) -> str:
    return f"{DEM_BUCKET}/{name}/{name}.tif"


def _tile_blob(name: str) -> bytes:
    path = cache_dir() / f"{name}.tif"
    if path.exists():
        return path.read_bytes()
    blob = http_get(tile_url(name), timeout=300)
    tmp = path.with_suffix(".part")
    tmp.write_bytes(blob)
    tmp.replace(path)
    return blob


def dem(box: geo.Box, n: int) -> np.ndarray:
    e, nn = box.cell_centres(n)
    lat, lon = geo.to_wgs84(e, nn)
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    out = np.full(e.shape, np.nan, dtype=np.float32)
    names = {tile_name(a, b) for a, b in zip(lat.ravel()[:: max(1, n // 8)], lon.ravel()[:: max(1, n // 8)])}
    names |= {tile_name(float(a), float(b)) for a, b in
              [(lat.min(), lon.min()), (lat.min(), lon.max()), (lat.max(), lon.min()), (lat.max(), lon.max())]}
    for name in sorted(names):
        r = read_geotiff(_tile_blob(name))
        vals = r.sample(lon, lat)
        out = np.where(np.isnan(out), vals, out)
    return fill_nan(out).astype(np.float32)


def ortho(box: geo.Box, n: int) -> np.ndarray:
    min_lon, min_lat, max_lon, max_lat = box.wgs84_bounds()
    # pad a little so bilinear sampling at the edges stays inside
    pad_lon = (max_lon - min_lon) * 0.02
    pad_lat = (max_lat - min_lat) * 0.02
    min_lon, max_lon, min_lat, max_lat = min_lon - pad_lon, max_lon + pad_lon, min_lat - pad_lat, max_lat + pad_lat
    aspect = (max_lon - min_lon) * math.cos(math.radians((min_lat + max_lat) / 2)) / (max_lat - min_lat)
    h = min(WMS_MAX_PX, n)
    w = min(WMS_MAX_PX, max(1, round(h * aspect)))
    params = {"SERVICE": "WMS", "VERSION": "1.1.1", "REQUEST": "GetMap", "LAYERS": EOX_LAYER,
              "STYLES": "", "SRS": "EPSG:4326", "BBOX": f"{min_lon},{min_lat},{max_lon},{max_lat}",
              "WIDTH": w, "HEIGHT": h, "FORMAT": "image/jpeg"}
    img = decode_image(http_get(EOX_WMS, params=params))
    return warp_lonlat_image(img, (min_lon, min_lat, max_lon, max_lat), box, n)


def warp_lonlat_image(img: np.ndarray, bounds: tuple[float, float, float, float],
                      box: geo.Box, n: int) -> np.ndarray:
    """Resample an EPSG:4326 north-up image with outer bounds onto the SWEREF square."""
    min_lon, min_lat, max_lon, max_lat = bounds
    h, w = img.shape[:2]
    e, nn = box.cell_centres(n)
    lat, lon = geo.to_wgs84(e, nn)
    col = (np.asarray(lon) - min_lon) / (max_lon - min_lon) * w - 0.5
    row = (max_lat - np.asarray(lat)) / (max_lat - min_lat) * h - 0.5
    out = bilinear(img, col, row)
    if np.isnan(out).all():
        raise SourceError("ortho image does not cover the site")
    return np.clip(np.nan_to_num(out, nan=0.0), 0, 255).astype(np.uint8)


def fetch(box: geo.Box, n_dem: int, n_ortho: int) -> tuple[np.ndarray, np.ndarray]:
    return dem(box, n_dem), ortho(box, n_ortho)
