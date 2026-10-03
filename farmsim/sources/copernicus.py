"""Copernicus fallback: GLO-30 DEM from the public AWS bucket, Sentinel-2 cloudless ortho.

UNTESTED against the live services (the build sandbox blocks them); tests
replace the network with fake files. The GLO-30 URL pattern is verified.

DEM: Copernicus DEM GLO-30 (30 m, a surface model, so it includes trees and
buildings) as 1x1 degree Cloud Optimized GeoTIFFs, no auth:

    https://copernicus-dem-30m.s3.amazonaws.com/
      Copernicus_DSM_COG_10_N59_00_E017_00_DEM/Copernicus_DSM_COG_10_N59_00_E017_00_DEM.tif

The tile name is the floor of the SW corner's latitude and longitude ("N59",
"E017"). Tiles are EPSG:4326 float32, DEFLATE with a floating-point predictor
(reading them needs the `imagecodecs` package). Whole tiles (about 30-50 MB)
are cached under $LOBBOT_CACHE (default ~/.cache/lobbot).

At 30 m a 300 m site is only about 10 source cells: use Lantmäteriet when you can.

Ortho: EOX Sentinel-2 cloudless 2024 (10 m, CC BY-NC-SA 4.0, non-commercial,
"Sentinel-2 cloudless - https://s2maps.eu by EOX IT Services GmbH"): WMTS
tiles in Web Mercator stitched and warped onto the SWEREF square, with a WMS
1.1.1 GetMap in EPSG:4326 as the fallback.
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
EOX_WMTS = os.environ.get(
    "EOX_WMTS_URL",
    "https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2024_3857/default/GoogleMapsCompatible/{z}/{y}/{x}.jpg")
EOX_WMS = os.environ.get("EOX_WMS_URL", "https://tiles.maps.eox.at/wms")
EOX_LAYER = os.environ.get("EOX_LAYER", "s2cloudless-2024")
WMS_MAX_PX = 2048
WMTS_MAX_ZOOM = 17
WMTS_MAX_TILES = 64
ATTRIBUTION = ("Copernicus DEM GLO-30 (c) DLR e.V. 2010-2014 and (c) Airbus 2014-2018, provided under "
               "COPERNICUS by the European Union and ESA; Sentinel-2 cloudless 2024 - https://s2maps.eu "
               "by EOX IT Services GmbH (contains modified Copernicus Sentinel data 2024), "
               "CC BY-NC-SA 4.0 (non-commercial)")
NOTE = ("DEM is 30 m (a 300 m site spans only about 10 source cells, so terrain is heavily "
        "smoothed); ortho is 10 m Sentinel-2. Use the lantmateriet source for 1 m data.")


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
    """Sentinel-2 cloudless from the WMTS tiles; falls back to the WMS on failure."""
    try:
        return ortho_wmts(box, n)
    except SourceError as wmts_err:
        try:
            return ortho_wms(box, n)
        except SourceError as wms_err:
            raise SourceError(f"EOX WMTS: {wmts_err}; WMS: {wms_err}") from wms_err


def _merc(lat, lon, z: int):
    """WGS84 -> global Web Mercator pixel coordinates at zoom z (256 px tiles)."""
    scale = 256 * 2 ** z
    x = (np.asarray(lon) + 180.0) / 360.0 * scale
    lr = np.radians(np.asarray(lat))
    y = (1.0 - np.log(np.tan(lr) + 1.0 / np.cos(lr)) / np.pi) / 2.0 * scale
    return x, y


def ortho_wmts(box: geo.Box, n: int) -> np.ndarray:
    """Stitch EOX WMTS tiles (GoogleMapsCompatible, EPSG:3857) and warp them onto the square."""
    min_lon, min_lat, max_lon, max_lat = box.wgs84_bounds()
    # zoom where the box spans roughly n pixels, capped (Sentinel-2 is 10 m, z16 is ~1.2 m/px at 60N)
    for z in range(WMTS_MAX_ZOOM, 0, -1):
        x0, y0 = _merc(max_lat, min_lon, z)
        x1, y1 = _merc(min_lat, max_lon, z)
        tiles = (int(x1 // 256) - int(x0 // 256) + 1) * (int(y1 // 256) - int(y0 // 256) + 1)
        if (x1 - x0) <= 1.5 * n and tiles <= WMTS_MAX_TILES:
            break
    tx0, ty0, tx1, ty1 = int(x0 // 256), int(y0 // 256), int(x1 // 256), int(y1 // 256)
    mosaic = np.zeros(((ty1 - ty0 + 1) * 256, (tx1 - tx0 + 1) * 256, 3), np.uint8)
    for ty in range(ty0, ty1 + 1):
        for tx in range(tx0, tx1 + 1):
            tile = decode_image(http_get(EOX_WMTS.format(z=z, x=tx, y=ty)))
            if tile.shape[:2] != (256, 256):
                raise SourceError(f"unexpected WMTS tile shape {tile.shape}")
            mosaic[(ty - ty0) * 256:(ty - ty0 + 1) * 256, (tx - tx0) * 256:(tx - tx0 + 1) * 256] = tile
    e, nn = box.cell_centres(n)
    lat, lon = geo.to_wgs84(e, nn)
    px, py = _merc(lat, lon, z)
    out = bilinear(mosaic, px - tx0 * 256 - 0.5, py - ty0 * 256 - 0.5)
    return np.clip(np.nan_to_num(out, nan=0.0), 0, 255).astype(np.uint8)


def ortho_wms(box: geo.Box, n: int) -> np.ndarray:
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
