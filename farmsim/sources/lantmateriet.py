"""Lantmäteriet: Markhöjdmodell (1 m grid DEM) and orthophoto for a SWEREF 99 TM square.

UNTESTED against the live API (the build sandbox blocks api.lantmateriet.se).
Endpoints are constants below and every one can be overridden by env var, so
they can be corrected without a code change.

Credentials, in order of preference:
- LANTMATERIET_TOKEN: a ready OAuth2 access token (sent as Bearer)
- LANTMATERIET_CONSUMER_KEY + LANTMATERIET_CONSUMER_SECRET: OAuth2 client
  credentials, exchanged at TOKEN_URL for a Bearer token
- LANTMATERIET_USER + LANTMATERIET_PASSWORD: Geotorget account, sent as HTTP
  Basic (the STAC download services use this)

Elevation: a STAC search over DEM_STAC_URL for items intersecting the site,
then each item's GeoTIFF asset (EPSG:3006) is downloaded and sampled onto the
grid. Ortho: one WMS 1.3.0 GetMap in EPSG:3006 for exactly the square.
"""

from __future__ import annotations

import base64
import json
import os
import time

import numpy as np

from farmsim import geo
from farmsim.sources.common import SourceError, decode_image, fill_nan, http_get, read_geotiff, resize_rgb

NAME = "lantmateriet"

TOKEN_URL = os.environ.get("LANTMATERIET_TOKEN_URL", "https://apimanager.lantmateriet.se/oauth2/token")
DEM_STAC_URL = os.environ.get("LANTMATERIET_DEM_STAC_URL", "https://api.lantmateriet.se/stac-hojd/v1")
DEM_COLLECTION = os.environ.get("LANTMATERIET_DEM_COLLECTION", "mhm-grid1")
ORTHO_WMS_URL = os.environ.get("LANTMATERIET_ORTHO_WMS_URL",
                               "https://api.lantmateriet.se/ortofoto/v1/wms")
ORTHO_LAYER = os.environ.get("LANTMATERIET_ORTHO_LAYER", "Ortofoto_0.5")
WMS_MAX_PX = 4096
MAX_ITEMS = 16

_token_cache: dict = {}


def has_credentials() -> bool:
    env = os.environ
    return bool(env.get("LANTMATERIET_TOKEN")
                or (env.get("LANTMATERIET_CONSUMER_KEY") and env.get("LANTMATERIET_CONSUMER_SECRET"))
                or (env.get("LANTMATERIET_USER") and env.get("LANTMATERIET_PASSWORD")))


def auth_headers() -> dict:
    """Authorization header from the environment. Raises SourceError when none is set."""
    env = os.environ
    if env.get("LANTMATERIET_TOKEN"):
        return {"Authorization": f"Bearer {env['LANTMATERIET_TOKEN']}"}
    key, secret = env.get("LANTMATERIET_CONSUMER_KEY"), env.get("LANTMATERIET_CONSUMER_SECRET")
    if key and secret:
        return {"Authorization": f"Bearer {_client_token(key, secret)}"}
    user, pw = env.get("LANTMATERIET_USER"), env.get("LANTMATERIET_PASSWORD")
    if user and pw:
        return {"Authorization": "Basic " + base64.b64encode(f"{user}:{pw}".encode()).decode()}
    raise SourceError("Lantmäteriet credentials missing: set LANTMATERIET_TOKEN, or "
                      "LANTMATERIET_CONSUMER_KEY and LANTMATERIET_CONSUMER_SECRET, or "
                      "LANTMATERIET_USER and LANTMATERIET_PASSWORD")


def _client_token(key: str, secret: str) -> str:
    cached = _token_cache.get(key)
    if cached and cached[1] > time.time() + 30:
        return cached[0]
    basic = base64.b64encode(f"{key}:{secret}".encode()).decode()
    body = http_get(TOKEN_URL, data=b"grant_type=client_credentials",
                    headers={"Authorization": f"Basic {basic}",
                             "Content-Type": "application/x-www-form-urlencoded"})
    try:
        doc = json.loads(body)
        token = doc["access_token"]
    except (ValueError, KeyError) as e:
        raise SourceError(f"bad token response from {TOKEN_URL}: {body[:200]!r}") from e
    _token_cache[key] = (token, time.time() + float(doc.get("expires_in", 3600)))
    return token


def _stac_items(box: geo.Box, headers: dict) -> list[dict]:
    min_lon, min_lat, max_lon, max_lat = box.wgs84_bounds()
    query = {"collections": [DEM_COLLECTION], "bbox": [min_lon, min_lat, max_lon, max_lat], "limit": MAX_ITEMS}
    body = http_get(f"{DEM_STAC_URL}/search", data=json.dumps(query).encode(),
                    headers={**headers, "Content-Type": "application/json", "Accept": "application/geo+json"})
    try:
        feats = json.loads(body).get("features", [])
    except ValueError as e:
        raise SourceError(f"bad STAC response: {body[:200]!r}") from e
    if not feats:
        raise SourceError(f"no {DEM_COLLECTION} items cover the site")
    return feats


def _tiff_href(item: dict) -> str:
    assets = item.get("assets", {})
    for key in ("data", "dem", "elevation"):
        if key in assets:
            return assets[key]["href"]
    for a in assets.values():
        if "tiff" in a.get("type", "") or a.get("href", "").lower().endswith((".tif", ".tiff")):
            return a["href"]
    raise SourceError(f"STAC item {item.get('id')} has no GeoTIFF asset")


def dem(box: geo.Box, n: int, headers: dict | None = None) -> np.ndarray:
    headers = headers if headers is not None else auth_headers()
    e, nn = box.cell_centres(n)
    out = np.full(e.shape, np.nan, dtype=np.float32)
    for item in _stac_items(box, headers):
        r = read_geotiff(http_get(_tiff_href(item), headers=headers, timeout=300))
        out = np.where(np.isnan(out), r.sample(e, nn), out)
    return fill_nan(out).astype(np.float32)


def ortho(box: geo.Box, n: int, headers: dict | None = None) -> np.ndarray:
    headers = headers if headers is not None else auth_headers()
    px = min(WMS_MAX_PX, n)
    # WMS 1.3.0 with EPSG:3006 uses northing,easting axis order in BBOX
    params = {"SERVICE": "WMS", "VERSION": "1.3.0", "REQUEST": "GetMap", "LAYERS": ORTHO_LAYER,
              "STYLES": "", "CRS": "EPSG:3006",
              "BBOX": f"{box.south},{box.west},{box.north},{box.east}",
              "WIDTH": px, "HEIGHT": px, "FORMAT": "image/png"}
    img = decode_image(http_get(ORTHO_WMS_URL, params=params, headers=headers))
    return resize_rgb(img, n)


def fetch(box: geo.Box, n_dem: int, n_ortho: int) -> tuple[np.ndarray, np.ndarray]:
    headers = auth_headers()
    return dem(box, n_dem, headers), ortho(box, n_ortho, headers)
