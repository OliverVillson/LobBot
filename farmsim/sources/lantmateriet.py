"""Lantmäteriet: Markhöjdmodell (1 m grid DEM) and orthophoto for a SWEREF 99 TM square.

UNTESTED against the live API (the build sandbox blocks api.lantmateriet.se);
tests replace the network with fake responses. Every endpoint below can be
overridden by env var.

Elevation: the STAC catalogue at DEM_STAC_URL (anonymous) has one collection
per zone, mhm-<zone> (e.g. mhm-65_6): 1 m grid, 2.5 km COG tiles in EPSG:5845
(SWEREF 99 TM plane coordinates plus RH2000 heights). We list /collections, keep
the mhm-* ones whose extent meets the site, search each for items (GET with
bbox, falling back to POST), filter items by their own bbox client-side, then
download the GeoTIFF assets (auth required) and mosaic them onto the grid.

Ortho: one WMS 1.3.0 GetMap in EPSG:3006 for exactly the square. The STAC
image files are about 700 MB each, so they are not used.

Credentials, in order of preference:
- LANTMATERIET_USER + LANTMATERIET_PASSWORD: Geotorget account, HTTP Basic
- LANTMATERIET_TOKEN: a ready access token, sent as Bearer
- LANTMATERIET_CONSUMER_KEY + LANTMATERIET_CONSUMER_SECRET: OAuth2 client
  credentials exchanged at TOKEN_URL for a Bearer token (unverified)

Licence: CC BY 4.0, attribution "© Lantmäteriet, CC BY 4.0".
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
ATTRIBUTION = "© Lantmäteriet, CC BY 4.0"
NOTE = "Markhöjdmodell 1 m grid (RH2000) and Lantmäteriet orthophoto."

TOKEN_URL = os.environ.get("LANTMATERIET_TOKEN_URL", "https://apimanager.lantmateriet.se/oauth2/token")
DEM_STAC_URL = os.environ.get("LANTMATERIET_DEM_STAC_URL", "https://api.lantmateriet.se/stac-hojd/v1").rstrip("/")
DEM_COLLECTION_PREFIX = os.environ.get("LANTMATERIET_DEM_COLLECTION_PREFIX", "mhm-")
ORTHO_WMS_URL = os.environ.get("LANTMATERIET_ORTHO_WMS_URL", "https://maps.lantmateriet.se/ortofoto/wms/v1.3")
ORTHO_LAYER = os.environ.get("LANTMATERIET_ORTHO_LAYER", "Ortofoto_0.25")
# WMS 1.3.0 with EPSG:3006 uses northing,easting ("ne"); set "en" if the server wants x,y
ORTHO_AXIS_ORDER = os.environ.get("LANTMATERIET_ORTHO_AXIS_ORDER", "ne")
WMS_MAX_PX = 4096
SEARCH_LIMIT = 50

_token_cache: dict = {}


def has_credentials() -> bool:
    env = os.environ
    return bool((env.get("LANTMATERIET_USER") and env.get("LANTMATERIET_PASSWORD"))
                or env.get("LANTMATERIET_TOKEN")
                or (env.get("LANTMATERIET_CONSUMER_KEY") and env.get("LANTMATERIET_CONSUMER_SECRET")))


def auth_headers() -> dict:
    """Authorization header from the environment. Raises SourceError when none is set."""
    env = os.environ
    user, pw = env.get("LANTMATERIET_USER"), env.get("LANTMATERIET_PASSWORD")
    if user and pw:
        return {"Authorization": "Basic " + base64.b64encode(f"{user}:{pw}".encode()).decode()}
    if env.get("LANTMATERIET_TOKEN"):
        return {"Authorization": f"Bearer {env['LANTMATERIET_TOKEN']}"}
    key, secret = env.get("LANTMATERIET_CONSUMER_KEY"), env.get("LANTMATERIET_CONSUMER_SECRET")
    if key and secret:
        return {"Authorization": f"Bearer {_client_token(key, secret)}"}
    raise SourceError("Lantmäteriet credentials missing: set LANTMATERIET_USER and LANTMATERIET_PASSWORD "
                      "(Geotorget), or LANTMATERIET_TOKEN, or LANTMATERIET_CONSUMER_KEY and "
                      "LANTMATERIET_CONSUMER_SECRET")


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


def _json(body: bytes, what: str) -> dict:
    try:
        return json.loads(body)
    except ValueError as e:
        raise SourceError(f"bad {what} response: {body[:200]!r}") from e


def _overlaps(a: list[float], b: list[float]) -> bool:
    """Do two [min_x, min_y, max_x, max_y] boxes intersect?"""
    return a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]


def _collections(bbox: list[float]) -> list[str]:
    """mhm-* collection ids whose spatial extent meets the WGS84 bbox."""
    doc = _json(http_get(f"{DEM_STAC_URL}/collections", headers={"Accept": "application/json"}), "collections")
    out = []
    for c in doc.get("collections", []):
        cid = c.get("id", "")
        if not cid.startswith(DEM_COLLECTION_PREFIX):
            continue
        boxes = c.get("extent", {}).get("spatial", {}).get("bbox") or []
        if not boxes or any(_overlaps(list(b)[:4], bbox) for b in boxes):
            out.append(cid)
    if not out:
        raise SourceError(f"no {DEM_COLLECTION_PREFIX}* collection covers the site")
    return out


def _item_bbox_wgs84(item: dict) -> list[float] | None:
    b = item.get("bbox")
    if not b or len(b) < 4:
        return None
    b = [float(v) for v in (b[:4] if len(b) == 4 else [b[0], b[1], b[3], b[4]])]
    if abs(b[0]) > 360 or abs(b[1]) > 360:  # projected coordinates, convert the corners
        lat0, lon0 = geo.to_wgs84(b[0], b[1])
        lat1, lon1 = geo.to_wgs84(b[2], b[3])
        b = [float(lon0), float(lat0), float(lon1), float(lat1)]
    return b


def _search(collection: str, bbox: list[float]) -> list[dict]:
    """Items of one collection meeting the bbox. GET search first, POST if GET fails."""
    bbox_s = ",".join(f"{v:.6f}" for v in bbox)
    try:
        body = http_get(f"{DEM_STAC_URL}/search", headers={"Accept": "application/geo+json"},
                        params={"collections": collection, "bbox": bbox_s, "limit": SEARCH_LIMIT})
    except SourceError:
        query = {"collections": [collection], "bbox": bbox, "limit": SEARCH_LIMIT}
        body = http_get(f"{DEM_STAC_URL}/search", data=json.dumps(query).encode(),
                        headers={"Content-Type": "application/json", "Accept": "application/geo+json"})
    feats = _json(body, "STAC search").get("features", [])
    # the server may ignore bbox, so filter by each item's own bbox
    return [f for f in feats if (b := _item_bbox_wgs84(f)) is None or _overlaps(b, bbox)]


def find_dem_items(box: geo.Box) -> list[dict]:
    min_lon, min_lat, max_lon, max_lat = box.wgs84_bounds()
    bbox = [min_lon, min_lat, max_lon, max_lat]
    items = [it for c in _collections(bbox) for it in _search(c, bbox)]
    if not items:
        raise SourceError("no Markhöjdmodell tiles cover the site")
    return items


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
    """Mosaic of the 1 m tiles meeting the site, sampled onto the n x n grid."""
    headers = headers if headers is not None else auth_headers()
    e, nn = box.cell_centres(n)
    out = np.full(e.shape, np.nan, dtype=np.float32)
    for item in find_dem_items(box):
        r = read_geotiff(http_get(_tiff_href(item), headers=headers, timeout=300))
        out = np.where(np.isnan(out), r.sample(e, nn), out)
        if not np.isnan(out).any():
            break
    return fill_nan(out).astype(np.float32)


def ortho(box: geo.Box, n: int, headers: dict | None = None) -> np.ndarray:
    headers = headers if headers is not None else auth_headers()
    px = min(WMS_MAX_PX, n)
    if ORTHO_AXIS_ORDER == "ne":
        bbox = f"{box.south},{box.west},{box.north},{box.east}"
    else:
        bbox = f"{box.west},{box.south},{box.east},{box.north}"
    params = {"SERVICE": "WMS", "VERSION": "1.3.0", "REQUEST": "GetMap", "LAYERS": ORTHO_LAYER,
              "STYLES": "", "CRS": "EPSG:3006", "BBOX": bbox,
              "WIDTH": px, "HEIGHT": px, "FORMAT": "image/png"}
    img = decode_image(http_get(ORTHO_WMS_URL, params=params, headers=headers))
    return resize_rgb(img, n)


def fetch(box: geo.Box, n_dem: int, n_ortho: int) -> tuple[np.ndarray, np.ndarray]:
    headers = auth_headers()
    return dem(box, n_dem, headers), ortho(box, n_ortho, headers)
