"""Helpers shared by the data sources: HTTP, GeoTIFF reading and grid resampling.

Every source module exposes

    fetch(box: geo.Box, n_dem: int, n_ortho: int) -> tuple[np.ndarray, np.ndarray]

returning a float32 DEM [n_dem, n_dem] in metres and a uint8 RGB ortho
[n_ortho, n_ortho, 3], both north up and covering exactly `box`.

HTTP uses urllib only. urllib's default opener reads HTTPS_PROXY / HTTP_PROXY
from the environment; SSL_CERT_FILE or REQUESTS_CA_BUNDLE pick the CA bundle.
"""

from __future__ import annotations

import io
import os
import ssl
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

import numpy as np

USER_AGENT = "LobBot-farmsim/0.1"
TIMEOUT_S = 60.0


class SourceError(RuntimeError):
    """A source could not deliver data (no credentials, HTTP error, bad file)."""


def _ssl_context() -> ssl.SSLContext:
    cafile = os.environ.get("SSL_CERT_FILE") or os.environ.get("REQUESTS_CA_BUNDLE")
    return ssl.create_default_context(cafile=cafile if cafile and os.path.exists(cafile) else None)


def http_get(url: str, params: dict | None = None, headers: dict | None = None,
             data: bytes | None = None, timeout: float = TIMEOUT_S) -> bytes:
    """GET (or POST when data is given) and return the body. Raises SourceError."""
    if params:
        url = url + ("&" if "?" in url else "?") + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, data=data, headers={"User-Agent": USER_AGENT, **(headers or {})})
    opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=_ssl_context()))
    try:
        with opener.open(req, timeout=timeout) as resp:
            return resp.read()
    except urllib.error.HTTPError as e:
        body = e.read()[:300].decode("utf-8", "replace")
        raise SourceError(f"HTTP {e.code} for {url.split('?')[0]}: {body}") from e
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise SourceError(f"request failed for {url.split('?')[0]}: {e}") from e


@dataclass
class Raster:
    """A north-up raster with an affine pixel grid: pixel (r, c) centre at
    x = x0 + (c + 0.5) * dx, y = y0 - (r + 0.5) * dy (x0, y0 = outer top-left corner)."""

    data: np.ndarray
    x0: float
    y0: float
    dx: float
    dy: float
    nodata: float | None = None

    def sample(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Bilinear sample at map coordinates; NaN outside or on nodata."""
        return bilinear(self.data, (x - self.x0) / self.dx - 0.5, (self.y0 - y) / self.dy - 0.5, self.nodata)


def read_geotiff(blob: bytes) -> Raster:
    """Read a single-image GeoTIFF (first page, first band) with its pixel grid tags."""
    import tifffile

    try:
        with tifffile.TiffFile(io.BytesIO(blob)) as tif:
            page = tif.pages[0]
            data = page.asarray()
            tags = page.tags
            scale = tags["ModelPixelScaleTag"].value
            tie = tags["ModelTiepointTag"].value
            nodata_tag = tags.get("GDAL_NODATA")
            raster_type = 1
            geokeys = tags.get("GeoKeyDirectoryTag")
            if geokeys is not None:
                vals = list(geokeys.value)
                for k in range(4, len(vals) - 3, 4):
                    if vals[k] == 1025:  # GTRasterTypeGeoKey: 1 = PixelIsArea, 2 = PixelIsPoint
                        raster_type = vals[k + 3]
    except KeyError as e:
        raise SourceError(f"GeoTIFF is missing tag {e}") from e
    except Exception as e:  # tifffile raises many types on bad input
        raise SourceError(f"could not read GeoTIFF: {e}") from e
    if data.ndim == 3:
        data = data[..., 0] if data.shape[-1] <= 4 else data[0]
    dx, dy = float(scale[0]), float(scale[1])
    i, j, x, y = float(tie[0]), float(tie[1]), float(tie[3]), float(tie[4])
    x0, y0 = x - i * dx, y + j * dy
    if raster_type == 2:  # tie point is a pixel centre, shift to the outer corner
        x0, y0 = x0 - dx / 2, y0 + dy / 2
    nodata = None
    if nodata_tag is not None:
        try:
            nodata = float(str(nodata_tag.value).strip("\x00 "))
        except ValueError:
            nodata = None
    return Raster(data.astype(np.float32), x0, y0, dx, dy, nodata)


def bilinear(img: np.ndarray, col: np.ndarray, row: np.ndarray, nodata: float | None = None) -> np.ndarray:
    """Bilinear interpolation of a 2-D (or HxWxC) array at fractional pixel indices.

    Points outside the array, or touching a nodata pixel, come back as NaN.
    """
    h, w = img.shape[:2]
    a = img.astype(np.float32)
    if nodata is not None:
        a = np.where(a == nodata, np.nan, a)
    inside = (col >= -0.5) & (col <= w - 0.5) & (row >= -0.5) & (row <= h - 0.5)
    c = np.clip(col, 0, w - 1)
    r = np.clip(row, 0, h - 1)
    c0 = np.clip(np.floor(c).astype(int), 0, max(w - 2, 0))
    r0 = np.clip(np.floor(r).astype(int), 0, max(h - 2, 0))
    c1 = np.minimum(c0 + 1, w - 1)
    r1 = np.minimum(r0 + 1, h - 1)
    fc = c - c0
    fr = r - r0
    if a.ndim == 3:
        fc = fc[..., None]
        fr = fr[..., None]
    out = (a[r0, c0] * (1 - fc) * (1 - fr) + a[r0, c1] * fc * (1 - fr)
           + a[r1, c0] * (1 - fc) * fr + a[r1, c1] * fc * fr)
    if a.ndim == 3:
        out[~inside] = np.nan
    else:
        out = np.where(inside, out, np.nan)
    return out


def fill_nan(a: np.ndarray) -> np.ndarray:
    """Replace NaNs by the nearest valid value along rows, then columns. Raises if all NaN."""
    if np.isnan(a).all():
        raise SourceError("no valid data covers the site")
    out = a.copy()
    for axis in (1, 0):
        out = _ffill_bfill(out, axis)
    return out


def _ffill_bfill(a: np.ndarray, axis: int) -> np.ndarray:
    a = np.moveaxis(a, axis, -1).copy()
    for flip in (False, True):
        b = a[..., ::-1] if flip else a
        idx = np.where(~np.isnan(b), np.arange(b.shape[-1]), 0)
        np.maximum.accumulate(idx, axis=-1, out=idx)
        filled = np.take_along_axis(b, idx, axis=-1)
        b = np.where(np.isnan(b), filled, b)
        a = b[..., ::-1] if flip else b
    return np.moveaxis(a, -1, axis)


def decode_image(blob: bytes) -> np.ndarray:
    """PNG/JPEG bytes -> uint8 RGB array. Raises SourceError on a non-image (e.g. a WMS error XML)."""
    from PIL import Image

    try:
        return np.asarray(Image.open(io.BytesIO(blob)).convert("RGB"))
    except Exception as e:
        raise SourceError(f"not an image: {blob[:200]!r}") from e


def resize_rgb(img: np.ndarray, n: int) -> np.ndarray:
    """Resize an RGB array to n x n with Pillow."""
    from PIL import Image

    if img.shape[:2] == (n, n):
        return img
    return np.asarray(Image.fromarray(img).resize((n, n), Image.BILINEAR))
