"""Coordinates and sun: WGS84 <-> SWEREF 99 TM, square site boxes, solar position.

SWEREF 99 TM (EPSG:3006) is the Swedish national grid: easting and northing in
metres. Every site is a square in that grid, so DEM cells are square metres.
The solar position is the NOAA spreadsheet algorithm (about 0.01 deg accurate
for current dates), with no dependencies beyond the standard library.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache

import numpy as np
from pyproj import Transformer

WGS84 = "EPSG:4326"
SWEREF99TM = "EPSG:3006"


@lru_cache(maxsize=None)
def _transformer(src: str, dst: str) -> Transformer:
    return Transformer.from_crs(src, dst, always_xy=True)


def to_sweref(lat, lon):
    """WGS84 lat/lon (deg) -> SWEREF 99 TM (easting, northing) in metres. Accepts arrays."""
    e, n = _transformer(WGS84, SWEREF99TM).transform(lon, lat)
    return e, n


def to_wgs84(e, n):
    """SWEREF 99 TM (easting, northing) -> WGS84 (lat, lon) in degrees. Accepts arrays."""
    lon, lat = _transformer(SWEREF99TM, WGS84).transform(e, n)
    return lat, lon


@dataclass(frozen=True)
class Box:
    """A north-up square in SWEREF 99 TM. (west, north) is the NW corner."""

    west: float
    south: float
    east: float
    north: float

    @property
    def size_m(self) -> float:
        return self.east - self.west

    def cell_centres(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Easting and northing grids [n, n] of cell centres; row 0 north, col 0 west."""
        res = self.size_m / n
        e = self.west + (np.arange(n) + 0.5) * res
        nn = self.north - (np.arange(n) + 0.5) * res
        return np.meshgrid(e, nn)

    def wgs84_bounds(self) -> tuple[float, float, float, float]:
        """(min_lon, min_lat, max_lon, max_lat) enclosing the square, from a dense edge sample."""
        t = np.linspace(0.0, 1.0, 21)
        es = np.concatenate([self.west + t * self.size_m, np.full(21, self.east),
                             self.west + t * self.size_m, np.full(21, self.west)])
        ns = np.concatenate([np.full(21, self.north), self.south + t * self.size_m,
                             np.full(21, self.south), self.south + t * self.size_m])
        lat, lon = to_wgs84(es, ns)
        return float(np.min(lon)), float(np.min(lat)), float(np.max(lon)), float(np.max(lat))


def square_box(lat: float, lon: float, size_m: float) -> Box:
    """Square of side size_m (m) centred on a WGS84 point, in SWEREF 99 TM."""
    e, n = to_sweref(lat, lon)
    h = size_m / 2.0
    return Box(west=e - h, south=n - h, east=e + h, north=n + h)


def solar_position(lat: float, lon: float, when: datetime) -> tuple[float, float]:
    """Sun (elevation_deg, azimuth_deg) at a place and time. Azimuth is clockwise from north.

    A naive datetime is taken as UTC. Elevation includes a simple refraction term.
    """
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    when = when.astimezone(timezone.utc)
    jd = when.timestamp() / 86400.0 + 2440587.5
    t = (jd - 2451545.0) / 36525.0

    l0 = (280.46646 + t * (36000.76983 + 0.0003032 * t)) % 360.0
    m = 357.52911 + t * (35999.05029 - 0.0001537 * t)
    ecc = 0.016708634 - t * (0.000042037 + 0.0000001267 * t)
    mr = math.radians(m)
    c = (math.sin(mr) * (1.914602 - t * (0.004817 + 0.000014 * t))
         + math.sin(2 * mr) * (0.019993 - 0.000101 * t) + math.sin(3 * mr) * 0.000289)
    true_long = l0 + c
    omega = 125.04 - 1934.136 * t
    app_long = true_long - 0.00569 - 0.00478 * math.sin(math.radians(omega))
    eps0 = 23.0 + (26.0 + (21.448 - t * (46.815 + t * (0.00059 - t * 0.001813))) / 60.0) / 60.0
    eps = math.radians(eps0 + 0.00256 * math.cos(math.radians(omega)))
    decl = math.asin(math.sin(eps) * math.sin(math.radians(app_long)))

    y = math.tan(eps / 2) ** 2
    l0r = math.radians(l0)
    eq_time = 4 * math.degrees(y * math.sin(2 * l0r) - 2 * ecc * math.sin(mr)
                               + 4 * ecc * y * math.sin(mr) * math.cos(2 * l0r)
                               - 0.5 * y * y * math.sin(4 * l0r) - 1.25 * ecc * ecc * math.sin(2 * mr))
    minutes = when.hour * 60 + when.minute + when.second / 60.0 + when.microsecond / 6e7
    tst = (minutes + eq_time + 4 * lon) % 1440.0
    ha = math.radians(tst / 4.0 - 180.0)

    latr = math.radians(lat)
    cos_zen = math.sin(latr) * math.sin(decl) + math.cos(latr) * math.cos(decl) * math.cos(ha)
    zen = math.acos(max(-1.0, min(1.0, cos_zen)))
    elev = 90.0 - math.degrees(zen)
    az = math.degrees(math.atan2(math.sin(ha), math.cos(ha) * math.sin(latr) - math.tan(decl) * math.cos(latr)))
    az = (az + 180.0) % 360.0
    return elev + _refraction(elev), az


def _refraction(elev: float) -> float:
    """Atmospheric refraction correction in degrees (NOAA approximation)."""
    if elev > 85.0:
        return 0.0
    te = math.tan(math.radians(elev))
    if elev > 5.0:
        r = 58.1 / te - 0.07 / te ** 3 + 0.000086 / te ** 5
    elif elev > -0.575:
        r = 1735.0 + elev * (-518.2 + elev * (103.4 + elev * (-12.79 + elev * 0.711)))
    else:
        r = -20.774 / te
    return r / 3600.0


def noon_elevation(lat: float, lon: float, day: datetime) -> float:
    """Highest sun elevation (deg) on the UTC date of `day`, sampled every 5 minutes."""
    base = datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp()
    best = -90.0
    for k in range(0, 24 * 12):
        when = datetime.fromtimestamp(base + k * 300, tz=timezone.utc)
        best = max(best, solar_position(lat, lon, when)[0])
    return best
