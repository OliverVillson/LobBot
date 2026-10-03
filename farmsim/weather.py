"""Weather for a site: SMHI forecast and observations, farmer advisories, a climate sampler.

    fc = forecast(59.858, 17.639)                 # next hours: temp_c, precip_mm, wind_ms
    days, station = recent_precip(59.858, 17.639) # last ~4 months of daily rain (mm)
    farm = farm_summary(dem, res_m, lat, lon)     # what site.json carries under summary["farm"]
    w = sample_weather(rng, lat, lon, farm["climate"])   # per-episode sim weather

Data is SMHI open data (no key, CC BY 4.0):
- point forecast, SNOW1g v1 (the 2025 successor of PMP3g v2):
  https://opendata-download-metfcst.smhi.se/api/category/snow1g/version/1/geotype/point/lon/{lon}/lat/{lat}/data.json
  timeSeries[] = {"time", "intervalParametersStartTime", "data": {"air_temperature", "wind_speed",
  "wind_speed_of_gust", "precipitation_amount_mean", ...}}
- observations, metobs v1.0, parameter 5 (daily precipitation, 06-06 UTC, mm):
  .../parameter/5.json lists stations (id, latitude, longitude, active);
  .../parameter/5/station/{id}/period/latest-months/data.json has value[] = {"ref": "YYYY-MM-DD", "value": "1.2"}

Responses are cached under $LOBBOT_CACHE/smhi (default ~/.cache/lobbot). A failed
request falls back to a stale cache entry when one exists. Nothing here raises
into build_site: farm_summary records weather_error instead. LOBBOT_WEATHER=off
skips the network entirely (climate falls back to a Swedish growing-season default).
"""

from __future__ import annotations

import json
import math
import os
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from farmsim import geo
from farmsim.sources.common import SourceError, http_get

FORECAST_URL = os.environ.get(
    "SMHI_FORECAST_URL",
    "https://opendata-download-metfcst.smhi.se/api/category/snow1g/version/1/geotype/point"
    "/lon/{lon}/lat/{lat}/data.json")
METOBS_BASE = os.environ.get("SMHI_METOBS_URL", "https://opendata-download-metobs.smhi.se/api/version/1.0")
PRECIP_DAILY = 5          # metobs parameter: precipitation, 24 h sum, once a day at 06 UTC
TIMEOUT_S = 8.0
ATTRIBUTION = "Weather: SMHI open data (CC BY 4.0), forecast SNOW1g and metobs"

# advisory thresholds
SPRAY_DRY_H = 6           # hours without rain needed for a spray window
SPRAY_MAX_WIND = 4.0      # m/s
SPRAY_RAIN_MM = 0.1       # mm per hour that still counts as dry
SOFT_GROUND_MM = 15.0     # rain over the last 3 days that makes the ground soft

# offline climate: a generic Swedish growing season (Apr-Sep), about 1 day in 3 wet,
# 5-6 mm on a wet day (SMHI normals for central Sweden are 40-75 mm per summer month)
DEFAULT_CLIMATE = {"source": "default", "p_wet": 0.35, "wet_mean_mm": 5.5, "months": [4, 5, 6, 7, 8, 9]}
WET_DAY_MM = 1.0
RAIN_HOURS_PER_WET_DAY = 5.0   # a wet day's rain falls over about this many hours
RAINING_SHARE = 0.5            # share of a wet day's daylight episodes that see rain
MIN_SUN_ELEV = 10.0


# ------------------------------------------------------------------ fetching
def enabled() -> bool:
    return os.environ.get("LOBBOT_WEATHER", "on").lower() not in ("0", "off", "false", "no")


def cache_dir() -> Path:
    d = Path(os.environ.get("LOBBOT_CACHE", Path.home() / ".cache" / "lobbot")) / "smhi"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _get_json(url: str, name: str, max_age_s: float) -> dict:
    """Fetch JSON with a file cache. A fresh entry skips the network; a stale one is the fallback."""
    path = cache_dir() / f"{name}.json"
    if path.exists() and time.time() - path.stat().st_mtime < max_age_s:
        try:
            return json.loads(path.read_text())
        except ValueError:
            pass
    try:
        body = http_get(url, timeout=TIMEOUT_S)
        doc = json.loads(body)
    except (SourceError, ValueError) as e:
        if path.exists():
            try:
                return json.loads(path.read_text())
            except ValueError:
                pass
        raise SourceError(f"SMHI: {e}") from e
    tmp = path.with_suffix(".part")
    tmp.write_text(json.dumps(doc))
    tmp.replace(path)
    return doc


def _t(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def parse_forecast(doc: dict) -> list[dict]:
    """SNOW1g JSON -> [{time, hours, temp_c, precip_mm, wind_ms, gust_ms, cloud}] in time order.

    precip_mm is the amount over the step's interval (intervalParametersStartTime .. time).
    Also reads the old PMP3g shape (parameters[] with name/values) so either product works.
    """
    out = []
    for e in doc.get("timeSeries", []):
        if "data" in e:
            d = e["data"]
        else:  # PMP3g v2: parameters = [{"name": "t", "values": [..]}]
            p = {q["name"]: q["values"][0] for q in e.get("parameters", [])}
            d = {"air_temperature": p.get("t"), "wind_speed": p.get("ws"), "wind_speed_of_gust": p.get("gust"),
                 "precipitation_amount_mean": p.get("pmean"), "cloud_area_fraction": p.get("tcc_mean")}
        t = _t(e["validTime"] if "validTime" in e else e["time"])
        start = _t(e["intervalParametersStartTime"]) if e.get("intervalParametersStartTime") else t - timedelta(hours=1)
        hours = max((t - start).total_seconds() / 3600.0, 1.0)
        pm = d.get("precipitation_amount_mean")
        if "data" not in e and pm is not None:
            pm = pm * hours  # PMP3g pmean is mm/h
        out.append({"time": t, "hours": hours,
                    "temp_c": d.get("air_temperature"),
                    "precip_mm": float(pm) if pm is not None else 0.0,
                    "wind_ms": d.get("wind_speed"),
                    "gust_ms": d.get("wind_speed_of_gust"),
                    "cloud": d.get("cloud_area_fraction")})
    out.sort(key=lambda r: r["time"])
    return out


def forecast(lat: float, lon: float) -> list[dict]:
    url = FORECAST_URL.format(lat=f"{lat:.4f}", lon=f"{lon:.4f}")
    return parse_forecast(_get_json(url, f"fc_{lat:.3f}_{lon:.3f}", 3600))


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin((p2 - p1) / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return 6371.0 * 2 * math.asin(math.sqrt(a))


def nearest_stations(doc: dict, lat: float, lon: float, k: int = 3) -> list[dict]:
    """The k nearest active stations of a metobs parameter listing, each with distance_km."""
    rows = []
    for s in doc.get("station", []):
        if not s.get("active") or s.get("latitude") is None:
            continue
        rows.append({"id": s["id"], "name": s.get("name", str(s["id"])),
                     "lat": s["latitude"], "lon": s["longitude"],
                     "distance_km": round(_haversine_km(lat, lon, s["latitude"], s["longitude"]), 1)})
    rows.sort(key=lambda r: r["distance_km"])
    return rows[:k]


def parse_daily(doc: dict) -> list[tuple[str, float]]:
    """metobs data.json -> [(YYYY-MM-DD, mm)] in date order, skipping missing values."""
    out = []
    for v in doc.get("value") or []:
        try:
            day = v.get("ref") or datetime.fromtimestamp(v["date"] / 1000, tz=timezone.utc).date().isoformat()
            out.append((day, float(v["value"])))
        except (KeyError, TypeError, ValueError):
            continue
    out.sort()
    return out


def recent_precip(lat: float, lon: float) -> tuple[list[tuple[str, float]], dict]:
    """Daily rain (mm) for the latest months from the nearest active SMHI station that has data."""
    listing = _get_json(f"{METOBS_BASE}/parameter/{PRECIP_DAILY}.json", f"stations_{PRECIP_DAILY}", 7 * 86400)
    errors = []
    for st in nearest_stations(listing, lat, lon):
        url = f"{METOBS_BASE}/parameter/{PRECIP_DAILY}/station/{st['id']}/period/latest-months/data.json"
        try:
            days = parse_daily(_get_json(url, f"obs_{PRECIP_DAILY}_{st['id']}", 6 * 3600))
        except SourceError as e:
            errors.append(str(e))
            continue
        if days:
            return days, st
    raise SourceError(f"no SMHI station with recent precipitation near {lat:.3f},{lon:.3f}: {errors}")


# ------------------------------------------------------------------ summarising
def forecast_summary(fc: list[dict], now: datetime | None = None) -> dict:
    """Headline numbers for the next 24 h and the first spray window in the next 48 h."""
    now = now or datetime.now(timezone.utc)
    nxt = [r for r in fc if now < r["time"] <= now + timedelta(hours=48)]
    day = [r for r in nxt if r["time"] <= now + timedelta(hours=24)]
    if not day:
        raise SourceError("forecast has no steps in the next 24 h")
    rain24 = sum(r["precip_mm"] for r in day)
    winds = [r["wind_ms"] for r in day if r["wind_ms"] is not None]
    temps = [r["temp_c"] for r in day if r["temp_c"] is not None]
    out = {"rain_24h_mm": round(rain24, 1),
           "rain_48h_mm": round(sum(r["precip_mm"] for r in nxt), 1),
           "wind_max_24h_ms": round(max(winds), 1) if winds else None,
           "temp_min_24h_c": round(min(temps), 1) if temps else None,
           "temp_max_24h_c": round(max(temps), 1) if temps else None,
           "spray_window": spray_window(nxt)}
    return out


def spray_window(steps: list[dict]) -> str | None:
    """Start (ISO UTC) of the first run of >= SPRAY_DRY_H hours with no rain and wind < SPRAY_MAX_WIND."""
    run_start, run_h = None, 0.0
    for r in steps:
        ok = (r["wind_ms"] is not None and r["wind_ms"] < SPRAY_MAX_WIND
              and r["precip_mm"] / r["hours"] < SPRAY_RAIN_MM)
        if ok:
            if run_start is None:
                run_start = r["time"] - timedelta(hours=r["hours"])
            run_h += r["hours"]
            if run_h >= SPRAY_DRY_H:
                return run_start.isoformat(timespec="minutes").replace("+00:00", "Z")
        else:
            run_start, run_h = None, 0.0
    return None


def climate_from_history(days: list[tuple[str, float]]) -> dict:
    """Wet-day probability and wet-day mean from daily rain, plus the months the record covers."""
    if len(days) < 14:
        return dict(DEFAULT_CLIMATE)
    mm = np.array([v for _, v in days])
    wet = mm >= WET_DAY_MM
    months = sorted({int(d[5:7]) for d, _ in days})
    return {"source": "smhi", "p_wet": round(float(wet.mean()), 3),
            "wet_mean_mm": round(float(mm[wet].mean()) if wet.any() else 0.0, 2),
            "months": months, "days": len(days), "from": days[0][0], "to": days[-1][0]}


def daylight_hours(lat: float, lon: float, day: date) -> float:
    """Hours with the sun above the horizon on a UTC date, sampled every 10 minutes."""
    base = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
    up = sum(geo.solar_position(lat, lon, base + timedelta(minutes=10 * k))[0] > 0.0 for k in range(144))
    return round(up / 6.0, 1)


def slope_classes(dem: np.ndarray, res_m: float) -> dict:
    """Percent of the area by slope: < 2 deg (flat), 2-5 deg (gentle), > 5 deg (steep)."""
    gy, gx = np.gradient(dem.astype(np.float64), res_m)
    s = np.degrees(np.arctan(np.hypot(gx, gy)))
    n = s.size
    return {"flat_lt2_pct": round(100.0 * float((s < 2).sum()) / n, 1),
            "gentle_2to5_pct": round(100.0 * float(((s >= 2) & (s <= 5)).sum()) / n, 1),
            "steep_gt5_pct": round(100.0 * float((s > 5).sum()) / n, 1)}


def advisories(fc_sum: dict | None, rain_3d: float | None, slopes: dict) -> list[str]:
    """Plain-words advice for the farmer."""
    out = []
    if fc_sum is not None:
        w = fc_sum["spray_window"]
        if w:
            out.append(f"Spray window from {w[:16].replace('T', ' ')} UTC: {SPRAY_DRY_H} h dry with wind "
                       f"under {SPRAY_MAX_WIND:g} m/s.")
        else:
            out.append(f"No spray window in the next 48 h (needs {SPRAY_DRY_H} h dry and wind under "
                       f"{SPRAY_MAX_WIND:g} m/s).")
        if fc_sum["rain_24h_mm"] >= 10:
            out.append(f"Heavy rain expected: {fc_sum['rain_24h_mm']:g} mm in the next 24 h.")
    if rain_3d is not None:
        if rain_3d > SOFT_GROUND_MM:
            out.append(f"Soft ground: {rain_3d:g} mm of rain in the last 3 days. Expect wheel slip and "
                       "compaction; wait before heavy machinery.")
        else:
            out.append(f"Ground should carry machinery: {rain_3d:g} mm of rain in the last 3 days.")
    if slopes["steep_gt5_pct"] >= 10:
        out.append(f"{slopes['steep_gt5_pct']:g}% of the field is steeper than 5 degrees: drive across the "
                   "slope with care and watch for erosion.")
    return out


def farm_summary(dem: np.ndarray, res_m: float, lat: float, lon: float,
                 now: datetime | None = None) -> dict:
    """Everything site.json carries under summary["farm"]. Never raises on weather trouble."""
    now = now or datetime.now(timezone.utc)
    slopes = slope_classes(dem, res_m)
    out: dict = {"slope_classes": slopes, "daylight_h_today": daylight_hours(lat, lon, now.date()),
                 "attribution": ATTRIBUTION}
    errors = []
    fc_sum = rain_3d = None
    climate = dict(DEFAULT_CLIMATE)
    if not enabled():
        errors.append("weather disabled (LOBBOT_WEATHER=off)")
    else:
        try:
            fc_sum = forecast_summary(forecast(lat, lon), now)
            out["forecast"] = fc_sum
            wind = fc_sum["wind_max_24h_ms"]
            out["headline"] = (f"rain {fc_sum['rain_24h_mm']:g} mm next 24 h"
                               + (f", wind up to {wind:g} m/s" if wind is not None else "")
                               + (f", {fc_sum['temp_min_24h_c']:g} to {fc_sum['temp_max_24h_c']:g} C"
                                  if fc_sum["temp_min_24h_c"] is not None else ""))
        except Exception as e:  # noqa: BLE001 - weather is optional
            errors.append(f"forecast: {e}")
        try:
            days, station = recent_precip(lat, lon)
            rain_3d = round(sum(v for _, v in days[-3:]), 1)
            out["observed"] = {"station": station, "rain_last_3d_mm": rain_3d,
                               "rain_last_30d_mm": round(sum(v for _, v in days[-30:]), 1),
                               "last_day": days[-1][0], "daily": days[-100:]}
            climate = climate_from_history(days)
        except Exception as e:  # noqa: BLE001
            errors.append(f"observations: {e}")
    climate.setdefault("year", now.year)
    out["climate"] = climate
    out["advisories"] = advisories(fc_sum, rain_3d, slopes)
    if errors:
        out["weather_error"] = "; ".join(errors)[:1000]
    return out


# ------------------------------------------------------------------ climate sampler
def sample_weather(rng: np.random.Generator, lat: float, lon: float, climate: dict | None = None) -> dict:
    """Per-episode {rain_mm_h, sun_elev_deg, sun_azim_deg} from the site's climate.

    Rain: an episode is wet with the wet-day probability times RAINING_SHARE; the rate is exponential around the wet-day mean spread over
    RAIN_HOURS_PER_WET_DAY hours. Sun: the solar position at a random daylight time
    (sun at least MIN_SUN_ELEV high) on a random day in the climate's months.
    Deterministic given the rng state.
    """
    c = climate or DEFAULT_CLIMATE
    p_wet = float(c.get("p_wet", DEFAULT_CLIMATE["p_wet"]))
    wet_mean = float(c.get("wet_mean_mm", DEFAULT_CLIMATE["wet_mean_mm"]))
    months = [int(m) for m in (c.get("months") or DEFAULT_CLIMATE["months"])]
    year = int(c.get("year", 2026))

    p_rain_now = min(1.0, p_wet * RAINING_SHARE)
    rain = 0.0
    if rng.random() < p_rain_now and wet_mean > 0:
        rain = float(np.clip(rng.exponential(wet_mean / RAIN_HOURS_PER_WET_DAY), 0.2, 12.0))

    elev, azim = 40.0, 180.0
    for _ in range(64):
        m = months[int(rng.integers(len(months)))]
        dim = (date(year + (m == 12), m % 12 + 1, 1) - date(year, m, 1)).days
        when = datetime(year, m, int(rng.integers(1, dim + 1)), tzinfo=timezone.utc) \
            + timedelta(minutes=float(rng.uniform(0, 24 * 60)))
        e, a = geo.solar_position(lat, lon, when)
        if e >= MIN_SUN_ELEV:
            elev, azim = e, a
            break
    return {"rain_mm_h": round(rain, 3), "sun_elev_deg": round(float(elev), 2), "sun_azim_deg": round(float(azim), 2)}
