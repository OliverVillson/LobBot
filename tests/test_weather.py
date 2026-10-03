"""SMHI weather, farmer summary and climate sampler, with recorded-shape fake responses."""

import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from farmsim import weather
from farmsim.site import build_site, load_site
from farmsim.sources.common import SourceError

LAT, LON = 59.8581, 17.6389
NOW = datetime(2026, 10, 3, 12, 0, tzinfo=timezone.utc)


def _forecast_doc(rain_hours=(), wind=2.5, start=NOW):
    """SNOW1g v1 shape: hourly steps for 48 h. rain_hours: hour offsets with 1 mm."""
    ts = []
    for h in range(1, 49):
        t = start + timedelta(hours=h)
        ts.append({"time": t.isoformat().replace("+00:00", "Z"),
                   "intervalParametersStartTime": (t - timedelta(hours=1)).isoformat().replace("+00:00", "Z"),
                   "data": {"air_temperature": 10.0 + (h % 24) / 4, "wind_from_direction": 214,
                            "wind_speed": wind(h) if callable(wind) else wind, "wind_speed_of_gust": 7.8,
                            "relative_humidity": 80, "cloud_area_fraction": 3,
                            "precipitation_amount_mean": 1.0 if h in rain_hours else 0.0}})
    return {"createdTime": "2026-10-03T11:00:00Z", "referenceTime": "2026-10-03T11:00:00Z",
            "geometry": {"type": "Point", "coordinates": [[LON, LAT]]}, "timeSeries": ts}


STATIONS = {"key": "5", "title": "Nederbördsmängd", "unit": "millimeter", "station": [
    {"key": "188790", "name": "Abisko Aut", "id": 188790, "latitude": 68.3538, "longitude": 18.8164,
     "active": True, "from": 486432000000, "to": 1791061200000},
    {"key": "97510", "name": "Uppsala Aut", "id": 97510, "latitude": 59.8471, "longitude": 17.6320,
     "active": True, "from": 486432000000, "to": 1791061200000},
    {"key": "97500", "name": "Uppsala (closed)", "id": 97500, "latitude": 59.858, "longitude": 17.639,
     "active": False, "from": 0, "to": 1000},
]}


def _obs_doc(last3=(6.0, 5.0, 7.0), n=100):
    vals = []
    d0 = NOW.date() - timedelta(days=n)
    for i in range(n):
        v = 4.0 if i % 3 == 0 else 0.0
        if i >= n - 3:
            v = last3[i - (n - 3)]
        day = (d0 + timedelta(days=i)).isoformat()
        vals.append({"from": 0, "to": 0, "ref": day, "value": f"{v:.1f}", "quality": "G"})
    return {"updated": 0, "parameter": {"key": "5"}, "station": {"key": "97510"},
            "period": {"key": "latest-months", "sampling": "24 timmar"}, "value": vals}


@pytest.fixture
def fake_smhi(tmp_path, monkeypatch):
    monkeypatch.setenv("LOBBOT_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("LOBBOT_WEATHER", "on")
    docs = {"fc": _forecast_doc(), "obs": _obs_doc()}
    calls = []

    def fake_get(url, params=None, timeout=None, **kw):
        calls.append(url)
        if "metfcst" in url:
            return json.dumps(docs["fc"]).encode()
        if url.endswith("/parameter/5.json"):
            return json.dumps(STATIONS).encode()
        if "/station/97510/" in url:
            return json.dumps(docs["obs"]).encode()
        raise SourceError(f"HTTP 404 for {url}")

    monkeypatch.setattr(weather, "http_get", fake_get)
    return docs, calls


def test_parse_forecast_and_spray_window():
    fc = weather.parse_forecast(_forecast_doc(rain_hours={1, 2, 3}))
    assert len(fc) == 48 and fc[0]["precip_mm"] == 1.0 and fc[0]["hours"] == 1.0
    s = weather.forecast_summary(fc, NOW)
    assert s["rain_24h_mm"] == 3.0 and s["wind_max_24h_ms"] == 2.5
    assert s["spray_window"] == "2026-10-03T15:00Z"   # first dry hour after the rain
    windy = weather.parse_forecast(_forecast_doc(wind=6.0))
    assert weather.forecast_summary(windy, NOW)["spray_window"] is None


def test_parse_old_pmp3g_shape():
    doc = {"timeSeries": [{"validTime": "2026-10-03T13:00:00Z", "parameters": [
        {"name": "t", "values": [9.5]}, {"name": "ws", "values": [3.0]}, {"name": "pmean", "values": [0.4]}]}]}
    (r,) = weather.parse_forecast(doc)
    assert r["temp_c"] == 9.5 and r["wind_ms"] == 3.0 and r["precip_mm"] == pytest.approx(0.4)


def test_nearest_active_station():
    st = weather.nearest_stations(STATIONS, LAT, LON)
    assert st[0]["id"] == 97510 and all(s["id"] != 97500 for s in st)


def test_farm_summary_soft_ground_and_cache(fake_smhi):
    docs, calls = fake_smhi
    dem = np.zeros((64, 64), np.float32)
    dem[:, 32:] = np.arange(32)[None, :] * 0.2   # east half ~11 deg slope at 1 m cells
    farm = weather.farm_summary(dem, 1.0, LAT, LON, now=NOW)
    assert "weather_error" not in farm
    assert farm["headline"].startswith("rain 0 mm next 24 h, wind up to 2.5 m/s")
    assert farm["observed"]["rain_last_3d_mm"] == 18.0 and farm["observed"]["station"]["id"] == 97510
    text = " ".join(farm["advisories"])
    assert "Soft ground" in text and "Spray window" in text and "steeper than 5" in text
    sc = farm["slope_classes"]
    assert abs(sc["flat_lt2_pct"] + sc["gentle_2to5_pct"] + sc["steep_gt5_pct"] - 100) < 0.5
    assert sc["steep_gt5_pct"] > 40
    c = farm["climate"]
    assert c["source"] == "smhi" and 0.3 < c["p_wet"] < 0.4 and c["months"]
    assert 10 < farm["daylight_h_today"] < 13   # early October, Uppsala
    n = len(calls)
    weather.farm_summary(dem, 1.0, LAT, LON, now=NOW)
    assert len(calls) == n   # served from $LOBBOT_CACHE


def test_dry_ground_advice(fake_smhi):
    docs, _ = fake_smhi
    docs["obs"] = _obs_doc(last3=(1.0, 0.0, 2.0))
    farm = weather.farm_summary(np.zeros((16, 16), np.float32), 1.0, LAT, LON, now=NOW)
    assert any("Ground should carry" in a for a in farm["advisories"])


def test_offline_never_fails_build(tmp_path, monkeypatch):
    monkeypatch.setenv("LOBBOT_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("LOBBOT_WEATHER", "on")

    def boom(*a, **k):
        raise SourceError("CONNECT tunnel failed, response 403")

    monkeypatch.setattr(weather, "http_get", boom)
    site = build_site(LAT, LON, 150, source="synthetic", sites_dir=tmp_path)
    farm = load_site(site.id, tmp_path).summary["farm"]
    assert "403" in farm["weather_error"]
    assert farm["climate"]["source"] == "default" and "slope_classes" in farm


def test_build_site_with_weather(tmp_path, fake_smhi):
    site = build_site(LAT, LON, 150, source="synthetic", sites_dir=tmp_path)
    farm = load_site(site.id, tmp_path).summary["farm"]
    assert farm["climate"]["source"] == "smhi" and farm["advisories"]


def test_sampler_deterministic_and_daylight():
    c = dict(weather.DEFAULT_CLIMATE, year=2026)
    a = [weather.sample_weather(np.random.default_rng(s), LAT, LON, c) for s in range(200)]
    b = [weather.sample_weather(np.random.default_rng(s), LAT, LON, c) for s in range(200)]
    assert a == b
    assert all(w["sun_elev_deg"] >= weather.MIN_SUN_ELEV for w in a)
    assert all(w["sun_elev_deg"] < 56 for w in a)        # never above the June noon sun at 59.9 N
    wet = sum(w["rain_mm_h"] > 0 for w in a) / len(a)
    assert 0.08 < wet < 0.3
    dry = weather.sample_weather(np.random.default_rng(1), LAT, LON, dict(c, p_wet=0.0))
    assert dry["rain_mm_h"] == 0.0


def test_env_uses_site_climate(tmp_path, monkeypatch):
    monkeypatch.setenv("LOBBOT_WEATHER", "off")
    from farmsim.env import FarmEnv

    site = build_site(LAT, LON, 150, source="synthetic", sites_dir=tmp_path)
    assert site.summary["farm"]["climate"]["source"] == "default"
    env = FarmEnv(site, seed=3, img_size=32)
    w1 = dict(env.weather)
    env.reset(3)
    assert env.weather == w1
    want = weather.sample_weather(np.random.default_rng([3, 104729]), LAT, LON, site.summary["farm"]["climate"])
    assert w1 == want
    env.close()
