"""Site building with the synthetic source, geo round trips and the sun."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pytest

from farmsim import geo
from farmsim.site import build_site, list_sites, load_site, site_id
from farmsim.sources import copernicus, lantmateriet, synthetic
from farmsim.sources.common import SourceError

LAT, LON = 59.8581, 17.6389  # Ultuna, Uppsala


def test_synthetic_build_and_load(tmp_path):
    site = build_site(LAT, LON, size_m=300, name="Ultuna", source="synthetic", sites_dir=tmp_path)
    assert site.id == site_id(LAT, LON, 300) and site.id.isalnum()
    assert site.source == "synthetic" and site.crs == "EPSG:3006"
    z = site.dem()
    assert z.dtype == np.float32 and z.shape == (300, 300)
    assert site.res_m == pytest.approx(1.0)
    assert 0.5 < z.max() - z.min() < 20
    for p in (site.ortho_path, site.preview_path, site.dir / "site.json"):
        assert p.exists()
    s = site.summary
    for k in ("elev_min_m", "elev_max_m", "elev_mean_m", "slope_mean_deg", "slope_max_deg",
              "sun_noon_elev_jun21_deg", "sun_noon_elev_today_deg", "area_ha", "source", "res_m"):
        assert k in s
    assert s["area_ha"] == pytest.approx(9.0)

    again = load_site(site.id, sites_dir=tmp_path)
    assert again.id == site.id and again.name == "Ultuna"
    assert again.origin_e == pytest.approx(site.origin_e)
    assert np.array_equal(again.dem(), z)
    assert load_site(site.dir).id == site.id
    meta = json.loads((site.dir / "site.json").read_text())
    assert meta["source_errors"] == {} and meta["dem_shape"] == [300, 300]


def test_synthetic_is_deterministic(tmp_path):
    box = geo.square_box(LAT, LON, 200)
    a = synthetic.fetch(box, 64, 128)
    b = synthetic.fetch(box, 64, 128)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert a[1].shape == (128, 128, 3) and a[1].dtype == np.uint8


def test_large_site_caps_grid(tmp_path):
    site = build_site(LAT, LON, size_m=1000, source="synthetic", sites_dir=tmp_path)
    assert site.dem().shape == (512, 512)
    assert site.res_m == pytest.approx(1000 / 512)


def test_list_sites(tmp_path):
    assert list_sites(tmp_path / "missing") == []
    build_site(LAT, LON, 200, source="synthetic", sites_dir=tmp_path)
    build_site(57.7, 11.97, 300, source="synthetic", sites_dir=tmp_path)
    (tmp_path / "junk").mkdir()
    ids = {s.id for s in list_sites(tmp_path)}
    assert ids == {site_id(LAT, LON, 200), site_id(57.7, 11.97, 300)}


def test_auto_falls_back_to_synthetic(tmp_path, monkeypatch):
    for k in ("LANTMATERIET_TOKEN", "LANTMATERIET_CONSUMER_KEY", "LANTMATERIET_USER"):
        monkeypatch.delenv(k, raising=False)

    def boom(*a, **k):
        raise SourceError("offline")

    monkeypatch.setattr(copernicus, "fetch", boom)
    site = build_site(LAT, LON, 150, source="auto", sites_dir=tmp_path)
    assert site.source == "synthetic"
    errs = json.loads((site.dir / "site.json").read_text())["source_errors"]
    assert set(errs) == {"lantmateriet", "copernicus"} and "offline" in errs["copernicus"]


def test_lantmateriet_needs_credentials(monkeypatch):
    for k in ("LANTMATERIET_TOKEN", "LANTMATERIET_CONSUMER_KEY", "LANTMATERIET_CONSUMER_SECRET",
              "LANTMATERIET_USER", "LANTMATERIET_PASSWORD"):
        monkeypatch.delenv(k, raising=False)
    assert not lantmateriet.has_credentials()
    with pytest.raises(SourceError, match="credentials"):
        lantmateriet.fetch(geo.square_box(LAT, LON, 100), 16, 16)


def test_copernicus_tile_name():
    assert copernicus.tile_name(59.86, 17.64) == "Copernicus_DSM_COG_10_N59_00_E017_00_DEM"
    assert copernicus.tile_url("X").endswith("/X/X.tif")


def test_geo_round_trip():
    lat = np.array([55.4, 59.8581, 63.8, 67.9])
    lon = np.array([13.0, 17.6389, 20.3, 20.2])
    e, n = geo.to_sweref(lat, lon)
    lat2, lon2 = geo.to_wgs84(e, n)
    e2, n2 = geo.to_sweref(lat2, lon2)
    assert np.max(np.hypot(e2 - e, n2 - n)) < 0.01
    # SWEREF 99 TM central meridian is 15 E with false easting 500 km
    e15, _ = geo.to_sweref(62.0, 15.0)
    assert abs(e15 - 500000.0) < 0.01


def test_square_box():
    box = geo.square_box(LAT, LON, 300)
    assert box.size_m == pytest.approx(300)
    assert box.north - box.south == pytest.approx(300)
    e, n = box.cell_centres(3)
    assert n[0, 0] > n[2, 0] and e[0, 2] > e[0, 0]  # row 0 north, col 0 west


def test_solar_elevation():
    elev = geo.noon_elevation(59.3293, 18.0686, datetime(2026, 6, 21, tzinfo=timezone.utc))
    assert 53.5 < elev < 54.8
    winter = geo.noon_elevation(59.3293, 18.0686, datetime(2026, 12, 21, tzinfo=timezone.utc))
    assert 6.5 < winter < 8.0
    el, az = geo.solar_position(59.3293, 18.0686, datetime(2026, 6, 21, 10, 50, tzinfo=timezone.utc))
    assert 170 < az < 190 and el > 53
    _, az_am = geo.solar_position(59.3293, 18.0686, datetime(2026, 6, 21, 6, 0))
    assert 60 < az_am < 120  # morning sun in the east


def test_dem_orientation(tmp_path, monkeypatch):
    """A DEM that rises northward must have its high values in row 0."""

    def north_ramp(box, n_dem, n_ortho):
        _, nn = box.cell_centres(n_dem)
        return (nn - box.south).astype(np.float32) * 0.01, np.zeros((n_ortho, n_ortho, 3), np.uint8)

    monkeypatch.setattr(synthetic, "fetch", north_ramp)
    site = build_site(LAT, LON, 200, source="synthetic", sites_dir=tmp_path)
    z = site.dem()
    assert z[0].mean() > z[-1].mean()
    assert np.allclose(z[:, 0], z[:, -1])


def _geotiff(data: np.ndarray, x0: float, y0: float, dx: float, dy: float) -> bytes:
    import io

    import tifffile

    buf = io.BytesIO()
    tags = [(33550, "d", 3, (dx, dy, 0.0)), (33922, "d", 6, (0.0, 0.0, 0.0, x0, y0, 0.0))]
    tifffile.imwrite(buf, data.astype(np.float32), extratags=tags)
    return buf.getvalue()


def test_copernicus_offline(tmp_path, monkeypatch):
    """Copernicus DEM sampling and ortho warp, with the network replaced by fake files."""
    import io

    from PIL import Image

    # a 1x1 degree tile whose value is latitude * 100, so north rows must be higher
    k = 360
    lat_c = 60 - (np.arange(k) + 0.5) / k
    tile = np.repeat((lat_c * 100)[:, None], k, axis=1)
    blob = _geotiff(tile, 17.0, 60.0, 1 / k, 1 / k)
    monkeypatch.setattr(copernicus, "_tile_blob", lambda name: blob)

    def fake_get(url, params=None, **kw):
        w, h = (int(params["WIDTH"]), int(params["HEIGHT"])) if params else (256, 256)
        buf = io.BytesIO()
        Image.new("RGB", (w, h), (10, 200, 30)).save(buf, "PNG")
        return buf.getvalue()

    monkeypatch.setattr(copernicus, "http_get", fake_get)
    box = geo.square_box(LAT, LON, 300)
    z, img = copernicus.fetch(box, 64, 128)
    assert z.shape == (64, 64) and np.isfinite(z).all()
    assert z[0].mean() > z[-1].mean()
    assert abs(z.mean() - LAT * 100) < 1.0
    assert img.shape == (128, 128, 3) and abs(int(img[64, 64, 1]) - 200) < 3


def test_lantmateriet_offline(tmp_path, monkeypatch):
    """STAC discovery, client-side bbox filter, two-tile mosaic and WMS, with fake responses."""
    import io

    from PIL import Image

    monkeypatch.setenv("LANTMATERIET_USER", "u")
    monkeypatch.setenv("LANTMATERIET_PASSWORD", "p")
    box = geo.square_box(LAT, LON, 300)
    mid_e = round((box.west + box.east) / 2)  # tile edge through the site
    min_lon, min_lat, max_lon, max_lat = box.wgs84_bounds()
    k = 2500
    # west tile: z = 10, east tile: z = 20; both 1 m cells, 2.5 km square, north edge above the site
    top = box.north + 1000
    tiles = {"w.tif": _geotiff(np.full((k, 1000), 10.0), mid_e - 1000, top, 1.0, 1.0),
             "e.tif": _geotiff(np.full((k, 1000), 20.0), mid_e, top, 1.0, 1.0)}
    item = lambda name, b: {"id": name, "bbox": b, "assets": {"data": {"href": "https://dl/" + name}}}
    seen = []

    def fake_get(url, params=None, headers=None, **kw):
        seen.append((url, headers))
        if url.endswith("/collections"):
            return json.dumps({"collections": [
                {"id": "mhm-65_6", "extent": {"spatial": {"bbox": [[min_lon - 1, min_lat - 1, max_lon + 1, max_lat + 1]]}}},
                {"id": "mhm-far", "extent": {"spatial": {"bbox": [[0, 0, 1, 1]]}}},
                {"id": "other", "extent": {}}]}).encode()
        if url.endswith("/search"):
            assert params["collections"] == "mhm-65_6"
            return json.dumps({"features": [
                item("w.tif", [min_lon - 0.01, min_lat - 0.01, (min_lon + max_lon) / 2 + 0.001, max_lat + 0.01]),
                item("e.tif", [(min_lon + max_lon) / 2 - 0.001, min_lat - 0.01, max_lon + 0.01, max_lat + 0.01]),
                item("far.tif", [0, 0, 1, 1])]}).encode()
        if url.startswith("https://dl/"):
            assert headers["Authorization"].startswith("Basic ")
            return tiles[url.rsplit("/", 1)[1]]
        assert params["CRS"] == "EPSG:3006" and params["BBOX"].startswith(f"{box.south}")
        buf = io.BytesIO()
        Image.new("RGB", (int(params["WIDTH"]), int(params["HEIGHT"])), (1, 2, 3)).save(buf, "PNG")
        return buf.getvalue()

    monkeypatch.setattr(lantmateriet, "http_get", fake_get)
    z, img = lantmateriet.fetch(box, 300, 64)
    assert not any("far.tif" in u for u, _ in seen)
    assert np.isfinite(z).all()
    assert z[:, 0].mean() == pytest.approx(10.0) and z[:, -1].mean() == pytest.approx(20.0)
    assert img.shape == (64, 64, 3)

    monkeypatch.setattr(copernicus, "fetch", lambda *a: (_ for _ in ()).throw(AssertionError("not reached")))
    site = build_site(LAT, LON, 300, source="auto", sites_dir=tmp_path)
    meta = json.loads((site.dir / "site.json").read_text())
    assert site.source == "lantmateriet" and meta["attribution"] == "© Lantmäteriet, CC BY 4.0"
