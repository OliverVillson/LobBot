"""FarmEnv, expert, video and LeRobot export on a small synthetic site.

Run with MUJOCO_GL=osmesa where there is no GPU. Images and rows are kept
small so the suite stays fast on CPU rendering.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("MUJOCO_GL", "osmesa")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PIL import Image  # noqa: E402

pytest.importorskip("mujoco")

from farmsim.env import FarmEnv  # noqa: E402
from farmsim.expert import ExpertPolicy  # noqa: E402
from farmsim.video import rollout, title_card, write_mp4  # noqa: E402

SIZE_M = 100.0


def make_site(d: Path, size_m: float = SIZE_M, n: int = 101, relief: float = 1.0) -> SimpleNamespace:
    """Site-like stand-in: rolling DEM (row 0 north) and a striped ortho PNG."""
    yy, xx = np.mgrid[0:n, 0:n] / (n - 1) * size_m
    dem = (30 + relief * (3 * np.sin(xx / 25) + 2 * np.cos(yy / 30)) + 0.03 * xx).astype(np.float32)
    img = np.zeros((256, 256, 3), np.uint8)
    img[...] = (60, 130, 50)
    img[:, (np.arange(256) // 8) % 2 == 0] = (150, 110, 60)
    img[:24] = (230, 230, 230)  # north edge marker
    path = d / "ortho.png"
    Image.fromarray(img).save(path)
    return SimpleNamespace(id="test0", name="test", lat=59.8, lon=17.6, size_m=size_m,
                           res_m=size_m / (n - 1), source="synthetic", summary={}, dir=d,
                           dem=lambda: dem, ortho_path=path)


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    return make_site(tmp_path_factory.mktemp("site"))


def test_reset_step_shapes(site):
    env = FarmEnv(site, seed=1, img_size=32, row_length_m=20)
    try:
        obs = env.reset(1)
        assert obs["front"].shape == (32, 32, 3) and obs["front"].dtype == np.uint8
        assert obs["state"].shape == (6,) and obs["state"].dtype == np.float32
        assert isinstance(obs["instruction"], str) and obs["instruction"].startswith("drive row")
        assert obs["front"].std() > 5, "front cam rendered a uniform image"
        assert env.rows and all(r.ndim == 2 and r.shape[1] == 2 for r in env.rows)
        obs, done, info = env.step(np.array([0.0, 2.0], np.float32))
        assert not done and obs["state"][3] > 0
        chase = env.render_chase(160, 90)
        assert chase.shape == (90, 160, 3) and chase.std() > 5
        m = env.metrics()
        assert set(m) >= {"success", "cte_mean_m", "cte_max_m", "progress", "steps", "time_s"}
    finally:
        env.close()


def test_texture_maps_across_terrain(site):
    """Top-down render shows ortho stripes, not one flat colour."""
    import mujoco

    env = FarmEnv(site, seed=0, img_size=0, row_length_m=20)
    try:
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [0, 0, 0]
        cam.distance, cam.elevation, cam.azimuth = 120.0, -90.0, 90.0
        img = env.render_camera(cam, 96, 96).astype(float)
        assert img.std() > 15
        # columns alternate brown/green from the striped ortho
        assert img[48, :, 0].std() > 10
    finally:
        env.close()


def test_expert_success_rate(site):
    env = FarmEnv(site, seed=0, img_size=0, row_length_m=40)
    pol = ExpertPolicy(env)
    wins = 0
    for seed in range(10):
        obs = env.reset(seed)
        done = False
        while not done:
            obs, done, _ = env.step(pol.act(obs))
        wins += env.metrics()["success"]
    assert wins >= 10 * 0.95


def test_rain_slips_more(site):
    """Driving straight with no steering drifts further across the slope in rain."""
    drift = {}
    for rain in (0.0, 8.0):
        env = FarmEnv(site, seed=3, img_size=0, row_length_m=40,
                      weather={"rain_mm_h": rain, "sun_elev_deg": 40, "sun_azim_deg": 180})
        obs = env.reset(3)
        for _ in range(150):
            obs, done, _ = env.step(np.array([0.0, 2.0], np.float32))
            if done:
                break
        drift[rain] = abs(env.cte)
    assert drift[8.0] > drift[0.0]


def test_determinism(site):
    def run(seed):
        env = FarmEnv(site, seed=seed, img_size=0, row_length_m=20)
        pol = ExpertPolicy(env)
        obs = env.reset(seed)
        traj = [obs["state"].copy()]
        done = False
        while not done:
            obs, done, _ = env.step(pol.act(obs))
            traj.append(obs["state"].copy())
        return np.stack(traj), env.weather, env.instruction

    a, b, c = run(5), run(5), run(6)
    assert np.array_equal(a[0], b[0]) and a[1] == b[1] and a[2] == b[2]
    assert a[0].shape != c[0].shape or not np.array_equal(a[0], c[0])


def test_video_rollout_mp4(site, tmp_path):
    import imageio.v2 as imageio

    env = FarmEnv(site, seed=2, img_size=32, row_length_m=8)
    pol = ExpertPolicy()
    pol.size_label = "6.1 GB"
    try:
        metrics, frames = rollout(env, pol, seed=2, max_steps=60)
    finally:
        env.close()
    assert frames and frames[0].shape == (360, 640, 3)
    assert metrics["steps"] > 0
    frames = title_card("LobBot\nfarm demo", seconds=0.5) + frames
    path = write_mp4(frames, tmp_path / "clip.mp4", fps=10)
    reader = imageio.get_reader(str(path))
    try:
        got = [f for f in reader]
        assert len(got) == len(frames)
        assert got[0].shape == (360, 640, 3)
    finally:
        reader.close()


def test_lerobot_export(site, tmp_path):
    import pandas as pd

    out = tmp_path / "ds"
    stats = export_episodes_small(site, out)
    assert stats["episodes"] == 2 and stats["frames"] > 0
    for rel in ("meta/info.json", "meta/episodes.jsonl", "meta/tasks.jsonl", "meta/modality.json",
                "farm_tractor_config.py", "data/chunk-000/episode_000000.parquet",
                "data/chunk-000/episode_000001.parquet",
                "videos/chunk-000/observation.images.front/episode_000000.mp4"):
        assert (out / rel).exists(), rel
    df = pd.read_parquet(out / "data/chunk-000/episode_000001.parquet")
    for col in ("observation.state", "action", "timestamp", "frame_index", "episode_index", "index",
                "task_index", "annotation.human.task_description", "next.done"):
        assert col in df.columns, col
    assert len(df.iloc[0]["observation.state"]) == 6 and len(df.iloc[0]["action"]) == 2
    assert df["episode_index"].iloc[0] == 1 and df["index"].iloc[0] > 0
    info = json.loads((out / "meta/info.json").read_text())
    assert info["total_frames"] == stats["frames"] and info["fps"] == 10
    modality = json.loads((out / "meta/modality.json").read_text())
    assert list(modality["state"]) == ["x", "y", "heading", "speed", "steer", "row_progress"]
    assert list(modality["action"]) == ["steer", "speed"]
    assert modality["video"]["front"]["original_key"] == "observation.images.front"
    compile((out / "farm_tractor_config.py").read_text(), "farm_tractor_config.py", "exec")
    import imageio.v2 as imageio
    n_frames = sum(1 for _ in imageio.get_reader(str(out / "videos/chunk-000/observation.images.front/episode_000001.mp4")))
    assert n_frames == len(df)


def export_episodes_small(site, out):
    from farmsim.lerobot import export_episodes

    return export_episodes(site, None, 2, out, seed=0, img_size=32, row_length_m=8)


def test_real_synthetic_site(tmp_path):
    """The real farmsim.site synthetic source, when that module is present."""
    site_mod = pytest.importorskip("farmsim.site")
    try:
        s = site_mod.build_site(59.8581, 17.6389, size_m=120.0, source="synthetic", sites_dir=tmp_path)
    except Exception as e:  # noqa: BLE001 - another module, report but do not mask our tests
        pytest.skip(f"build_site failed: {e}")
    env = FarmEnv(s, seed=0, img_size=0, row_length_m=40)
    pol = ExpertPolicy(env)
    wins = 0
    for seed in range(5):
        obs = env.reset(seed)
        done = False
        while not done:
            obs, done, _ = env.step(pol.act(obs))
        wins += env.metrics()["success"]
    assert wins >= 4
