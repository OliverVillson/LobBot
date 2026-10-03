"""FarmEnv: a fast tractor row-driving sim on a real (or synthetic) site.

The scene is MuJoCo MJCF built in code: a heightfield from the site DEM,
textured with the orthophoto, plus a mocap tractor with a cab camera. MuJoCo
is used only for rendering; the vehicle is a kinematic bicycle model in
Python (no contact physics), which is fast, deterministic and robust.

Frames: local metres, x east and y north from the SW corner of the site.
MuJoCo world = local shifted so the site centre is the origin, z = DEM - min.
heading_rad is measured counter-clockwise from east (pi/2 = north).
"""

from __future__ import annotations

import math
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

WHEELBASE_M = 2.5
STEER_MAX = 0.6
SPEED_MAX = 3.0
TAU_STEER_S = 0.25
TAU_SPEED_S = 0.6
ROW_SPACING_M = 6.0
ROW_STEP_M = 1.0
SLIP_K = 0.25  # lateral slip speed per unit cross-grade per unit speed, dry
CTE_LIMIT_M = 3.0
SUCCESS_CTE_MEAN_M = 0.5
SUCCESS_CTE_MAX_M = 1.5
MAX_HFIELD_CELLS = 512  # per side; larger DEMs are subsampled for rendering only


def _bilinear(grid: np.ndarray, r: float, c: float) -> float:
    h, w = grid.shape
    r = min(max(r, 0.0), h - 1.0)
    c = min(max(c, 0.0), w - 1.0)
    r0, c0 = int(r), int(c)
    r1, c1 = min(r0 + 1, h - 1), min(c0 + 1, w - 1)
    fr, fc = r - r0, c - c0
    top = grid[r0, c0] * (1 - fc) + grid[r0, c1] * fc
    bot = grid[r1, c0] * (1 - fc) + grid[r1, c1] * fc
    return float(top * (1 - fr) + bot * fr)


def _quat_from_ypr(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """MuJoCo quat (w, x, y, z) of R = Rz(yaw) Ry(-pitch) Rx(roll); pitch > 0 = nose up."""
    q = np.zeros(4)
    mat = np.zeros(9)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(-pitch), math.sin(-pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    mat[:] = (rz @ ry @ rx).ravel()
    mujoco.mju_mat2Quat(q, mat)
    return q


def sample_weather(rng: np.random.Generator) -> dict:
    rain = 0.0 if rng.random() < 0.6 else float(rng.uniform(0.5, 8.0))
    return {"rain_mm_h": rain,
            "sun_elev_deg": float(rng.uniform(15.0, 60.0)),
            "sun_azim_deg": float(rng.uniform(90.0, 270.0))}


class FarmEnv:
    """Drive one field row per episode. See docs/robot-mvp.md for the contract.

    Extra keyword arguments beyond the contract (all optional):
      row_length_m: cap on each row's length (None = full field minus margins)
      row_spacing_m, margin_m: row layout
      max_steps: episode step budget (default row length / 1.5 m/s * 1.5)
    """

    DT = 0.1

    def __init__(self, site, seed: int = 0, img_size: int = 224, weather: dict | None = None,
                 row_length_m: float | None = 80.0, row_spacing_m: float = ROW_SPACING_M,
                 margin_m: float | None = None, max_steps: int | None = None):
        self.site = site
        self.seed = seed
        self.img_size = int(img_size)
        self.fixed_weather = dict(weather) if weather else None
        self.size_m = float(site.size_m)
        dem = np.asarray(site.dem(), dtype=np.float32)
        self.dem = dem
        self.z_min = float(dem.min())
        self._z = dem - self.z_min  # row 0 north
        self._h, self._w = dem.shape
        # gradients in local frame (per metre): d/dx east, d/dy north
        dr = self.size_m / max(self._h - 1, 1)
        dc = self.size_m / max(self._w - 1, 1)
        gr, gc = np.gradient(self._z.astype(np.float64), dr, dc)
        self._gx = gc
        self._gy = -gr  # row index increases southward
        self.rows = self._make_rows(row_spacing_m, margin_m, row_length_m)
        row_len = float(np.max([self._row_len(r) for r in self.rows]))
        self.max_steps = int(max_steps or math.ceil(row_len / 1.5 * 1.5 / self.DT) + 50)
        self._build_model()
        self._renderers: dict[tuple[int, int], mujoco.Renderer] = {}
        self._chase_yaw: float | None = None
        self.reset(seed)

    # ---------------------------------------------------------------- terrain
    def _rc(self, x: float, y: float) -> tuple[float, float]:
        return ((1.0 - y / self.size_m) * (self._h - 1), x / self.size_m * (self._w - 1))

    def height(self, x: float, y: float) -> float:
        """Terrain height above the site minimum at local (x, y)."""
        return _bilinear(self._z, *self._rc(x, y))

    def gradient(self, x: float, y: float) -> tuple[float, float]:
        r, c = self._rc(x, y)
        return _bilinear(self._gx, r, c), _bilinear(self._gy, r, c)

    # ------------------------------------------------------------------- rows
    def _make_rows(self, spacing: float, margin: float | None, length: float | None) -> list[np.ndarray]:
        margin = float(margin if margin is not None else max(8.0, 0.08 * self.size_m))
        x0, x1 = margin, self.size_m - margin
        y0, y1 = margin, self.size_m - margin
        if x1 - x0 < spacing or y1 - y0 < 10.0:
            margin = 0.1 * self.size_m
            x0, x1, y0, y1 = margin, self.size_m - margin, margin, self.size_m - margin
        n = max(1, int((x1 - x0) // spacing) + 1)
        xs = x0 + (x1 - x0 - (n - 1) * spacing) / 2.0 + spacing * np.arange(n)
        full = y1 - y0
        L = min(full, float(length)) if length else full
        rows = []
        for i, x in enumerate(xs):
            m = max(2, int(round(L / ROW_STEP_M)) + 1)
            if i % 2 == 0:  # serpentine: even rows northbound from the south margin
                ys = np.linspace(y0, y0 + L, m)
            else:
                ys = np.linspace(y1, y1 - L, m)
            rows.append(np.stack([np.full(m, x), ys], axis=1).astype(np.float64))
        return rows

    @staticmethod
    def _row_len(row: np.ndarray) -> float:
        return float(np.sum(np.linalg.norm(np.diff(row, axis=0), axis=1)))

    def project(self, x: float, y: float, row: np.ndarray | None = None) -> tuple[float, float]:
        """(arc length s along the row, signed cross-track error, + = left of travel)."""
        row = self.row if row is None else row
        a, b = row[:-1], row[1:]
        ab = b - a
        seg = np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-12)
        p = np.array([x, y])
        t = np.clip(np.einsum("ij,ij->i", p - a, ab) / seg, 0.0, 1.0)
        proj = a + ab * t[:, None]
        d = np.linalg.norm(p - proj, axis=1)
        k = int(np.argmin(d))
        cross = ab[k, 0] * (y - a[k, 1]) - ab[k, 1] * (x - a[k, 0])
        return float(self._cum[k] + t[k] * math.sqrt(seg[k])), float(math.copysign(d[k], cross))

    def point_at(self, s: float, row: np.ndarray | None = None) -> np.ndarray:
        """Point at arc length s along the row (extrapolated past the end)."""
        row = self.row if row is None else row
        cum = self._cum if row is self.row else np.concatenate(
            [[0.0], np.cumsum(np.linalg.norm(np.diff(row, axis=0), axis=1))])
        if s >= cum[-1]:
            d = row[-1] - row[-2]
            return row[-1] + d / max(np.linalg.norm(d), 1e-9) * (s - cum[-1])
        s = max(s, 0.0)
        return np.array([np.interp(s, cum, row[:, 0]), np.interp(s, cum, row[:, 1])])

    # ------------------------------------------------------------------ scene
    def _build_model(self) -> None:
        half = self.size_m / 2.0
        zmax = float(self._z.max())
        zscale = max(zmax, 0.01)
        hz = self._z
        step = int(math.ceil(max(hz.shape) / MAX_HFIELD_CELLS))
        if step > 1:
            hz = hz[::step, ::step]
        nrow, ncol = hz.shape
        ortho = os.path.abspath(str(self.site.ortho_path))
        xml = f"""
<mujoco model="farm">
  <visual>
    <global offwidth="1280" offheight="720" fovy="50"/>
    <quality shadowsize="2048"/>
    <map fogstart="40" fogend="400" znear="0.002" zfar="6"/>
    <rgba fog="0.75 0.8 0.85 1" haze="0.75 0.8 0.85 1"/>
    <headlight ambient="0.25 0.25 0.25" diffuse="0.3 0.3 0.3" specular="0 0 0"/>
  </visual>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.55 0.72 0.92" rgb2="0.9 0.93 0.97" width="256" height="256"/>
    <texture name="ortho" type="2d" file="{ortho}"/>
    <material name="ground" texture="ortho" specular="0" shininess="0" reflectance="0"/>
    <material name="paint" rgba="0.12 0.5 0.15 1" specular="0.3"/>
    <material name="cabglass" rgba="0.35 0.45 0.5 1" specular="0.6"/>
    <material name="tyre" rgba="0.08 0.08 0.08 1"/>
    <hfield name="terrain" nrow="{nrow}" ncol="{ncol}" size="{half} {half} {zscale} 1.0"/>
  </asset>
  <worldbody>
    <light name="sun" directional="true" castshadow="true" pos="0 0 100" dir="0 0 -1" diffuse="0.8 0.8 0.75" ambient="0.3 0.3 0.3"/>
    <geom name="ground" type="hfield" hfield="terrain" material="ground" contype="0" conaffinity="0"/>
    <body name="tractor" mocap="true" pos="0 0 0">
      <geom type="box" size="1.45 0.7 0.45" pos="0.45 0 0.95" material="paint" contype="0" conaffinity="0"/>
      <geom type="box" size="0.6 0.7 0.6" pos="-0.6 0 1.95" material="cabglass" contype="0" conaffinity="0"/>
      <geom type="box" size="0.65 0.75 0.05" pos="-0.6 0 2.6" material="paint" contype="0" conaffinity="0"/>
      <geom type="cylinder" size="0.8 0.25" pos="-1.0 0.95 0.8" euler="90 0 0" material="tyre" contype="0" conaffinity="0"/>
      <geom type="cylinder" size="0.8 0.25" pos="-1.0 -0.95 0.8" euler="90 0 0" material="tyre" contype="0" conaffinity="0"/>
      <geom type="cylinder" size="0.5 0.2" pos="1.5 0.9 0.5" euler="90 0 0" material="tyre" contype="0" conaffinity="0"/>
      <geom type="cylinder" size="0.5 0.2" pos="1.5 -0.9 0.5" euler="90 0 0" material="tyre" contype="0" conaffinity="0"/>
      <camera name="cab" pos="0.2 0 2.9" xyaxes="0 -1 0 0.259 0 0.966" fovy="60"/>
    </body>
  </worldbody>
</mujoco>"""
        self.model = mujoco.MjModel.from_xml_string(xml)
        hid = self.model.hfield("terrain").id
        self.model.hfield_data[:] = (np.flipud(hz) / zscale).astype(np.float32).ravel()  # row 0 = south
        ext = max(float(self.model.stat.extent), 1e-6)  # clip planes are relative to extent
        self.model.vis.map.znear = 0.2 / ext
        self.model.vis.map.zfar = 3.0 * self.size_m / ext
        self.data = mujoco.MjData(self.model)
        self._mocap = int(self.model.body("tractor").mocapid[0])
        self._sun = self.model.light("sun").id
        self._cab = self.model.camera("cab").id
        self._hfield_id = hid

    def _apply_weather(self) -> None:
        w = self.weather
        el = math.radians(float(np.clip(w["sun_elev_deg"], 3.0, 89.0)))
        az = math.radians(float(w["sun_azim_deg"]))  # compass, 0 = north, 90 = east
        to_sun = np.array([math.sin(az) * math.cos(el), math.cos(az) * math.cos(el), math.sin(el)])
        self.model.light_dir[self._sun] = -to_sun
        self.model.light_pos[self._sun] = to_sun * 100.0
        rain = float(w["rain_mm_h"])
        wet = min(rain / 8.0, 1.0)
        bright = (0.35 + 0.55 * math.sin(el)) * (1.0 - 0.5 * wet)
        self.model.light_diffuse[self._sun] = np.array([1.0, 0.97, 0.9]) * bright
        self.model.light_ambient[self._sun] = np.full(3, 0.25 + 0.1 * (1 - wet))
        self.model.vis.headlight.ambient[:] = 0.2 + 0.1 * wet
        self.model.vis.headlight.diffuse[:] = 0.25
        grey = 0.8 - 0.25 * wet
        self.model.vis.rgba.fog[:] = [grey, grey + 0.02, grey + 0.04, 1.0]
        self.model.vis.rgba.haze[:] = [grey, grey + 0.02, grey + 0.04, 1.0]
        ext = max(float(self.model.stat.extent), 1e-6)  # fog distances are relative to extent
        self.model.vis.map.fogstart = (120.0 - 100.0 * wet) / ext
        self.model.vis.map.fogend = (600.0 - 480.0 * wet) / ext
        self._fog = rain > 0.0

    def _sample_weather(self, rng: np.random.Generator) -> dict:
        """Seeded weather: the site's climate sampler when site.json has one, else the generic one.

        The generic draw always consumes the episode rng, so the rest of the episode
        (row, offsets, slip) is the same either way; the climate draw uses its own seeded rng.
        """
        w = sample_weather(rng)
        climate = ((getattr(self.site, "summary", None) or {}).get("farm") or {}).get("climate")
        if climate:
            from farmsim import weather as wx

            w = wx.sample_weather(np.random.default_rng([int(self.seed) % 2**32, 104729]),
                                  float(self.site.lat), float(self.site.lon), climate)
        return w

    # ------------------------------------------------------------------- API
    def reset(self, seed: int | None = None) -> dict:
        """Start an episode. seed=None uses the next seed after the last episode's."""
        if seed is not None:
            self.seed = int(seed)
        elif getattr(self, "_episodes", 0) > 0:
            self.seed += 1
        self._episodes = getattr(self, "_episodes", 0) + 1
        rng = np.random.default_rng(self.seed)
        self.rng = rng
        self.weather = dict(self.fixed_weather) if self.fixed_weather else self._sample_weather(rng)
        self.weather.setdefault("rain_mm_h", 0.0)
        self.weather.setdefault("sun_elev_deg", 40.0)
        self.weather.setdefault("sun_azim_deg", 180.0)
        self._apply_weather()
        self.row_index = int(rng.integers(len(self.rows)))
        self.row = self.rows[self.row_index]
        self._cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(self.row, axis=0), axis=1))])
        self.row_length = float(self._cum[-1])
        heading_dir = "northbound" if self.row[-1, 1] > self.row[0, 1] else "southbound"
        self.instruction = f"drive row {self.row_index + 1} of {len(self.rows)} {heading_dir}"
        d = self.row[1] - self.row[0]
        row_heading = math.atan2(d[1], d[0])
        offset = float(rng.uniform(-0.3, 0.3))
        left = np.array([-math.sin(row_heading), math.cos(row_heading)])
        p = self.row[0] + left * offset
        self.x, self.y = float(p[0]), float(p[1])
        self.heading = row_heading + float(rng.uniform(-1.0, 1.0)) * math.radians(3.0)
        self.speed = 0.0
        self.steer = 0.0
        self._slip_noise = 0.0
        self.traction = 1.0 / (1.0 + float(self.weather["rain_mm_h"]) / 4.0)
        self.steps = 0
        self._ctes: list[float] = []
        self._s_max = 0.0
        self.done = False
        self.done_reason = ""
        self._chase_yaw = None
        s, cte = self.project(self.x, self.y)
        self.s, self.cte = s, cte
        self._s_max = max(s, 0.0)
        self._pose_tractor()
        return self._obs()

    def step(self, action: np.ndarray) -> tuple[dict, bool, dict]:
        if self.done:
            return self._obs(), True, self._info()
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        steer_cmd = float(np.clip(a[0], -STEER_MAX, STEER_MAX)) if a.size > 0 and np.isfinite(a[0]) else 0.0
        speed_cmd = float(np.clip(a[1], 0.0, SPEED_MAX)) if a.size > 1 and np.isfinite(a[1]) else 0.0
        dt = self.DT
        self.steer += (steer_cmd - self.steer) * min(dt / TAU_STEER_S, 1.0)
        gx, gy = self.gradient(self.x, self.y)
        ch, sh = math.cos(self.heading), math.sin(self.heading)
        grade = gx * ch + gy * sh  # + uphill along travel
        cross = -gx * sh + gy * ch  # + = left side uphill
        slope_factor = float(np.clip(1.0 - 2.0 * max(grade, 0.0) - 0.5 * max(-grade, 0.0), 0.4, 1.0))
        self.speed += (speed_cmd * slope_factor - self.speed) * min(dt / TAU_SPEED_S, 1.0)
        self.speed = max(self.speed, 0.0)
        v = self.speed
        # lateral slip: downhill on cross-slope, worse in rain; plus OU noise
        loss = 1.0 / self.traction
        self._slip_noise += -0.3 * self._slip_noise + 0.15 * float(self.rng.standard_normal())
        v_lat = (-SLIP_K * cross * loss + 0.03 * loss * self._slip_noise) * v
        self.x += (v * ch - v_lat * sh) * dt
        self.y += (v * sh + v_lat * ch) * dt
        self.heading += v / WHEELBASE_M * math.tan(self.steer) * dt
        self.heading = (self.heading + math.pi) % (2 * math.pi) - math.pi
        self.steps += 1
        self.s, self.cte = self.project(self.x, self.y)
        self._s_max = max(self._s_max, self.s)
        self._ctes.append(abs(self.cte))
        if self.s >= self.row_length - 0.25:
            self.done, self.done_reason = True, "row_end"
        elif not (0.0 <= self.x <= self.size_m and 0.0 <= self.y <= self.size_m):
            self.done, self.done_reason = True, "left_site"
        elif abs(self.cte) > CTE_LIMIT_M:
            self.done, self.done_reason = True, "cte"
        elif self.steps >= self.max_steps:
            self.done, self.done_reason = True, "max_steps"
        self._pose_tractor()
        return self._obs(), self.done, self._info()

    def _info(self) -> dict:
        return {"cte": self.cte, "progress": self.progress, "reason": self.done_reason,
                "row": self.row_index, "weather": self.weather}

    @property
    def progress(self) -> float:
        return float(np.clip(self._s_max / max(self.row_length, 1e-9), 0.0, 1.0))

    def state(self) -> np.ndarray:
        return np.array([self.x, self.y, self.heading, self.speed, self.steer,
                         float(np.clip(self.s / max(self.row_length, 1e-9), 0.0, 1.0))], dtype=np.float32)

    def _obs(self) -> dict:
        front = self.render_camera(self._cab, self.img_size, self.img_size) if self.img_size > 0 \
            else np.zeros((0, 0, 3), np.uint8)
        return {"front": front, "state": self.state(), "instruction": self.instruction}

    def metrics(self) -> dict:
        ctes = np.asarray(self._ctes) if self._ctes else np.asarray([abs(self.cte)])
        mean, mx = float(ctes.mean()), float(ctes.max())
        success = bool(self.done_reason == "row_end" and mean < SUCCESS_CTE_MEAN_M and mx < SUCCESS_CTE_MAX_M)
        return {"success": success, "cte_mean_m": mean, "cte_max_m": mx, "progress": self.progress,
                "steps": self.steps, "time_s": round(self.steps * self.DT, 3), "reason": self.done_reason}

    # -------------------------------------------------------------- rendering
    def _pose_tractor(self) -> None:
        gx, gy = self.gradient(self.x, self.y)
        ch, sh = math.cos(self.heading), math.sin(self.heading)
        pitch = math.atan(gx * ch + gy * sh)
        roll = math.atan(-gx * sh + gy * ch)
        half = self.size_m / 2.0
        self.z = self.height(self.x, self.y)
        self.data.mocap_pos[self._mocap] = [self.x - half, self.y - half, self.z]
        self.data.mocap_quat[self._mocap] = _quat_from_ypr(self.heading, pitch, roll)
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_camlight(self.model, self.data)

    def _renderer(self, width: int, height: int) -> mujoco.Renderer:
        key = (width, height)
        r = self._renderers.get(key)
        if r is None:
            r = mujoco.Renderer(self.model, height=height, width=width)
            self._renderers[key] = r
        return r

    def render_camera(self, camera, width: int, height: int) -> np.ndarray:
        r = self._renderer(width, height)
        r.update_scene(self.data, camera=camera)
        r.scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = self._fog
        r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = True
        return r.render().copy()

    def render_front(self, size: int | None = None) -> np.ndarray:
        size = size or self.img_size or 224
        return self.render_camera(self._cab, size, size)

    def render_chase(self, width: int = 640, height: int = 360) -> np.ndarray:
        yaw = math.degrees(self.heading)
        if self._chase_yaw is None:
            self._chase_yaw = yaw
        else:  # smooth the camera yaw so slip noise does not shake the view
            dy = (yaw - self._chase_yaw + 180.0) % 360.0 - 180.0
            self._chase_yaw += 0.2 * dy
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        half = self.size_m / 2.0
        cam.lookat[:] = [self.x - half + 4.0 * math.cos(self.heading),
                         self.y - half + 4.0 * math.sin(self.heading), self.z + 1.0]
        cam.distance = 16.0
        cam.azimuth = self._chase_yaw  # camera sits behind the tractor, looking along heading
        cam.elevation = -22.0
        return self.render_camera(cam, width, height)

    def close(self) -> None:
        for r in self._renderers.values():
            r.close()
        self._renderers.clear()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001 - interpreter shutdown
            pass
