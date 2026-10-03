"""ExpertPolicy: pure-pursuit row follower that sees the true row (the demonstrator).

It needs the env it drives: pass it to the constructor, or call bind(env)
(rollout() and export_episodes() bind it for you).
"""

from __future__ import annotations

import math

import numpy as np

from farmsim.env import SLIP_K, STEER_MAX, WHEELBASE_M


class ExpertPolicy:
    name = "expert"

    def __init__(self, env=None, lookahead_m: float = 4.0, target_speed: float = 2.0):
        self.env = env
        self.lookahead_m = lookahead_m
        self.target_speed = target_speed

    def bind(self, env) -> "ExpertPolicy":
        self.env = env
        return self

    def reset(self) -> None:
        pass

    def act(self, obs: dict) -> np.ndarray:
        env = self.env
        if env is None:
            raise RuntimeError("ExpertPolicy needs an env: ExpertPolicy(env) or policy.bind(env)")
        x, y, heading = (float(v) for v in obs["state"][:3])
        s, _ = env.project(x, y)
        tx, ty = env.point_at(s + self.lookahead_m)
        dx, dy = tx - x, ty - y
        ld = max(math.hypot(dx, dy), 1e-3)
        alpha = math.atan2(dy, dx) - heading
        alpha = (alpha + math.pi) % (2 * math.pi) - math.pi
        # crab into the cross-slope to cancel the downhill slip the expert knows about
        gx, gy = env.gradient(x, y)
        cross = -gx * math.sin(heading) + gy * math.cos(heading)
        lat = float(np.clip(SLIP_K * cross / max(env.traction, 1e-3), -0.5, 0.5))
        alpha += math.asin(lat)
        steer = math.atan2(2.0 * WHEELBASE_M * math.sin(alpha), ld)
        # slow down on steep ground (along-track or cross slope)
        slope = math.hypot(gx, gy)
        speed = self.target_speed * float(np.clip(1.0 - 1.5 * slope, 0.6, 1.0))
        return np.array([np.clip(steer, -STEER_MAX, STEER_MAX), speed], dtype=np.float32)
