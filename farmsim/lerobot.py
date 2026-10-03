"""Export FarmEnv demonstrations as a GR00T-flavoured LeRobot v2 dataset.

Layout (per Isaac-GR00T getting_started/data_preparation.md and its
demo_data/cube_to_bowl_5 example, fetched 2026-10):

    meta/info.json  meta/episodes.jsonl  meta/tasks.jsonl  meta/modality.json
    data/chunk-000/episode_000000.parquet
    videos/chunk-000/observation.images.front/episode_000000.mp4
    farm_tractor_config.py     # GR00T modality config for NEW_EMBODIMENT

meta/stats.json is not written: launch_finetune.py computes stats.json and
relative_stats.json itself (gr00t/data/dataset/factory.py -> gr00t/data/stats.py).

Checked against Isaac-GR00T main (2026-08) by tests/test_gr00t_compat.py: its
LeRobotEpisodeLoader / ShardedSingleStepDataset load this export with
farm_tractor_config.py registered the way launch_finetune.py registers it, and
a CPU fine-tune of a tiny random GR00T N1.7 runs on it end to end.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from farmsim.env import FarmEnv
from farmsim.video import write_mp4

STATE_KEYS = ["x", "y", "heading", "speed", "steer", "row_progress"]
ACTION_KEYS = ["steer", "speed"]
VIDEO_KEY = "observation.images.front"
ANNOTATION_COL = "annotation.human.task_description"
ROBOT_TYPE = "farm_tractor"
CHUNKS_SIZE = 1000

CONFIG_PY = '''"""GR00T modality config for the LobBot farm tractor (NEW_EMBODIMENT).

Modelled on Isaac-GR00T examples/SO100/so100_config.py. Keys must match
meta/modality.json in the exported dataset. launch_finetune.py imports this file
for --modality-config-path <dataset>/farm_tractor_config.py (with
--embodiment-tag NEW_EMBODIMENT); importing registers the config, and a second
registration in the same process raises.
"""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)

farm_tractor_config = {
    # Video: current cab-camera frame only
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["front"],
    ),
    # State: x, y (m, local site frame), heading (rad), speed (m/s), steer (rad), row progress (0..1)
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=__STATE_KEYS__,
    ),
    # Action: 16-step chunk (1.6 s at 10 Hz) of target steering angle and target speed
    "action": ModalityConfig(
        delta_indices=list(range(0, 16)),
        modality_keys=__ACTION_KEYS__,
        action_configs=[
            # steer: ABSOLUTE target angle in rad
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # speed: ABSOLUTE target speed in m/s
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    # Language: task instruction ("drive row 3 of 8 northbound")
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

register_modality_config(farm_tractor_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
'''


def modality_json() -> dict:
    return {
        "state": {k: {"start": i, "end": i + 1} for i, k in enumerate(STATE_KEYS)},
        "action": {k: {"start": i, "end": i + 1} for i, k in enumerate(ACTION_KEYS)},
        "video": {"front": {"original_key": VIDEO_KEY}},
        # same form as Isaac-GR00T demo_data/cube_to_bowl_5; the parquet also has
        # the dedicated annotation.human.task_description column the docs require
        "annotation": {"human.task_description": {"original_key": "task_index"}},
    }


def config_py() -> str:
    return (CONFIG_PY.replace("__STATE_KEYS__", json.dumps(STATE_KEYS))
            .replace("__ACTION_KEYS__", json.dumps(ACTION_KEYS)))


def _scalar(dtype: str) -> dict:
    return {"dtype": dtype, "shape": [1], "names": None}


def info_json(n_episodes: int, n_frames: int, n_tasks: int, img: int, fps: int) -> dict:
    return {
        "codebase_version": "v2.1",
        "robot_type": ROBOT_TYPE,
        "total_episodes": n_episodes,
        "total_frames": n_frames,
        "total_tasks": n_tasks,
        "total_videos": n_episodes,
        "total_chunks": max(1, (n_episodes + CHUNKS_SIZE - 1) // CHUNKS_SIZE),
        "chunks_size": CHUNKS_SIZE,
        "fps": fps,
        "splits": {"train": f"0:{n_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "action": {"dtype": "float32", "shape": [len(ACTION_KEYS)], "names": ACTION_KEYS},
            "observation.state": {"dtype": "float32", "shape": [len(STATE_KEYS)], "names": STATE_KEYS},
            VIDEO_KEY: {
                "dtype": "video", "shape": [img, img, 3], "names": ["height", "width", "channels"],
                "info": {"video.height": img, "video.width": img, "video.codec": "h264",
                         "video.pix_fmt": "yuv420p", "video.is_depth_map": False, "video.fps": fps,
                         "video.channels": 3, "has_audio": False},
            },
            "timestamp": _scalar("float32"),
            "frame_index": _scalar("int64"),
            "episode_index": _scalar("int64"),
            "index": _scalar("int64"),
            "task_index": _scalar("int64"),
            ANNOTATION_COL: _scalar("int64"),
            "next.reward": _scalar("float32"),
            "next.done": _scalar("bool"),
        },
    }


def export_episodes(site, policy, n_episodes, out_dir, seed=0, img_size: int = 224,
                    only_success: bool = True, max_attempts: int | None = None,
                    progress=None, **env_kwargs) -> dict:
    """Drive `policy` for n_episodes seeded episodes and write a LeRobot v2 dataset.

    With only_success (default) failed episodes are dropped and more seeds are
    tried, up to max_attempts (default 2 * n_episodes). policy=None uses the
    ExpertPolicy. progress, if given, is called as progress(done, total).
    Returns stats: episodes, attempts, frames, success_rate, cte_mean_m, tasks, dir.
    """
    if img_size <= 0:
        raise ValueError("export needs camera frames: img_size must be > 0")
    if policy is None:
        from farmsim.expert import ExpertPolicy
        policy = ExpertPolicy()
    out = Path(out_dir)
    for sub in ("data", "videos", "meta"):
        if (out / sub).exists():
            shutil.rmtree(out / sub)
    (out / "meta").mkdir(parents=True, exist_ok=True)
    env = FarmEnv(site, seed=seed, img_size=img_size, **env_kwargs)
    if hasattr(policy, "bind"):
        policy.bind(env)
    fps = int(round(1.0 / env.DT))
    max_attempts = int(max_attempts or max(2 * n_episodes, n_episodes + 2))
    tasks: dict[str, int] = {}
    episodes: list[dict] = []
    total = 0
    attempts = 0
    successes = 0
    ctes: list[float] = []
    try:
        while len(episodes) < n_episodes and attempts < max_attempts:
            ep_seed = seed + attempts
            attempts += 1
            obs = env.reset(ep_seed)
            policy.reset()
            states, actions, frames = [], [], []
            done = False
            while not done:
                act = np.asarray(policy.act(obs), dtype=np.float32).reshape(-1)[:2]
                act = np.array([np.clip(act[0], -0.6, 0.6), np.clip(act[1], 0.0, 3.0)], dtype=np.float32)
                states.append(obs["state"].astype(np.float32))
                actions.append(act)
                frames.append(obs["front"])
                obs, done, _ = env.step(act)
            m = env.metrics()
            successes += int(m["success"])
            if only_success and not m["success"]:
                continue
            ctes.append(m["cte_mean_m"])
            ep = len(episodes)
            task = env.instruction
            ti = tasks.setdefault(task, len(tasks))
            n = len(states)
            chunk = ep // CHUNKS_SIZE
            reward = np.zeros(n, np.float32)
            reward[-1] = 1.0 if m["success"] else 0.0
            dn = np.zeros(n, bool)
            dn[-1] = True
            table = pa.table({
                "observation.state": pa.array([s.tolist() for s in states], type=pa.list_(pa.float32())),
                "action": pa.array([a.tolist() for a in actions], type=pa.list_(pa.float32())),
                "timestamp": pa.array((np.arange(n) * env.DT).astype(np.float32)),
                "frame_index": pa.array(np.arange(n, dtype=np.int64)),
                "episode_index": pa.array(np.full(n, ep, np.int64)),
                "index": pa.array(np.arange(total, total + n, dtype=np.int64)),
                "task_index": pa.array(np.full(n, ti, np.int64)),
                ANNOTATION_COL: pa.array(np.full(n, ti, np.int64)),
                "next.reward": pa.array(reward),
                "next.done": pa.array(dn),
            })
            dpath = out / f"data/chunk-{chunk:03d}/episode_{ep:06d}.parquet"
            dpath.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, dpath)
            write_mp4(frames, out / f"videos/chunk-{chunk:03d}/{VIDEO_KEY}/episode_{ep:06d}.mp4", fps=fps)
            episodes.append({"episode_index": ep, "tasks": [task], "length": n,
                             "seed": ep_seed, "success": bool(m["success"]), "cte_mean_m": m["cte_mean_m"]})
            total += n
            if progress:
                progress(len(episodes), n_episodes)
    finally:
        env.close()
    meta = out / "meta"
    with open(meta / "episodes.jsonl", "w") as f:
        for e in episodes:
            f.write(json.dumps(e) + "\n")
    with open(meta / "tasks.jsonl", "w") as f:
        for t, i in sorted(tasks.items(), key=lambda kv: kv[1]):
            f.write(json.dumps({"task_index": i, "task": t}) + "\n")
    (meta / "info.json").write_text(json.dumps(info_json(len(episodes), total, len(tasks), img_size, fps), indent=2))
    (meta / "modality.json").write_text(json.dumps(modality_json(), indent=2))
    (out / "farm_tractor_config.py").write_text(config_py())
    return {"episodes": len(episodes), "attempts": attempts, "frames": total,
            "success_rate": successes / max(attempts, 1),
            "cte_mean_m": float(np.mean(ctes)) if ctes else None,
            "tasks": len(tasks), "site": getattr(site, "id", ""), "dir": str(out)}
