# Robot MVP: map to sim to compressed GR00T

Hackathon MVP for farm automation. You pick a place in Sweden on a map, and
LobBot builds a driving sim from Lantmäteriet elevation and orthophoto data.
It records scripted demos, fine-tunes GR00T N1.7 on them, compresses the
policy, and scores every candidate in the sim. The outputs are:

- a training video, `out/training.mp4`
- a graph of footprint against sim performance, `out/footprint_vs_performance.png`

The robot pipeline is separate from the LLM pipeline (`pipeline.py`). It
reuses the same job layout (`stages/_util.py:Job`) and the same progress
protocol (`common/progress.py`).

## Modules and owners

| Module | What |
|---|---|
| `farmsim/geo.py`, `farmsim/site.py`, `farmsim/sources/`, `farmsim/static/map.html` | Map pick to a site on disk: elevation grid, orthophoto, summary |
| `farmsim/env.py`, `farmsim/expert.py`, `farmsim/video.py`, `farmsim/lerobot.py` | Driving sim, scripted expert, video recording, LeRobot v2 export |
| `robot/gr00t_policy.py`, `robot/finetune.py`, `robot/compress.py`, `robot/footprint.py` | GR00T N1.7 adapter, fine-tune wrapper, compression candidates, footprint |
| `common/robotspec.py`, `robot_pipeline.py`, `robot/simeval.py`, `robot/report.py`, API and CLI | Orchestration, eval, graph, stitched video, HTTP endpoints |

## Shared interfaces (the contract)

### Site (`farmsim/site.py`)

```python
SITES: Path  # env LOBBOT_SITES, default /mnt/nvme/sites

@dataclass
class Site:
    id: str            # e.g. "s59p8581n17p6389" from lat/lon/size, letters and digits only
    name: str
    lat: float; lon: float     # WGS84 centre
    size_m: float              # square side length in metres
    crs: str                   # "EPSG:3006" (SWEREF 99 TM)
    origin_e: float; origin_n: float   # SWEREF 99 TM coords of the NORTH-WEST corner
    res_m: float               # metres per DEM cell
    source: str                # "lantmateriet" | "copernicus" | "synthetic"
    summary: dict              # elevation min/max/mean m, mean/max slope deg, noon sun elevation deg, area ha
    dir: Path

    def dem(self) -> np.ndarray      # float32 [H, W] metres; row 0 = north, col 0 = west
    @property
    def ortho_path(self) -> Path     # RGB PNG covering exactly the same square, north up
    @property
    def preview_path(self) -> Path   # PNG: ortho with hillshade, for the map page

def build_site(lat, lon, size_m=300.0, name="", source="auto", sites_dir=None) -> Site
def load_site(id_or_dir, sites_dir=None) -> Site
def list_sites(sites_dir=None) -> list[Site]
```

On disk, a site lives in `<sites_dir>/<id>/` as `site.json`, `dem.npy`,
`ortho.png` and `preview.png`.

`source="auto"` tries the sources in order: Lantmäteriet (when credentials are
set), then Copernicus DEM with Sentinel-2 cloudless, then synthetic. The
`synthetic` source is deterministic given lat/lon, needs no network, and is
what tests and dry runs use.

### Sim (`farmsim/env.py`)

```python
class FarmEnv:
    DT = 0.1  # s, 10 Hz control
    def __init__(self, site: Site, seed: int = 0, img_size: int = 224, weather: dict | None = None): ...
    def reset(self, seed: int | None = None) -> dict
    def step(self, action: np.ndarray) -> tuple[dict, bool, dict]
    def render_chase(self, width: int = 640, height: int = 360) -> np.ndarray   # uint8 RGB
    def metrics(self) -> dict    # success, cte_mean_m, cte_max_m, progress (0..1), steps, time_s
    rows: list[np.ndarray]       # field rows to drive, each [N, 2] local metres (x east, y north from the SW corner)
    instruction: str             # e.g. "drive row 3 of 8 northbound"
```

`weather` is `{"rain_mm_h": float, "sun_elev_deg": float, "sun_azim_deg": float}`.
Rain lowers traction (more slip), and the sun angle sets the light. When it is
None, each `reset` samples it from the seed.

Observations are a dict:

```python
{"front": uint8[img, img, 3],        # cab camera looking ahead
 "state": float32[6],                # x, y, heading_rad, speed_mps, steer_rad, row_progress
 "instruction": str}
```

An action is `float32[2]`: target steering angle in rad, clipped to ±0.6, and
target speed in m/s, clipped to 0..3.

An episode ends on any of these:
- the end of the row (success when `cte_mean_m < 0.5` and `cte_max_m < 1.5`)
- leaving the site
- `cte > 3 m`
- running out of steps

### Policy protocol (any policy)

```python
class Policy:
    name: str
    def reset(self) -> None
    def act(self, obs: dict) -> np.ndarray   # action float32[2]
```

`farmsim/expert.py:ExpertPolicy` is a pure-pursuit controller on the true row
(the demonstrator). `robot/gr00t_policy.py:Gr00tPolicy` wraps a GR00T
checkpoint plus a compression config.

### Video (`farmsim/video.py`)

```python
def rollout(env, policy, seed, max_steps=600, frames=True) -> tuple[dict, list[np.ndarray]]   # metrics, chase frames with front-cam inset and HUD
def write_mp4(frames, path, fps=10) -> Path
def title_card(text, width=640, height=360, seconds=1.5, fps=10) -> list[np.ndarray]
```

### LeRobot export (`farmsim/lerobot.py`)

```python
def export_episodes(site, policy, n_episodes, out_dir, seed=0) -> dict   # stats
```

This writes a LeRobot v2 dataset (parquet, mp4 and meta), plus
`meta/modality.json` and `farm_tractor_config.py`, a GR00T modality config for
`NEW_EMBODIMENT`.

### Compression candidates (`robot/compress.py`, `robot/footprint.py`)

```python
@dataclass
class Candidate:
    name: str                      # "teacher", "steps2", "fp8", "mixed", "pruned"...
    denoise_steps: int = 4
    drop_dit_blocks: list[int] = []
    precision: dict[str, str] = {} # layer-group name -> "bf16" | "fp8" | "nvfp4"
    note: str = ""

def footprint(model_or_shapes, cand: Candidate) -> dict   # params, size_gb, bits_per_weight
```

`work/candidates.json` is a list of `asdict(Candidate)` plus a `footprint`.

### Eval output (`out/eval.json`)

```json
{"kind": "robot", "dry_run": false, "site": "<id>", "episodes": 20,
 "candidates": [{"name": "teacher", "size_gb": 6.1, "params": 3.0e9, "bits_per_weight": 16,
                 "latency_ms": 41.0, "success_rate": 0.9, "cte_mean_m": 0.21, "progress": 0.98,
                 "video": "out/clips/teacher.mp4"}],
 "expert": {"success_rate": 1.0, "cte_mean_m": 0.05}}
```

## Robot pipeline stages

```
site      build or load the site from robotspec.json (lat, lon, size_m)
demos     the expert drives N episodes; LeRobot v2 export; demo clip
finetune  GR00T N1.7 NEW_EMBODIMENT fine-tune on the demos (Isaac-GR00T launch_finetune.py)
compress  build candidates: fewer denoising steps, sim-scored per-group FP8/NVFP4, DiT block drops
simeval   every candidate drives the same seeded episodes; latency; out/eval.json; one clip each
report    out/footprint_vs_performance.png and out/training.mp4 (title cards plus clips)
```

```bash
python robot_pipeline.py --job <dir> [--from <stage>] [--only <stage>]
```

`<dir>/robotspec.json` must exist. With `LOBBOT_DRY_RUN=1`, finetune and
compress do no GPU work. Candidates are then the expert plus noise and lag
scaled to each candidate's footprint, and `eval.json` and the graph are
labelled DRY RUN. Never present dry-run numbers as results.

## Running locally without a GPU

MuJoCo renders headless with `MUJOCO_GL=osmesa` (CPU) or `MUJOCO_GL=egl` (GPU).
On the B200, use EGL. Isaac Sim needs RT cores and is not used.
