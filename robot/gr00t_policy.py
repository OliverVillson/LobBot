"""GR00T N1.7 checkpoint as a FarmEnv policy, with a compression candidate applied.

Runs the model in-process through Isaac-GR00T's own policy class
(gr00t/policy/gr00t_policy.py:Gr00tPolicy), which validates the observation,
runs the processor and model.get_action, and un-normalises the action chunk.
On top of it this adapter:

- maps FarmEnv obs (front image, state[6], instruction) to the GR00T nested
  obs dict, with the keys of the checkpoint's modality config;
- keeps an action-chunk queue: one inference per `chunk_exec` env steps;
- applies a Candidate in place: denoising steps
  (model.action_head.num_inference_timesteps, read at sampling time, the same
  knob gr00t/eval/open_loop_eval.py sets for --denoising-steps), DiT block
  drops (model.action_head.model.transformer_blocks[j] -> identity, which
  keeps the residual stream since each BasicTransformerBlock returns
  hidden_states + attn + ff, gr00t/model/modules/dit.py), and fake-quant of the
  weights per layer group (FP8 e4m3 per output channel, NVFP4 emulation).
  The original weights stay on the device so set_candidate can switch freely.

torch and gr00t are imported lazily; this module imports without them.
"""

from __future__ import annotations

import time

import numpy as np

from robot.footprint import NVFP4_BLOCK, Candidate, footprint, group_of, is_quantizable, shapes_from_model

# FarmEnv obs -> GR00T modality keys. MUST match the modality config that
# farmsim/lerobot.py writes (meta/modality.json and farm_tractor_config.py).
# The adapter reads the real keys from the checkpoint's modality config at load
# and maps by these names; a single state key takes the whole state[6], and a
# single video key takes the front camera.
GR00T_KEYS = {
    "video": "front",
    "state": ["x", "y", "heading", "speed", "steer", "row_progress"],  # FarmEnv state[6] order
    "action": ["steer", "speed"],  # FarmEnv action[2] order
    "language": "annotation.human.task_description",
}

E2M1_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
FP8_MAX = 448.0  # largest finite float8_e4m3fn


# ---------------------------------------------------------------------------
# Fake quantisation (torch tensors in, same dtype out)
# ---------------------------------------------------------------------------
def _as_rows(w):
    """View a weight as [out_channels, in_features]. CategorySpecificLinear.W
    (gr00t/model/modules/embodiment_conditioned_mlp.py) is [cats, in, out], so
    it is transposed first; conv kernels flatten to [out, in*k]."""
    if w.ndim == 3:
        return w.transpose(1, 2).reshape(-1, w.shape[1]), lambda r: r.reshape(w.shape[0], w.shape[2], w.shape[1]).transpose(1, 2)
    if w.ndim == 2:
        return w, lambda r: r
    return w.reshape(w.shape[0], -1), lambda r: r.reshape(w.shape)


def fake_fp8(w):
    """FP8 e4m3 round trip with one scale per output channel."""
    import torch

    rows, back = _as_rows(w.float())
    scale = rows.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / FP8_MAX
    q = (rows / scale).to(torch.float8_e4m3fn).float() * scale
    return back(q).to(w.dtype)


def fake_nvfp4(w, block: int = NVFP4_BLOCK):
    """NVFP4 emulation: blocks of 16 along the input dim, e2m1 values, one
    e4m3 scale per block under a per-tensor fp32 scale (as in NVIDIA's format)."""
    import torch

    rows, back = _as_rows(w.float())
    n_out, n_in = rows.shape
    pad = (-n_in) % block
    x = torch.nn.functional.pad(rows, (0, pad)).reshape(n_out, -1, block)
    amax = x.abs().amax(dim=2, keepdim=True)
    g = amax.max().clamp_min(1e-12) / (6.0 * FP8_MAX)  # per-tensor scale so block scales fit e4m3
    s = (amax / 6.0 / g).to(torch.float8_e4m3fn).float() * g
    s = torch.where(s > 0, s, torch.ones_like(s))
    y = x / s
    grid = torch.tensor(E2M1_GRID, device=w.device)
    mids = (grid[1:] + grid[:-1]) / 2
    idx = torch.bucketize(y.abs().clamp(max=6.0), mids)
    q = torch.sign(y) * grid[idx] * s
    q = q.reshape(n_out, -1)[:, :n_in]
    return back(q).to(w.dtype)


FAKE_QUANT = {"fp8": fake_fp8, "nvfp4": fake_nvfp4}


def skip_block():
    """Stand-in for a dropped DiT block: returns hidden_states unchanged."""
    import torch

    class SkipBlock(torch.nn.Module):
        def forward(self, hidden_states, *args, **kwargs):
            return hidden_states

    return SkipBlock()


def apply_precision(model, precision: dict[str, str], originals: dict | None = None) -> int:
    """Fake-quantise the quantizable weights of each named group in place.
    With `originals` (name -> original tensor), quantises from the original so
    repeated calls do not compound. Returns the number of tensors changed."""
    import torch

    n = 0
    with torch.no_grad():
        for name, p in model.named_parameters():
            g = group_of(name)
            prec = precision.get(g or "", "bf16")
            if not g or prec == "bf16" or not is_quantizable(name, p.ndim):
                continue
            src = originals[name] if originals is not None else p.data
            p.data.copy_(FAKE_QUANT[prec](src))
            n += 1
    return n


def dit_blocks(model):
    """The DiT ModuleList (Gr00tN1d7.action_head.model.transformer_blocks)."""
    return model.action_head.model.transformer_blocks


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------
class Gr00tPolicy:
    """Policy protocol: name, reset(), act(obs) -> float32[2]."""

    def __init__(
        self,
        checkpoint_dir,
        candidate: Candidate | None = None,
        device: str = "cuda",
        embodiment_tag: str = "NEW_EMBODIMENT",
        chunk_exec: int = 8,
        noise_seed: int | None = 0,
    ):
        import torch
        from gr00t.policy.gr00t_policy import Gr00tPolicy as _Gr00t  # Isaac-GR00T

        self.device = device
        self.chunk_exec = chunk_exec
        self.noise_seed = noise_seed  # fixed flow-matching noise: comparable candidates
        self._gr00t = _Gr00t(embodiment_tag=embodiment_tag, model_path=str(checkpoint_dir), device=device)
        self.model = self._gr00t.model
        self._resolve_keys(self._gr00t.modality_configs)
        # Pristine copies to restore from when switching candidates.
        self._orig = {n: p.detach().clone() for n, p in self.model.named_parameters()}
        self._orig_blocks = list(dit_blocks(self.model))
        self._orig_steps = int(self.model.action_head.num_inference_timesteps)
        self.shapes = shapes_from_model(self.model)
        self.latency_ms: list[float] = []
        self._queue: list[np.ndarray] = []
        self._dirty: set[str] = set()
        self._torch = torch
        self.set_candidate(candidate or Candidate("teacher"))

    # -- modality mapping ---------------------------------------------------
    def _resolve_keys(self, mc: dict) -> None:
        def keys(m):
            return list(mc[m].modality_keys)

        self.video_key = GR00T_KEYS["video"] if GR00T_KEYS["video"] in keys("video") else keys("video")[0]
        self.video_t = len(mc["video"].delta_indices)
        self.state_t = len(mc["state"].delta_indices)
        sk = keys("state")
        if len(sk) == 1:
            self.state_map = {sk[0]: list(range(6))}
        else:
            missing = [k for k in sk if k not in GR00T_KEYS["state"]]
            if missing:
                raise ValueError(f"state keys {missing} not in GR00T_KEYS; fix to match farmsim/lerobot.py")
            self.state_map = {k: [GR00T_KEYS["state"].index(k)] for k in sk}
        ak = keys("action")
        if len(ak) > 1 and all(k in GR00T_KEYS["action"] for k in ak):
            ak = sorted(ak, key=GR00T_KEYS["action"].index)
        self.action_keys = ak
        self.language_key = keys("language")[0]

    def to_gr00t_obs(self, obs: dict) -> dict:
        img = np.asarray(obs["front"], dtype=np.uint8)
        state = np.asarray(obs["state"], dtype=np.float32)
        return {
            "video": {self.video_key: np.repeat(img[None, None], self.video_t, axis=1)},
            "state": {
                k: np.repeat(state[idx][None, None], self.state_t, axis=1).astype(np.float32)
                for k, idx in self.state_map.items()
            },
            "language": {self.language_key: [[str(obs.get("instruction", ""))]]},
        }

    # -- candidate ----------------------------------------------------------
    def set_candidate(self, cand: Candidate) -> None:
        torch = self._torch
        with torch.no_grad():
            params = dict(self.model.named_parameters())
            for n in self._dirty:
                params[n].data.copy_(self._orig[n])
        self._dirty = set()
        blocks = dit_blocks(self.model)
        for j, b in enumerate(self._orig_blocks):
            blocks[j] = b
        for j in cand.drop_dit_blocks:
            blocks[j] = skip_block()
        self.model.action_head.num_inference_timesteps = int(cand.denoise_steps or self._orig_steps)
        if cand.precision:
            apply_precision(self.model, cand.precision, self._orig)
            self._dirty = {
                n for n, p in self.model.named_parameters()
                if cand.precision.get(group_of(n) or "", "bf16") != "bf16"
            }
        self.candidate = cand
        self.footprint = footprint(self.shapes, cand)
        self.name = cand.name
        self.reset()

    @property
    def size_label(self) -> str:
        return f"{self.footprint['size_gb']:.2f} GB"

    # -- inference ----------------------------------------------------------
    def predict_chunk(self, obs: dict) -> np.ndarray:
        """Full action chunk [horizon, 2] for one FarmEnv obs; records latency."""
        torch = self._torch
        g_obs = self.to_gr00t_obs(obs)
        if self.noise_seed is not None:
            torch.manual_seed(self.noise_seed)
        cuda = str(self.device).startswith("cuda")
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        action, _info = self._gr00t.get_action(g_obs)
        if cuda:
            torch.cuda.synchronize()
        self.latency_ms.append((time.perf_counter() - t0) * 1000)
        chunk = np.concatenate([np.asarray(action[k], dtype=np.float32)[0] for k in self.action_keys], axis=-1)
        return chunk[:, :2]

    def reset(self) -> None:
        self._queue = []
        if hasattr(self._gr00t, "reset"):
            self._gr00t.reset()

    def act(self, obs: dict) -> np.ndarray:
        if not self._queue:
            self._queue = list(self.predict_chunk(obs)[: self.chunk_exec])
        a = self._queue.pop(0)
        return np.array([np.clip(a[0], -0.6, 0.6), np.clip(a[1], 0.0, 3.0)], dtype=np.float32)
