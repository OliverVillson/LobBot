import json
from types import SimpleNamespace

import numpy as np
import pytest

from robot import compress
from robot import footprint as fp
from robot.footprint import Candidate


class FakeJob:
    def __init__(self, root, max_size_gb=8.0):
        self.root = root
        (root / "work").mkdir(exist_ok=True)
        self.spec = SimpleNamespace(target=SimpleNamespace(max_size_gb=max_size_gb))

    def path(self, *parts):
        return self.root.joinpath(*parts)


# ------------------------------------------------------------------ footprint


def test_footprint_bits():
    M = 1_000_000
    shapes = {"vision": 1000 * M, "llm.0.qkv": 1000 * M, "dit.0.attn": 500 * M, "dit.0.ff": 500 * M}
    t = fp.footprint(shapes, Candidate("teacher"))
    assert t["params"] == 3000 * M and t["bits_per_weight"] == 16.0
    assert t["size_gb"] == pytest.approx(3000 * M * 2 / 1e9)
    q = fp.footprint(shapes, Candidate("q", precision={"vision": "nvfp4", "llm.0.qkv": "fp8"}))
    expect = M * (1000 * 4.5 + 1000 * fp.BPW["fp8"] + 1000 * 16)
    assert q["size_gb"] == pytest.approx(expect / 8 / 1e9)
    assert 8.0 < fp.BPW["fp8"] < 8.1
    d = fp.footprint(shapes, Candidate("d", drop_dit_blocks=[0]))
    assert d["params"] == 2000 * M


def test_default_shapes_are_gr00t_sized():
    s = fp.DEFAULT_SHAPES
    assert 1.8e9 < sum(s.values()) < 3.2e9
    assert sum(1 for g in s if g.startswith("dit.")) == 2 * fp.DIT_BLOCKS
    assert sum(1 for g in s if g.endswith(".down")) == fp.LLM_LAYERS


def test_group_names():
    assert fp.group_of("backbone.model.model.language_model.layers.3.self_attn.k_proj.weight") == "llm.3.qkv"
    assert fp.group_of("backbone.model.model.language_model.layers.3.self_attn.o_proj.weight") == "llm.3.o"
    assert fp.group_of("backbone.model.model.language_model.layers.11.mlp.up_proj.weight") == "llm.11.gate_up"
    assert fp.group_of("backbone.model.model.language_model.layers.11.mlp.down_proj.weight") == "llm.11.down"
    assert fp.group_of("backbone.model.model.visual.blocks.0.attn.qkv.weight") == "vision"
    assert fp.group_of("backbone.model.model.language_model.embed_tokens.weight") == "llm.embed"
    assert fp.group_of("action_head.model.transformer_blocks.5.attn1.to_q.weight") == "dit.5.attn"
    assert fp.group_of("action_head.model.transformer_blocks.5.ff.net.0.proj.weight") == "dit.5.ff"
    assert fp.group_of("action_head.state_encoder.layer1.W") == "proj"
    assert fp.group_of("backbone.model.lm_head.weight") is None
    assert fp.allowed("llm.2.o") == ["fp8", "bf16"] and fp.allowed("dit.0.attn") == ["fp8", "bf16"]
    assert fp.allowed("llm.2.qkv")[0] == "nvfp4"


# ------------------------------------------------------------------ allocation


def _sens(shapes, hot=()):
    out = {}
    for g in shapes:
        w = 50.0 if g in hot else 1.0
        out[g] = {p: (0.01 if p == "nvfp4" else 0.001) * w for p in fp.allowed(g) if p != "bf16"}
    return out


@pytest.mark.parametrize("frac", [0.40, 0.45, 0.6])
def test_allocation_respects_budget(frac):
    shapes = fp.DEFAULT_SHAPES
    budget = frac * fp.size_gb(shapes, {})
    prec = compress.allocate(shapes, _sens(shapes), budget)
    assert fp.size_gb(shapes, prec) <= budget + 1e-6
    for g, p in prec.items():
        assert p in fp.allowed(g)


def test_sensitive_groups_upgrade_first():
    shapes = fp.DEFAULT_SHAPES
    budget = 0.42 * fp.size_gb(shapes, {})
    prec = compress.allocate(shapes, _sens(shapes, hot={"llm.11.qkv"}), budget)
    assert prec["llm.11.qkv"] != "nvfp4"
    assert prec["llm.5.qkv"] == "nvfp4" or prec["llm.11.qkv"] == "bf16"


def test_budget_below_floor_raises():
    with pytest.raises(ValueError):
        compress.allocate(fp.DEFAULT_SHAPES, _sens(fp.DEFAULT_SHAPES), 0.5)


# ------------------------------------------------------------------ dry run


def test_dry_run_candidates(tmp_path):
    job = FakeJob(tmp_path)
    cands = compress.build_candidates(job, dry_run=True)
    names = [c.name for c in cands]
    assert names == ["teacher", "steps2", "steps1", "fp8", "nvfp4", "mixed", "pruned"]
    by = {c.name: c for c in cands}
    size = {n: c.footprint["size_gb"] for n, c in by.items()}
    assert size["teacher"] == size["steps2"] == size["steps1"]
    assert size["nvfp4"] <= size["mixed"] < size["fp8"] < size["teacher"]
    assert size["pruned"] < size["mixed"] and len(by["pruned"].drop_dit_blocks) == compress.N_PRUNE
    assert by["steps1"].denoise_steps == 1
    assert all("dry_run" in c.note for c in cands)
    sens = json.loads((tmp_path / "work" / "sensitivity.json").read_text())
    assert sens["dry_run"] is True and sens["budget_gb"] >= size["mixed"]
    rows = json.loads((tmp_path / "work" / "candidates.json").read_text())
    assert rows[0]["name"] == "teacher" and "footprint" in rows[0]
    Candidate(**rows[-1])  # round trip


# ------------------------------------------------------------------ measured path with fakes


class FakeEnv:
    def __init__(self, seed):
        self.t = 0

    def reset(self, seed=None):
        self.t = 0
        return self._obs()

    def _obs(self):
        return {"front": np.zeros((8, 8, 3), np.uint8), "state": np.full(6, self.t, np.float32), "instruction": "x"}

    def step(self, action):
        self.t += 1
        return self._obs(), self.t >= 20, {}


class FakePolicy:
    """Action error grows with each group's quantisation and each dropped block."""

    name = "fake"

    def __init__(self):
        self.shapes = {"vision": 10_000_000, "llm.0.qkv": 10_000_000, "llm.0.o": 5_000_000,
                       "dit.0.attn": 5_000_000, "dit.0.ff": 5_000_000, "dit.1.attn": 5_000_000,
                       "dit.1.ff": 5_000_000, "proj": 2_000_000}
        self.model = SimpleNamespace(action_head=SimpleNamespace(model=SimpleNamespace(transformer_blocks=[0, 1])))
        self.cand = Candidate("teacher")

    def set_candidate(self, c):
        self.cand = c

    def reset(self):
        pass

    def act(self, obs):
        return np.zeros(2, np.float32)

    def predict_chunk(self, obs):
        off = sum({"fp8": 0.01, "nvfp4": 0.1}.get(p, 0) * (3 if g == "vision" else 1) for g, p in self.cand.precision.items())
        off += sum(0.05 * (j + 1) for j in self.cand.drop_dit_blocks)
        return np.full((16, 2), off, np.float32)


def test_measured_path_with_fakes(tmp_path):
    job = FakeJob(tmp_path)
    pol = FakePolicy()
    cands = compress.build_candidates(job, lambda c: pol, FakeEnv, dry_run=False, n_states=6, n_prune=1, expert=FakePolicy())
    sens = json.loads((tmp_path / "work" / "sensitivity.json").read_text())
    assert sens["groups"]["vision"]["nvfp4"] > sens["groups"]["llm.0.qkv"]["nvfp4"] > sens["groups"]["llm.0.qkv"]["fp8"]
    assert sens["dit_block_drop"][0] < sens["dit_block_drop"][1]
    assert {c.name: c for c in cands}["pruned"].drop_dit_blocks == [0]
    assert sens["candidate_action_error"]["teacher"] == 0.0


# ------------------------------------------------------------------ fake quant (torch)


def test_fake_quant_error_ordering():
    torch = pytest.importorskip("torch")
    from robot.gr00t_policy import fake_fp8, fake_nvfp4

    torch.manual_seed(0)
    w = torch.randn(64, 256, dtype=torch.bfloat16)

    def rel(q):
        return ((q.float() - w.float()) ** 2).sum().item() / (w.float() ** 2).sum().item()

    e16, e8, e4 = 0.0, rel(fake_fp8(w)), rel(fake_nvfp4(w))
    assert e16 < e8 < e4 < 0.05
    assert fake_fp8(w).dtype == torch.bfloat16
    # NVFP4 values per 16-block lie on the scaled e2m1 grid: at most 15 distinct magnitudes (+0)
    blk = fake_nvfp4(w.float())[0, :16].abs().unique()
    assert len(blk) <= 8
    # CategorySpecificLinear-style [cats, in, out] weights keep their shape
    w3 = torch.randn(4, 32, 48)
    assert fake_nvfp4(w3).shape == w3.shape and fake_fp8(w3).shape == w3.shape


def test_apply_precision_and_skip_block():
    torch = pytest.importorskip("torch")
    from robot.gr00t_policy import apply_precision, skip_block

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attn1 = torch.nn.Linear(32, 32)
            self.ff = torch.nn.Linear(32, 32)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.action_head = torch.nn.Module()
            self.action_head.model = torch.nn.Module()
            self.action_head.model.transformer_blocks = torch.nn.ModuleList([Block(), Block()])

    m = Model()
    orig = {n: p.detach().clone() for n, p in m.named_parameters()}
    n = apply_precision(m, {"dit.0.ff": "nvfp4"}, orig)
    assert n == 1  # weight only, bias stays
    changed = {k for k, p in m.named_parameters() if not torch.equal(p, orig[k])}
    assert changed == {"action_head.model.transformer_blocks.0.ff.weight"}
    shapes = fp.shapes_from_model(m)
    assert shapes["dit.0.attn"] == 32 * 32 and shapes["other"] == 4 * 32
    x = torch.randn(2, 5, 32)
    assert torch.equal(skip_block()(x, encoder_hidden_states=None, temb=None), x)


def test_gr00t_policy_adapter_with_fake_gr00t(monkeypatch):
    """Gr00tPolicy against a stand-in for Isaac-GR00T's in-process policy."""
    torch = pytest.importorskip("torch")
    import sys
    import types

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attn1 = torch.nn.Linear(16, 16)
            self.ff = torch.nn.Linear(16, 16)

        def forward(self, h, **kw):
            return h + self.ff(self.attn1(h))

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.action_head = torch.nn.Module()
            self.action_head.model = torch.nn.Module()
            self.action_head.model.transformer_blocks = torch.nn.ModuleList([Block(), Block()])
            self.action_head.num_inference_timesteps = 4

    MC = lambda keys, n=1: SimpleNamespace(modality_keys=keys, delta_indices=list(range(n)))  # noqa: E731
    seen = {}

    class FakeGr00t:
        def __init__(self, embodiment_tag, model_path, device):
            self.model = Model()
            self.modality_configs = {"video": MC(["front"]), "state": MC(["x", "y", "heading", "speed", "steer", "row_progress"]),
                                     "action": MC(["speed", "steer"], 40), "language": MC(["annotation.human.task_description"])}

        def get_action(self, obs):
            seen["obs"] = obs
            h = torch.ones(1, 3, 16)
            for b in self.model.action_head.model.transformer_blocks:
                h = b(h)
            v = float(h.mean().detach())
            return {"steer": np.full((1, 40, 1), 0.1 * v, np.float32), "speed": np.full((1, 40, 1), 1.0, np.float32)}, {}

    mod = types.ModuleType("gr00t.policy.gr00t_policy")
    mod.Gr00tPolicy = FakeGr00t
    for name in ("gr00t", "gr00t.policy"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "gr00t.policy.gr00t_policy", mod)
    from robot.gr00t_policy import Gr00tPolicy

    p = Gr00tPolicy("/nowhere", device="cpu", chunk_exec=4)
    obs = {"front": np.zeros((8, 8, 3), np.uint8), "state": np.arange(6, dtype=np.float32), "instruction": "drive row 1"}
    a = p.act(obs)
    assert a.dtype == np.float32 and a.shape == (2,) and a[1] == 1.0  # steer first, speed second
    g = seen["obs"]
    assert g["video"]["front"].shape == (1, 1, 8, 8, 3) and g["state"]["steer"].shape == (1, 1, 1)
    assert g["state"]["speed"][0, 0, 0] == 3.0 and g["language"]["annotation.human.task_description"] == [["drive row 1"]]
    for _ in range(3):
        p.act(obs)
    assert len(p.latency_ms) == 1  # one inference per 4 steps
    teacher = p.predict_chunk(obs)
    p.set_candidate(Candidate("x", denoise_steps=2, drop_dit_blocks=[1], precision={"dit.0.ff": "nvfp4"}))
    assert p.model.action_head.num_inference_timesteps == 2 and p.name == "x"
    assert not np.allclose(p.predict_chunk(obs), teacher)
    assert p.footprint["params"] < fp.footprint(p.shapes, Candidate("t"))["params"]
    p.set_candidate(Candidate("teacher"))
    assert np.allclose(p.predict_chunk(obs), teacher)  # originals restored
    assert p.size_label.endswith("GB")
