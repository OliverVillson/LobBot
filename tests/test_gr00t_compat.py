"""LobBot against a real Isaac-GR00T checkout, without a GPU.

Skipped unless the checkout is at $LOBBOT_GR00T_REPO (default
/mnt/nvme/Isaac-GR00T). Three levels:

- source checks (stdlib only): launch_finetune.py flags, output layout, the
  Gr00tPolicy constructor, and the model attributes robot/gr00t_policy.py and
  robot/footprint.py touch;
- dataset (needs GR00T's CPU deps: torch, pandas, torchcodec, tyro, ...): a
  farmsim LeRobot export loaded by GR00T's own stats + ShardedSingleStepDataset
  with farm_tractor_config.py registered as launch_finetune.py registers it;
- end to end (also transformers, torchvision, diffusers): a tiny random
  Cosmos-Reason2 stand-in and base GR00T N1.7, the real launch_finetune.py run
  on CPU with robot/finetune.py's argv, then robot/gr00t_policy.py loading the
  result and driving with every kind of compression candidate.

The heavy parts run in subprocesses: they register modality configs and set
HF_HOME/HF_HUB_OFFLINE, which must not leak into other tests.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GR00T = Path(os.environ.get("LOBBOT_GR00T_REPO", "/mnt/nvme/Isaac-GR00T"))
pytestmark = pytest.mark.skipif(
    not (GR00T / "gr00t" / "experiment" / "launch_finetune.py").exists(),
    reason=f"no Isaac-GR00T checkout at {GR00T} (set LOBBOT_GR00T_REPO)",
)

sys.path.insert(0, str(ROOT))


def _src(rel: str) -> str:
    return (GR00T / rel).read_text()


def _dataclass_fields(rel: str, cls: str) -> dict[str, bool]:
    """Field name -> has a default, for a dataclass in the checkout."""
    tree = ast.parse(_src(rel))
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == cls)
    return {s.target.id: s.value is not None for s in node.body
            if isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name)}


def _assigned_default(rel: str, cls: str, name: str):
    tree = ast.parse(_src(rel))
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == cls)
    for s in node.body:
        if isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name) and s.target.id == name:
            return s.value
    raise KeyError(name)


# ---------------------------------------------------------------------------
# Source checks
# ---------------------------------------------------------------------------
def test_finetune_flags_exist_in_FinetuneConfig():
    from robot.finetune import command

    fields = _dataclass_fields("gr00t/configs/finetune_config.py", "FinetuneConfig")
    cmd = command("/d", "/o", 100, "/base", num_gpus=1, gr00t_repo=GR00T)
    flags = [a[2:].replace("-", "_") for a in cmd if a.startswith("--")]
    assert flags, cmd
    unknown = [f for f in flags if f not in fields]
    assert not unknown, f"launch_finetune.py has no {unknown}"
    required = [f for f, has_default in fields.items() if not has_default]
    assert set(required) <= set(flags), f"missing required {set(required) - set(flags)}"
    assert cmd[cmd.index("--embodiment-tag") + 1] == "NEW_EMBODIMENT"
    tags = _src("gr00t/data/embodiment_tags.py")
    assert 'NEW_EMBODIMENT = "new_embodiment"' in tags
    multi = command("/d", "/o", 100, "/base", num_gpus=2, batch=32, gr00t_repo=GR00T)
    assert "torch.distributed.run" in multi and multi[multi.index("--num-gpus") + 1] == "2"


def test_modality_config_is_registered_from_path():
    lf = _src("gr00t/experiment/launch_finetune.py")
    assert "load_modality_config(ft_config.modality_config_path)" in lf
    assert "importlib.import_module(path.stem)" in lf
    reg = _src("gr00t/configs/data/embodiment_configs.py")
    assert "def register_modality_config(" in reg
    from farmsim.lerobot import config_py

    cfg = config_py()
    compile(cfg, "farm_tractor_config.py", "exec")
    for imp in ("from gr00t.configs.data.embodiment_configs import register_modality_config",
                "from gr00t.data.embodiment_tags import EmbodimentTag",
                "from gr00t.data.types import ("):
        assert imp in cfg
    types_src = _src("gr00t/data/types.py")
    for name in ("class ActionConfig", "class ModalityConfig", "class ActionRepresentation",
                 "class ActionType", "class ActionFormat"):
        assert name in types_src


def test_output_layout_matches_find_checkpoint():
    exp = _src("gr00t/experiment/experiment.py")
    assert 'processor_dir = output_dir / "processor"' in exp
    assert "trainer.save_model()" in exp
    assert "output_dir = Path(config.training.output_dir)\n" in exp  # experiment_name unset
    utils = _src("gr00t/experiment/utils.py")
    assert "shutil.copytree(self.processor_dir, checkpoint_dir" in utils
    pol = _src("gr00t/policy/gr00t_policy.py")
    assert 'model_dir / "processor"' in pol


def test_policy_constructor_and_model_knobs():
    tree = ast.parse(_src("gr00t/policy/gr00t_policy.py"))
    cls = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "Gr00tPolicy")
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    assert [a.arg for a in init.args.args] == ["self", "embodiment_tag", "model_path"]
    assert "device" in [a.arg for a in init.args.kwonlyargs]
    pol = _src("gr00t/policy/gr00t_policy.py")
    assert "self.model.get_action(" in pol and "self.modality_configs" in pol
    model = _src("gr00t/model/gr00t_n1d7/gr00t_n1d7.py")
    assert "self.num_inference_timesteps = config.num_inference_timesteps" in model
    assert "range(self.num_inference_timesteps)" in model  # read at sampling time
    assert "self.model = AlternateVLDiT(" in model and "self.action_head = Gr00tN1d7ActionHead(" in model
    assert "self.backbone = backbone_cls(" in model
    dit = _src("gr00t/model/modules/dit.py")
    assert "self.transformer_blocks = nn.ModuleList(" in dit
    assert "hidden_states = ff_output + hidden_states" in dit  # residual: a skipped block is identity
    bb = _src("gr00t/model/modules/qwen3_backbone.py")
    assert "self.model.language_model.layers.pop(-1)" in bb  # LLM truncated to select_layer
    mlp = _src("gr00t/model/modules/embodiment_conditioned_mlp.py")
    assert "torch.randn(num_categories, input_dim, hidden_dim)" in mlp  # W is [cats, in, out]


def test_footprint_defaults_match_model_config():
    from robot.footprint import DIT_BLOCKS, LLM_LAYERS, Candidate

    rel, cls = "gr00t/configs/model/gr00t_n1d7.py", "Gr00tN1d7Config"
    assert ast.literal_eval(_assigned_default(rel, cls, "select_layer")) == LLM_LAYERS
    assert ast.literal_eval(_assigned_default(rel, cls, "num_inference_timesteps")) == Candidate("t").denoise_steps
    assert ast.literal_eval(_assigned_default(rel, cls, "backbone_embedding_dim")) == 2048
    dit_cfg = _src(rel)
    assert '"num_layers": 16' in dit_cfg and DIT_BLOCKS == 16
    assert '"num_attention_heads": 32' in dit_cfg and '"attention_head_dim": 48' in dit_cfg
    assert ast.literal_eval(_assigned_default(rel, cls, "action_horizon")) >= 16  # our chunk


# ---------------------------------------------------------------------------
# Dataset and end to end (subprocesses)
# ---------------------------------------------------------------------------
def _need(*mods):
    """Skip unless importable; checked without importing (torchcodec can crash
    a process that already loaded other native libs)."""
    import importlib.util

    missing = [m for m in mods if importlib.util.find_spec(m) is None]
    if missing:
        pytest.skip(f"needs {missing} (Isaac-GR00T's venv has them)")


def _run(script: str, *args, env: dict | None = None, timeout: int = 1200) -> str:
    e = {**os.environ, "PYTHONPATH": os.pathsep.join([str(GR00T), str(ROOT)]),
         "NO_ALBUMENTATIONS_UPDATE": "1", "WANDB_MODE": "disabled", **(env or {})}
    e.setdefault("MUJOCO_GL", "osmesa")
    p = subprocess.run([sys.executable, "-c", textwrap.dedent(script), *map(str, args)], env=e,
                       capture_output=True, text=True, timeout=timeout, cwd=str(GR00T))
    assert p.returncode == 0, f"stdout:\n{p.stdout[-4000:]}\nstderr:\n{p.stderr[-6000:]}"
    return p.stdout


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    _need("mujoco", "pyarrow", "imageio")
    out = tmp_path_factory.mktemp("lerobot")
    _run("""
        import sys
        from farmsim.site import build_site
        from farmsim.lerobot import export_episodes
        out = sys.argv[1]
        site = build_site(59.0, 17.0, size_m=120, source="synthetic", sites_dir=out + "/sites")
        st = export_episodes(site, None, 2, out + "/ds", seed=0, img_size=64)
        assert st["episodes"] == 2, st
    """, out)
    return out / "ds"


LOAD = """
import json, sys, importlib.util
import numpy as np
ds = sys.argv[1]
spec = importlib.util.spec_from_file_location("lf", "gr00t/experiment/launch_finetune.py")
lf = importlib.util.module_from_spec(spec); spec.loader.exec_module(lf)
lf.load_modality_config(ds + "/farm_tractor_config.py")          # as launch_finetune.py does
from gr00t.configs.base_config import get_default_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.stats import generate_rel_stats, generate_stats
from gr00t.data.dataset.sharded_single_step_dataset import ShardedSingleStepDataset, extract_step_data
tag = EmbodimentTag.resolve("NEW_EMBODIMENT")
config = get_default_config().load_dict({"data": {"download_cache": False, "datasets": [
    {"dataset_paths": [ds], "mix_ratio": 1.0, "embodiment_tag": tag.value}]}})
config.validate()
generate_stats(ds); generate_rel_stats(ds, tag)
mc = config.data.modality_configs[tag.value]
d = ShardedSingleStepDataset(dataset_path=ds, embodiment_tag=tag, modality_configs=mc,
                             shard_size=config.data.shard_size,
                             episode_sampling_rate=config.data.episode_sampling_rate,
                             seed=config.data.seed, allow_padding=config.data.allow_padding)
ep = d.episode_loader[0]
s = extract_step_data(ep, 3, mc, tag)
stats = d.get_dataset_statistics()
print(json.dumps({
    "shards": int(len(d)),
    "keys": {m: list(mc[m].modality_keys) for m in ("video", "state", "action", "language")},
    "horizon": {m: len(mc[m].delta_indices) for m in ("video", "state", "action")},
    "video": [list(np.asarray(v[0]).shape) + [str(np.asarray(v[0]).dtype)] for v in s.images.values()],
    "state": {k: list(v.shape) for k, v in s.states.items()},
    "action": {k: list(v.shape) for k, v in s.actions.items()},
    "text": s.text,
    "stats": sorted(stats["state"]) + sorted(stats["action"]),
}))
"""


def test_dataset_loads_with_gr00t_loader(dataset):
    _need("torch", "pandas", "torchcodec", "tyro", "omegaconf", "transformers", "torchvision")
    from robot.gr00t_policy import GR00T_KEYS

    r = json.loads(_run(LOAD, dataset).strip().splitlines()[-1])
    assert r["shards"] >= 1
    assert r["keys"]["video"] == [GR00T_KEYS["video"]]
    assert r["keys"]["state"] == GR00T_KEYS["state"]
    assert r["keys"]["action"] == GR00T_KEYS["action"]
    assert r["keys"]["language"] == [GR00T_KEYS["language"]]
    assert r["video"] == [[64, 64, 3, "uint8"]]
    assert r["state"] == {k: [1, 1] for k in GR00T_KEYS["state"]}
    assert r["action"] == {k: [16, 1] for k in GR00T_KEYS["action"]}
    assert r["text"].startswith("drive row ")
    assert (dataset / "meta" / "stats.json").exists()


TINY = """
# Tiny random stand-ins: nvidia/Cosmos-Reason2-2B in a fake offline HF cache, and
# a base GR00T N1.7 checkpoint with a pretrain-style processor.
import os, sys
from pathlib import Path
root = Path(sys.argv[1]); hf = Path(os.environ["HF_HOME"])
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import Qwen2TokenizerFast, Qwen3VLConfig, Qwen3VLForConditionalGeneration, Qwen3VLProcessor
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode
from transformers.models.qwen2_vl.image_processing_qwen2_vl_fast import Qwen2VLImageProcessorFast
from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor
repo = hf / "hub" / "models--nvidia--Cosmos-Reason2-2B"
snap = repo / "snapshots" / ("0" * 40); snap.mkdir(parents=True, exist_ok=True)
(repo / "refs").mkdir(exist_ok=True); (repo / "refs" / "main").write_text("0" * 40)
specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>",
            "<|image_pad|>", "<|video_pad|>"]
vocab = {c: i for i, c in enumerate(bytes_to_unicode().values())}
tok = Tokenizer(models.BPE(vocab=vocab, merges=[]))
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False); tok.decoder = decoders.ByteLevel()
tok.add_special_tokens(specials)
ids = {s: tok.token_to_id(s) for s in specials}
tmpl = ("{% for message in messages %}<|im_start|>{{ message['role'] }}\\n"
        "{% if message['content'] is string %}{{ message['content'] }}{% else %}"
        "{% for c in message['content'] %}{% if c['type'] == 'image' %}<|vision_start|><|image_pad|><|vision_end|>"
        "{% elif c['type'] == 'text' %}{{ c['text'] }}{% endif %}{% endfor %}{% endif %}<|im_end|>\\n{% endfor %}")
hf_tok = Qwen2TokenizerFast(tokenizer_object=tok, unk_token=None, bos_token=None, eos_token="<|im_end|>",
                            pad_token="<|endoftext|>", chat_template=tmpl)
Qwen3VLProcessor(image_processor=Qwen2VLImageProcessorFast(patch_size=16, merge_size=2, temporal_patch_size=2),
                 tokenizer=hf_tok, video_processor=Qwen3VLVideoProcessor(), chat_template=tmpl).save_pretrained(snap)
H = 64
cfg = Qwen3VLConfig(
    text_config=dict(hidden_size=H, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4,
                     num_key_value_heads=2, head_dim=16, vocab_size=len(vocab) + len(specials),
                     max_position_embeddings=4096,
                     rope_scaling={"rope_type": "default", "mrope_section": [2, 3, 3], "mrope_interleaved": True}),
    vision_config=dict(depth=2, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=H,
                       patch_size=16, spatial_merge_size=2, temporal_patch_size=2,
                       deepstack_visual_indexes=[0], num_position_embeddings=256),
    image_token_id=ids["<|image_pad|>"], video_token_id=ids["<|video_pad|>"],
    vision_start_token_id=ids["<|vision_start|>"], vision_end_token_id=ids["<|vision_end|>"])
torch.manual_seed(0)
Qwen3VLForConditionalGeneration(cfg).save_pretrained(snap)
from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7
from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor
m = Gr00tN1d7Config(backbone_embedding_dim=H, select_layer=2, use_flash_attention=False, hidden_size=64,
                    input_embedding_dim=96, max_seq_len=256,
                    diffusion_model_cfg={"positional_embeddings": None, "num_layers": 4, "num_attention_heads": 2,
                                         "attention_head_dim": 48, "norm_type": "ada_norm", "dropout": 0.2,
                                         "final_dropout": True, "output_dim": 64, "interleave_self_attention": True})
base = root / "base"
Gr00tN1d7(m, transformers_loading_kwargs={}).save_pretrained(base)
Gr00tN1d7Processor(modality_configs={"libero_sim": MODALITY_CONFIGS["libero_sim"]}, max_state_dim=m.max_state_dim,
                   max_action_dim=m.max_action_dim, max_action_horizon=m.action_horizon, use_albumentations=True,
                   image_crop_size=m.image_crop_size, image_target_size=m.image_target_size,
                   transformers_loading_kwargs={}).save_pretrained(base)
"""

# launch_finetune.py exactly as robot/finetune.py runs it; the one patch is
# that tf32/bf16 (CUDA only) are switched off in TrainingArguments.
CPU_LAUNCH = """
import runpy, sys, transformers
_init = transformers.TrainingArguments.__init__
def init(self, *a, **kw):
    kw.update(tf32=False, bf16=False, use_cpu=True)
    _init(self, *a, **kw)
transformers.TrainingArguments.__init__ = init
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
"""

POLICY = """
import json, sys
import numpy as np
from robot.footprint import Candidate, group_of, is_quantizable
from robot.gr00t_policy import Gr00tPolicy, dit_blocks
p = Gr00tPolicy(sys.argv[1], device="cpu", chunk_exec=4)
obs = {"front": np.random.default_rng(0).integers(0, 255, (64, 64, 3), dtype=np.uint8),
       "state": np.array([50, 60, 1.57, 2.0, 0.01, 0.2], np.float32), "instruction": "drive row 3 of 8 northbound"}
ref = p.predict_chunk(obs)
names = [n for n, q in p.model.named_parameters() if is_quantizable(n, q.ndim)]
ungrouped = [n for n in names if group_of(n) is None and "lm_head" not in n]
groups = sorted({group_of(n) for n in names if group_of(n)})
out = {"chunk": list(ref.shape), "ungrouped": ungrouped, "groups": groups,
       "llm_layers": len(p.model.backbone.model.language_model.layers),
       "dit_blocks": len(dit_blocks(p.model)), "steps0": int(p.model.action_head.num_inference_timesteps),
       "keys": [p.video_key, list(p.state_map), p.action_keys, p.language_key]}
lower = {g: "fp8" for g in p.shapes if g != "other"}
for c in [Candidate("steps2", denoise_steps=2), Candidate("fp8", precision=lower),
          Candidate("nvfp4", precision={g: "nvfp4" for g in lower}), Candidate("drop", drop_dit_blocks=[1, 2])]:
    p.set_candidate(c)
    a = p.predict_chunk(obs)
    out[c.name] = {"shape": list(a.shape), "finite": bool(np.isfinite(a).all()),
                   "steps": int(p.model.action_head.num_inference_timesteps), "size_gb": p.footprint["size_gb"],
                   "skipped": sum(type(b).__name__ == "SkipBlock" for b in dit_blocks(p.model))}
p.set_candidate(Candidate("teacher"))
out["restored"] = float(np.abs(p.predict_chunk(obs) - ref).max())
out["teacher_size"] = p.footprint["size_gb"]
act = [p.act(obs) for _ in range(5)]
out["act"] = [list(a.shape) for a in act]
print(json.dumps(out))
"""


def test_finetune_and_policy_end_to_end(dataset, tmp_path):
    _need("torch", "pandas", "torchcodec", "tyro", "omegaconf", "transformers", "torchvision",
          "diffusers", "albumentations", "peft", "tokenizers")
    from robot.finetune import command, find_checkpoint

    env = {"HF_HOME": str(tmp_path / "hf"), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
           "CUDA_VISIBLE_DEVICES": ""}
    _run(TINY, tmp_path, env=env)
    out = tmp_path / "ft"
    cmd = command(dataset, out, 2, str(tmp_path / "base"), batch=2, workers=0, save_steps=2, gr00t_repo=GR00T)
    argv = cmd[cmd.index("gr00t/experiment/launch_finetune.py"):]
    _run(CPU_LAUNCH, *argv, env=env)
    assert (out / "config.json").exists() and (out / "model.safetensors").exists()
    assert (out / "processor" / "processor_config.json").exists()
    assert (out / "checkpoint-2" / "processor_config.json").exists()
    assert find_checkpoint(out) == out

    r = json.loads(_run(POLICY, out, env=env).strip().splitlines()[-1])
    assert r["chunk"] == [16, 2]
    assert r["ungrouped"] == [], r["ungrouped"]
    assert r["llm_layers"] == 2 and r["dit_blocks"] == 4 and r["steps0"] == 4
    assert {"vision", "llm.embed", "llm.1.down", "dit.0.attn", "dit.3.ff", "proj"} <= set(r["groups"])
    assert r["keys"] == ["front", ["x", "y", "heading", "speed", "steer", "row_progress"], ["steer", "speed"],
                         "annotation.human.task_description"]
    for name in ("steps2", "fp8", "nvfp4", "drop"):
        assert r[name]["shape"] == [16, 2] and r[name]["finite"], r[name]
    assert r["steps2"]["steps"] == 2 and r["drop"]["skipped"] == 2
    assert r["nvfp4"]["size_gb"] < r["fp8"]["size_gb"] < r["teacher_size"]
    assert r["restored"] == 0.0
    assert r["act"] == [[2]] * 5
