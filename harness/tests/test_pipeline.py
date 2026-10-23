"""blackhole_pipeline: bit packing, and a full run on a tiny random Flint (CPU) that reuses an
earlier REAPer job's calibration and scores."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("reaper_pipeline")

from blackhole_pipeline import STAGES  # noqa: E402
from blackhole_pipeline import pack  # noqa: E402
from blackhole_tui.catalog import find_schema  # noqa: E402
from blackhole_tui.jobs import parse_event  # noqa: E402

REAPER = Path(__import__("reaper_pipeline").__file__).resolve().parents[1]
HARNESS = Path(__file__).resolve().parents[1]


def test_int4_pack_roundtrip_and_layout():
    codes = torch.randint(-8, 8, (5, 40), dtype=torch.int8)
    p = pack.pack_int4(codes)
    assert p.dtype == torch.int32 and p.shape == (5, 5)
    assert torch.equal(pack.unpack_int4(p, 40), codes)
    one = torch.full((1, 8), -8, dtype=torch.int8)
    one[0, 1] = 7  # element 1 sits in bits 4..7, stored offset by 8
    assert pack.pack_int4(one).item() == 15 << 4


def test_int_codes_from_gptq_scales():
    sys.path.insert(0, str(REAPER / "gpu_perf"))
    from gptq import gptq_batched
    w = torch.randn(2, 16, 256)
    x = torch.randn(512, 256)
    H = (x.T @ x)[None].repeat(2, 1, 1)
    Q, s = gptq_batched(w, H, "int4_g128", return_scales=True)
    assert s.shape == (2, 16, 2)
    c = pack.int_codes(Q, s, 128)
    assert torch.allclose(c.float() * s.repeat_interleave(128, -1), Q, atol=1e-5)
    assert torch.equal(gptq_batched(w, H, "int4_g128"), Q)


def test_codebook_roundtrip():
    sys.path.insert(0, str(REAPER / "gpu_perf"))
    from headers import header_parts, round_to_table
    W = torch.randn(3, 8, 64)
    for L in (16, 8):
        S, Lv = header_parts(W, "expert", L, 32)
        Q = round_to_table(W, S, Lv)
        es = pack.expert_scale(W)
        back = pack.codebook_dequant(pack.pack_nibbles(pack.table_indices(Q, S, Lv)), 64,
                                     pack.block_codes(S, es, 32), es, Lv, 32)
        assert torch.allclose(back, Q, rtol=1e-5, atol=1e-7)


def test_lab_schema_stages_match_the_pipeline():
    sc = find_schema("reaq-codebook")
    assert sc.stages() == STAGES and "lab" in sc.tags


@pytest.mark.parametrize("donor", ["scores", "data-only"])
def test_full_run_on_a_tiny_flint(tmp_path, donor):
    pytest.importorskip("transformers")
    sys.path.insert(0, str(REAPER / "tests"))
    from tiny_moe import build
    model = build("flint", tmp_path / "flint-tiny")
    home = tmp_path / "home"
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(REAPER), str(HARNESS)]), "REAPER_HW": "builtin",
           "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "REAPER_HOME": str(home)}
    small = {"calib": "mix", "heldout": 4, "max_seq": 96, "answer_max_tokens": 12, "writer_max_tokens": 24,
             "teacher_backend": "hf"}

    # an earlier REAPer run at 50% on the same model and purpose (like ~/.reaper/jobs/flint-prune50)
    old = home / "jobs" / "flint-tiny-prune50"
    old.mkdir(parents=True)
    (old / "config.json").write_text(json.dumps({"model": str(model), "purpose": "", "prune": 0.5,
                                                 "calib_samples": 8, **small}))
    p = subprocess.run([sys.executable, "-m", "reaper_pipeline", "--job", str(old)], env=env, cwd=tmp_path,
                       capture_output=True, text=True, timeout=900)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]

    if donor == "data-only":  # e.g. a job whose scoring never finished: reuse the teacher answers only
        (old / ".done" / "score").unlink()
        (old / "work" / "teacher_ref.pt").unlink()

    sc = find_schema("reaq-codebook")
    job = tmp_path / "bh" / "flint-tiny-reaq"
    job.mkdir(parents=True)
    (job / "config.json").write_text(json.dumps({
        "model": str(model), "schema": sc.id, "target": "rtx4090", "purpose": "",
        "steps": [s.__dict__ for s in sc.steps], "quant": sc.quant, "calib_samples": 8, **small}))
    p = subprocess.run([sys.executable, "-m", "blackhole_pipeline", "--job", str(job)], env=env, cwd=tmp_path,
                       capture_output=True, text=True, timeout=900)
    events = [e for e in map(parse_event, p.stdout.splitlines()) if e]
    assert p.returncode == 0, p.stdout[-4000:] + p.stderr[-3000:]
    by = {}
    for e in events:
        by.setdefault(e["stage"], []).append(e)
    assert by["calibrate"][-1]["status"] == "skipped" and "reused" in by["calibrate"][-1]["msg"]
    if donor == "scores":
        assert all(e["status"] == "skipped" for e in by["score"])  # scores reused, only prune ran
    else:
        assert by["score"][-1]["status"] == "done"
        assert "calibration usable, REAP scores not" in p.stdout
    for s in ("detect", "plan", "prune", "eval", "gptq", "reaq", "quant", "package"):
        assert by[s][-1]["status"] == "done", (s, by[s][-1])
    ev = by["eval"][-1]
    assert {"top1", "kl", "ppl_teacher", "ppl_student"} <= set(ev)
    assert "kl" in by["gptq"][-1] and "kl" in by["reaq"][-1]
    final = by["pipeline"][-1]
    assert final["status"] == "done" and final["outcome"] == "done"
    gdir = Path(final["model"])
    assert gdir.name.endswith("-REAP-6e-GPTQ-W4A16")  # 8 experts, 25% pruned

    # the int4 checkpoint: compressed-tensors config, packed experts that dequantize near the bf16 ones
    from safetensors import safe_open
    cfg = json.loads((gdir / "config.json").read_text())
    assert cfg["quantization_config"]["format"] == "pack-quantized" and cfg["num_experts"] == 6
    idx = json.loads((gdir / "model.safetensors.index.json").read_text())["weight_map"]
    bf16 = gdir.with_name(gdir.name[:-len("-GPTQ-W4A16")])
    bidx = json.loads((bf16 / "model.safetensors.index.json").read_text())["weight_map"] \
        if (bf16 / "model.safetensors.index.json").exists() else None
    key = next(k for k in idx if k.endswith("experts.0.gate_proj.weight_packed"))
    base = key[:-len(".weight_packed")]
    assert f"{base}.weight" not in idx
    with safe_open(gdir / idx[key], "pt") as f:
        packed, scale, shape = f.get_tensor(key), f.get_tensor(base + ".weight_scale"), f.get_tensor(base + ".weight_shape")
    out, n = shape.tolist()
    deq = pack.unpack_int4(packed, n).float() * scale.float().repeat_interleave(128, -1)[:, :n]
    bfile = bf16 / (bidx[base + ".weight"] if bidx else "model.safetensors")
    with safe_open(bfile, "pt") as f:
        w = f.get_tensor(base + ".weight").float()
    assert deq.shape == w.shape and (deq - w).norm() / w.norm() < 0.25

    # the REAQ codebook folder: tiers by REAP rank, dequantizes near the bf16 experts
    qdir = gdir.with_name(bf16.name + "-REAQ-codebook")
    assert (qdir / "REAQ_LAYOUT.txt").exists()
    qidx = json.loads((qdir / "model.safetensors.index.json").read_text())["weight_map"]
    t16 = [k for k in qidx if ".tier16.experts" in k]
    t8 = [k for k in qidx if ".tier8.experts" in k]
    assert t16 and len(t16) == len(t8)
    rec = json.loads((job / "receipt.json").read_text())
    assert rec["results"]["reaq_codebook"]["bpw"] == pytest.approx(3.75)
    assert rec["results"]["gptq_int4"]["bpw"] == pytest.approx(4.125)
    assert (job / "out" / gdir.name).is_symlink() and (job / "RESULTS.md").exists()
