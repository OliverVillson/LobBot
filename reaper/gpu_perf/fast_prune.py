"""Zero-copy REAP checkpoint rewrite: the same output as reaper_pipeline.ckpt.prune_checkpoint,
without decoding a single tensor.

ckpt.prune_checkpoint reads every tensor into a torch tensor (dropped ones too, to count
bytes), then serialises the kept ones again, one shard after another. A safetensors file is
an 8-byte little-endian header length, a JSON header {name: {dtype, shape, data_offsets}},
then raw bytes. Pruning only drops tensors, renames tensors, or keeps whole rows of a
row-major [E, ...] tensor, and every one of those is a byte range of the source file. So
this module writes the new header and copies byte ranges with os.copy_file_range (in-kernel,
no user-space copy; a reflink on XFS/btrfs), shards in parallel threads.

It needs no torch and no GPU, keeps fp8 / mxfp4 blocks bit-exact by construction, and runs
at disk speed: for an 80 GB checkpoint on one NVMe, roughly the time to read the kept half.

    python fast_prune.py SRC DST saliency.json [--workers 8]

saliency.json is REAPer's work/saliency.json (layers[].layer / layers[].kept).
"""

from __future__ import annotations

import json
import os
import shutil
import struct
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

COPY_CHUNK = 1 << 30


def read_header(path: Path) -> tuple[dict, int]:
    """(header dict, byte offset where tensor data starts)."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def _copy(src_fd: int, dst_fd: int, off: int, length: int) -> None:
    """Copy length bytes from src_fd at off to the current position of dst_fd."""
    while length > 0:
        try:
            n = os.copy_file_range(src_fd, dst_fd, min(length, COPY_CHUNK), offset_src=off)
        except (AttributeError, OSError):
            n = 0
        if n == 0:  # filesystem or kernel without copy_file_range: plain pread/write
            buf = os.pread(src_fd, min(length, 64 << 20), off)
            n = os.write(dst_fd, buf)
        off += n
        length -= n


def rewrite_shard(src: Path, dst: Path, actions: dict, num_experts: int) -> tuple[dict, int, int]:
    """Write the pruned shard. Returns ({name: header entry}, bytes_before, bytes_after)."""
    hdr, base = read_header(src)
    meta = hdr.pop("__metadata__", None)
    before = sum(v["data_offsets"][1] - v["data_offsets"][0] for v in hdr.values())
    plan = []          # (new name, dtype, shape, [(src_off, length), ...])
    for name, v in sorted(hdr.items(), key=lambda kv: kv[1]["data_offsets"][0]):
        a = actions.get(name, "copy")
        s, e = v["data_offsets"]
        if a == "drop":
            continue
        if a == "copy" or a[0] == "rename":
            plan.append((name if a == "copy" else a[1], v["dtype"], v["shape"], [(base + s, e - s)]))
            continue
        rows = a[1]
        if not v["shape"] or v["shape"][0] != num_experts:
            plan.append((name, v["dtype"], v["shape"], [(base + s, e - s)]))
            continue
        row = (e - s) // v["shape"][0]
        ranges = []
        for r in rows:  # merge runs of consecutive kept rows into one range
            o = base + s + r * row
            if ranges and ranges[-1][0] + ranges[-1][1] == o:
                ranges[-1] = (ranges[-1][0], ranges[-1][1] + row)
            else:
                ranges.append((o, row))
        plan.append((name, v["dtype"], [len(rows)] + v["shape"][1:], ranges))
    if not plan:
        return {}, before, 0

    out_hdr, off = {}, 0
    if meta is not None:
        out_hdr["__metadata__"] = meta
    for name, dt, shape, ranges in plan:
        n = sum(l for _, l in ranges)
        out_hdr[name] = {"dtype": dt, "shape": shape, "data_offsets": [off, off + n]}
        off += n
    blob = json.dumps(out_hdr, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)  # safetensors pads the header to 8-byte alignment
    sfd = os.open(src, os.O_RDONLY)
    dfd = os.open(dst, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        os.write(dfd, struct.pack("<Q", len(blob)) + blob)
        for _, _, _, ranges in plan:
            for o, l in ranges:
                _copy(sfd, dfd, o, l)
    finally:
        os.close(sfd)
        os.close(dfd)
    entries = {k: v for k, v in out_hdr.items() if k != "__metadata__"}
    return entries, before, off


def prune_checkpoint(src, dst, keep: dict[int, list[int]], num_experts: int, num_layers: int,
                     workers: int = 8, on_progress=None) -> dict:
    """Drop-in for ckpt.prune_checkpoint (same arguments, same summary)."""
    from reaper_pipeline import ckpt  # key planning and config edits are reused as they are (pip install -e reaper)

    src, dst = Path(src), Path(dst)
    n_keep = len(next(iter(keep.values())))
    if any(len(v) != n_keep for v in keep.values()):
        raise ValueError("every layer must keep the same number of experts")
    shards = {}
    idx = src / "model.safetensors.index.json"
    if idx.exists():
        for k, f in json.loads(idx.read_text())["weight_map"].items():
            shards.setdefault(f, []).append(k)
    else:
        shards = {"model.safetensors": [k for k in read_header(src / "model.safetensors")[0] if k != "__metadata__"]}
    actions, notes = ckpt.plan_keys([k for ks in shards.values() for k in ks], keep, num_layers)

    tmp = dst.with_name(dst.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    weight_map, before, total, done = {}, 0, 0, 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futs = {pool.submit(rewrite_shard, src / f, tmp / f, actions, num_experts): f for f in shards}
        for fut in futs:
            entries, b, a = fut.result()
            f = futs[fut]
            before += b
            total += a
            if not entries:
                (tmp / f).unlink(missing_ok=True)
            weight_map.update({k: f for k in entries})
            done += 1
            if on_progress:
                on_progress(done, len(shards), f)

    if idx.exists() or len(shards) > 1:
        (tmp / "model.safetensors.index.json").write_text(json.dumps(
            {"metadata": {"total_size": total}, "weight_map": dict(sorted(weight_map.items()))}, indent=2) + "\n")
    for p in src.iterdir():
        if p.name in ckpt.SKIP_DIRS or p.is_dir() or p.name.endswith(ckpt.WEIGHT_SUFFIXES) \
                or p.name in ("model.safetensors.index.json", "config.json"):
            continue
        shutil.copy2(p, tmp / p.name)
    cfg = json.loads((src / "config.json").read_text())
    dropped_mtp = any("multi-token" in n for n in notes)
    (tmp / "config.json").write_text(json.dumps(ckpt.prune_config(cfg, n_keep, dropped_mtp), indent=2) + "\n")
    shutil.rmtree(dst, ignore_errors=True)
    tmp.rename(dst)
    return {"bytes_before": before, "bytes_after": total, "notes": notes}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("saliency")
    ap.add_argument("--num-layers", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    s = json.loads(Path(a.saliency).read_text())
    keep = {l["layer"]: l["kept"] for l in s["layers"]}
    cfg = json.loads((Path(a.src) / "config.json").read_text())
    tc = cfg.get("text_config", cfg)
    t0 = time.time()
    info = prune_checkpoint(a.src, a.dst, keep, s["experts"], a.num_layers or tc["num_hidden_layers"], a.workers,
                            lambda i, n, f: print(f"shard {i}/{n} {f}", flush=True))
    gb = 1024 ** 3
    print(f"{info['bytes_before'] / gb:.1f} GB -> {info['bytes_after'] / gb:.1f} GB in {time.time() - t0:.1f}s")
