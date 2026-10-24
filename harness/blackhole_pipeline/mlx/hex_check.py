"""Check the HEX Metal kernel on an Apple-silicon Mac: same numbers as the reference path, and its
speed next to MLX's own 3/4-bit expert matmul (gather_qmm) on Flint-sized experts.

    pip install mlx
    python hex_check.py            # from this folder (it imports flint_mlx.py next to it)
"""

import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent))
import flint_mlx as F  # noqa: E402


def random_hex(E, out, inp, bits):
    nblk = inp // F.HEX_BLOCK
    idx = mx.random.randint(0, 2 ** 31, (E, out, nblk * bits)).astype(mx.uint32)
    codes = mx.random.randint(40, 70, (E, out, nblk)).astype(mx.uint8)       # fp8 e4m3, about 0.1 to 1.0
    es = mx.random.uniform(0.5, 1.5, (E,))
    levels = mx.sort(mx.random.uniform(-1, 1, (E, 2 ** bits)), axis=-1)
    return idx, codes, es, levels


def bench(fn, n=20, chain=48):
    """Microseconds per call, `chain` calls per evaluation (as in decoding: one per layer)."""
    for _ in range(3):
        mx.eval([fn() for _ in range(chain)])
    t = time.perf_counter()
    for _ in range(n):
        mx.eval([fn() for _ in range(chain)])
    return (time.perf_counter() - t) / (n * chain) * 1e6


def main():
    assert mx.metal.is_available(), "needs an Apple-silicon GPU"
    E, H, I, k = 160, 2048, 512, 6
    ok = True
    for bits in (3, 4):
        for inp, out in ((H, I), (I, H)):
            lin = F.HexSwitchLinear(inp, out, E, bits)
            lin.idx, lin.block_scale, lin.expert_scale, lin.levels = random_hex(E, out, inp, bits)
            for T, shared in ((1, True), (7, True), (5, False)):
                R = T * k
                x = mx.random.normal((T if shared else R, inp)).astype(mx.bfloat16)
                ids = mx.random.randint(0, E, (R,)).astype(mx.uint32)
                ids = mx.where(mx.arange(R) % 4 == 3, mx.array(F.HEX_SKIP, dtype=mx.uint32), ids)
                got = lin(x, ids, shared).astype(mx.float32)
                with mx.stream(mx.cpu):
                    live = ids != F.HEX_SKIP
                    e = mx.where(live, ids, 0)
                    W = F.hex_dequant(lin.idx[e], lin.block_scale[e], lin.expert_scale[e], lin.levels[e], bits)
                    xr = mx.repeat(x, R // x.shape[0], axis=0).astype(mx.float32)
                    ref = mx.where(live[:, None], (W @ xr[..., None])[..., 0], 0.0)
                err = float(mx.abs(got - ref).max() / (mx.abs(ref).max() + 1e-6))
                print(f"bits {bits} in {inp:4d} out {out:4d} rows {R:3d}: max rel error {err:.2e}")
                ok &= err < 2e-2
            # decode speed: one token, 6 experts
            x = mx.random.normal((1, inp)).astype(mx.bfloat16)
            ids = mx.random.randint(0, E, (k,)).astype(mx.uint32)
            t_hex = bench(lambda: lin(x, ids, True))
            w = mx.random.normal((E, out, inp)).astype(mx.bfloat16)
            wq, s, b = mx.quantize(w, group_size=64, bits=bits)
            xi = x.reshape(1, 1, inp)
            t_aff = bench(lambda: mx.gather_qmm(xi, wq, s, b, rhs_indices=ids.reshape(1, k), transpose=True,
                                                group_size=64, bits=bits))
            gb = k * out * inp * (bits + 0.25) / 8 / 1e9
            print(f"  decode, {k} experts: HEX {t_hex:7.1f} us ({gb / (t_hex * 1e-6):5.1f} GB/s) · "
                  f"MLX affine {bits}-bit {t_aff:7.1f} us")
    print("HEX kernel OK" if ok else "HEX kernel MISMATCH: send this output")


if __name__ == "__main__":
    main()
