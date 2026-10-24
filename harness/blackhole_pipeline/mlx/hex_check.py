"""Check the HEX Metal kernels on an Apple-silicon Mac: same numbers as the reference path, and
their speed next to MLX's own 3/4-bit expert matmul (gather_qmm) on Flint-sized experts, for one
decoded token and for a long prompt.

    pip install mlx
    python hex_check.py            # from this folder (it imports flint_mlx.py next to it)
"""

import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import flint_mlx as F  # noqa: E402


def random_hex(lin):
    E, out, nw = lin.idx.shape
    lin.idx = mx.random.randint(0, 2 ** 31, (E, out, nw)).astype(mx.uint32)
    lin.block_scale = mx.random.randint(40, 70, (E, out, nw // lin.bits)).astype(mx.uint8)  # fp8 e4m3 ~0.1 to 1
    lin.expert_scale = mx.random.uniform(0.5, 1.5, (E,))
    lin.levels = mx.sort(mx.random.uniform(-1, 1, (E, 2 ** lin.bits)), axis=-1)


def reference(lin, x, ids):
    with mx.stream(mx.cpu):
        live = ids != F.HEX_SKIP
        e = mx.where(live, ids, 0)
        W = F.hex_dequant(lin.idx[e], lin.block_scale[e], lin.expert_scale[e], lin.levels[e], lin.bits)
        xr = mx.repeat(x, ids.shape[0] // x.shape[0], axis=0).astype(mx.float32)
        return mx.where(live[:, None], (W @ xr[..., None])[..., 0], 0.0)


def rel(got, ref):
    return float(mx.abs(got.astype(mx.float32) - ref).max() / (mx.abs(ref).max() + 1e-6))


def bench(fn, n=20, chain=48):
    """Microseconds per call, `chain` calls per evaluation (as in decoding: one per layer)."""
    for _ in range(3):
        mx.eval([fn() for _ in range(chain)])
    t = time.perf_counter()
    for _ in range(n):
        mx.eval([fn() for _ in range(chain)])
    return (time.perf_counter() - t) / (n * chain) * 1e6


def affine_glu(E, H, I, bits):
    ws = [mx.quantize(mx.random.normal((E, o, i)).astype(mx.bfloat16) * 0.02, group_size=64, bits=bits)
          for o, i in ((I, H), (I, H), (H, I))]

    def run(x, ids, sort=False):   # x [N, H], ids [N, k], like mlx-lm's SwitchGLU
        k = ids.shape[-1]
        xe = x[:, None, None, :]
        idx = ids
        if sort:
            flat = ids.reshape(-1)
            order = mx.argsort(flat)
            idx, xe = flat[order], x[order // k][:, None, :]
        mm = lambda a, w: mx.gather_qmm(a, *w, rhs_indices=idx, transpose=True, group_size=64,  # noqa: E731
                                        bits=bits, sorted_indices=sort)
        a = nn.silu(mm(xe, ws[0])) * mm(xe, ws[1])
        y = mm(a, ws[2])
        return y[mx.argsort(order)] if sort else y
    return run


def main():
    assert mx.metal.is_available(), "needs an Apple-silicon GPU"
    E, H, I, k = 160, 2048, 512, 6
    ok = True
    for bits in (3, 4):
        glu = F.HexSwitchGLU(H, I, E, bits)
        for lin in (glu.gate_proj, glu.up_proj, glu.down_proj):
            random_hex(lin)
        errs = []
        for T, skip in ((1, False), (7, True), (40, False)):
            R = T * k
            x = mx.random.normal((T, H)).astype(mx.bfloat16)
            ids = mx.random.randint(0, E, (R,)).astype(mx.uint32)
            if skip:
                ids = mx.where(mx.arange(R) % 4 == 3, mx.array(F.HEX_SKIP, dtype=mx.uint32), ids)
            g, u = reference(glu.gate_proj, x, ids), reference(glu.up_proj, x, ids)
            errs.append(rel(glu.gate_proj(x, ids), g))
            a_ref = g / (1 + mx.exp(-g)) * u
            errs.append(rel(glu.gate_proj._matvec(x, ids, glu.up_proj), a_ref))
            a = a_ref.astype(mx.bfloat16)
            errs.append(rel(glu.down_proj(a, ids, False), reference(glu.down_proj, a, ids)))
            yd = glu._dense(x, ids, k)
            yr = reference(glu.down_proj, (nn.silu(g) * u).astype(mx.bfloat16), ids)
            errs.append(rel(yd, yr))
        with mx.stream(mx.cpu):
            wr = F.hex_dequant(*glu.down_proj.parts(), bits)
        errs.append(rel(glu.down_proj.dequant(mx.float32), wr))
        err = max(errs)
        print(f"bits {bits}: max rel error over matvec / fused gate-up / down / prompt path / dequant: {err:.2e}")
        ok &= err < 2e-2

        aff = affine_glu(E, H, I, bits)
        x = mx.random.normal((1, H)).astype(mx.bfloat16)
        ids = mx.random.randint(0, E, (1, k)).astype(mx.uint32)
        t_hex = bench(lambda: glu(x, ids))
        t_aff = bench(lambda: aff(x, ids))
        gb = 3 * k * H * I * (bits + 0.25) / 8 / 1e9
        print(f"  decode, one token, {k} experts (gate+up+down): HEX {t_hex:6.1f} us ({gb / (t_hex * 1e-6):5.1f} GB/s)"
              f" · MLX affine {bits}-bit {t_aff:6.1f} us")
        T = 1024
        x = mx.random.normal((T, H)).astype(mx.bfloat16)
        ids = mx.random.randint(0, E, (T, k)).astype(mx.uint32)
        t_dense = bench(lambda: glu(x, ids), n=5, chain=8)
        saved = glu.dequant_min_per_expert
        glu.dequant_min_per_expert = 1 << 30
        t_mv = bench(lambda: glu(x, ids), n=2, chain=4)
        glu.dequant_min_per_expert = saved
        t_aff = bench(lambda: aff(x, ids, sort=True), n=5, chain=8)
        print(f"  prompt, {T} tokens: HEX dequant+matmul {t_dense / 1e3:6.1f} ms · HEX matvec {t_mv / 1e3:6.1f} ms"
              f" · MLX affine {bits}-bit {t_aff / 1e3:6.1f} ms  (per MoE layer)")
    print("HEX kernels OK" if ok else "HEX kernel MISMATCH: send this output")


if __name__ == "__main__":
    main()
