"""Where does prompt processing (prefill) time go in an exported Flint MLX model? Run on the Mac:

    python prefill_profile.py ~/flint/flint-REAP-128e-MLX-9.1GB

Times a 1024-token prefill of the whole model, then the same with parts switched off, and one MoE
layer's experts alone (both REAQ tiers as exported vs every slot through a single tier).
"""

import sys
import time

import mlx.core as mx
from mlx_lm.utils import load_model


def timed(fn, n=3):
    mx.eval(fn())
    t = time.perf_counter()
    for _ in range(n):
        mx.eval(fn())
    return (time.perf_counter() - t) / n


def main():
    path = sys.argv[1]
    T = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
    from pathlib import Path
    model, _ = load_model(Path(path).expanduser(), trust_remote_code=True)
    ids = mx.random.randint(0, min(20000, model.args.vocab_size), (1, T))
    full = timed(lambda: model(ids))
    print(f"whole model, {T}-token prefill: {full:.2f} s = {T / full:.0f} tok/s")

    moes = [l.mlp for l in model.model.layers if hasattr(l.mlp, "gate")]
    cls = type(moes[0])
    orig = cls.__call__
    cls.__call__ = lambda self, x: self.shared_expert(x)          # routed experts off
    no_exp = timed(lambda: model(ids))
    cls.__call__ = orig
    print(f"  without routed experts:        {no_exp:.2f} s  -> routed experts cost {full - no_exp:.2f} s")

    m = moes[0]
    x = mx.random.normal((1, T, model.args.hidden_size)).astype(mx.bfloat16)
    layer = timed(lambda: m(x), 5)
    scores = mx.sigmoid(m.gate(x).astype(mx.float32))
    k = m.k
    ids_ = mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]
    gp = m.experts.gate_proj
    n_cold = (gp.weight if hasattr(gp, "weight") else gp.idx).shape[0]
    single = timed(lambda: m.experts(x, mx.minimum(ids_, n_cold - 1)), 5)
    print(f"one MoE layer as exported: {layer * 1e3:.1f} ms; all slots through one tier: {single * 1e3:.1f} ms"
          f"  (x{len(moes)} layers)")
    print(f"tiers: {getattr(m, 'n_hot', 0)} hot experts; expert class {type(m.experts).__name__}")


if __name__ == "__main__":
    main()
