"""Where does prompt processing (prefill) time go in an exported Flint MLX model? Run on the Mac:

    python prefill_profile.py ~/flint/flint-REAP-128e-MLX-9.1GB [tokens] [--wire]

Prefills real text (mlx-lm's sources) and reports: the whole model; without routed experts; then
every MoE layer timed on its own real input, as exported and with variants (both tiers on all
slots, one tier only), plus how many distinct experts the tokens use.
"""

import inspect
import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm.utils import load_model, load_tokenizer


def timed(fn, n=3):
    mx.eval(fn())
    t = time.perf_counter()
    for _ in range(n):
        mx.eval(fn())
    return (time.perf_counter() - t) / n


def main():
    path = Path(sys.argv[1]).expanduser()
    nums = [a for a in sys.argv[2:] if a.isdigit()]
    T = int(nums[0]) if nums else 1024
    info = mx.metal.device_info() if hasattr(mx.metal, "device_info") else mx.device_info()
    rec = int(info.get("max_recommended_working_set_size", 0))
    print(f"RAM {info.get('memory_size', 0) / 2**30:.1f} GB, GPU working set limit {rec / 2**30:.1f} GB")
    if "--wire" in sys.argv:   # pin the model in memory (what mlx_lm.generate does for big models)
        mx.set_wired_limit(rec)
        print(f"wired limit set to {rec / 2**30:.1f} GB")
    model, _ = load_model(path, trust_remote_code=True)
    mx.eval(model.parameters())
    print(f"model in memory: {mx.get_active_memory() / 2**30:.2f} GB")
    tok = load_tokenizer(path)
    import mlx_lm.generate as g
    import mlx_lm.utils as u
    text = inspect.getsource(g) + inspect.getsource(u)
    ids = mx.array(tok.encode(text)[:T])[None]
    T = ids.shape[1]
    full = timed(lambda: model(ids))
    print(f"whole model, {T}-token prefill of real text: {full:.2f} s = {T / full:.0f} tok/s")

    moes = [l.mlp for l in model.model.layers if hasattr(l.mlp, "gate")]
    cls = type(moes[0])
    orig = cls.__call__
    inputs = []

    def record(self, x):
        inputs.append(x)
        return orig(self, x)
    cls.__call__ = lambda self, x: self.shared_expert(x)
    no_exp = timed(lambda: model(ids))
    cls.__call__ = record
    mx.eval(model(ids))
    cls.__call__ = orig
    print(f"  without routed experts: {no_exp:.2f} s -> routed experts {full - no_exp:.2f} s")

    tot = {"as exported": 0.0, "both tiers, all slots": 0.0, "one tier only": 0.0}
    for li, (m, x) in enumerate(zip(moes, inputs)):
        scores = mx.sigmoid(m.gate(x).astype(mx.float32))
        e = mx.argpartition(-scores, kth=m.k - 1, axis=-1)[..., :m.k]
        uniq = len(set(e.reshape(-1).tolist()))
        a = timed(lambda: m(x))
        saved = m.split_min
        m.split_min = 1 << 30
        b = timed(lambda: m(x))
        m.split_min = saved
        gp = m.experts.gate_proj
        n_cold = (gp.weight if hasattr(gp, "weight") else gp.idx).shape[0]
        c = timed(lambda: m.experts(x, mx.minimum(e, n_cold - 1)))
        tot["as exported"] += a
        tot["both tiers, all slots"] += b
        tot["one tier only"] += c
        if li < 4 or li % 12 == 0:
            print(f"  layer {li:2d}: {uniq:3d} distinct experts · as exported {a * 1e3:6.1f} ms · both tiers "
                  f"{b * 1e3:6.1f} ms · one tier {c * 1e3:6.1f} ms")
    for k_, v in tot.items():
        print(f"sum over {len(moes)} MoE layers, {k_}: {v:.2f} s")
    print(f"mlx {mx.__version__}; expert class {type(moes[0].experts).__name__}; "
          f"cold tier bits {getattr(moes[0].experts.gate_proj, 'bits', '?')}")


if __name__ == "__main__":
    main()
