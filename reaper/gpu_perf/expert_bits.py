"""Per-expert bit allocation from REAP scores: pruning and quantization as one decision.

Every routed expert j of layer l gets one format from a menu (drop, 2/3/4/8-bit, bf16), so
REAP pruning is the 0-bit tier of the same problem. The allocator minimises the expected
damage to the layer output under two budgets a personal PC actually has:

  storage   sum_j params_j * bpw_j / 8                 must fit RAM/VRAM   (GB on disk)
  traffic   sum_j p_j * params_j * bpw_j / 8           bytes read per token (decode speed)

where p_j = tokens routed to j / tokens (per layer the p_j sum to top-k). Rarely routed
experts dominate storage but barely touch traffic, frequently routed ones the opposite, so
the two budgets pull the allocation in different directions and both matter.

Damage of storing expert j in format b:
  D_j(b) = p_j * E[g_j^2 ||f_j - f_j^b||^2]       measured by probe() on reservoir samples
  D_j(b) ~ p_j * S_j^2 * rel_mse(b)               from REAP saliency alone (no probe)
with S_j = mean over routed tokens of g_j ||f_j|| (REAP) and rel_mse(drop) = 1.

Solved by Lagrangian relaxation of the multiple-choice knapsack: each expert independently
picks argmin_b D_j(b) + lam * storage_j(b) + nu * traffic_j(b), with lam and nu bisected
to meet the two budgets, then a greedy pass that spends what is left on the best upgrades. Runtime
constraints from the serving kernels are enforced afterwards: at most `max_tiers` formats
per layer (each tier is one grouped-GEMM launch) and, optionally, the same number of
dropped experts in every layer (Hugging Face configs carry one expert count).

    python expert_bits.py saliency.json config.json --budget-gb 14 --menu gguf
    python expert_bits.py saliency.json config.json --budget-gb 40 --menu blackwell
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

# relative weight MSE per format. GGUF rows: LobBot's bits.py, measured on Qwen3-MoE experts
# with llama.cpp (2026-10-03, no imatrix). Others: fakequant.py round-to-nearest on Gaussian
# weights (test_fake_quant_error_ladder); replace with probe() measurements per model.
REL_MSE = {
    "drop": 1.0,
    "q2_k": 0.0876, "q3_k": 0.0227, "q4_k": 0.00507, "q5_k": 0.00130, "q6_k": 0.000314, "q8_0": 0.0000196,
    "int2_g64": 0.256, "int3_g64": 0.063, "int4_g128": 0.0138, "mxfp4": 0.0131, "nvfp4": 0.0092,
    "int8_g128": 0.000042, "fp8_block": 0.00070, "bf16": 0.0,
}
BPW = {
    "drop": 0.0,
    "q2_k": 2.625, "q3_k": 3.4375, "q4_k": 4.5, "q5_k": 5.5, "q6_k": 6.5625, "q8_0": 8.5,
    "int2_g64": 2.25, "int3_g64": 3.25, "int4_g128": 4.125, "mxfp4": 4.25, "nvfp4": 4.5,
    "int8_g128": 8.125, "fp8_block": 8.002, "bf16": 16.0,
}
# what each runtime / GPU generation can serve as an expert tier with a fast fused MoE kernel
MENUS = {
    "gguf":      ["drop", "q2_k", "q3_k", "q4_k", "q5_k", "q6_k", "q8_0"],   # llama.cpp (needs the tier patch)
    "ada":       ["drop", "int4_g128", "fp8_block", "bf16"],                  # Marlin W4A16, FP8 (sm_89)
    "hopper":    ["drop", "int4_g128", "mxfp4", "fp8_block", "bf16"],         # Marlin/Machete, triton mxfp4, DeepGEMM fp8
    "blackwell": ["drop", "nvfp4", "mxfp4", "fp8_block", "bf16"],             # native FP4/FP8 tensor cores
}


@dataclass
class Expert:
    layer: int
    j: int
    p: float          # routing probability per token (sum over a layer = top-k)
    s: float          # REAP saliency
    params: int
    measured: dict[str, float] | None = None   # fmt -> D_j(b) from probe()

    def damage(self, fmt: str) -> float:
        if self.measured and fmt in self.measured:
            return self.measured[fmt]
        return self.p * self.s * self.s * REL_MSE[fmt]

    def storage(self, fmt: str) -> float:
        return self.params * BPW[fmt] / 8

    def traffic(self, fmt: str) -> float:
        return self.p * self.storage(fmt)


@dataclass
class Allocation:
    choice: dict[tuple[int, int], str] = field(default_factory=dict)
    storage: float = 0.0
    traffic: float = 0.0
    damage: float = 0.0


def experts_from_saliency(sal: dict, cfg: dict) -> list[Expert]:
    c = cfg.get("text_config", cfg)
    H = c["hidden_size"]
    inter = c.get("moe_intermediate_size") or c["intermediate_size"]
    params = 3 * H * inter
    out = []
    for L in sal["layers"]:
        tot = sum(L["freq"]) or 1.0
        k = c.get("num_experts_per_tok", 1)
        for j, (s, f) in enumerate(zip(L["saliency"], L["freq"])):
            out.append(Expert(L["layer"], j, k * f / tot, s, params))
    return out


def _evaluate(experts, choice) -> Allocation:
    a = Allocation(choice=choice)
    for e in experts:
        f = choice[(e.layer, e.j)]
        a.storage += e.storage(f)
        a.traffic += e.traffic(f)
        a.damage += e.damage(f)
    return a


def _pick(experts, menu, lam, nu):
    return {(e.layer, e.j): min(menu, key=lambda f: e.damage(f) + lam * e.storage(f) + nu * e.traffic(f))
            for e in experts}


def _fit_lam(experts, menu, budget, nu):
    """Smallest storage price lam whose choice fits the storage budget (given traffic price nu)."""
    lo, hi = 0.0, 1e-12
    while _evaluate(experts, _pick(experts, menu, hi, nu)).storage > budget and hi < 1e30:
        hi *= 8
    for _ in range(50):
        mid = math.sqrt(lo * hi) if lo > 0 else hi / 1e6
        if _evaluate(experts, _pick(experts, menu, mid, nu)).storage > budget:
            lo = mid
        else:
            hi = mid
    return hi


def allocate(experts: list[Expert], menu: list[str], budget_bytes: float, traffic_bytes: float | None = None,
             max_tiers: int = 3, uniform_drop: bool = False) -> Allocation:
    """Formats per expert minimising damage with storage <= budget_bytes and, when given,
    bytes read per token <= traffic_bytes (two Lagrange prices, lam for storage, nu for traffic)."""
    nu = 0.0
    lam = _fit_lam(experts, menu, budget_bytes, nu)
    if traffic_bytes is not None and _evaluate(experts, _pick(experts, menu, lam, nu)).traffic > traffic_bytes:
        lo, hi = 0.0, lam
        while True:
            l2 = _fit_lam(experts, menu, budget_bytes, hi)
            if _evaluate(experts, _pick(experts, menu, l2, hi)).traffic <= traffic_bytes or hi > 1e30:
                break
            lo, hi = hi, hi * 8
        for _ in range(30):
            mid = math.sqrt(lo * hi) if lo > 0 else hi / 1e6
            l2 = _fit_lam(experts, menu, budget_bytes, mid)
            if _evaluate(experts, _pick(experts, menu, l2, mid)).traffic > traffic_bytes:
                lo = mid
            else:
                hi = mid
        nu, lam = hi, _fit_lam(experts, menu, budget_bytes, hi)
    choice = _pick(experts, menu, lam, nu)
    choice = _limit_tiers(experts, choice, menu, max_tiers)
    if uniform_drop:
        choice = _uniform_drop(experts, choice, menu)
    return _greedy_fill(experts, choice, menu, budget_bytes, traffic_bytes or float("inf"))


def _limit_tiers(experts, choice, menu, max_tiers):
    """Merge the least used formats of a layer into the next format up (never more bytes saved
    than planned is lost: the expert moves to the nearest larger format)."""
    by_layer: dict[int, list[Expert]] = {}
    for e in experts:
        by_layer.setdefault(e.layer, []).append(e)
    order = {f: i for i, f in enumerate(menu)}
    for L, es in by_layer.items():
        while True:
            used = sorted({choice[(L, e.j)] for e in es if choice[(L, e.j)] != "drop"}, key=order.get)
            if len(used) <= max_tiers:
                break
            counts = {f: sum(1 for e in es if choice[(L, e.j)] == f) for f in used}
            f = min(used[:-1], key=counts.get)  # rarest non-top format
            up = used[used.index(f) + 1]
            for e in es:
                if choice[(L, e.j)] == f:
                    choice[(L, e.j)] = up
    return choice


def _uniform_drop(experts, choice, menu):
    """Same number of dropped experts per layer (the mean), lowest damage-if-dropped first."""
    by_layer: dict[int, list[Expert]] = {}
    for e in experts:
        by_layer.setdefault(e.layer, []).append(e)
    n = round(sum(1 for v in choice.values() if v == "drop") / len(by_layer))
    keep_fmt = next(f for f in menu if f != "drop")
    for L, es in by_layer.items():
        ranked = sorted(es, key=lambda e: e.damage("drop"))
        for i, e in enumerate(ranked):
            if i < n:
                choice[(L, e.j)] = "drop"
            elif choice[(L, e.j)] == "drop":
                choice[(L, e.j)] = keep_fmt
    return choice


def _greedy_fill(experts, choice, menu, budget, tbudget):
    """Spend leftover budget on the upgrades with the best damage saved per byte, staying
    within the formats each layer already uses (so the tier count does not grow)."""
    import heapq
    order = {f: i for i, f in enumerate(menu)}
    used: dict[int, set] = {}
    for (L, _), v in choice.items():
        used.setdefault(L, set()).add(v)

    def step(e, f, direction):
        seq = menu[order[f] + 1:] if direction > 0 else list(reversed(menu[:order[f]]))
        return next((g for g in seq if g in used[e.layer]), None)

    def extra(e, f, g):
        return e.storage(g) - e.storage(f)

    a = _evaluate(experts, choice)
    spent, tspent = a.storage, a.traffic
    if spent > budget or tspent > tbudget:  # tier merging overshot: step down where it hurts least per byte saved
        heap = []
        for e in experts:
            f = choice[(e.layer, e.j)]
            d = step(e, f, -1)
            if d is not None:
                heapq.heappush(heap, ((e.damage(d) - e.damage(f)) / max(-extra(e, f, d), 1e-9), id(e), e, f, d))
        while (spent > budget or tspent > tbudget) and heap:
            _, _, e, f, d = heapq.heappop(heap)
            if choice[(e.layer, e.j)] != f:
                continue
            spent += extra(e, f, d)
            tspent += e.p * extra(e, f, d)
            choice[(e.layer, e.j)] = d
            d2 = step(e, d, -1)
            if d2 is not None:
                heapq.heappush(heap, ((e.damage(d2) - e.damage(d)) / max(-extra(e, d, d2), 1e-9), id(e), e, d, d2))
    heap = []
    for e in experts:
        f = choice[(e.layer, e.j)]
        u = step(e, f, +1)
        if u is not None and extra(e, f, u) > 0:
            heapq.heappush(heap, (-(e.damage(f) - e.damage(u)) / extra(e, f, u), id(e), e, f, u))
    while heap:
        _, _, e, f, u = heapq.heappop(heap)
        if choice[(e.layer, e.j)] != f:
            continue
        x = extra(e, f, u)
        if spent + x > budget or tspent + e.p * x > tbudget:
            continue
        choice[(e.layer, e.j)] = u
        spent += x
        tspent += e.p * x
        u2 = step(e, u, +1)
        if u2 is not None and extra(e, u, u2) > 0:
            heapq.heappush(heap, (-(e.damage(u) - e.damage(u2)) / extra(e, u, u2), id(e), e, u, u2))
    return _evaluate(experts, choice)


def probe(expert_weights: dict, samples, gates, fmts: list[str], act=None) -> dict[str, float]:
    """Measured D_j(b)/p_j for one expert: mean over its reservoir tokens of g^2 ||f - f_b||^2.

    expert_weights: {"gate": [I, H], "up": [I, H], "down": [H, I]} float tensors
    samples: [n, H] routed inputs, gates: [n] router weights (FastSaliency reservoir)."""
    import torch

    from fakequant import fake_quant
    act = act or torch.nn.functional.silu
    with torch.no_grad():
        x = samples.float()

        def f(wg, wu, wd):
            return (act(x @ wg.T) * (x @ wu.T)) @ wd.T

        ref = f(expert_weights["gate"].float(), expert_weights["up"].float(), expert_weights["down"].float())
        g2 = gates.float().pow(2)
        out = {}
        for fmt in fmts:
            q = {k: fake_quant(v, fmt) for k, v in expert_weights.items()}
            out[fmt] = float((g2 * (ref - f(q["gate"], q["up"], q["down"])).pow(2).sum(-1)).mean())
    return out


def report(experts, alloc: Allocation, menu, dense_bytes: float, bandwidth_gbs: float | None) -> dict:
    gb = 1024 ** 3
    by_layer: dict[int, dict] = {}
    for e in experts:
        f = alloc.choice[(e.layer, e.j)]
        by_layer.setdefault(e.layer, {}).setdefault(f, []).append(e.j)
    counts = {f: sum(1 for v in alloc.choice.values() if v == f) for f in menu}
    params = sum(e.params for e in experts if alloc.choice[(e.layer, e.j)] != "drop")
    r = {
        "experts_gb": round(alloc.storage / gb, 2),
        "total_gb": round((alloc.storage + dense_bytes) / gb, 2),
        "avg_expert_bpw_kept": round(8 * alloc.storage / max(params, 1), 3),
        "active_expert_gb_per_token": round(alloc.traffic / gb, 4),
        "format_counts": counts,
        "relative_damage": alloc.damage,
        "layers": {L: {f: ids for f, ids in sorted(t.items())} for L, t in sorted(by_layer.items())},
    }
    if bandwidth_gbs:
        per_tok = alloc.traffic / gb + dense_bytes / gb  # attention, embeddings etc. read every token
        r["decode_tok_s_upper_bound"] = round(bandwidth_gbs / per_tok, 1)
    return r


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("saliency")
    ap.add_argument("config")
    ap.add_argument("--budget-gb", type=float, required=True, help="expert storage budget")
    ap.add_argument("--menu", default="gguf", choices=sorted(MENUS))
    ap.add_argument("--traffic-gb", type=float, default=None, help="expert bytes read per token (decode speed)")
    ap.add_argument("--max-tiers", type=int, default=3)
    ap.add_argument("--uniform-drop", action="store_true")
    ap.add_argument("--dense-gb", type=float, default=0.0, help="non-expert weights, for totals")
    ap.add_argument("--bandwidth", type=float, default=None, help="GB/s of the target, for a tok/s bound")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    sal = json.loads(Path(a.saliency).read_text())
    cfg = json.loads(Path(a.config).read_text())
    ex = experts_from_saliency(sal, cfg)
    alloc = allocate(ex, MENUS[a.menu], a.budget_gb * 1024 ** 3,
                     a.traffic_gb * 1024 ** 3 if a.traffic_gb else None, max_tiers=a.max_tiers,
                     uniform_drop=a.uniform_drop)
    rep = report(ex, alloc, MENUS[a.menu], a.dense_gb * 1024 ** 3, a.bandwidth)
    text = json.dumps(rep, indent=1)
    if a.out:
        Path(a.out).write_text(text)
    print(json.dumps({k: v for k, v in rep.items() if k != "layers"}, indent=1))
