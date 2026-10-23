"""One-pass REAP scoring: a drop-in for reaper_pipeline.moe.Saliency that never runs an
expert twice and never syncs the GPU per expert.

Where today's Saliency spends its time
  * Its router hook re-runs every routed expert on its tokens to get ||f_j(x)||, after (or
    before) the model runs the very same experts itself: the routed-expert FLOPs of the
    scoring pass are paid twice.
  * Per layer it loops over torch.unique(idx).tolist() and does torch.where(idx == j) per
    expert: E small kernels plus a host sync per expert per layer (128 experts x 48 layers
    = 6,144 syncs per forward on Qwen3-30B-A3B), and float64 accumulators on the GPU
    (1/64 rate on Ada and consumer cards).

What this does instead
  * ModuleList experts (transformers 4.x, remote code): a forward hook on each expert reads
    the output the model computes anyway. The token -> (slot, weight) mapping is recovered
    from the router output; the dispatch order (slot-major or token-major) is detected once
    per block by checking the expert's input rows against the router input, and a block
    whose order cannot be proven falls back to the old recompute path.
  * Fused experts (transformers 5.x, gpt-oss): the experts module's forward is replaced by
    a sorted dispatch that computes each expert once (with REAPer's validated strategy:
    the module's own kernel or the manual maths), returns sum_j g_j f_j(x) as the module
    would, and records the norms on the way. One host sync per layer (bincount).
  * Accumulators are float32 on the device per batch, folded into float64 on the CPU once
    per results() call.

Extra statistics from the same pass (all optional, all per expert):
  act_stats=True    sum of x^2 per input channel of each expert (the imatrix / AWQ statistic
                    for gate and up) and of the down_proj input when it is reachable, so the
                    quant stage needs no second calibration pass (llama-imatrix today)
  reservoir=R       keep up to R routed input rows per expert (CPU, bf16) for the
                    per-expert bit-sensitivity probe in expert_bits.py
"""

from __future__ import annotations

import torch
import torch.nn as nn

from reaper_pipeline import moe


class FastSaliency:
    def __init__(self, model: nn.Module, blocks: list[moe.MoeBlock] | None = None,
                 act_stats: bool = False, reservoir: int = 0):
        self.config = model.config
        self.blocks = blocks if blocks is not None else moe.find_blocks(model)
        if not self.blocks:
            raise RuntimeError("no MoE blocks found")
        self.mask: torch.Tensor | None = None
        self.act_stats, self.reservoir = act_stats, reservoir
        self.routed_from: dict[int, str] = {}
        self.mode: dict[int, str] = {}           # layer -> hook | fused | recompute
        self.handles, self._restore = [], []
        self._route: dict[int, tuple] = {}      # layer -> (x, idx, w, mask) of the current forward
        self._dev: dict[int, dict] = {}          # layer -> float32 device accumulators
        self._host: dict[int, dict] = {}         # layer -> float64 CPU totals
        self.samples: dict[int, list[list[torch.Tensor]]] = {}
        self.sample_gates: dict[int, list[list[torch.Tensor]]] = {}
        self._order: dict[int, str] = {}
        self._manual_ok: dict[int, bool] = {}
        self._grouped_ok: dict[int, bool] = {}
        self._res_batches: dict[int, int] = {}
        self.reservoir_batches = 8  # forwards per layer that fill the reservoir through the loop path
        self._fallback = moe.Saliency.__new__(moe.Saliency)  # reuse the reference hook for odd blocks
        self._fallback.config, self._fallback.mask = self.config, None
        self._fallback.sum, self._fallback.freq, self._fallback.routed_from = {}, {}, {}
        for b in self.blocks:
            E = b.num_experts
            self._host[b.layer] = {"sum": torch.zeros(E, dtype=torch.float64),
                                   "freq": torch.zeros(E, dtype=torch.float64)}
            if reservoir:
                self.samples[b.layer] = [[] for _ in range(E)]
                self.sample_gates[b.layer] = [[] for _ in range(E)]
            self.handles.append(b.router.register_forward_hook(self._router_hook(b), with_kwargs=True))
            if b.fused:
                self._patch_fused(b)
                self.mode[b.layer] = "fused"
            else:
                for j, ex in enumerate(b.experts):
                    self.handles.append(ex.register_forward_hook(self._expert_hook(b, j)))
                    if act_stats and isinstance(getattr(ex, "down_proj", None), nn.Module):
                        self.handles.append(ex.down_proj.register_forward_hook(self._down_hook(b, j)))
                self.mode[b.layer] = "hook"

    # ------------------------------------------------------------------ accumulators

    def _acc(self, blk: moe.MoeBlock, device) -> dict:
        a = self._dev.get(blk.layer)
        if a is None or a["sum"].device != device:
            if a is not None:
                self._fold(blk.layer)
            E = blk.num_experts
            a = {"sum": torch.zeros(E, device=device), "freq": torch.zeros(E, device=device)}
            self._dev[blk.layer] = a
        return a

    def _acc_act(self, blk, name, width, device) -> torch.Tensor:
        a = self._acc(blk, device)
        if name not in a:
            a[name] = torch.zeros(blk.num_experts, width, device=device)
        return a[name]

    def _fold(self, layer: int) -> None:
        a = self._dev.pop(layer, None)
        if not a:
            return
        h = self._host[layer]
        for k, v in a.items():
            v = v.double().cpu()
            h[k] = h[k] + v if k in h else v

    def _keep_samples(self, blk, j, x, g) -> None:
        """First R routed (input row, router weight) pairs per expert. Batches are length-sorted,
        so for an unbiased sample shuffle the calibration order or use a later batch."""
        if not self.reservoir:
            return
        bucket = self.samples[blk.layer][j]
        have = sum(t.shape[0] for t in bucket)
        if have < self.reservoir:
            n = self.reservoir - have
            bucket.append(x[:n].detach().to("cpu", torch.bfloat16))
            self.sample_gates[blk.layer][j].append(g[:n].detach().float().cpu())

    # ------------------------------------------------------------------ router: stash the routing

    def _router_hook(self, blk: moe.MoeBlock):
        @torch.no_grad()
        def hook(_mod, args, kwargs, out):
            x = args[0] if args else next(iter(kwargs.values()))
            x = x.reshape(-1, x.shape[-1])
            parsed = moe.parse_router_output(out, x.shape[0], blk.num_experts)
            if parsed is None:
                logits = out[0] if isinstance(out, (tuple, list)) else out
                idx, w = moe.route_from_logits(blk, self.config, logits.reshape(-1, blk.num_experts))
                self.routed_from[blk.layer] = "logits"
            else:
                idx, w = parsed
                self.routed_from[blk.layer] = "router"
            m = None
            if self.mask is not None and self.mask.numel() == x.shape[0]:
                m = self.mask.to(x.device)
            self._route[blk.layer] = (x, idx, w, m)
            if self.mode[blk.layer] == "recompute":
                self._fallback.mask = self.mask
                self._fallback._hook(blk)(_mod, args, kwargs, out)
        return hook

    # ------------------------------------------------------------------ ModuleList experts

    def _tokens_for(self, blk, j, xin) -> tuple[torch.Tensor, torch.Tensor] | None:
        """(token ids, slots) in the order the block fed expert j, or None if unprovable."""
        x, idx, _, _ = self._route[blk.layer]
        orders = [self._order[blk.layer]] if blk.layer in self._order else ["slot", "token"]
        for o in orders:
            if o == "slot":
                slot, tok = torch.where((idx == j).T)
            else:
                tok, slot = torch.where(idx == j)
            if tok.numel() != xin.shape[0]:
                continue
            if blk.layer not in self._order:  # prove it once per block, then trust it
                if not torch.equal(x[tok], xin):
                    continue
                self._order[blk.layer] = o
            return tok, slot
        return None

    def _expert_hook(self, blk: moe.MoeBlock, j: int):
        @torch.no_grad()
        def hook(_mod, args, out):
            if self.mode[blk.layer] != "hook" or blk.layer not in self._route:
                return
            xin = args[0].reshape(-1, args[0].shape[-1])
            y = (out[0] if isinstance(out, tuple) else out).reshape(-1, xin.shape[-1])
            got = self._tokens_for(blk, j, xin)
            if got is None:  # cannot map rows to tokens: use the reference recompute from now on
                self.mode[blk.layer] = "recompute"
                self._fallback.sum.setdefault(blk.layer, torch.zeros(blk.num_experts, dtype=torch.float64))
                self._fallback.freq.setdefault(blk.layer, torch.zeros(blk.num_experts, dtype=torch.float64))
                return
            tok, slot = got
            self._record(blk, j, tok, slot, xin, y)
        return hook

    def _down_hook(self, blk, j):
        @torch.no_grad()
        def hook(_mod, args, _out):
            if self.mode[blk.layer] != "hook" or blk.layer not in self._route:
                return
            # runs inside the expert, before its own hook proves the dispatch order: the
            # first batch of each block is skipped here, every later one is counted
            if blk.layer not in self._order:
                return
            h = args[0].reshape(-1, args[0].shape[-1])
            _, idx, _, m = self._route[blk.layer]
            if m is not None:
                tok = torch.where((idx == j).T)[1] if self._order[blk.layer] == "slot" else torch.where(idx == j)[0]
                h = h[m[tok]]
            self._acc_act(blk, "down_in_sq", h.shape[-1], h.device)[j] += h.float().pow(2).sum(0)
        return hook

    # ------------------------------------------------------------------ shared bookkeeping

    def _record(self, blk, j, tok, slot, xin, y) -> None:
        _, _, w, m = self._route[blk.layer]
        a = self._acc(blk, y.device)
        g = w[tok, slot].float()
        contrib = g * y.float().norm(dim=-1)
        if m is not None:
            keep = m[tok]
            contrib, xin, g = contrib[keep], xin[keep], g[keep]
            a["freq"][j] += keep.sum()
        else:
            a["freq"][j] += tok.numel()
        a["sum"][j] += contrib.sum()
        if self.act_stats:
            self._acc_act(blk, "in_sq", xin.shape[-1], xin.device)[j] += xin.float().pow(2).sum(0)
        self._keep_samples(blk, j, xin, g)

    # ------------------------------------------------------------------ fused experts

    def _patch_fused(self, blk: moe.MoeBlock) -> None:
        ex = blk.experts
        orig = ex.forward
        state = {"inside": False}
        me = self

        def forward(hidden_states, *args, **kwargs):
            if state["inside"] or blk.layer not in me._route:
                return orig(hidden_states, *args, **kwargs)
            x_full = hidden_states.reshape(-1, hidden_states.shape[-1])
            # the routing the block actually passes in (it may rescale after the router)
            ts = [t for t in (*args, *kwargs.values()) if isinstance(t, torch.Tensor)]
            idx = next((t for t in ts if not t.is_floating_point() and t.dim() == 2), None)
            w = None
            if idx is not None:
                w = next((t for t in ts if t.is_floating_point() and t.shape == idx.shape), None)
                if w is None:
                    dense = next((t for t in ts if t.is_floating_point() and t.dim() == 2
                                  and t.shape == (idx.shape[0], blk.num_experts)), None)
                    w = dense.gather(1, idx.long()) if dense is not None else None
            if w is None or idx.shape[0] != x_full.shape[0]:
                return orig(hidden_states, *args, **kwargs)
            idx = idx.long()
            rx, _, _, m = me._route[blk.layer]
            me._route[blk.layer] = (rx, idx, w, m)
            state["inside"] = True  # expert_forward may call the module itself (call_topk/dense)
            try:
                if me._use_grouped(blk, x_full, idx, w):
                    return me._grouped(blk, x_full, idx, w, record=True).reshape(hidden_states.shape)
                out = torch.zeros_like(x_full)
                k = idx.shape[1]
                flat = idx.reshape(-1)
                order = torch.argsort(flat, stable=True)
                counts = torch.bincount(flat, minlength=blk.num_experts).tolist()  # one sync per layer
                tok_all, slot_all = order // k, order % k
                start = 0
                for j, c in enumerate(counts):
                    if c == 0:
                        continue
                    tok, slot = tok_all[start:start + c], slot_all[start:start + c]
                    start += c
                    xin = x_full[tok]
                    y = me._expert(blk, j, xin)
                    out.index_add_(0, tok, (w[tok, slot, None].to(y.dtype) * y).to(out.dtype))
                    me._record(blk, j, tok, slot, xin, y)
            finally:
                state["inside"] = False
            return out.reshape(hidden_states.shape)

        ex.forward = forward
        self._restore.append((ex, orig))

    # ------------------------------------------------------------------ grouped GEMM path

    def _grouped_weights(self, blk):
        """(gate_up [E, H, 2I], down [E, I, H]) views for torch._grouped_mm, or None."""
        ex = blk.experts
        gu, dn = getattr(ex, "gate_up_proj", None), getattr(ex, "down_proj", None)
        if not (isinstance(gu, torch.Tensor) and isinstance(dn, torch.Tensor) and gu.dim() == 3 and dn.dim() == 3):
            return None
        if any(isinstance(getattr(ex, b, None), torch.Tensor) for b in ("gate_up_proj_bias", "down_proj_bias")):
            return None
        H = blk.router.weight.shape[1]
        gu = gu.transpose(1, 2) if gu.shape[-1] == H else gu      # stored [E, 2I, H] (F.linear style)
        dn = dn.transpose(1, 2) if dn.shape[1] == H else dn       # stored [E, H, I]
        return (gu, dn) if gu.shape[1] == H and dn.shape[-1] == H else None

    def _grouped(self, blk, x, idx, w, record: bool):
        """All routed (token, expert) pairs of the layer in two grouped GEMMs: no per-expert
        Python loop and no host sync. Same maths as moe._manual."""
        gu, dn = self._grouped_weights(blk)
        E, k = blk.num_experts, idx.shape[1]
        flat = idx.reshape(-1)
        order = torch.argsort(flat, stable=True)
        e_sorted = flat[order]
        tok = order // k
        offs = torch.cumsum(torch.bincount(flat, minlength=E), 0).to(torch.int32)
        xs = x[tok].to(gu.dtype)
        h = torch._grouped_mm(xs, gu, offs=offs)
        g, u = h.chunk(2, dim=-1)
        y = torch._grouped_mm((moe._act(blk.experts, self.config)(g) * u).to(dn.dtype), dn, offs=offs)
        ws = w.reshape(-1)[order]
        out = torch.zeros_like(x)
        out.index_add_(0, tok, (ws[:, None].to(y.dtype) * y).to(out.dtype))
        if record:
            _, _, _, m = self._route[blk.layer]
            a = self._acc(blk, y.device)
            contrib = ws.float() * y.float().norm(dim=-1)
            keep = m[tok].float() if m is not None else torch.ones_like(contrib)
            a["sum"].index_add_(0, e_sorted, contrib * keep)
            a["freq"].index_add_(0, e_sorted, keep)
            if self.act_stats:
                acc = self._acc_act(blk, "in_sq", x.shape[-1], x.device)
                acc.index_add_(0, e_sorted, xs.float().pow(2) * keep[:, None])
        return out

    def _use_grouped(self, blk, x, idx, w) -> bool:
        """Grouped path when torch has it, the weights fit it, the per-expert maths is proven
        (_manual_ok) and the grouped answer matches the loop once; while the reservoir is
        still filling, the loop path runs so it can keep samples."""
        if self.reservoir and self._res_batches.get(blk.layer, 0) < self.reservoir_batches:
            self._res_batches[blk.layer] = self._res_batches.get(blk.layer, 0) + 1
            return False
        ok = self._grouped_ok.get(blk.layer)
        if ok is not None:
            return ok
        ok = False
        if hasattr(torch, "_grouped_mm") and self._grouped_weights(blk) is not None:
            try:
                n = min(64, x.shape[0])
                xi, ii, wi = x[:n], idx[:n], w[:n]
                if self._expert(blk, int(ii[0, 0]), xi[:4]) is not None and self._manual_ok.get(blk.layer):
                    got = self._grouped(blk, xi, ii, wi, record=False).float()
                    ref = torch.zeros_like(xi, dtype=torch.float32)
                    for t in range(n):
                        for sl in range(ii.shape[1]):
                            ref[t] += wi[t, sl].float() * moe._manual(blk, self.config, int(ii[t, sl]), xi[t:t + 1])[0].float()
                    ok = torch.allclose(got, ref, rtol=2e-2, atol=2e-2 * (ref.abs().max().item() + 1e-6))
            except Exception:
                ok = False
        self._grouped_ok[blk.layer] = ok
        return ok

    def _expert(self, blk, j, x):
        """Expert j on x. The stacked-weight maths (_manual: two matmuls, no dispatch) is much
        cheaper than calling the fused module once per expert, so it is used whenever it
        reproduces the module's own answer on this block (checked once; gpt-oss's clamped,
        interleaved activation fails the check and keeps the module call)."""
        ok = self._manual_ok.get(blk.layer)
        if ok is None:
            ref = moe.expert_forward(blk, self.config, j, x[:4])
            try:
                ok = torch.allclose(moe._manual(blk, self.config, j, x[:4]).float(), ref.float(),
                                    rtol=1e-3, atol=1e-3 * (ref.abs().max().item() + 1e-6))
            except Exception:
                ok = False
            self._manual_ok[blk.layer] = ok
        return moe._manual(blk, self.config, j, x) if ok else moe.expert_forward(blk, self.config, j, x)

    # ------------------------------------------------------------------ results

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []
        for ex, orig in self._restore:
            ex.forward = orig
        self._restore = []

    def results(self) -> dict[int, dict]:
        for li in list(self._dev):
            self._fold(li)
        out = {}
        for li, h in self._host.items():
            s, f = h["sum"], h["freq"]
            if self.mode.get(li) == "recompute":
                s = s + self._fallback.sum.get(li, 0)
                f = f + self._fallback.freq.get(li, 0)
            r = {"saliency": (s / f.clamp(min=1)).tolist(), "freq": f.tolist()}
            if "in_sq" in h:
                r["in_sq"] = h["in_sq"]           # [E, hidden] float64, not JSON: torch.save it
            if "down_in_sq" in h:
                r["down_in_sq"] = h["down_in_sq"]
            out[li] = r
        return out
