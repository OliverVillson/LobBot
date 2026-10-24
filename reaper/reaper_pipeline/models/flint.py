"""Recursive's Flint (RecursiveSI/flint) in plain PyTorch, for REAP scoring and eval.

transformers has no Flint yet, so REAPer registers this one for model_type
"flint" (only when transformers does not know it). Written from the vLLM
implementation's semantics and the checkpoint's tensor names:

- 49 layers; `layer_types` picks sliding (window 513, counting the current
  token) or full attention; layers in `no_rope_layer_indices` (the full ones)
  use no positional encoding, the others NeoX RoPE.
- GQA attention with per-head q/k RMSNorm before RoPE and one learned sink
  logit per head (`self_attn.softmax_offset`), added to the softmax
  denominator.
- `mlp_only_layers` have a dense SwiGLU MLP; the others a MoE block: router
  `mlp.gate` [E, H], sigmoid scores, top-k, renormalised over the chosen
  experts; per-expert SwiGLU `mlp.experts.<j>.{gate,up,down}_proj`; an
  ungated shared expert `mlp.shared_expert`.
- RMSNorm in fp32. The LM head runs in the model dtype (the reference uses
  fp32): REAP never reads the logits, and fp32 logits of a 150k vocabulary
  over a full calibration batch would cost tens of GB.

No KV cache: forward always sees the whole sequence (generation recomputes;
use vLLM for long generations). The router returns (logits, weights, ids), so
REAP reads Flint's routing exactly as the model applies it.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GenerationMixin, PreTrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

Q_CHUNK = 1024  # query rows per attention chunk (bounds the [heads, q, k] score matrix)
STACKED_CHUNK = 4096  # tokens per batched-experts pass (bounds the [E, C, H] buffer)
Q_CHUNK_CACHED = 256  # the same for cached generation, where the batch is large


class FlintConfig(PreTrainedConfig):
    model_type = "flint"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(self, vocab_size=150144, hidden_size=2048, intermediate_size=6144, num_hidden_layers=49,
                 num_attention_heads=32, num_key_value_heads=4, head_dim=128, hidden_act="silu",
                 max_position_embeddings=262144, initializer_range=0.02, rms_norm_eps=1e-6, use_cache=False,
                 tie_word_embeddings=False, rope_theta=10000.0, rope_parameters=None, attention_bias=False,
                 attention_dropout=0.0, use_sliding_window=True, sliding_window=513, layer_types=None,
                 no_rope_layer_indices=None, moe_intermediate_size=512, shared_expert_intermediate_size=1024,
                 num_experts_per_tok=6, num_experts=256, norm_topk_prob=True, router_score_function="sigmoid",
                 mlp_only_layers=None, **kwargs):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = (rope_parameters or {}).get("rope_theta", rope_theta)
        self.rope_parameters = rope_parameters or {"rope_type": "default", "rope_theta": rope_theta}
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window if use_sliding_window else None
        self.layer_types = layer_types or ["full_attention"] * num_hidden_layers
        self.no_rope_layer_indices = list(no_rope_layer_indices or [])
        self.moe_intermediate_size = moe_intermediate_size
        self.shared_expert_intermediate_size = shared_expert_intermediate_size
        self.num_experts_per_tok = num_experts_per_tok
        self.num_experts = num_experts
        self.norm_topk_prob = norm_topk_prob
        self.router_score_function = router_score_function
        self.mlp_only_layers = list(mlp_only_layers or [])
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


class FlintRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        h = x.float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
        return (h * self.weight.float()).to(x.dtype)


class FlintMLP(nn.Module):
    def __init__(self, hidden: int, inter: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, inter, bias=False)
        self.up_proj = nn.Linear(hidden, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x).float()).to(x.dtype) * self.up_proj(x))


class FlintRouter(nn.Module):
    """Sigmoid top-k router. Returns (logits [N, E], weights [N, k], ids [N, k])."""

    def __init__(self, config: FlintConfig):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(config.num_experts, config.hidden_size))
        self.top_k = config.num_experts_per_tok
        self.norm = config.norm_topk_prob
        if config.router_score_function != "sigmoid":
            raise NotImplementedError(f"Flint router {config.router_score_function!r}")

    def forward(self, x):
        logits = F.linear(x, self.weight).float()
        s, ids = torch.sigmoid(logits).topk(self.top_k, dim=-1)
        if self.norm:
            s = s / (s.sum(-1, keepdim=True) + 1e-20)
        return logits, s.to(x.dtype), ids


class FlintSparseMoeBlock(nn.Module):
    def __init__(self, config: FlintConfig):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.gate = FlintRouter(config)
        self.experts = nn.ModuleList(FlintMLP(config.hidden_size, config.moe_intermediate_size)
                                     for _ in range(config.num_experts))
        self.shared_expert = FlintMLP(config.hidden_size, config.shared_expert_intermediate_size)

    @torch.no_grad()
    def stack_experts(self) -> None:
        """Put the experts' weights into stacked [E, 2I, H] (gate rows, then up) and [E, H, I]
        tensors, with each expert's Linear weights becoming views into them (no extra memory
        once done), so forward can run all experts as two batched matmuls instead of a loop."""
        if hasattr(self, "_gu"):
            return
        ex = self.experts
        I = ex[0].gate_proj.weight.shape[0]
        gu = torch.stack([torch.cat([e.gate_proj.weight, e.up_proj.weight]) for e in ex])
        dn = torch.stack([e.down_proj.weight for e in ex])
        for j, e in enumerate(ex):
            e.gate_proj.weight = nn.Parameter(gu[j, :I], requires_grad=False)
            e.up_proj.weight = nn.Parameter(gu[j, I:], requires_grad=False)
            e.down_proj.weight = nn.Parameter(dn[j], requires_grad=False)
        self._gu, self._dn, self._I, self._fixed_c = gu, dn, I, False

    def _stacked_forward(self, h, w, ids):
        """All experts at once: tokens scattered into a [E, C, H] buffer (C = most tokens any
        expert got), two bmm's, gathered back. Chunked so prefill buffers stay a few GB."""
        E, k, I = self._gu.shape[0], ids.shape[1], self._I
        if h.shape[0] * k * 2 <= E:  # few tokens (small-batch decode): read only the chosen experts
            flat = ids.reshape(-1)
            x = h.repeat_interleave(k, 0)[:, None, :]                              # [N, 1, H]
            g = torch.bmm(x, self._gu[flat].transpose(1, 2))
            a = F.silu(g[..., :I].float()).to(h.dtype) * g[..., I:]
            y = torch.bmm(a, self._dn[flat].transpose(1, 2))[:, 0].float()         # [N, H]
            return (y * w.reshape(-1, 1).float()).view(h.shape[0], k, -1).sum(1)
        out = torch.zeros(h.shape, dtype=torch.float32, device=h.device)
        for c0 in range(0, h.shape[0], STACKED_CHUNK):
            hc, wc, ic = h[c0:c0 + STACKED_CHUNK], w[c0:c0 + STACKED_CHUNK], ids[c0:c0 + STACKED_CHUNK]
            flat = ic.reshape(-1)
            order = torch.argsort(flat, stable=True)
            e_sorted = flat[order]
            if self._fixed_c:  # decode under a CUDA graph: no host sync, C = tokens (an expert gets a token once)
                counts = torch.zeros(E, dtype=torch.long, device=h.device).scatter_add_(0, flat, torch.ones_like(flat))
                C = hc.shape[0]
            else:
                counts = torch.bincount(flat, minlength=E)
            start = torch.cumsum(counts, 0) - counts
            rank = torch.arange(flat.numel(), device=h.device) - start[e_sorted]
            if not self._fixed_c:
                C = int(counts.max())
            tok = order // k
            X = torch.zeros(E, C, h.shape[1], dtype=h.dtype, device=h.device)
            X[e_sorted, rank] = hc[tok]
            g = torch.bmm(X, self._gu.transpose(1, 2))
            a = F.silu(g[..., :I].float()).to(h.dtype) * g[..., I:]
            Y = torch.bmm(a, self._dn.transpose(1, 2))
            out[c0:c0 + STACKED_CHUNK].index_add_(0, tok, Y[e_sorted, rank].float() * wc.reshape(-1)[order, None].float())
        return out

    def forward(self, x):
        shape = x.shape
        h = x.reshape(-1, shape[-1])
        _, w, ids = self.gate(h)
        if hasattr(self, "_gu"):
            return (self._stacked_forward(h, w, ids).to(x.dtype) + self.shared_expert(h)).reshape(shape)
        # group the (token, slot) pairs by expert with one sort and one host sync per layer
        # (a torch.where per expert syncs 256 times per layer, which dominates decoding)
        k = ids.shape[1]
        flat = ids.reshape(-1)
        order = torch.argsort(flat, stable=True)
        tok = order // k
        counts = torch.bincount(flat, minlength=len(self.experts)).tolist()
        hs, ws = h[tok], w.reshape(-1)[order]
        out = torch.zeros(h.shape, dtype=torch.float32, device=h.device)
        start = 0
        for j, c in enumerate(counts):
            if c:
                sl = slice(start, start + c)
                out.index_add_(0, tok[sl], (self.experts[j](hs[sl]) * ws[sl, None]).float())
                start += c
        return (out.to(x.dtype) + self.shared_expert(h)).reshape(shape)


def _rope(x, pos, theta: float):
    """NeoX rotary embedding on [B, heads, T, D] with positions [B, T]."""
    d = x.shape[-1]
    inv = 1.0 / (theta ** (torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d))
    ang = pos.float()[:, None, :, None] * inv
    cos, sin = ang.cos(), ang.sin()
    x1, x2 = x.float()[..., : d // 2], x.float()[..., d // 2:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1).to(x.dtype)


class FlintAttention(nn.Module):
    def __init__(self, config: FlintConfig, layer_idx: int):
        super().__init__()
        H, D = config.num_attention_heads, config.head_dim
        self.nh, self.nkv, self.d = H, config.num_key_value_heads, D
        self.q_proj = nn.Linear(config.hidden_size, H * D, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.nkv * D, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.nkv * D, bias=config.attention_bias)
        self.o_proj = nn.Linear(H * D, config.hidden_size, bias=False)
        self.q_norm = FlintRMSNorm(D, config.rms_norm_eps)
        self.k_norm = FlintRMSNorm(D, config.rms_norm_eps)
        self.softmax_offset = nn.Parameter(torch.zeros(H, dtype=torch.float32))
        self.rope = layer_idx not in config.no_rope_layer_indices
        self.theta = float(config.rope_parameters.get("rope_theta", config.rope_theta))
        sliding = config.layer_types[layer_idx] == "sliding_attention"
        self.window = config.sliding_window if sliding else None
        self.scale = D ** -0.5

    def forward(self, x, pos, key_mask, kv=None):
        B, T, _ = x.shape
        q = self.q_norm(self.q_proj(x).view(B, T, self.nh, self.d)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(B, T, self.nkv, self.d)).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.nkv, self.d).transpose(1, 2)
        if self.rope:
            q, k = _rope(q, pos, self.theta), _rope(k, pos, self.theta)
        if kv is not None:
            return self._cached(q, k, v, key_mask, kv)
        rep = self.nh // self.nkv
        k = k.repeat_interleave(rep, 1).float()
        v = v.repeat_interleave(rep, 1).float()
        sink = self.softmax_offset.float().view(1, self.nh, 1, 1)
        kpos = torch.arange(T, device=x.device)
        out = torch.empty(B, self.nh, T, self.d, device=x.device, dtype=torch.float32)
        for a in range(0, T, Q_CHUNK):
            b = min(T, a + Q_CHUNK)
            s = torch.matmul(q[:, :, a:b].float(), k.transpose(-1, -2)) * self.scale
            qpos = kpos[a:b, None]
            allowed = kpos[None, :] <= qpos
            if self.window:
                allowed &= kpos[None, :] > qpos - self.window
            allowed = allowed[None, None] & key_mask[:, None, None, :]
            s = s.masked_fill(~allowed, float("-inf"))
            m = torch.maximum(s.amax(-1, keepdim=True), sink)
            p = torch.exp(s - m)
            out[:, :, a:b] = torch.matmul(p, v) / (p.sum(-1, keepdim=True) + torch.exp(sink - m))
        out = out.to(x.dtype).transpose(1, 2).reshape(B, T, self.nh * self.d)
        return self.o_proj(out)

    def static_step(self, x, st: dict, li: int):
        """One decode token per slot against preallocated caches, every slot at its own position
        st["t"] [S] (a slot's tokens sit at cache index = position, no padding). Full-attention
        caches st["k"][li] [S, nkv, Lmax, D] are read only up to st["L"] (a bucket covering every
        slot's position); sliding layers use a ring of `window` slots with positions
        st["ring_pos"] [S, W]. Fixed shapes and no host syncs: one CUDA graph per bucket."""
        S, t, ar = x.shape[0], st["t"], st["ar"]
        q = self.q_norm(self.q_proj(x).view(S, 1, self.nh, self.d)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(S, 1, self.nkv, self.d)).transpose(1, 2)
        v = self.v_proj(x).view(S, 1, self.nkv, self.d).transpose(1, 2)
        if self.rope:
            q, k = _rope(q, t[:, None], self.theta), _rope(k, t[:, None], self.theta)
        kc, vc = st["k"][li], st["v"][li]
        if st["ring"][li]:
            at = t % kc.shape[2]
            kpos = st["ring_pos"]
        else:
            at = t
            kc, vc = kc[:, :, :st["L"]], vc[:, :, :st["L"]]
            kpos = st["kpos"][:st["L"]].expand(S, -1)
        st["k"][li][ar, :, at] = k[:, :, 0].to(kc.dtype)
        st["v"][li][ar, :, at] = v[:, :, 0].to(vc.dtype)
        G, rep = self.nkv, self.nh // self.nkv
        sc = torch.matmul(q.reshape(S, G, rep, self.d).to(kc.dtype), kc.transpose(-1, -2)).float() * self.scale
        allowed = (kpos <= t[:, None]) & (kpos >= 0)
        if self.window:
            allowed = allowed & (kpos > t[:, None] - self.window)
        sc = sc.masked_fill(~allowed[:, None, None, :], float("-inf"))
        sink = self.softmax_offset.float().view(1, G, rep, 1)
        m = torch.maximum(sc.amax(-1, keepdim=True), sink)
        pr = torch.exp(sc - m)
        pr = pr / (pr.sum(-1, keepdim=True) + torch.exp(sink - m))
        o = torch.matmul(pr.to(vc.dtype), vc)                              # [S, G, rep, D]
        return self.o_proj(o.reshape(S, 1, self.nh * self.d).to(x.dtype))

    def _cached(self, q, k, v, key_mask, kv: dict):
        """Attention for generation with a KV cache (kv holds this layer's k, v [B, nkv, Tk, D]).
        key_mask [B, Tk] covers cache + new tokens. Query heads are grouped onto their KV head, so
        the cache is never repeated per head; same masking, window and sink as forward()."""
        if "k" in kv:
            k, v = torch.cat([kv["k"], k], 2), torch.cat([kv["v"], v], 2)
        kv["k"], kv["v"] = k, v
        B, H, T, D = q.shape
        G, rep, Tk = self.nkv, self.nh // self.nkv, k.shape[2]
        off = Tk - T
        qg = q.reshape(B, G, rep, T, D)
        kt = k.transpose(-1, -2)
        kpos = torch.arange(Tk, device=q.device)
        sink = self.softmax_offset.float().view(1, G, rep, 1, 1)
        out = torch.empty(B, G, rep, T, D, device=q.device, dtype=q.dtype)
        for a in range(0, T, Q_CHUNK_CACHED):
            b = min(T, a + Q_CHUNK_CACHED)
            t = b - a
            s = torch.matmul(qg[:, :, :, a:b].reshape(B, G, rep * t, D), kt).float() * self.scale
            s = s.view(B, G, rep, t, Tk)
            qpos = kpos[off + a:off + b, None]
            allowed = kpos[None, :] <= qpos
            if self.window:
                allowed &= kpos[None, :] > qpos - self.window
            allowed = allowed[None, None, None] & key_mask[:, None, None, None, :]
            s = s.masked_fill(~allowed, float("-inf"))
            m = torch.maximum(s.amax(-1, keepdim=True), sink)
            p = torch.exp(s - m)
            den = p.sum(-1, keepdim=True) + torch.exp(sink - m)
            o = torch.matmul((p / den).to(v.dtype).view(B, G, rep * t, Tk), v)
            out[:, :, :, a:b] = o.view(B, G, rep, t, D)
        out = out.reshape(B, H, T, D).transpose(1, 2).reshape(B, T, H * D)
        return self.o_proj(out)


class FlintDecoderLayer(nn.Module):
    def __init__(self, config: FlintConfig, layer_idx: int):
        super().__init__()
        self.self_attn = FlintAttention(config, layer_idx)
        self.mlp = (FlintMLP(config.hidden_size, config.intermediate_size) if layer_idx in config.mlp_only_layers
                    else FlintSparseMoeBlock(config))
        self.input_layernorm = FlintRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = FlintRMSNorm(config.hidden_size, config.rms_norm_eps)

    def static_step(self, x, st: dict, li: int):
        x = x + self.self_attn.static_step(self.input_layernorm(x), st, li)
        return x + self.mlp(self.post_attention_layernorm(x))

    def forward(self, x, pos, key_mask, kv=None):
        x = x + self.self_attn(self.input_layernorm(x), pos, key_mask, kv)
        return x + self.mlp(self.post_attention_layernorm(x))


class FlintPreTrainedModel(PreTrainedModel):
    config_class = FlintConfig
    base_model_prefix = "model"
    _no_split_modules = ["FlintDecoderLayer"]
    _supports_cache_class = False

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(0.0, std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(0.0, std)
        elif isinstance(module, FlintRouter):
            module.weight.data.normal_(0.0, std)
        elif isinstance(module, FlintRMSNorm):
            module.weight.data.fill_(1.0)
        elif isinstance(module, FlintAttention):
            module.softmax_offset.data.zero_()


class FlintModel(FlintPreTrainedModel):
    def __init__(self, config: FlintConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(FlintDecoderLayer(config, i) for i in range(config.num_hidden_layers))
        self.norm = FlintRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None, **_):
        x = inputs_embeds if inputs_embeds is not None else self.embed_tokens(input_ids)
        B, T = x.shape[:2]
        if attention_mask is None or attention_mask.dim() != 2:
            key_mask = torch.ones(B, T, dtype=torch.bool, device=x.device)
        else:
            key_mask = attention_mask.to(x.device).bool()
        pos = (key_mask.long().cumsum(-1) - 1).clamp(min=0)  # left padding (batched generation) too
        for layer in self.layers:
            x = layer(x, pos, key_mask)
        return self.norm(x)


class FlintForCausalLM(FlintPreTrainedModel, GenerationMixin):
    def __init__(self, config: FlintConfig):
        super().__init__(config)
        self.model = FlintModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
        self.generation_config.use_cache = False  # no KV cache: every step sees the whole sequence

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None, labels=None, **kwargs):
        h = self.model(input_ids=input_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds)
        logits = self.lm_head(h)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1),
                                   ignore_index=-100)
        return CausalLMOutputWithPast(loss=loss, logits=logits)

    def stack_experts(self) -> None:
        """Batched experts for inference (FlintSparseMoeBlock.stack_experts); memory neutral."""
        for layer in self.model.layers:
            if isinstance(layer.mlp, FlintSparseMoeBlock):
                layer.mlp.stack_experts()

    @torch.no_grad()
    def fast_generate(self, input_ids, attention_mask=None, max_new_tokens: int = 256, temperature: float = 0.0,
                      top_p: float = 1.0, eos_token_id=None, generator=None, on_step=None, on_token=None):
        """Batched sampling with a KV cache (generate() recomputes the whole sequence every step).
        input_ids [B, T] left-padded. Returns new tokens [B, n]; a row is padded with its first
        eos token after it finishes. on_step(step, still_writing) is called every 16 tokens and
        on_token(tokens [B]) after every token; on_token returning True stops generation."""
        mask = attention_mask.bool() if attention_mask is not None else torch.ones_like(input_ids, dtype=torch.bool)
        prompts = [r[mk].tolist() for r, mk in zip(input_ids, mask)]
        outs = self.generate_many(prompts, max_new_tokens, temperature, top_p, eos_token_id, generator,
                                  slots=len(prompts), on_step=on_step, on_token=on_token)
        n = max((len(o) for o in outs), default=0)
        eos = sorted({eos_token_id} if isinstance(eos_token_id, int) else set(eos_token_id or []))
        fill = eos[0] if eos else 0
        return torch.tensor([o + [fill] * (n - len(o)) for o in outs], dtype=torch.long).view(len(outs), n)

    @torch.no_grad()
    def generate_many(self, prompts: list[list[int]], max_new_tokens: int = 256, temperature: float = 0.0,
                      top_p: float = 1.0, eos_token_id=None, generator=None, slots: int | None = None,
                      on_step=None, on_token=None, on_done=None, top_k: int = 0) -> list[list[int]]:
        """Continuous batching: `slots` sequences decode together and a finished slot takes the next
        prompt at once (a batch never waits for its longest answer). Returns each prompt's new
        tokens, ending with its eos token if it produced one. on_step(tokens so far, still writing);
        on_done(prompt index, tokens) as each answer finishes; on_token(tokens [S]) after every step
        (only meaningful when every prompt has its own slot; True stops generation)."""
        m = self.model
        self.stack_experts()
        dev = m.embed_tokens.weight.device
        N = len(prompts)
        S = max(1, min(slots or N, N))
        eos_l = sorted({eos_token_id} if isinstance(eos_token_id, int) else set(eos_token_id or []))
        eos = torch.tensor(eos_l, dtype=torch.long, device=dev)
        K = 1 if on_token is not None else 8                                # steps between host checks
        Lmax = max(len(p) for p in prompts) + max_new_tokens + K + 1
        attn = [l.self_attn for l in m.layers]
        W = next((a.window for a in attn if a.window), None)
        ring = [bool(a.window) and W is not None and W < Lmax for a in attn]
        cfg = self.config
        nkv, D = cfg.num_key_value_heads, cfg.head_dim
        dt = m.embed_tokens.weight.dtype
        st = {"k": [torch.zeros(S, nkv, W if r else Lmax, D, dtype=dt, device=dev) for r in ring],
              "ring": ring, "t": torch.zeros(S, dtype=torch.long, device=dev),
              "ar": torch.arange(S, device=dev), "kpos": torch.arange(Lmax, device=dev),
              "tok": torch.zeros(S, dtype=torch.long, device=dev), "L": Lmax}
        st["v"] = [torch.zeros_like(k) for k in st["k"]]
        if any(ring):
            st["ring_pos"] = torch.full((S, W), -1, dtype=torch.long, device=dev)
        logits = torch.zeros(S, cfg.vocab_size, dtype=torch.float32, device=dev)
        out_buf = torch.zeros(S, max_new_tokens + K + 1, dtype=torch.long, device=dev)
        n_gen = torch.zeros(S, dtype=torch.long, device=dev)
        done = torch.ones(S, dtype=torch.bool, device=dev)
        owner = [-1] * S                      # prompt index per slot (-1: free)
        t_host = [0] * S
        results: list[list[int] | None] = [None] * N
        queue = list(range(N))
        moe = [l.mlp for l in m.layers if isinstance(l.mlp, FlintSparseMoeBlock)]

        def prefill(free: list[int]) -> None:
            """Fill free slots with the next prompts: one batched forward per group of prompts."""
            while free and queue:
                grp = []
                while free and queue and len(grp) < 32:
                    grp.append((free.pop(0), queue.pop(0)))
                ps = [prompts[i] for _, i in grp]
                T = max(len(p) for p in ps)
                ids = torch.tensor([[0] * (T - len(p)) + p for p in ps], device=dev)
                km = torch.tensor([[False] * (T - len(p)) + [True] * len(p) for p in ps], device=dev)
                pos = (km.long().cumsum(-1) - 1).clamp(min=0)
                caches = [{} for _ in m.layers]
                x = m.embed_tokens(ids)
                for layer, kv in zip(m.layers, caches):
                    x = layer(x, pos, km, kv)
                lg = self.lm_head(m.norm(x[:, -1])).float()
                del x
                for j, (slot, i) in enumerate(grp):
                    n = len(ps[j])
                    for li, kv in enumerate(caches):
                        kk, vv = kv["k"][j, :, T - n:], kv["v"][j, :, T - n:]
                        if ring[li]:
                            keep = torch.arange(max(0, n - W), n, device=dev)
                            st["k"][li][slot, :, keep % W] = kk[:, keep]
                            st["v"][li][slot, :, keep % W] = vv[:, keep]
                        else:
                            st["k"][li][slot, :, :n] = kk
                            st["v"][li][slot, :, :n] = vv
                    if any(ring):
                        keep = torch.arange(max(0, n - W), n, device=dev)
                        st["ring_pos"][slot] = -1
                        st["ring_pos"][slot, keep % W] = keep
                    st["t"][slot] = n - 1
                    t_host[slot] = n - 1
                    logits[slot] = lg[j]
                    n_gen[slot] = 0
                    done[slot] = False
                    owner[slot] = i
                del caches

        def step() -> None:
            for b in moe:
                b._fixed_c = True
            try:
                st["t"].add_(1).clamp_(max=Lmax - 1)
                if "ring_pos" in st:
                    st["ring_pos"][st["ar"], st["t"] % W] = st["t"]
                h = m.embed_tokens(st["tok"][:, None])
                for li, layer in enumerate(m.layers):
                    h = layer.static_step(h, st, li)
                logits.copy_(self.lm_head(m.norm(h[:, -1])).float())
            finally:
                for b in moe:
                    b._fixed_c = False

        BUCKET = 1024
        graphs: dict[int, object] = {}
        use_graphs = dev.type == "cuda" and max_new_tokens > 8 and not os.environ.get("REAPER_NO_CUDA_GRAPH")
        pool = torch.cuda.graph_pool_handle() if use_graphs else None

        def run_step(L: int) -> None:
            nonlocal use_graphs
            st["L"] = L
            if not use_graphs:
                return step()
            if L not in graphs:
                try:
                    snap = [x.clone() for x in (st["t"], logits)]
                    side = torch.cuda.Stream()
                    side.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(side):
                        step()  # warm-up before capture
                    torch.cuda.current_stream().wait_stream(side)
                    st["t"].copy_(snap[0])
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g, pool=pool):
                        step()
                    graphs[L] = g
                    st["t"].copy_(snap[0])  # the capture does not run the step; replay below does
                    del snap
                except Exception as e:  # anything the capture does not support: decode eagerly
                    print(f"flint: CUDA graph capture failed ({type(e).__name__}: {e}); decoding eagerly", flush=True)
                    use_graphs = False
                    torch.cuda.synchronize()
                    return step()
            graphs[L].replay()

        def harvest(everything: bool = False) -> None:
            d = done.tolist()
            fin = [s_ for s_ in range(S) if owner[s_] >= 0 and (d[s_] or everything)]
            if not fin:
                return
            ng = n_gen.tolist()
            for s_, r in zip(fin, out_buf[fin].tolist()):
                r = r[:ng[s_]]
                cut = next((k_ + 1 for k_, tk in enumerate(r) if tk in eos_l), len(r))
                results[owner[s_]] = r[:cut]
                if on_done is not None:
                    on_done(owner[s_], r[:cut])
                owner[s_] = -1
                t_host[s_] = 0
                st["t"][s_] = 0

        prefill(list(range(S)))
        steps = 0
        while True:
            # finished slots hand over to the next prompts (checked every K steps: a host sync)
            if steps and steps % K == 0:
                harvest()
                prefill([s_ for s_ in range(S) if owner[s_] < 0])
                if all(o < 0 for o in owner):
                    break
            nxt = _sample(logits, temperature, top_p, generator, top_k)
            out_buf[st["ar"], n_gen.clamp(max=out_buf.shape[1] - 1)] = nxt
            n_gen.add_((~done).long())
            if len(eos):
                done |= torch.isin(nxt, eos)
            done |= n_gen >= max_new_tokens
            steps += 1
            if on_token is not None and on_token(nxt):
                harvest(everything=True)
                break
            if on_step is not None and steps % 16 == 0:
                on_step(steps, int((~done).sum()))
            st["tok"].copy_(nxt)
            for s_ in range(S):
                t_host[s_] = min(t_host[s_] + 1, Lmax - 1)
            run_step(min(Lmax, (max(t_host) // BUCKET + 1) * BUCKET))
        return [r if r is not None else [] for r in results]


def _sample(logits, temperature: float, top_p: float, generator=None, top_k: int = 0):
    if temperature <= 0:
        return logits.argmax(-1)
    if top_k and top_k < logits.shape[-1]:
        kth = logits.topk(top_k, -1).values[:, -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    probs = torch.softmax(logits / temperature, -1)
    if top_p < 1.0:
        sp, si = probs.sort(-1, descending=True)
        keep = sp.cumsum(-1) - sp < top_p
        sp = sp * keep
        return si.gather(1, torch.multinomial(sp / sp.sum(-1, keepdim=True), 1, generator=generator))[:, 0]
    return torch.multinomial(probs, 1, generator=generator)[:, 0]


def register() -> bool:
    """Make AutoConfig / AutoModelForCausalLM load model_type "flint". False when transformers has its own."""
    from transformers import AutoConfig, AutoModelForCausalLM
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    if "flint" in CONFIG_MAPPING:
        return CONFIG_MAPPING["flint"] is FlintConfig
    AutoConfig.register("flint", FlintConfig)
    AutoModelForCausalLM.register(FlintConfig, FlintForCausalLM)
    return True
