"""blackhole-serve: an OpenAI-compatible chat server for a model the pipeline made, for models no
serving engine supports yet (Flint: no llama.cpp / Ollama, no vLLM). Runs REAPer's PyTorch
model with its KV-cached generation on the GPU; one request at a time.

    blackhole-serve                      # newest job's GPTQ int4 model, on 127.0.0.1:8000
    blackhole-serve <model dir> --port 8000

Loads the bf16 REAP folder as is, the GPTQ W4A16 folder (compressed-tensors) and the REAQ
codebook folder by unpacking the experts to bf16, so what you talk to has exactly the weights
that were measured. Endpoints: GET /v1/models, POST /v1/chat/completions (stream or not,
tools passed to the chat template, <tool_call>{json}</tool_call> replies returned as
tool_calls). From a laptop: ssh -N -L 8000:localhost:8000 user@vm, then point opencode or pi
at http://localhost:8000/v1.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

TOOL_RE = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", re.S)
THINK_RE = re.compile(r"^\s*<think>(.*?)</think>\s*", re.S)


# --------------------------------------------------------------------------- loading


def _quant_method(path: Path) -> str:
    cfg = json.loads((path / "config.json").read_text())
    if cfg.get("model_file") == "flint_mlx.py":
        return "mlx"
    return str((cfg.get("quantization_config") or {}).get("quant_method", "")).lower()


def _unpacked_state(path: Path, dtype):
    """bf16 tensors of a GPTQ (compressed-tensors) or REAQ codebook folder, shard by shard."""
    import torch
    from safetensors import safe_open

    from blackhole_pipeline.pack import codebook_dequant, unpack_int4
    idx = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
    method = _quant_method(path)
    block = (json.loads((path / "config.json").read_text())["quantization_config"] or {}).get("block", 32)
    for shard in sorted(set(idx.values())):
        out = {}
        with safe_open(path / shard, "pt") as f:
            keys = set(f.keys())
            for k in keys:
                if method == "compressed-tensors" and k.endswith(".weight_packed"):
                    base = k[:-len(".weight_packed")]
                    n = f.get_tensor(base + ".weight_shape").tolist()[1]
                    sc = f.get_tensor(base + ".weight_scale").float()
                    g = -(-n // sc.shape[1])
                    w = unpack_int4(f.get_tensor(k), n).float() * sc.repeat_interleave(g, -1)[:, :n]
                    out[base + ".weight"] = w.to(dtype)
                elif method == "blackhole-reaq-codebook" and k.endswith(".experts") and ".tier" in k:
                    t = k[:-len(".experts")]
                    prefix = t.rsplit(".tier", 1)[0]
                    ids = f.get_tensor(k).tolist()
                    mats = {}
                    for mat in ("gate_up", "down"):
                        p = f.get_tensor(f"{t}.{mat}.idx")
                        n = p.shape[-1] * 2
                        codes = f.get_tensor(f"{t}.{mat}.block_scale")
                        n = min(n, codes.shape[-1] * block)
                        mats[mat] = codebook_dequant(p, n, codes, f.get_tensor(f"{t}.{mat}.expert_scale"),
                                                     f.get_tensor(f"{t}.{mat}.levels"), block)
                    I = mats["gate_up"].shape[1] // 2
                    for i, j in enumerate(ids):
                        out[f"{prefix}.{j}.gate_proj.weight"] = mats["gate_up"][i, :I].to(dtype)
                        out[f"{prefix}.{j}.up_proj.weight"] = mats["gate_up"][i, I:].to(dtype)
                        out[f"{prefix}.{j}.down_proj.weight"] = mats["down"][i].to(dtype)
                elif not re.search(r"\.(weight_scale|weight_shape)$|\.tier\d+\.", k):
                    out[k] = f.get_tensor(k)
        yield out


def _mlx_state(path: Path, dtype):
    """bf16 tensors (Hugging Face Flint names) of a BlackHole MLX folder: quantized modules are
    dequantized, the stacked hot/cold expert tensors split back into one module per expert (the
    router rows are already in the stacked order)."""
    import torch
    from safetensors.torch import load_file

    from blackhole_pipeline.pack import mlx_dequant, mlx_unpack
    cfg = json.loads((path / "config.json").read_text())
    q = cfg.get("quantization") or {}
    tiers = {int(k): v for k, v in (cfg.get("expert_tiers") or {}).items()}
    raw = {}
    for f in sorted(path.glob("model*.safetensors")):
        raw.update(load_file(str(f)))
    out = {}
    for k in [k for k in raw if k.endswith(".levels")]:  # HEX experts: per-expert level tables
        mod = k[:-len(".levels")]
        lv = raw.pop(k).float()
        bits = int(lv.shape[-1]).bit_length() - 1
        words = raw.pop(f"{mod}.idx")
        n, o, nw = words.shape
        u = mlx_unpack(words.reshape(n, o, nw // bits, bits), bits).reshape(n, o, -1)
        vals = torch.gather(lv, 1, u.reshape(n, -1)).view(u.shape)
        sc = raw.pop(f"{mod}.block_scale").view(torch.float8_e4m3fn).float()
        es = raw.pop(f"{mod}.expert_scale").float()
        raw[f"{mod}.weight"] = (vals * sc.repeat_interleave(32, -1) * es[:, None, None]).to(dtype)
    for k in list(raw):
        if not k.endswith(".weight") or k not in raw:
            continue
        mod = k[:-len(".weight")]
        if f"{mod}.scales" not in raw:
            continue
        spec = q.get(mod, {"bits": q.get("bits", 4), "group_size": q.get("group_size", 64)})
        u = mlx_unpack(raw.pop(k), spec["bits"])
        raw[k] = mlx_dequant(u, raw.pop(f"{mod}.scales"), raw.pop(f"{mod}.biases"), spec["group_size"]).to(dtype)
    ids = raw.pop("lm_head_ids", None)
    if ids is not None:  # output-trimmed LM head: back to full size, the dropped rows zero (see mask_head)
        ids = ids.to(torch.long)
        W = raw["lm_head.weight"]
        full = torch.zeros(cfg["vocab_size"], W.shape[1], dtype=W.dtype)
        full[ids] = W
        raw["lm_head.weight"] = full
        out["lm_head_ids"] = ids
    for k, v in raw.items():
        m = re.match(r"model\.layers\.(\d+)\.mlp\.(experts_hot|experts)\.(gate_proj|up_proj|down_proj)\.weight$", k)
        if not m:
            out[k] = v.to(dtype) if v.is_floating_point() and not k.endswith("softmax_offset") else v
            continue
        layer, tier, name = int(m.group(1)), m.group(2), m.group(3)
        base = 0 if tier == "experts_hot" else tiers.get(layer, 0)
        for j in range(v.shape[0]):
            out[f"model.layers.{layer}.mlp.experts.{base + j}.{name}.weight"] = v[j].to(dtype)
    return out


HEAD_FILL = -30.0   # logit of a token an output-trimmed LM head cannot write (finite, so KL stays finite)


def mask_head(model, ids) -> None:
    """Output-only vocab trim on a full-size PyTorch LM head: tokens outside ids get HEAD_FILL."""
    import torch
    V = model.lm_head.out_features
    keep = torch.zeros(V, dtype=torch.bool)
    keep[torch.as_tensor(ids, dtype=torch.long)] = True
    drop = ~keep

    def hook(mod, args, out):
        return out.masked_fill(drop.to(out.device), HEAD_FILL)
    model.lm_head.register_forward_hook(hook)


def load(path: Path, device: str):
    """Model and tokenizer; quantized folders are unpacked to bf16 on the way in."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    from reaper_pipeline.load import load_model, load_tokenizer
    from reaper_pipeline.models import register_all
    register_all()
    tok = load_tokenizer(path)
    method = _quant_method(path)
    if method not in ("compressed-tensors", "blackhole-reaq-codebook", "mlx"):
        prof = {"device": device, "dtype": "bfloat16" if device == "cuda" else "float32"}
        return load_model(path, prof), tok
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    cfg = AutoConfig.from_pretrained(path)
    for c in (cfg, getattr(cfg, "text_config", None)):
        for a in ("quantization_config", "quantization", "model_file", "expert_tiers", "head_vocab"):
            if c is not None and hasattr(c, a):
                delattr(c, a)
    try:
        from transformers.initialization import no_init_weights
    except ImportError:  # transformers 4.x
        from transformers.modeling_utils import no_init_weights
    with no_init_weights(), torch.device(device):
        model = AutoModelForCausalLM.from_config(cfg, dtype=dtype)
    missing = {n for n, _ in model.named_parameters()}
    head_ids = None
    for part in ([_mlx_state(path, dtype)] if method == "mlx" else _unpacked_state(path, dtype)):
        head_ids = part.pop("lm_head_ids", head_ids)
        model.load_state_dict(part, strict=False, assign=False)
        missing -= set(part)
    if missing - {"lm_head.weight"}:
        raise RuntimeError(f"{len(missing)} weights missing from {path}, e.g. {sorted(missing)[:3]}")
    if "lm_head.weight" in missing:
        model.tie_weights()
    if head_ids is not None:
        mask_head(model, head_ids)
    return model.eval(), tok


# --------------------------------------------------------------------------- chat


def to_template_messages(messages: list[dict]) -> list[dict]:
    """OpenAI chat messages -> what HF chat templates expect (text content, dict tool arguments)."""
    out = []
    for m in messages:
        m = dict(m)
        c = m.get("content")
        if isinstance(c, list):
            m["content"] = "".join(p.get("text", "") for p in c if isinstance(p, dict))
        elif c is None:
            m["content"] = ""
        if m.get("tool_calls"):
            calls = []
            for tc in m["tool_calls"]:
                fn = dict(tc.get("function") or {})
                if isinstance(fn.get("arguments"), str):
                    try:
                        fn["arguments"] = json.loads(fn["arguments"] or "{}")
                    except ValueError:
                        pass
                calls.append({**tc, "function": fn})
            m["tool_calls"] = calls
        out.append(m)
    return out


def parse_reply(text: str) -> tuple[str, str, list[dict]]:
    """(content, reasoning, tool_calls) from generated text: a leading <think> block is reasoning;
    <tool_call>{"name": ..., "arguments": {...}}</tool_call> blocks are tool calls."""
    reasoning = ""
    m = THINK_RE.match(text)
    if m:
        reasoning, text = m.group(1).strip(), text[m.end():]
    calls = []
    for raw in TOOL_RE.findall(text):
        try:
            obj = json.loads(raw)
        except ValueError:
            continue
        args = obj.get("arguments", obj.get("parameters", {}))
        calls.append({"id": f"call_{uuid.uuid4().hex[:12]}", "type": "function",
                      "function": {"name": obj.get("name", ""),
                                   "arguments": args if isinstance(args, str) else json.dumps(args)}})
    content = TOOL_RE.sub("", text).strip() if calls else text.strip()
    return content, reasoning, calls


class Engine:
    def __init__(self, model, tok, name: str):
        self.model, self.tok, self.name = model, tok, name
        self.lock = threading.Lock()
        gc = getattr(model, "generation_config", None)
        e = getattr(gc, "eos_token_id", None) if gc is not None else None
        e = [] if e is None else ([e] if isinstance(e, int) else list(e))
        self.eos = sorted({t for t in [tok.eos_token_id, *e] if t is not None})

    def prompt(self, req: dict) -> str:
        msgs = to_template_messages(req.get("messages") or [])
        kw = {"tools": req["tools"]} if req.get("tools") else {}
        try:
            return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
        except Exception:
            if kw:  # a template without tool support: describe the tools in a system message
                tools = json.dumps([t.get("function", t) for t in req["tools"]])
                sys_msg = {"role": "system", "content": "You can call these tools. To call one, reply with "
                           f"<tool_call>{{\"name\": ..., \"arguments\": {{...}}}}</tool_call>.\nTools: {tools}"}
                return self.tok.apply_chat_template([sys_msg] + msgs, tokenize=False, add_generation_prompt=True)
            raise

    def generate(self, req: dict, on_text=None) -> tuple[str, int, int, str]:
        """Generated text, prompt tokens, completion tokens, finish reason. on_text(full_text_so_far)."""
        import torch
        text = self.prompt(req)
        ids = self.tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"]
        max_new = int(req.get("max_tokens") or req.get("max_completion_tokens") or 2048)
        temp = float(req.get("temperature", 0.6) if req.get("temperature") is not None else 0.6)
        top_p = float(req.get("top_p") or 0.95)
        stops = req.get("stop") or []
        stops = [stops] if isinstance(stops, str) else list(stops)
        got: list[int] = []
        state = {"text": "", "reason": "length"}

        def on_token(nxt):
            t = int(nxt[0])
            if t in self.eos:
                state["reason"] = "stop"
                return True
            got.append(t)
            state["text"] = self.tok.decode(got, skip_special_tokens=True)
            for s in stops:
                if s and s in state["text"]:
                    state["text"] = state["text"].split(s)[0]
                    state["reason"] = "stop"
                    return True
            if on_text:
                on_text(state["text"])
            return False

        with self.lock, torch.no_grad():
            if hasattr(self.model, "fast_generate"):
                self.model.fast_generate(ids, torch.ones_like(ids), max_new, temp, top_p, self.eos, on_token=on_token)
            else:  # transformers generate: no token-by-token streaming
                dev = self.model.get_input_embeddings().weight.device
                g = self.model.generate(ids.to(dev), max_new_tokens=max_new, do_sample=temp > 0,
                                        temperature=max(temp, 1e-5), top_p=top_p, pad_token_id=self.tok.pad_token_id)
                for t in g[0, ids.shape[1]:].tolist():
                    if on_token(torch.tensor([t])):
                        break
        return state["text"], ids.shape[1], len(got), state["reason"]


# --------------------------------------------------------------------------- http


def make_handler(engine: Engine):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            sys.stderr.write(f"{time.strftime('%H:%M:%S')} {fmt % args}\n")

        def _json(self, code: int, obj) -> None:
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.rstrip("/") in ("/v1/models", "/models"):
                self._json(200, {"object": "list", "data": [{"id": engine.name, "object": "model",
                                                              "owned_by": "blackhole"}]})
            else:
                self._json(404, {"error": {"message": "not found"}})

        def do_POST(self):
            if self.path.rstrip("/") not in ("/v1/chat/completions", "/chat/completions"):
                return self._json(404, {"error": {"message": "only /v1/chat/completions"}})
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            except ValueError:
                return self._json(400, {"error": {"message": "bad JSON"}})
            cid, created = f"chatcmpl-{uuid.uuid4().hex[:16]}", int(time.time())
            if not req.get("stream"):
                text, pt, ct, reason = engine.generate(req)
                content, reasoning, calls = parse_reply(text)
                msg = {"role": "assistant", "content": content}
                if reasoning:
                    msg["reasoning_content"] = reasoning
                if calls:
                    msg["tool_calls"], reason = calls, "tool_calls"
                return self._json(200, {"id": cid, "object": "chat.completion", "created": created,
                                        "model": engine.name, "choices": [{"index": 0, "message": msg,
                                                                           "finish_reason": reason}],
                                        "usage": {"prompt_tokens": pt, "completion_tokens": ct,
                                                  "total_tokens": pt + ct}})
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True

            def send(delta: dict, finish=None):
                chunk = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": engine.name,
                         "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.flush()

            send({"role": "assistant", "content": ""})
            sent = {"n": 0}

            def on_text(full: str):
                # stream plain text; hold back anything from a <think> or <tool_call> tag on
                # (sent once parsed at the end), and a possible partial tag at the tail
                cut = len(full)
                for tag in ("<tool_call>", "<think>"):
                    i = full.find(tag)
                    if i >= 0:
                        cut = min(cut, i)
                j = full.rfind("<", sent["n"], cut)
                if j >= 0 and ">" not in full[j:cut]:
                    cut = j
                if cut > sent["n"]:
                    send({"content": full[sent["n"]:cut]})
                    sent["n"] = cut

            try:
                text, pt, ct, reason = engine.generate(req, on_text)
                content, reasoning, calls = parse_reply(text)
                # anything held back (after a think block, or a trailing partial tag)
                streamed = text[:sent["n"]]
                rest = content[len(streamed):] if content.startswith(streamed) else ""
                if reasoning:
                    send({"reasoning_content": reasoning})
                if rest:
                    send({"content": rest})
                if calls:
                    send({"tool_calls": [{**c, "index": i} for i, c in enumerate(calls)]})
                    reason = "tool_calls"
                send({}, reason)
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

    return Handler


def newest_model() -> Path | None:
    from blackhole_tui.jobs import home
    outs = sorted(home().glob("jobs/*/out/*"), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
    for pref in ("-GPTQ-W4A16", "-REAQ-codebook", ""):
        for p in outs:
            if p.name.endswith(pref) and (p / "config.json").exists() and not p.name.endswith("-FP8"):
                return p.resolve()
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", nargs="?", help="model folder (default: the newest job's GPTQ int4 model)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--name", help="model id clients use (default: the folder name)")
    ap.add_argument("--device", default=None, help="cuda (default when available) or cpu")
    a = ap.parse_args(argv)
    path = Path(a.model).expanduser().resolve() if a.model else newest_model()
    if path is None or not (path / "config.json").exists():
        ap.error("no model folder given and none found under ~/.blackhole/jobs/*/out")
    import torch
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"loading {path} on {device}…", flush=True)
    t = time.time()
    model, tok = load(path, device)
    if not hasattr(model, "fast_generate"):
        print("note: this model has no cached generation; replies will be slow", flush=True)
    engine = Engine(model, tok, a.name or path.name)
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(engine))
    print(f"loaded in {time.time() - t:.0f}s · serving {engine.name} at http://{a.host}:{a.port}/v1", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
