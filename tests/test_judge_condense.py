"""Gemini judge and condense.chat routing, against a local fake server."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace

import pytest

from stages import condense, gemini
from stages import eval as ev

SPEC = SimpleNamespace(description="d", output_format="json", eval_criteria="c")


@pytest.fixture
def server():
    seen = []

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["content-length"])))
            seen.append({"path": self.path, "headers": {k.lower(): v for k, v in self.headers.items()}, "body": body})
            if self.path.startswith("/broken"):
                self.send_response(404)
                self.send_header("content-length", "0")
                self.end_headers()
                return
            if self.path == "/v1/compress":
                out = {"model": body["model"], "messages": [
                    {"role": m["role"], "content": " ".join(m["content"].split()[::2])} for m in body["messages"]]}
            elif self.path.endswith("/chat/completions"):
                out = {"choices": [{"message": {"role": "assistant", "content": "8"}}]}
            else:
                out = {"id": "m", "type": "message", "role": "assistant", "model": body["model"],
                       "content": [{"type": "text", "text": "7"}], "stop_reason": "end_turn",
                       "usage": {"input_tokens": 5, "output_tokens": 1}}
            data = json.dumps(out).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}", seen
    srv.shutdown()


@pytest.fixture(autouse=True)
def fresh_gemini(monkeypatch):
    monkeypatch.setattr(gemini, "_condense_off", False)
    monkeypatch.delenv("CONDENSE_API_KEY", raising=False)
    monkeypatch.delenv("CONDENSE_GEMINI_URL", raising=False)
    monkeypatch.delenv("CONDENSE_COMPRESS_URL", raising=False)


def test_gemini_judge(server, monkeypatch):
    url, seen = server
    monkeypatch.setattr(gemini, "DIRECT_URL", url + "/v1beta/openai/chat/completions")
    monkeypatch.setenv("GEMINI_API_KEY", "g-key")
    assert ev.judge_key("gemini-3.8-flash") == "g-key"
    assert ev.judge(SPEC, "gemini-3.8-flash", ["a", "b"], ["x", "y"]) == 0.8
    assert seen[0]["headers"]["authorization"] == "Bearer g-key"
    assert seen[0]["body"]["model"] == "gemini-3.8-flash"


def test_claude_judge_through_condense(server, monkeypatch):
    pytest.importorskip("anthropic")
    url, seen = server
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("CONDENSE_API_KEY", "ak_test")
    monkeypatch.setenv("CONDENSE_BASE_URL", url + "/anthropic")
    assert ev.judge(SPEC, "claude-sonnet-5-5", ["a"], ["x"]) == 0.7
    req = seen[0]
    assert req["path"].startswith("/anthropic/v1/messages")
    assert req["headers"]["x-condense-auth-token"] == "ak_test"
    assert req["headers"]["x-api-key"] == "sk-test"


def test_condense_off_is_plain_anthropic(monkeypatch):
    pytest.importorskip("anthropic")
    monkeypatch.delenv("CONDENSE_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert condense.headers() == {}
    assert "condense" not in str(condense.anthropic_client().base_url)


def test_no_key_means_no_judge(monkeypatch):
    for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    assert ev.judge_key("gemini-3.8-flash") is None


def test_gemini_4xx_stops_judging(monkeypatch):
    import httpx

    calls = []

    def fake_post(url, **kw):
        calls.append(1)
        return httpx.Response(400, request=httpx.Request("POST", url), json={"error": "bad model"})

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setenv("GEMINI_API_KEY", "g-key")
    assert ev.judge(SPEC, "gemini-nope", ["a"] * 50, ["x"] * 50) is None
    assert len(calls) < 50


def test_gemini_through_condense(server, monkeypatch):
    url, seen = server
    monkeypatch.setattr(gemini, "DIRECT_URL", url + "/direct/chat/completions")
    monkeypatch.setenv("GEMINI_API_KEY", "g-key")
    monkeypatch.setenv("CONDENSE_API_KEY", "ak_test")
    monkeypatch.setenv("CONDENSE_GEMINI_URL", url + "/google/v1beta/openai/chat/completions")
    assert gemini.chat("gemini-3.8-flash", [{"role": "user", "content": "hi"}]) == "8"
    assert gemini.last_route == "condense"
    assert seen[0]["path"].startswith("/google/") and seen[0]["headers"]["x-condense-auth-token"] == "ak_test"
    assert seen[0]["headers"]["authorization"] == "Bearer g-key"


def test_condense_failure_falls_back_to_direct_once(server, monkeypatch):
    url, seen = server
    monkeypatch.setattr(gemini, "DIRECT_URL", url + "/direct/chat/completions")
    monkeypatch.setenv("GEMINI_API_KEY", "g-key")
    monkeypatch.setenv("CONDENSE_API_KEY", "ak_test")
    monkeypatch.setenv("CONDENSE_GEMINI_URL", url + "/broken/chat/completions")
    for _ in range(3):
        assert gemini.chat("gemini-3.8-flash", [{"role": "user", "content": "hi"}]) == "8"
    assert gemini.last_route == "direct"
    # condense tried once, then switched off; every call answered directly
    assert [r["path"] for r in seen] == ["/broken/chat/completions"] + ["/direct/chat/completions"] * 3
    assert "x-condense-auth-token" not in seen[1]["headers"]


def test_compress(server, monkeypatch):
    url, seen = server
    monkeypatch.setenv("CONDENSE_API_KEY", "ak_test")
    monkeypatch.setenv("CONDENSE_COMPRESS_URL", url + "/v1/compress")
    assert condense.compress(["a b c d", "e f g h"]) == ["a c", "e g"]
    assert seen[0]["body"]["model"] == "helene-1.1" and seen[0]["headers"]["x-condense-auth-token"] == "ak_test"
    assert len(seen[0]["body"]["messages"]) == 2


def test_compress_falls_back_to_original(server, monkeypatch):
    url, seen = server
    texts = ["keep this text"]
    assert condense.compress(texts) == texts and not seen  # condense off: no request
    monkeypatch.setenv("CONDENSE_API_KEY", "ak_test")
    monkeypatch.setenv("CONDENSE_COMPRESS_URL", url + "/broken/v1/compress")
    assert condense.compress(texts) == texts  # error: originals


def test_gemini_proxy_is_opt_in(monkeypatch):
    monkeypatch.setenv("CONDENSE_API_KEY", "ak_test")
    assert gemini.condense_url() is None


def test_testgen_shows_compressed_examples(server, monkeypatch):
    from stages import testgen
    from stages.data import norm_key, parse_array

    url, seen = server
    monkeypatch.setenv("CONDENSE_API_KEY", "ak_test")
    monkeypatch.setenv("CONDENSE_COMPRESS_URL", url + "/v1/compress")
    prompts = []
    monkeypatch.setattr(gemini, "chat", lambda model, msgs, **kw: prompts.append(msgs[0]["content"]) or "[]")
    ex = [SimpleNamespace(input=f"one two three four example {i}") for i in range(10)]
    spec = SimpleNamespace(description="d", input_format="email", seed_examples=ex)
    testgen.held_out_inputs(spec, SimpleNamespace(testgen_model="gemini-3.8-flash"), 10, set(), norm_key, parse_array)
    assert len(seen[0]["body"]["messages"]) == 7  # 8 examples, the first sent uncompressed
    p = prompts[0]
    assert "one two three four example 0" in p  # the uncompressed one
    assert "one three example" in p and "one two three four example 1" not in p
    assert p.index("example 0") < p.index("shortened by a compression tool") < p.index("one three example")


def test_testgen_without_condense_has_no_compression_note(monkeypatch):
    from stages import testgen
    from stages.data import norm_key, parse_array

    prompts = []
    monkeypatch.setattr(gemini, "chat", lambda model, msgs, **kw: prompts.append(msgs[0]["content"]) or "[]")
    ex = [SimpleNamespace(input=f"full example {i}") for i in range(10)]
    spec = SimpleNamespace(description="d", input_format="email", seed_examples=ex)
    testgen.held_out_inputs(spec, SimpleNamespace(testgen_model="gemini-3.8-flash"), 10, set(), norm_key, parse_array)
    assert "compression tool" not in prompts[0] and "full example 3" in prompts[0] and "full example 4" not in prompts[0]
