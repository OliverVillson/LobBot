"""Gemini judge and condense.chat routing, against a local fake server."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace

import pytest

from stages import condense
from stages import eval as ev

SPEC = SimpleNamespace(description="d", output_format="json", eval_criteria="c")


@pytest.fixture
def server():
    seen = []

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["content-length"])))
            seen.append({"path": self.path, "headers": {k.lower(): v for k, v in self.headers.items()}, "body": body})
            if self.path.endswith("/chat/completions"):
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


def test_gemini_judge(server, monkeypatch):
    url, seen = server
    monkeypatch.setattr(ev, "GEMINI_URL", url + "/v1beta/openai/chat/completions")
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
