"""lobbot chat / ask: JSON answers shown as fields and code, files written, C compiled."""

import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from lobbot import answer

ROOT = Path(__file__).resolve().parents[1]

HEADER = """#ifndef OBSERVER_H
#define OBSERVER_H

typedef void (*observer_fn)(void *ctx, int event);

typedef struct {
    observer_fn fns[8];
    void *ctxs[8];
    int count;
} subject_t;

void subject_init(subject_t *s);
int subject_attach(subject_t *s, observer_fn fn, void *ctx);
void subject_notify(subject_t *s, int event);

#endif /* OBSERVER_H */
"""
IMPL = """#include "observer.h"

void subject_init(subject_t *s) { s->count = 0; }

int subject_attach(subject_t *s, observer_fn fn, void *ctx) {
    if (s->count >= 8) return -1;
    s->fns[s->count] = fn;
    s->ctxs[s->count++] = ctx;
    return 0;
}

void subject_notify(subject_t *s, int event) {
    for (int i = 0; i < s->count; i++) s->fns[i](s->ctxs[i], event);
}
"""
ANSWER = json.dumps({"pattern_name": "Observer", "header": HEADER, "implementation": IMPL})


def test_render_json_answer():
    shown, codes = answer.render(ANSWER)
    assert [c.name for c in codes] == ["observer.h", "observer.c"]
    assert "pattern_name  Observer" in shown
    assert "── header · observer.h" in shown and "── implementation · observer.c" in shown
    assert '#include "observer.h"' in shown and "\\n" not in shown  # real newlines, no escapes


def test_parse_tolerates_fences_and_chatter():
    assert answer.parse("```json\n" + ANSWER + "\n```")["pattern_name"] == "Observer"
    assert answer.parse("Here you go:\n" + ANSWER + "\nHope it helps")["pattern_name"] == "Observer"
    assert answer.parse("just text") is None
    shown, codes = answer.render('{"pattern_name": "Obs", "header": "#ifndef')  # cut off mid-answer
    assert "not valid JSON" in shown and codes == []
    shown, _ = answer.render("Plain prose answer.")
    assert shown == "Plain prose answer."


def test_file_names():
    # No include guard and no #include: names come from the name-like field and pair up.
    codes = answer.code_fields({"name": "Ring Buffer", "header": "typedef int rb;\nint rb_len(void);",
                                "implementation": "int rb_len(void) {\n  return 0;\n}"})
    assert [c.name for c in codes] == ["ring_buffer.h", "ring_buffer.c"]
    codes = answer.code_fields({"script": "import os\nprint(os.getcwd())", "summary": "prints cwd"})
    assert [c.name for c in codes] == ["script.py"]


@pytest.mark.skipif(answer.c_compiler() is None, reason="no C compiler")
def test_compile_check():
    _, codes = answer.render(ANSWER)
    good, cmd, _ = answer.compile_check(codes)
    assert good and "-fsyntax-only" in cmd
    broken = [answer.Code("implementation", IMPL.replace("return 0;", "return zero;"), "observer.c"), codes[0]]
    good, _, out = answer.compile_check(broken)
    assert not good and "zero" in out
    good, _, _ = answer.compile_check([codes[0]])  # header only
    assert good
    assert answer.compile_check(answer.code_fields({"script": "import os\nprint(1)"})) is None


class FakeOllama(BaseHTTPRequestHandler):
    reply = ANSWER

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if body["model"] != "lobbot-arch":
            self.send_response(404)
            self.end_headers()
            self.wfile.write(json.dumps({"error": f"model '{body['model']}' not found"}).encode())
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.end_headers()
        text = self.reply
        for i in range(0, len(text), 40):
            self.wfile.write(json.dumps({"message": {"content": text[i:i + 40]}, "done": False}).encode() + b"\n")
        self.wfile.write(json.dumps({"done": True, "eval_count": 300, "eval_duration": 6_000_000_000}).encode() + b"\n")

    def log_message(self, *a):
        pass


@pytest.fixture()
def ollama():
    srv = HTTPServer(("127.0.0.1", 0), FakeOllama)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"127.0.0.1:{srv.server_port}"
    srv.shutdown()


def lobbot(ollama, *args, stdin=None, cwd=None):
    env = {**os.environ, "OLLAMA_HOST": ollama, "PYTHONPATH": str(ROOT)}
    return subprocess.run([sys.executable, "-m", "lobbot.app", *args], env=env, input=stdin, cwd=cwd,
                          capture_output=True, text=True, timeout=60)


def test_ask_writes_files_and_compiles(ollama, tmp_path):
    p = lobbot(ollama, "ask", "lobbot-arch", "an event bus for sensors", "--out", str(tmp_path / "out"))
    assert p.returncode == 0, p.stderr
    assert "── implementation · observer.c" in p.stdout and "300 tokens · 50.0 tok/s" in p.stdout
    assert (tmp_path / "out/observer.h").read_text() == HEADER
    assert (tmp_path / "out/observer.c").read_text() == IMPL
    assert "wrote observer.h, observer.c" in p.stdout
    if answer.c_compiler():
        assert "compiles" in p.stdout and "does not compile" not in p.stdout
    raw = lobbot(ollama, "ask", "lobbot-arch", "x", "--raw")
    assert raw.stdout.startswith(ANSWER)
    missing = lobbot(ollama, "ask", "nope", "x")
    assert missing.returncode == 1 and "not found" in missing.stderr and "ollama list" in missing.stderr


def test_chat_session(ollama):
    p = lobbot(ollama, "chat", "lobbot-arch", stdin='"""\nan event bus\nfor sensors\n"""\n/bye\n')
    assert p.returncode == 0, p.stderr
    assert "answered on its own" in p.stdout and "pattern_name  Observer" in p.stdout


def test_no_ollama():
    p = lobbot("127.0.0.1:9", "ask", "lobbot-arch", "x")
    assert p.returncode == 1 and "cannot reach Ollama" in p.stderr
