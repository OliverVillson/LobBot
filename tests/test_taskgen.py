import json

import pytest

from common.taskspec import TaskSpec
from lobbot import taskgen

SECRET = "sk-test-SECRET-KEY-123456"


def make_spec(n=5, **over):
    spec = {
        "task_name": "Review Topic Classifier",
        "description": "Classify an app store review by topic.",
        "input_format": "One app store review.",
        "output_format": 'A JSON object with key "topic".',
        "seed_examples": [
            {"input": f"review {i}", "output": json.dumps({"topic": "ux"})} for i in range(n)
        ],
        "eval_criteria": "Topic matches the review.",
    }
    spec.update(over)
    return spec


def gemini_resp(text):
    return {"candidates": [{"content": {"role": "model", "parts": [{"text": text}]}}]}


def anthropic_resp(text):
    return {"content": [{"type": "text", "text": text}], "stop_reason": "end_turn"}


@pytest.fixture
def clean_env(monkeypatch):
    for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY", "LOBBOT_TASKGEN_MODEL"):
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


def fake_post(responses, calls):
    def _post(url, headers, body, timeout):
        calls.append({"url": url, "headers": headers, "body": body, "timeout": timeout})
        return responses.pop(0)
    return _post


def test_success_via_gemini(clean_env):
    clean_env.setenv("GEMINI_API_KEY", SECRET)
    clean_env.setenv("ANTHROPIC_API_KEY", "other")
    calls = []
    clean_env.setattr(taskgen, "_post_json", fake_post([gemini_resp(json.dumps(make_spec()))], calls))

    d = taskgen.draft_taskspec("classify app reviews", n_seeds=5)

    TaskSpec.from_dict(d)
    assert d["task_name"] == "review-topic-classifier"
    assert d["target"] == {"max_size_gb": 7, "min_tok_s": 40, "laptop_ram_gb": 16}
    assert d["version"] == 1
    assert len(calls) == 1
    call = calls[0]
    assert call["url"] == ("https://generativelanguage.googleapis.com/v1beta/models/"
                           "gemini-3.8-flash:generateContent")
    assert call["headers"]["x-goog-api-key"] == SECRET
    assert SECRET not in call["url"]
    assert call["body"]["generationConfig"]["responseMimeType"] == "application/json"
    prompt = call["body"]["contents"][0]["parts"][0]["text"]
    assert "classify app reviews" in prompt and "support-email-to-ticket" in prompt
    assert "exactly 5" in prompt


def test_model_env_override(clean_env):
    clean_env.setenv("GOOGLE_API_KEY", SECRET)
    clean_env.setenv("LOBBOT_TASKGEN_MODEL", "gemini-custom")
    calls = []
    clean_env.setattr(taskgen, "_post_json", fake_post([gemini_resp(json.dumps(make_spec()))], calls))
    taskgen.draft_taskspec("x")
    assert "/models/gemini-custom:generateContent" in calls[0]["url"]


def test_anthropic_fenced_json(clean_env):
    clean_env.setenv("ANTHROPIC_API_KEY", SECRET)
    spec = make_spec(n=4, target={"max_size_gb": 7}, version=None)
    # dict-valued outputs get serialized to JSON strings
    spec["seed_examples"][0]["output"] = {"topic": "bugs"}
    text = "Here you go:\n```json\n" + json.dumps(spec, indent=2) + "\n```\nHope that helps!"
    calls = []
    clean_env.setattr(taskgen, "_post_json", fake_post([anthropic_resp(text)], calls))

    d = taskgen.draft_taskspec("classify app reviews", n_seeds=4)

    TaskSpec.from_dict(d)
    assert d["version"] == 1
    assert d["target"]["laptop_ram_gb"] == 16
    assert json.loads(d["seed_examples"][0]["output"]) == {"topic": "bugs"}
    call = calls[0]
    assert call["url"] == "https://api.anthropic.com/v1/messages"
    assert call["headers"]["x-api-key"] == SECRET
    assert call["headers"]["anthropic-version"] == "2023-06-01"
    assert call["body"]["model"] == "claude-sonnet-5-5"


def test_parse_outermost_object_without_fence():
    obj = taskgen._parse_json_object('Sure! {"a": {"b": 1}} trailing')
    assert obj == {"a": {"b": 1}}


def test_retry_after_invalid_first_answer(clean_env):
    clean_env.setenv("GEMINI_API_KEY", SECRET)
    bad = json.dumps(make_spec(n=1))  # too few seeds
    good = json.dumps(make_spec(n=6))
    calls = []
    clean_env.setattr(taskgen, "_post_json", fake_post([gemini_resp(bad), gemini_resp(good)], calls))

    d = taskgen.draft_taskspec("classify app reviews", n_seeds=6)

    assert len(d["seed_examples"]) == 6
    assert len(calls) == 2
    contents = calls[1]["body"]["contents"]
    assert [c["role"] for c in contents] == ["user", "model", "user"]
    assert "3 to 50" in contents[2]["parts"][0]["text"]


def test_retry_after_unparseable_answer_then_gives_up(clean_env):
    clean_env.setenv("GEMINI_API_KEY", SECRET)
    calls = []
    clean_env.setattr(taskgen, "_post_json",
                      fake_post([gemini_resp("not json"), gemini_resp("{still not")], calls))
    with pytest.raises(taskgen.TaskgenError, match="after 2 attempts"):
        taskgen.draft_taskspec("x")
    assert len(calls) == 2


def test_no_key_error(clean_env):
    with pytest.raises(taskgen.TaskgenError) as ei:
        taskgen.draft_taskspec("x")
    msg = str(ei.value)
    assert "GEMINI_API_KEY" in msg and "ANTHROPIC_API_KEY" in msg


def test_explicit_provider_missing_key(clean_env):
    clean_env.setenv("GEMINI_API_KEY", SECRET)
    with pytest.raises(taskgen.TaskgenError, match="ANTHROPIC_API_KEY"):
        taskgen.draft_taskspec("x", provider="anthropic")


def test_http_error_hides_key(clean_env):
    clean_env.setenv("GEMINI_API_KEY", SECRET)
    import io

    seen = {}

    def fake_urlopen(req, timeout):
        seen["headers"] = dict(req.header_items())
        raise taskgen.urllib.error.HTTPError(
            req.full_url, 403, "Forbidden", {},
            io.BytesIO(b'{"error": {"code": 403, "message": "API key not valid"}}'))

    clean_env.setattr(taskgen.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(taskgen.TaskgenError) as ei:
        taskgen.draft_taskspec("x")
    msg = str(ei.value)
    assert "403" in msg and "API key not valid" in msg
    assert SECRET not in msg
    assert SECRET not in repr(ei.value)
    assert ei.value.__cause__ is None
    assert any(v == SECRET for v in seen["headers"].values())  # key was sent in a header


def test_main_writes_file(clean_env, tmp_path, capsys):
    clean_env.setenv("GEMINI_API_KEY", SECRET)
    clean_env.setattr(taskgen, "_post_json", fake_post([gemini_resp(json.dumps(make_spec()))], []))
    out = tmp_path / "spec.json"
    assert taskgen.main(["classify app reviews", "-o", str(out), "--seeds", "5"]) == 0
    TaskSpec.load(out)
    err = capsys.readouterr().err
    assert "review-topic-classifier" in err and "seeds:     5" in err
    assert SECRET not in err
