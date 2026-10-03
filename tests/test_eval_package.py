import json

from stages.eval import agreement
from stages.package import modelfile, template_family

REF = json.dumps({"category": "billing", "priority": "high",
                  "summary": "Customer was double-charged and wants a refund.", "customer_sentiment": "negative"})


def test_agreement_exact_json():
    assert agreement(REF, REF) == 1.0


def test_agreement_partial_and_fenced():
    ans = "```json\n" + json.dumps({"category": "billing", "priority": "low",
                                     "summary": "Customer was double-charged and wants a refund.",
                                     "customer_sentiment": "negative"}) + "\n```"
    assert agreement(ans, REF) == 0.75


def test_agreement_invalid_json_is_zero():
    assert agreement("sure! here is your ticket", REF) == 0.0


def test_agreement_plain_text_f1():
    assert 0 < agreement("the cat sat", "the cat sat down") < 1


def test_modelfile_chatml():
    mf = modelfile("Be terse.", "{% for m in messages %}<|im_start|>{{ m.role }}...")
    assert mf.startswith("FROM ./model.gguf")
    assert 'PARAMETER stop "<|im_end|>"' in mf and 'SYSTEM """Be terse."""' in mf


def test_modelfile_gemma4():
    mf = modelfile("Be terse.", "{{ bos_token }}{{ '<|turn>' + role + '\\n' }}...{{ '<turn|>\\n' }}")
    assert "<|turn>system\n{{ .System }}<turn|>" in mf
    assert mf.split('TEMPLATE """')[1].split('"""')[0].endswith("<|turn>model\n<|channel>thought\n<channel|>")
    assert 'PARAMETER stop "<turn|>"' in mf and "<|im_end|>" not in mf


def test_modelfile_gemma3_folds_system_into_user():
    mf = modelfile("Be terse.", "{{ '<start_of_turn>' + role }}...")
    assert "<start_of_turn>user" in mf and 'PARAMETER stop "<end_of_turn>"' in mf
    assert "<start_of_turn>system" not in mf


def test_template_family_falls_back_to_model_id():
    assert template_family("", "google/gemma-4-E4B-it") == "gemma4"
    assert template_family("", "google/gemma-3-4b-it") == "gemma3"
    assert template_family("", "Qwen/Qwen3-4B-Instruct-2507") == "chatml"
    assert template_family("{{ unknown }}", "google/gemma-4-E4B-it") is None  # readable but unknown: leave it to Ollama


def test_eval_generates_at_least_as_long_as_teacher_answers():
    from stages import data, eval as ev
    assert ev.MAX_TOKENS >= data.ANSWER_MAX_TOKENS
    assert ev.CTX_PER_SLOT >= ev.MAX_TOKENS + data.MAX_INPUT_CHARS // 3


def test_eval_cap_follows_raised_data_cap():
    import os, subprocess, sys
    code = "from stages import data, eval as ev; print(data.ANSWER_MAX_TOKENS, ev.MAX_TOKENS, ev.CTX_PER_SLOT)"
    env = {k: v for k, v in os.environ.items() if not k.startswith("LOBBOT_EVAL_")}
    out = subprocess.run([sys.executable, "-c", code], env={**env, "LOBBOT_DATA_ANSWER_MAX_TOKENS": "4096"},
                         capture_output=True, text=True, check=True).stdout.split()
    data_cap, eval_cap, ctx = map(int, out)
    assert data_cap == 4096 and eval_cap >= data_cap and ctx >= eval_cap + 2000
    out = subprocess.run([sys.executable, "-c", code], env={**env, "LOBBOT_EVAL_MAX_TOKENS": "1000"},
                         capture_output=True, text=True, check=True).stdout.split()
    assert int(out[1]) == 1000  # explicit override wins


def test_eval_cap_follows_per_job_data_cap(monkeypatch):
    from stages import data, eval as ev
    from stages._util import Config

    monkeypatch.delenv("LOBBOT_EVAL_MAX_TOKENS", raising=False)
    monkeypatch.delenv("LOBBOT_EVAL_CTX", raising=False)
    cfg = Config(data_answer_max_tokens=6000, data_max_len=16384)
    assert data.answer_max_tokens(cfg) == 6000 and data.max_model_len(cfg) == 16384
    assert ev.limits(cfg) == (6000, 6000 + 4096)
    assert ev.limits(Config())[0] == max(2048, data.ANSWER_MAX_TOKENS)  # no override: env knob / default
