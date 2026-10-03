import json

from stages.eval import agreement
from stages.package import modelfile

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
