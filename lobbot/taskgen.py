"""Draft a LobBot TaskSpec from a short plain-English task description.

Calls an LLM (Gemini or Anthropic) over plain HTTPS with the standard library,
parses its JSON answer, fills defaults and validates it with
``common.taskspec.TaskSpec``. One retry is made if the first answer does not
parse or validate.

    python -m lobbot.taskgen "classify app reviews by topic" -o spec.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

from common.taskspec import TaskSpec

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
DEFAULT_MODELS = {"gemini": "gemini-3.8-flash", "anthropic": "claude-sonnet-5-5"}
DEFAULT_TARGET = {"max_size_gb": 7, "min_tok_s": 40, "laptop_ram_gb": 16}
PROVIDERS = ("gemini", "anthropic")

_EXAMPLE_PATH = Path(__file__).resolve().parent.parent / "examples" / "support-tickets.taskspec.json"

# Fallback reference used when the repo's examples/ dir is not available
# (e.g. an installed wheel). Trimmed copy of examples/support-tickets.taskspec.json.
_FALLBACK_EXAMPLE = {
    "version": 1,
    "task_name": "support-email-to-ticket",
    "description": "Turn a customer support email into a structured JSON ticket for a SaaS helpdesk.",
    "input_format": "The free-text body of one customer email.",
    "output_format": "A single JSON object with keys: category (one of billing, bug, account, feature_request, other), priority (low, medium, high, urgent), summary (one sentence, max 25 words), customer_sentiment (negative, neutral, positive).",
    "seed_examples": [
        {"input": "Hi, I was charged twice for my Pro subscription this month. Can you refund the extra charge? Order #48213.", "output": "{\"category\": \"billing\", \"priority\": \"high\", \"summary\": \"Customer was double-charged for the Pro subscription this month and requests a refund for order #48213.\", \"customer_sentiment\": \"negative\"}"},
        {"input": "The export to CSV button does nothing in Firefox. Works fine in Chrome.", "output": "{\"category\": \"bug\", \"priority\": \"medium\", \"summary\": \"CSV export button is unresponsive in Firefox but works in Chrome.\", \"customer_sentiment\": \"neutral\"}"},
        {"input": "Our whole team is locked out since this morning, nobody can log in and we have a client demo in an hour!!!", "output": "{\"category\": \"account\", \"priority\": \"urgent\", \"summary\": \"Entire team cannot log in since this morning, blocking a client demo in one hour.\", \"customer_sentiment\": \"negative\"}"},
        {"input": "Love the new dashboard! Would be great if we could also schedule reports to go out by email every Monday.", "output": "{\"category\": \"feature_request\", \"priority\": \"low\", \"summary\": \"Customer likes the new dashboard and requests scheduled weekly email reports.\", \"customer_sentiment\": \"positive\"}"},
    ],
    "eval_criteria": "Output is a single valid JSON object with exactly the four keys; category and priority are from the allowed sets and match the email; summary is faithful, specific and at most 25 words; sentiment matches the tone.",
    "target": dict(DEFAULT_TARGET),
}


class TaskgenError(RuntimeError):
    """Raised when a TaskSpec cannot be drafted (no key, HTTP error, bad output)."""


# --------------------------------------------------------------------------- HTTP


def _post_json(url: str, headers: dict, body: dict, timeout: float) -> dict:
    """POST a JSON body and return the decoded JSON response.

    Uses urllib, so HTTPS_PROXY and the system CA bundle are honoured and TLS is
    verified. Error messages never include request headers (which carry keys).
    """
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={"content-type": "application/json", **headers},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as e:
        try:
            excerpt = e.read().decode("utf-8", "replace")
        except Exception:
            excerpt = ""
        excerpt = " ".join(excerpt.split())[:300]
        raise TaskgenError(f"HTTP {e.code} from {_host(url)}: {excerpt}") from None
    except urllib.error.URLError as e:
        raise TaskgenError(f"request to {_host(url)} failed: {e.reason}") from None
    except TimeoutError:
        raise TaskgenError(f"request to {_host(url)} timed out after {timeout:g}s") from None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        raise TaskgenError(f"non-JSON response from {_host(url)}: {raw[:200]!r}") from None


def _host(url: str) -> str:
    return url.split("/")[2] if "://" in url else url


# ---------------------------------------------------------------------- providers


def _resolve_provider(provider: str | None) -> tuple[str, str]:
    """Return (provider, api_key)."""
    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if provider is None:
        if gemini_key:
            return "gemini", gemini_key
        if anthropic_key:
            return "anthropic", anthropic_key
        raise TaskgenError(
            "no LLM API key found: set GEMINI_API_KEY (or GOOGLE_API_KEY) for Gemini, "
            "or ANTHROPIC_API_KEY for Anthropic"
        )
    provider = provider.lower()
    if provider == "gemini":
        if not gemini_key:
            raise TaskgenError("provider 'gemini' needs GEMINI_API_KEY (or GOOGLE_API_KEY) to be set")
        return provider, gemini_key
    if provider == "anthropic":
        if not anthropic_key:
            raise TaskgenError("provider 'anthropic' needs ANTHROPIC_API_KEY to be set")
        return provider, anthropic_key
    raise TaskgenError(f"unknown provider {provider!r}; choose one of: {', '.join(PROVIDERS)}")


def _call_gemini(messages: list[dict], system: str, model: str, key: str, timeout: float) -> str:
    contents = [
        {"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
        for m in messages
    ]
    body = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": contents,
        "generationConfig": {"responseMimeType": "application/json", "temperature": 0.8},
    }
    resp = _post_json(GEMINI_URL.format(model=model), {"x-goog-api-key": key}, body, timeout)
    try:
        parts = resp["candidates"][0]["content"]["parts"]
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
    except (KeyError, IndexError, TypeError):
        reason = ""
        if isinstance(resp, dict):
            fb = resp.get("promptFeedback") or (resp.get("candidates") or [{}])[0].get("finishReason")
            reason = f" ({fb})" if fb else ""
        raise TaskgenError(f"unexpected Gemini response shape{reason}") from None
    if not text.strip():
        raise TaskgenError("Gemini returned an empty answer")
    return text


def _call_anthropic(messages: list[dict], system: str, model: str, key: str, timeout: float) -> str:
    body = {
        "model": model,
        "max_tokens": 16000,
        "system": system,
        "messages": [{"role": m["role"], "content": m["content"]} for m in messages],
    }
    headers = {"x-api-key": key, "anthropic-version": "2023-06-01"}
    resp = _post_json(ANTHROPIC_URL, headers, body, timeout)
    try:
        text = "".join(b.get("text", "") for b in resp["content"] if b.get("type") == "text")
    except (KeyError, TypeError):
        raise TaskgenError("unexpected Anthropic response shape") from None
    if not text.strip():
        raise TaskgenError(f"Anthropic returned an empty answer (stop_reason={resp.get('stop_reason')})")
    return text


_CALLERS = {"gemini": _call_gemini, "anthropic": _call_anthropic}


# ------------------------------------------------------------------------- prompt


def _example_spec() -> dict:
    try:
        return json.loads(_EXAMPLE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _FALLBACK_EXAMPLE


SYSTEM_PROMPT = (
    "You design training specifications (TaskSpecs) for LobBot, which distils a large LLM "
    "into a small task-specialised model that runs on a laptop. You reply with exactly one "
    "JSON object and nothing else."
)


def _build_prompt(description: str, n_seeds: int) -> str:
    example = json.dumps(_example_spec(), indent=2, ensure_ascii=False)
    return f"""Write a TaskSpec JSON object for this task:

<task>
{description.strip()}
</task>

Here is a complete, high-quality reference TaskSpec for a different task. Match its structure and its level of precision:

<reference>
{example}
</reference>

Requirements:
- Keys: version (1), task_name, description, input_format, output_format, seed_examples, eval_criteria, target.
- task_name: a short kebab-case slug (lowercase letters, digits and hyphens, 2-5 words), e.g. "support-email-to-ticket".
- description: one or two sentences saying what the model does and for whom.
- input_format: what exactly one input looks like.
- output_format: a precise, unambiguous spec of the expected output (exact keys, allowed values, length limits, style). If the output is structured, prefer a single JSON object with a fixed set of keys.
- seed_examples: exactly {n_seeds} objects, each {{"input": "...", "output": "..."}} with both values as strings.
  - Inputs must be diverse and realistic, as a real user would send them: vary length (one line to several paragraphs), tone, difficulty and topic; include edge cases (ambiguous, messy, typos, off-topic or minimal inputs) where they make sense.
  - Each output must be the literal, exact text the trained model should produce for that input, of the best possible quality, strictly following output_format. No commentary or explanation around it.
  - When the output is JSON, the "output" value is a JSON *string* containing the serialized object (escaped quotes), exactly like the reference.
- eval_criteria: how a judge should decide whether an output is correct and good, checkable per example.
- target: {json.dumps(DEFAULT_TARGET)}

Return only the JSON object."""


# ------------------------------------------------------------------------ parsing


_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*\n?(.*?)```", re.S)


def _parse_json_object(text: str) -> dict:
    """Extract one JSON object from an LLM answer (fences, prose around it, etc.)."""
    text = text.strip()
    candidates = [text]
    m = _FENCE_RE.search(text)
    if m:
        candidates.insert(0, m.group(1).strip())
    for cand in list(candidates):
        start, end = cand.find("{"), cand.rfind("}")
        if start != -1 and end > start:
            candidates.append(cand[start : end + 1])
    last_err: Exception | None = None
    for cand in candidates:
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError as e:
            last_err = e
            continue
        if isinstance(obj, dict):
            return obj
        last_err = ValueError(f"expected a JSON object, got {type(obj).__name__}")
    raise ValueError(f"could not parse a JSON object from the answer: {last_err}")


def _as_text(v) -> str:
    if isinstance(v, str):
        return v
    return json.dumps(v, ensure_ascii=False)


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return re.sub(r"-{2,}", "-", slug)[:60].strip("-")


def _normalize(d: dict) -> dict:
    """Fill defaults and coerce types so the dict fits TaskSpec.from_dict."""
    out = dict(d)
    if isinstance(out.get("taskspec"), dict) and "seed_examples" not in out:
        out = dict(out["taskspec"])
    out["version"] = out.get("version") or 1
    target = dict(DEFAULT_TARGET)
    if isinstance(out.get("target"), dict):
        target.update({k: v for k, v in out["target"].items() if k in DEFAULT_TARGET})
    out["target"] = target
    if out.get("task_name"):
        out["task_name"] = _slugify(str(out["task_name"]))
    for key in ("description", "input_format", "output_format", "eval_criteria"):
        if key in out and not isinstance(out[key], str):
            out[key] = _as_text(out[key])
    seeds = out.get("seed_examples")
    if isinstance(seeds, list):
        norm = []
        for e in seeds:
            if not isinstance(e, dict) or "input" not in e or "output" not in e:
                raise ValueError(f"each seed example needs 'input' and 'output' keys, got {e!r:.120}")
            norm.append({"input": _as_text(e["input"]), "output": _as_text(e["output"])})
        out["seed_examples"] = norm
    known = {"version", "task_name", "description", "input_format", "output_format",
             "seed_examples", "eval_criteria", "target"}
    return {k: v for k, v in out.items() if k in known}


def _validate(text: str) -> dict:
    d = _normalize(_parse_json_object(text))
    try:
        TaskSpec.from_dict(d)
    except KeyError as e:
        raise ValueError(f"missing required key {e}") from None
    except TypeError as e:
        raise ValueError(f"bad field shape: {e}") from None
    return d


# --------------------------------------------------------------------------- API


def draft_taskspec(
    description: str,
    n_seeds: int = 12,
    provider: str | None = None,
    model: str | None = None,
    timeout: float = 120,
) -> dict:
    """Draft a TaskSpec dict for ``description`` using an LLM.

    The result passes ``TaskSpec.from_dict``. Raises ``TaskgenError`` on missing
    keys, HTTP failures, or if the model's answer is invalid twice in a row.
    """
    if not description or not description.strip():
        raise TaskgenError("task description is empty")
    if not 3 <= n_seeds <= 50:
        raise TaskgenError("n_seeds must be between 3 and 50")
    provider, key = _resolve_provider(provider)
    model = model or os.environ.get("LOBBOT_TASKGEN_MODEL") or DEFAULT_MODELS[provider]
    call = _CALLERS[provider]

    messages = [{"role": "user", "content": _build_prompt(description, n_seeds)}]
    last_err: Exception | None = None
    for attempt in range(2):
        text = call(messages, SYSTEM_PROMPT, model, key, timeout)
        try:
            return _validate(text)
        except ValueError as e:
            last_err = e
            messages = messages + [
                {"role": "assistant", "content": text},
                {"role": "user", "content": (
                    f"That answer was rejected: {e}. Fix it and reply with the complete, corrected "
                    f"TaskSpec JSON object only ({n_seeds} seed examples, string inputs and outputs)."
                )},
            ]
    raise TaskgenError(f"{provider} model {model} did not produce a valid TaskSpec after 2 attempts: {last_err}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m lobbot.taskgen",
                                description="Draft a LobBot TaskSpec from a plain-English task description.")
    p.add_argument("description", help="what the small model should do, in plain English")
    p.add_argument("-o", "--out", help="write the TaskSpec JSON here (default: stdout)")
    p.add_argument("--seeds", type=int, default=12, help="number of seed examples (3-50, default 12)")
    p.add_argument("--provider", choices=PROVIDERS, help="LLM provider (default: auto from env keys)")
    p.add_argument("--model", help="model id (default: LOBBOT_TASKGEN_MODEL or provider default)")
    args = p.parse_args(argv)

    try:
        spec = draft_taskspec(args.description, n_seeds=args.seeds, provider=args.provider, model=args.model)
    except TaskgenError as e:
        print(f"taskgen: error: {e}", file=sys.stderr)
        return 1

    text = json.dumps(spec, indent=2, ensure_ascii=False) + "\n"
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    first = spec["seed_examples"][0]
    print(f"task_name: {spec['task_name']}", file=sys.stderr)
    print(f"seeds:     {len(spec['seed_examples'])}", file=sys.stderr)
    print(f"first:     {first['input'][:120]!r} -> {first['output'][:120]!r}", file=sys.stderr)
    if args.out:
        print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
