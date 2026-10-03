"""Gemini over its OpenAI-compatible endpoint, optionally through condense.chat.

GEMINI_API_KEY (or GOOGLE_API_KEY) is required. When CONDENSE_API_KEY is also
set, calls go to condense.chat's provider passthrough first. If that route
fails for any reason, it is switched off for the rest of the process and the
call goes straight to Gemini, so condense can never block a stage.

The condense route is CONDENSE_GEMINI_URL, or the first of CONDENSE_GEMINI_ROUTES
that `python -m stages.condense --gemini` finds working.
"""

from __future__ import annotations

import os
import threading
import time

DIRECT_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
# condense.chat documents a passthrough, ANY /{provider}/..., beside its
# /anthropic and /openai dialects. These are the candidate provider names for Gemini.
CONDENSE_GEMINI_ROUTES = [
    "https://api.condense.chat/google/v1beta/openai/chat/completions",
    "https://api.condense.chat/gemini/v1beta/openai/chat/completions",
]

_lock = threading.Lock()
_condense_off = False  # set after the first condense failure
last_route = ""  # "condense" or "direct", for logs and the check command


def api_key() -> str | None:
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def condense_url() -> str | None:
    from stages import condense

    if _condense_off or not condense.enabled():
        return None
    return os.environ.get("CONDENSE_GEMINI_URL") or CONDENSE_GEMINI_ROUTES[0]


def _post(url: str, extra_headers: dict, body: dict, retries: int):
    import httpx

    headers = {"Authorization": f"Bearer {api_key()}", **extra_headers}
    for attempt in range(retries + 1):
        r = httpx.post(url, timeout=180, headers=headers, json=body)
        if r.status_code in (429, 500, 502, 503, 504) and attempt < retries:
            time.sleep(2 ** attempt)
            continue
        r.raise_for_status()
        return r.json()["choices"][0]["message"].get("content") or ""
    return ""


def chat(model: str, messages: list[dict], *, temperature: float = 0.0, max_tokens: int = 1024,
         reasoning_effort: str = "low", retries: int = 5) -> str:
    """One chat completion; returns the reply text."""
    global _condense_off, last_route
    from stages import condense

    body = {"model": model, "messages": messages, "temperature": temperature,
            "max_tokens": max_tokens, "reasoning_effort": reasoning_effort}
    url = condense_url()
    if url:
        try:
            text = _post(url, condense.headers(), body, retries=1)
            last_route = "condense"
            return text
        except Exception as e:
            with _lock:
                if not _condense_off:
                    _condense_off = True
                    print(f"[gemini] condense route {url} failed ({e}); calling Gemini directly from now on", flush=True)
    text = _post(DIRECT_URL, {}, body, retries)
    last_route = "direct"
    return text
