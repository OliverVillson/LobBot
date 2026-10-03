"""condense.chat in front of the Anthropic API.

condense.chat is a proxy that compresses the aged part of a conversation
before it reaches the model provider, so long multi-turn sessions bill fewer
input tokens. The request is otherwise unchanged: the caller's own
ANTHROPIC_API_KEY still pays Anthropic, and condense adds one header.

Set CONDENSE_API_KEY (an ak_... key from condense.chat) and every Claude call
made through anthropic_client() goes through https://api.condense.chat/anthropic.
Without the key the client talks to Anthropic directly, so nothing breaks.
Optional: CONDENSE_USER_ID, CONDENSE_SESSION_ID, CONDENSE_BASE_URL.

Check the wiring with one real request:
  python -m stages.condense
"""

from __future__ import annotations

import os

BASE_URL = "https://api.condense.chat/anthropic"


def enabled() -> bool:
    return bool(os.environ.get("CONDENSE_API_KEY"))


def headers() -> dict[str, str]:
    """The condense headers for a request, or {} when condense is off."""
    if not enabled():
        return {}
    h = {"x-condense-auth-token": os.environ["CONDENSE_API_KEY"]}
    if os.environ.get("CONDENSE_USER_ID"):
        h["x-condense-user-id"] = os.environ["CONDENSE_USER_ID"]
    if os.environ.get("CONDENSE_SESSION_ID"):
        h["x-condense-session-id"] = os.environ["CONDENSE_SESSION_ID"]
    return h


def anthropic_client(**kw):
    """anthropic.Anthropic, routed through condense.chat when CONDENSE_API_KEY is set."""
    import anthropic

    if enabled():
        kw.setdefault("base_url", os.environ.get("CONDENSE_BASE_URL", BASE_URL))
        kw["default_headers"] = {**kw.get("default_headers", {}), **headers()}
    return anthropic.Anthropic(**kw)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Send one Claude request through condense.chat and print the result.")
    ap.add_argument("--model", default="claude-haiku-4-5-20251001")
    args = ap.parse_args()
    if not enabled():
        raise SystemExit("CONDENSE_API_KEY is not set; nothing to check")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is not set; condense forwards the call to Anthropic with it")

    client = anthropic_client(max_retries=2)
    raw = client.messages.with_raw_response.create(
        model=args.model, max_tokens=20,
        messages=[{"role": "user", "content": "Reply with the single word: condensed"}])
    msg = raw.parse()
    print(f"via {client.base_url}")
    print(f"reply: {msg.content[0].text.strip()!r}")
    print(f"usage: {msg.usage.input_tokens} in, {msg.usage.output_tokens} out")
    for k, v in raw.headers.items():
        if k.lower().startswith("x-condense"):
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
