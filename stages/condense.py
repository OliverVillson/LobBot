"""condense.chat in front of the Anthropic API (and Gemini, see stages/gemini.py).

condense.chat is a proxy that compresses the aged part of a conversation
before it reaches the model provider, so long multi-turn sessions bill fewer
input tokens. The request is otherwise unchanged: the caller's own
ANTHROPIC_API_KEY still pays Anthropic, and condense adds one header.

Set CONDENSE_API_KEY (an ak_... key from condense.chat) and every Claude call
made through anthropic_client() goes through https://api.condense.chat/anthropic.
Without the key the client talks to Anthropic directly, so nothing breaks.
Optional: CONDENSE_USER_ID, CONDENSE_SESSION_ID, CONDENSE_BASE_URL.

Check the wiring with one real request:
  python -m stages.condense            # Claude through condense
  python -m stages.condense --gemini   # Gemini: direct, then each condense route
Gemini calls (the judge and the held-out test writer) go through stages/gemini.py.
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


def check_claude(model: str) -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is not set; condense forwards the call to Anthropic with it")
    client = anthropic_client(max_retries=2)
    raw = client.messages.with_raw_response.create(
        model=model, max_tokens=20,
        messages=[{"role": "user", "content": "Reply with the single word: condensed"}])
    msg = raw.parse()
    print(f"via {client.base_url}")
    print(f"reply: {msg.content[0].text.strip()!r}")
    print(f"usage: {msg.usage.input_tokens} in, {msg.usage.output_tokens} out")
    for k, v in raw.headers.items():
        if k.lower().startswith("x-condense"):
            print(f"{k}: {v}")


def check_gemini(model: str) -> None:
    """Try Gemini directly, then each condense route; print which works."""
    import httpx

    from stages import gemini

    if not gemini.api_key():
        raise SystemExit("GEMINI_API_KEY is not set")
    body = {"model": model, "max_tokens": 256, "reasoning_effort": "low",
            "messages": [{"role": "user", "content": "Reply with the single word: condensed"}]}
    routes = [("direct", gemini.DIRECT_URL, {})]
    custom = os.environ.get("CONDENSE_GEMINI_URL")
    routes += [("condense", u, headers()) for u in ([custom] if custom else gemini.CONDENSE_GEMINI_ROUTES)]
    working = None
    for name, url, extra in routes:
        try:
            r = httpx.post(url, timeout=60, json=body,
                           headers={"Authorization": f"Bearer {gemini.api_key()}", **extra})
            cond = {k: v for k, v in r.headers.items() if k.lower().startswith("x-condense")}
            if r.status_code == 200:
                text = (r.json()["choices"][0]["message"].get("content") or "").strip()
                print(f"OK    {name:8} {url}\n      reply {text!r} {cond or ''}")
                if name == "condense" and working is None:
                    working = url
            else:
                print(f"FAIL  {name:8} {url}\n      HTTP {r.status_code}: {r.text[:300]}")
        except Exception as e:
            print(f"FAIL  {name:8} {url}\n      {e}")
    if working:
        print(f"\ncondense fronts Gemini at {working}" +
              ("" if working == gemini.CONDENSE_GEMINI_ROUTES[0] else f"\nset CONDENSE_GEMINI_URL={working}"))
    else:
        print("\nno condense route reached Gemini; the judge and testgen call Gemini directly")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Send one real request through condense.chat and print the result.")
    ap.add_argument("--gemini", action="store_true", help="check the Gemini routes instead of Claude")
    ap.add_argument("--model", help="default claude-haiku-4-5-20251001, or gemini-3.8-flash with --gemini")
    args = ap.parse_args()
    if not enabled():
        raise SystemExit("CONDENSE_API_KEY is not set; nothing to check")
    if args.gemini:
        check_gemini(args.model or "gemini-3.8-flash")
    else:
        check_claude(args.model or "claude-haiku-4-5-20251001")


if __name__ == "__main__":
    main()
