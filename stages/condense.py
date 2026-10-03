"""condense.chat: its compression API, and its proxy in front of the Anthropic API.

condense.chat is a proxy that compresses the aged part of a conversation
before it reaches the model provider, so long multi-turn sessions bill fewer
input tokens. The request is otherwise unchanged: the caller's own
ANTHROPIC_API_KEY still pays Anthropic, and condense adds one header.

Set CONDENSE_API_KEY (an ak_... key from condense.chat) and every Claude call
made through anthropic_client() goes through https://api.condense.chat/anthropic.
Without the key the client talks to Anthropic directly, so nothing breaks.
Optional: CONDENSE_USER_ID, CONDENSE_SESSION_ID, CONDENSE_BASE_URL.

condense also has a compression API (compress() below): it shortens text
before we send it to any provider. Gemini's test writer uses it on the
examples it is shown (stages/testgen.py). condense's proxy only forwards
Gemini to Vertex AI (Google Cloud OAuth), not to AI Studio keys, so Gemini
calls themselves go direct unless CONDENSE_GEMINI_URL is set.

Check the wiring with one real request:
  python -m stages.condense --compress # compression API on a sample email
  python -m stages.condense            # Claude through condense
  python -m stages.condense --gemini   # Gemini: direct, then each condense route
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


COMPRESS_URL = "https://api.condense.chat/v1/compress"
COMPRESS_MODELS = ("helene-1.1", "adeline-1")  # extractive, fast / abstractive, deeper


def compress(texts: list[str], model: str | None = None) -> list[str]:
    """Compress each text with condense.chat's compression model (one request).
    Returns the originals unchanged if condense is off or anything goes wrong,
    so callers can always use the result. Model: CONDENSE_MODEL or helene-1.1."""
    if not enabled() or not texts:
        return texts
    import httpx

    model = model or os.environ.get("CONDENSE_MODEL", COMPRESS_MODELS[0])
    try:
        r = httpx.post(os.environ.get("CONDENSE_COMPRESS_URL", COMPRESS_URL), timeout=60, headers=headers(),
                       json={"model": model, "messages": [{"role": "user", "content": t} for t in texts]})
        r.raise_for_status()
        out = [m.get("content") or "" for m in r.json().get("messages", [])]
    except Exception as e:
        print(f"[condense] compression failed ({e}); using the uncompressed text", flush=True)
        return texts
    if len(out) != len(texts) or not all(out):
        print(f"[condense] compression returned {len(out)} of {len(texts)} texts; using the uncompressed text", flush=True)
        return texts
    before, after = sum(len(t.split()) for t in texts), sum(len(t.split()) for t in out)
    print(f"[condense] {model}: {before} -> {after} words ({1 - after / max(before, 1):.0%} smaller)", flush=True)
    return out


def check_compress(model: str) -> None:
    text = ("Hi team, our invoice #9921 was charged in USD although the account is set to EUR. This is the "
            "third time this happens. We already contacted support twice last month and got no answer. "
            "Please fix the invoice and confirm by email. Thanks, Anna")
    out = compress([text], model)[0]
    print(f"before: {text}\nafter:  {out}")
    if out == text:
        raise SystemExit("compression did not run (see the line above)")


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
    ap.add_argument("--gemini", action="store_true", help="check the Gemini proxy routes instead of Claude")
    ap.add_argument("--compress", action="store_true", help="check the compression API on a sample email")
    ap.add_argument("--model", help="default claude-haiku-4-5-20251001, or gemini-3.8-flash with --gemini")
    args = ap.parse_args()
    if not enabled():
        raise SystemExit("CONDENSE_API_KEY is not set; nothing to check")
    if args.compress:
        check_compress(args.model or COMPRESS_MODELS[0])
    elif args.gemini:
        check_gemini(args.model or "gemini-3.8-flash")
    else:
        check_claude(args.model or "claude-haiku-4-5-20251001")


if __name__ == "__main__":
    main()
