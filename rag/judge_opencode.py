#!/usr/bin/env python3
"""
judge_opencode.py — the OpenCode Zen judge backend, plus the judge plumbing both eval
harnesses share.

eval_camus.py and eval_memory.py differ in their judge prompt and their score schema, not in
how they reach the gateway, so the transport lives here once: key resolution, the four headers
Zen rejects requests without, the usage-cap abort, and reply unwrapping. The harnesses keep
their own JUDGE_SYS / judge_prompt / parse_judge.

    import judge_opencode as jz
    scores = parse_judge(jz.judge(model, JUDGE_SYS, judge_prompt(probe, convo, answer)))

Key: $OPENCODE_API_KEY, else ["opencode-go"]["key"] in ~/.local/share/opencode/auth.json.
It is never printed, logged or written, and every message produced here is scrubbed of it.

Failures split in two, deliberately: HTTP 429 raises UsageCapReached (a SystemExit, so a
harness's `except Exception` cannot turn a dead run into a row full of judge errors), while
everything else — other HTTP errors, timeouts, unparseable replies — raises normally so the
harness records it as a judge error and keeps going. 429 is never retried; a run that hit the
cap must stop, not spend a second call proving it.

The default model is a reasoning model, so it can spend the whole token budget thinking and
answer with nothing — or run out mid-object. judge() therefore sends the identical request
once more when the first reply is empty or carries no complete JSON object, and only then
raises what the first attempt's failure would have raised.
"""
import argparse, json, os, re, secrets

import requests

ENDPOINT      = "https://opencode.ai/zen/go/v1/chat/completions"
DEFAULT_MODEL = "space-bunny-free"
USER_AGENT    = "camus-gpt-eval/1.0"
TIMEOUT       = 120
MAX_TOKENS    = 4000
ATTEMPTS      = 2       # the first call, plus exactly one retry of the same request
AUTH_PATH     = os.path.expanduser("~/.local/share/opencode/auth.json")

# One session id per process, so the gateway can group a whole run's judge calls.
SESSION_ID = "ses_" + secrets.token_hex(16)

DEFAULT_JUDGE       = "opencode"
JUDGE_CHOICES       = (DEFAULT_JUDGE, "anthropic", "ollama", "none")
DEFAULT_JUDGE_MODEL = {DEFAULT_JUDGE: DEFAULT_MODEL, "anthropic": "claude-sonnet-4-6"}


class UsageCapReached(SystemExit):
    """HTTP 429. Subclasses SystemExit, not Exception, so it stops the run."""


def load_key(auth_path=AUTH_PATH):
    """The API key: $OPENCODE_API_KEY first, else the opencode-go entry in auth.json.

    An unset-but-empty OPENCODE_API_KEY counts as unset rather than as a usable key."""
    key = (os.environ.get("OPENCODE_API_KEY") or "").strip()
    if key:
        return key
    try:
        with open(auth_path, encoding="utf-8") as fh:
            entry = (json.load(fh) or {}).get("opencode-go")
    except FileNotFoundError:
        raise SystemExit(
            f"no OpenCode key: set OPENCODE_API_KEY, or log in (`opencode auth login`) so "
            f"{auth_path} exists — or run with --judge anthropic / --judge ollama / --judge none"
        ) from None
    except (OSError, ValueError) as e:
        raise SystemExit(f"could not read an OpenCode key from {auth_path}: {e}") from None
    key = (entry or {}).get("key") if isinstance(entry, dict) else None
    if not (key or "").strip():
        raise SystemExit(
            f"no OpenCode key in {auth_path} (expected [\"opencode-go\"][\"key\"]) — set "
            f"OPENCODE_API_KEY, or run with --judge anthropic / --judge ollama / --judge none"
        )
    return key.strip()


def headers(key):
    """Zen rejects the request without any of these four."""
    return {"Authorization": f"Bearer {key}",
            "x-opencode-session": SESSION_ID,
            "User-Agent": USER_AGENT,
            "Content-Type": "application/json"}


def judge(model, system, prompt, key=None, auth_path=AUTH_PATH):
    """Return the judge's reply text, once it is known to carry a complete JSON object.

    A reply that arrives empty, null, truncated mid-object, or not chat/completions-shaped at
    all gets the identical request sent once more — the model may just have run out of budget.
    If the second attempt is no better, the failure is raised exactly as before (a ValueError
    from first_json_object, or the shape RuntimeError), so the harness still records a judge
    error and keeps going.

    Raises UsageCapReached on 429 — raised from the first attempt, never retried — and raises
    normally otherwise.
    """
    key = key if key is not None else load_key(auth_path)
    payload = {"model": model,
               "messages": [{"role": "system", "content": system},
                            {"role": "user", "content": prompt}],
               "max_tokens": MAX_TOKENS, "temperature": 0}
    sent = headers(key)
    for attempt in range(ATTEMPTS):
        last = attempt == ATTEMPTS - 1
        try:
            r = requests.post(ENDPOINT, headers=sent, json=payload, timeout=TIMEOUT)
        except requests.RequestException as e:
            raise RuntimeError(_scrub(f"opencode judge request failed: {e}", key)) from None
        if r.status_code == 429:
            raise UsageCapReached(_scrub(
                f"opencode judge hit the usage cap (HTTP 429) on model {model} — stopping the run. "
                f"Everything past this point would be unscored, so nothing is recorded: top up or "
                f"wait out the cap, then re-run. ({_body(r)})", key))
        if r.status_code >= 400:
            raise RuntimeError(_scrub(f"opencode judge HTTP {r.status_code}: {_body(r)}", key))
        content = _content(r)
        if content is None:  # no choices[0].message: nothing here this call could judge
            if last:
                raise RuntimeError(_scrub("opencode judge reply was not chat/completions-shaped: "
                                         f"{_body(r)}", key)) from None
            continue
        try:
            first_json_object(content)  # exactly what the harness's parse_judge goes on to do
            return content
        except ValueError:
            if last:
                raise
    raise RuntimeError(_scrub(f"opencode judge reply unusable after {ATTEMPTS} attempts", key))


def _content(r):
    """choices[0].message.content, or None when the body is not chat/completions-shaped.

    A null or missing content is empty content, not a shape error: a reasoning model can
    spend every token on reasoning_content and hand back an empty answer."""
    try:
        message = r.json()["choices"][0]["message"]
    except (ValueError, KeyError, IndexError, TypeError):
        return None
    content = message.get("content") if isinstance(message, dict) else None
    return content if isinstance(content, str) else ""


def add_judge_args(ap):
    """The --judge / --judge-model flags, identical in both harnesses so they cannot drift."""
    ap.add_argument("--judge", choices=list(JUDGE_CHOICES), default=DEFAULT_JUDGE,
                    help=f"judge backend (default: {DEFAULT_JUDGE})")
    ap.add_argument("--judge-model", default=None,
                    help=f"override the judge model (opencode default: {DEFAULT_MODEL}; "
                         f"anthropic default: claude-sonnet-4-6; ollama requires one)")
    return ap


def resolve_model(judge_name, override):
    """--judge-model wins; otherwise the backend's default. Ollama has none of its own."""
    if override:
        return override
    if judge_name == "none":
        return None
    if judge_name == "ollama":
        raise SystemExit("--judge ollama requires --judge-model (a DIFFERENT model than camus)")
    return DEFAULT_JUDGE_MODEL[judge_name]


def first_json_object(text):
    """The first {...} object in the reply: fences stripped, braces matched so nested objects
    and trailing prose both survive. Raises ValueError when there is no object."""
    t = _strip_fences(text)
    start = t.find("{")
    if start < 0:
        raise ValueError(f"no JSON in judge output: {t[:120]}")
    depth, in_str, esc = 0, False, False
    for i, ch in enumerate(t[start:], start):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(t[start:i + 1])
    raise ValueError(f"unterminated JSON in judge output: {t[:120]}")


_FENCE = re.compile(r"^[ \t]*```[A-Za-z0-9_+-]*[ \t]*$", re.M)


def _strip_fences(text):
    return _FENCE.sub("", str(text or "")).strip()


def _body(r, limit=200):
    try:
        return str(r.text or "")[:limit]
    except Exception:
        return ""


def _scrub(text, key):
    """Error text ends up in the harness's judge_error field; the key must not ride along."""
    return str(text).replace(key, "[redacted]") if key else str(text)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="check the OpenCode judge backend's wiring")
    ap.add_argument("--auth-path", default=AUTH_PATH, help="read auth.json from here instead")
    add_judge_args(ap)
    a = ap.parse_args()
    load_key(a.auth_path)
    print(f"endpoint   {ENDPOINT}\nmodel      {resolve_model(a.judge, a.judge_model)}\n"
          f"session    {SESSION_ID}\nkey        found (never printed)")