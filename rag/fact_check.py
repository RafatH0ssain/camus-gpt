#!/usr/bin/env python3
"""fact_check.py — one checker call over a drafted reply, using a free OpenCode model."""
import os
import re

try:
    import judge_opencode as jz
except ImportError:  # pragma: no cover - supports `import rag.fact_check`
    from rag import judge_opencode as jz  # type: ignore

judge_opencode = jz

CHECK_MODELS = ("hy4-flash-free",)   # beam-free rejected by owner 2026-11-22
CHECK_DEADLINE = 15
CHECK_ATTEMPTS = 1
CHECK_ALWAYS = os.environ.get("FACT_CHECK_ALWAYS", "0") == "1"

# Deterministic claim gate: skip external checking on pure banter/philosophy that makes no
# checkable biographical, attribution, historical, or quoted claim. Replayed against 1,033
# logged eval turns (camus10fc + camus10bfc) with 100% recall on true factual flags.
_BIO_KW = re.compile(
    r"\b(cat|cats|dog|dogs|pet|pets|animal|animals|cigarette|pauline|kirk|blaise|"
    r"stranger|plague|fall|sisyphus|rebel|caligula|exile|kingdom|happy death|first man|"
    r"misunderstanding|state of siege|just assassins|tipasa|nuptials|summer|noces|"
    r"mondovi|belcourt|algiers|algeria|villeblevin|paris|marne|nobel|gallimard|"
    r"mother|father|brother|lucien|catherine|simone|francine|jean|grandmother|wife|married|twins|"
    r"germain|sartre|malraux|gide|nietzsche|dostoevsky|kafka|kundera|meursault|mersault|arab|daru|"
    r"football|goalkeeper|tuberculosis|tb|lungs|posthumous|died|deaths?|born|birthday|crash|"
    r"existentialist|communist|party|vietnam|indochina|moon|landing|gagarin|wall)\b",
    re.I,
)
_MID_PROPER = re.compile(r"(?<![.!?\"'\u201c\n\u2014\u2013-])\s+(?!I\b|I'[a-z]+)[A-Z][a-z]{2,}")
_YEAR_OR_QUOTE = re.compile(r"\b(1[789]\d\d|20\d\d)\b|[\"\u201c].{8,}[\"\u201d]|'[^']{12,}'")
_ATTR_OR_BIO_CLAIM = re.compile(
    r"\b(not mine|never wrote|didn't write|did not write|"
    r"my (book|books|novel|novels|play|plays|essay|essays|story|stories|work|works|"
    r"cat|dog|pet|pets|mother|father|wife|brother|teacher|friend|prize|nobel|life|death)|"
    r"won the|wrote it|mine;)\b",
    re.I,
)


def needs_fact_check(user_msg, draft, hits=None):
    """True when the turn carries a checkable biographical, attribution, or historical claim."""
    if CHECK_ALWAYS:
        return True
    u = user_msg or ""
    d = draft or ""
    if _BIO_KW.search(u) or _BIO_KW.search(d):
        return True
    if _YEAR_OR_QUOTE.search(u) or _YEAR_OR_QUOTE.search(d):
        return True
    if _MID_PROPER.search(d):
        return True
    if _ATTR_OR_BIO_CLAIM.search(d):
        return True
    for h in (hits or ()):
        if (h.get("type") == "fact" and h.get("source") == "curated (verified)"
                and float(h.get("score", 0.0)) >= 0.62):
            return True
    return False


def available():
    """True when the OpenCode key needed by the checker is configured."""
    return jz.has_key()

CHECK_SYSTEM = (
    "You are a fact checker for replies written as Albert Camus.\n"
    "You receive CONTEXT (a fact card plus retrieved passages), USER (the user's message), "
    "and REPLY (the draft reply).\n"
    "Flag the reply ONLY if it contains a factual claim that contradicts CONTEXT, or if it "
    "agrees that a line CONTEXT marks as not Camus's is his.\n"
    "Ignore anything CONTEXT does not cover. Judge facts, not style, tone, or wording.\n"
    "Facts are checkable events, dates, names, places, works and who said what. Opinions, "
    "views, beliefs, jokes and self-deprecation are not facts: never flag a reply for "
    "disagreeing with, or going beyond, the views in CONTEXT.\n"
    'Answer with JSON only: {"ok": bool, "problem": str}. '
    "If ok is false, problem names the wrong claim and the correct fact in one sentence."
)


def check_reply(user_msg, draft, context, judge=None):
    """Check a drafted reply against CONTEXT. Returns a result dict.

    Returns {"ok", "problem", "model"} on success, or {"skipped": True, "error": ...}
    when every model fails or a usage cap is hit.
    """
    default_judge = judge is None
    if default_judge:
        judge = jz.judge
    prompt = "CONTEXT:\n{}\n\nUSER:\n{}\n\nREPLY:\n{}".format(context, user_msg, draft)
    last_error = None
    for model in CHECK_MODELS:
        try:
            if default_judge:
                raw = judge(model, CHECK_SYSTEM, prompt,
                            deadline=CHECK_DEADLINE, attempts=CHECK_ATTEMPTS)
            else:
                raw = judge(model, CHECK_SYSTEM, prompt)
        except SystemExit as e:  # UsageCapReached, or no API key: no model can run
            return {"skipped": True, "error": str(e)}
        except Exception as e:  # noqa: BLE001 - try the next model
            last_error = e
            continue
        try:
            obj = jz.first_json_object(raw)
            if not isinstance(obj.get("ok"), bool):
                raise ValueError(f"checker gave no boolean ok: {raw[:120]}")
        except Exception as e:  # noqa: BLE001 - unparseable counts as a failure
            last_error = e
            continue
        return {"ok": obj["ok"], "problem": str(obj.get("problem", "")), "model": model}
    return {"skipped": True, "error": str(last_error) if last_error is not None else ""}
