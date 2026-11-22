#!/usr/bin/env python3
"""fact_check.py — one checker call over a drafted reply, using a free OpenCode model."""

try:
    import judge_opencode as jz
except ImportError:  # pragma: no cover - supports `import rag.fact_check`
    from rag import judge_opencode as jz  # type: ignore

judge_opencode = jz

CHECK_MODELS = ("beam-free", "hy4-flash-free")

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
    if judge is None:
        judge = jz.judge
    prompt = "CONTEXT:\n{}\n\nUSER:\n{}\n\nREPLY:\n{}".format(context, user_msg, draft)
    last_error = None
    for model in CHECK_MODELS:
        try:
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
