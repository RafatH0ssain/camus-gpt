"""Spot-the-bot reference rows and blind pair construction for camus10.

Reference corpora live in data/reference/ (gitignored); only the pairing code
is committed.
"""
import json
import random

import tells

REGISTERS = ("text", "call")
_REQUIRED = ("prompt", "reply", "register", "source")


def register_of(message: str) -> str:
    return "text" if len(str(message or "").split()) <= 4 else "call"


def load_refs(path: str) -> list[dict]:
    refs = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {lineno}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"line {lineno}: expected an object")
            for field in _REQUIRED:
                if field not in row:
                    raise ValueError(f"line {lineno}: missing field {field!r}")
            if row["register"] not in REGISTERS:
                raise ValueError(f"line {lineno}: unknown register {row['register']!r}")
            refs.append(row)
    return refs


def _has_answer(row) -> bool:
    if not isinstance(row, dict) or row.get("error"):
        return False
    answer = row.get("answer")
    return isinstance(answer, str) and answer.strip() != ""


def make_pairs(model_rows: list[dict], refs: list[dict], n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    by_register = {reg: [r for r in refs if r.get("register") == reg] for reg in REGISTERS}
    eligible = [row for row in (model_rows or []) if _has_answer(row)]
    chosen = rng.sample(eligible, min(n, len(eligible))) if n > 0 else []

    pairs = []
    for row in chosen:
        prompt = tells.last_user(row.get("turns", []))
        register = register_of(prompt)
        pool = by_register[register]
        if not pool:
            raise ValueError(f"no reference rows for register {register!r}")
        human = rng.choice(pool)
        row_id = row.get("id", "")
        answer = row["answer"]
        # Each side keeps its own message: the human reply answered someone else's message,
        # and showing it under the model's prompt would make the jury judge fit, not humanness.
        mine, theirs = (prompt, answer), (human["prompt"], human["reply"])
        for order, (a, b) in enumerate(((mine, theirs), (theirs, mine))):
            pairs.append({"pid": f"{row_id}:{order}", "register": register,
                          "prompt_a": a[0], "a": a[1], "prompt_b": b[0], "b": b[1],
                          "machine": "ab"[order]})
    return pairs
