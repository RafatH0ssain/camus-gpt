"""Spot-the-bot reference rows and blind pair construction for camus10.

Reference corpora live in data/reference/ (gitignored); only the pairing code
is committed.
"""
import argparse
import json
import os
import random

import tells
import judge_opencode as jz

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


JURY_MODELS = ("beam-free", "hy4-flash-free", "longcat-2.5-free")

JURY_SYS = """You will see two short exchanges, A and B. In each, someone sent a message and got a reply.
One of the two replies was written by a person; the other was written by an AI imitating a
person. The two exchanges are unrelated. Decide which reply is the AI's. Judge only how human
the reply sounds as speech or chat, not whether it is wise or correct.
Answer with one JSON object and nothing else: {"machine": "A"} or {"machine": "B"}.
"""


def ask(model: str, pair: dict, judge=jz.judge) -> str:
    prompt = ("A\nMESSAGE: {prompt_a}\nREPLY: {a}\n\n"
              "B\nMESSAGE: {prompt_b}\nREPLY: {b}").format(**pair)
    try:
        text = judge(model, JURY_SYS, prompt)
    except jz.UsageCapReached:
        raise
    except Exception:
        return "abstain"
    try:
        obj = jz.first_json_object(text)
    except ValueError:
        return "abstain"
    if not isinstance(obj, dict):
        return "abstain"
    machine = obj.get("machine")
    if not isinstance(machine, str):
        return "abstain"
    m = machine.strip().lower()
    if m in ("a", "b"):
        return m
    return "abstain"


def run_jury(pairs: list[dict], models=JURY_MODELS, judge=jz.judge) -> list[dict]:
    results = []
    for pair in pairs:
        for model in models:
            guess = ask(model, pair, judge=judge)
            results.append({"pid": pair["pid"], "model": model,
                            "guess": guess, "machine": pair["machine"],
                            "register": pair["register"]})
    return results


def fool_rate(results: list[dict]) -> dict:
    results = results or []
    abstains = sum(1 for r in results if r.get("guess") == "abstain")
    non_abstain = [r for r in results if r.get("guess") != "abstain"]

    def _rate(rows):
        if not rows:
            return 0.0
        fooled = sum(1 for r in rows if r.get("guess") != r.get("machine"))
        return fooled / len(rows)

    by_model = {}
    for m in sorted({r.get("model") for r in results}):
        rows = [r for r in non_abstain if r.get("model") == m]
        by_model[m] = _rate(rows)
    by_register = {}
    for rreg in sorted({r.get("register") for r in results}):
        rows = [r for r in non_abstain if r.get("register") == rreg]
        by_register[rreg] = _rate(rows)
    overall = _rate(non_abstain)
    return {"overall": overall, "by_model": by_model,
            "by_register": by_register, "abstains": abstains}


def main(argv=None):
    import sys
    argv = argv if argv is not None else sys.argv[1:]
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    jury = sub.add_parser("jury")
    jury.add_argument("--model-file", required=True)
    jury.add_argument("--refs", required=True)
    jury.add_argument("--n", type=int, default=40)
    jury.add_argument("--seed", type=int, default=1)
    jury.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "jury":
        jz.load_key()
        model_rows = tells._load_rows(args.model_file)
        refs = load_refs(args.refs)
        pairs = make_pairs(model_rows, refs, n=args.n, seed=args.seed)
        results = run_jury(pairs)
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "jury.jsonl"), "w", encoding="utf-8") as f:
            for row in results:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        fr = fool_rate(results)
        with open(os.path.join(args.out, "fool_rate.json"), "w", encoding="utf-8") as f:
            json.dump(fr, f, ensure_ascii=False)
            f.write("\n")
        print(json.dumps(fr, ensure_ascii=False))


if __name__ == "__main__":
    main()
