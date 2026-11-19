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
TRUST_THRESHOLD = 0.70


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


def write_sheet(pairs: list[dict], sheet_path: str, key_path: str) -> None:
    """Write the owner-facing markdown sheet and its number->pid key.

    One of each pair's two orders is shown, chosen at random (fixed seed) so the AI's reply
    is not always on the same side, renumbered 1..k. Each entry shows both sides' message
    and reply; no source is printed.
    """
    rng = random.Random(0)
    by_row = {}
    for p in pairs or []:
        by_row.setdefault(p["pid"].rsplit(":", 1)[0], []).append(p)
    ordered = [rng.choice(both) for both in by_row.values()]
    lines = ["# Owner calibration sheet", ""]
    key = {}
    for number, pair in enumerate(ordered, 1):
        key[number] = pair["pid"]
        lines.append(f"## {number}")
        lines.append(f"A MESSAGE: {pair['prompt_a']}")
        lines.append(f"A REPLY: {pair['a']}")
        lines.append(f"B MESSAGE: {pair['prompt_b']}")
        lines.append(f"B REPLY: {pair['b']}")
        lines.append("")
    with open(sheet_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")
    with open(key_path, "w", encoding="utf-8") as f:
        json.dump(key, f, ensure_ascii=False)
        f.write("\n")


def parse_owner(text: str, expected: int) -> dict[int, str]:
    """Parse owner answers of the form '<number> <a|b|A|B>'.

    Blank lines are ignored. Raises ValueError listing every missing number
    (1..expected) and every duplicated number.
    """
    answers = {}
    duplicates = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 2:
            raise ValueError(f"bad owner line: {line!r}")
        number = int(parts[0])
        choice = parts[1].lower()
        if choice not in ("a", "b"):
            raise ValueError(f"bad owner answer: {line!r}")
        if number in answers:
            duplicates.append(number)
        answers[number] = choice
    missing = [i for i in range(1, expected + 1) if i not in answers]
    if missing or duplicates:
        problems = []
        if missing:
            problems.append(f"missing {missing}")
        if duplicates:
            problems.append(f"duplicate {sorted(set(duplicates))}")
        raise ValueError(" ".join(problems))
    return answers


def agreement(key: dict, owner: dict[int, str], results: list[dict]) -> dict:
    """Score the jury against the owner's blind picks, per model.

    Returns each model's share of numbered pairs where its guess matches the
    owner, plus owner_correct (owner vs the machine truth) and trusted (mean
    model agreement >= TRUST_THRESHOLD).
    """
    key_n = {str(k): v for k, v in (key or {}).items()}
    owner_n = {str(k): str(v).lower() for k, v in (owner or {}).items()}
    numbers = sorted(key_n, key=int)

    truth = {}
    guesses = {}
    for row in results or []:
        pid = row.get("pid")
        truth[pid] = str(row.get("machine")).lower()
        guesses.setdefault(row.get("model"), {})[pid] = str(row.get("guess")).lower()

    by_model = {}
    for model, by_pid in guesses.items():
        correct = 0
        total = 0
        for number in numbers:
            guess = by_pid.get(key_n[number])
            choice = owner_n.get(number)
            if guess is None or choice is None:
                continue
            total += 1
            if guess == choice:
                correct += 1
        by_model[model] = correct / total if total else 0.0

    correct = 0
    total = 0
    for number in numbers:
        choice = owner_n.get(number)
        actual = truth.get(key_n[number])
        if choice is None or actual is None:
            continue
        total += 1
        if choice == actual:
            correct += 1
    owner_correct = correct / total if total else 0.0

    rates = list(by_model.values())
    mean = sum(rates) / len(rates) if rates else 0.0
    out = dict(by_model)
    out["owner_correct"] = owner_correct
    out["trusted"] = mean >= TRUST_THRESHOLD
    return out


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
    jury.add_argument("--models", nargs="+", default=list(JURY_MODELS))
    sheet = sub.add_parser("sheet")
    sheet.add_argument("--model-file", required=True)
    sheet.add_argument("--refs", required=True)
    sheet.add_argument("--n", type=int, default=40)
    sheet.add_argument("--seed", type=int, default=1)
    sheet.add_argument("--out", required=True)
    sheet.add_argument("--models", nargs="+", default=list(JURY_MODELS))
    cal = sub.add_parser("calibrate")
    cal.add_argument("--dir", required=True)
    cal.add_argument("--answers", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "jury":
        jz.load_key()
        model_rows = tells._load_rows(args.model_file)
        refs = load_refs(args.refs)
        pairs = make_pairs(model_rows, refs, n=args.n, seed=args.seed)
        results = run_jury(pairs, models=tuple(args.models))
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "jury.jsonl"), "w", encoding="utf-8") as f:
            for row in results:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        fr = fool_rate(results)
        with open(os.path.join(args.out, "fool_rate.json"), "w", encoding="utf-8") as f:
            json.dump(fr, f, ensure_ascii=False)
            f.write("\n")
        print(json.dumps(fr, ensure_ascii=False))
    elif args.cmd == "sheet":
        jz.load_key()
        model_rows = tells._load_rows(args.model_file)
        refs = load_refs(args.refs)
        pairs = make_pairs(model_rows, refs, n=args.n, seed=args.seed)
        os.makedirs(args.out, exist_ok=True)
        write_sheet(pairs, os.path.join(args.out, "sheet.md"),
                    os.path.join(args.out, "key.json"))
        results = run_jury(pairs, models=tuple(args.models))
        with open(os.path.join(args.out, "jury.jsonl"), "w", encoding="utf-8") as f:
            for row in results:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    elif args.cmd == "calibrate":
        with open(os.path.join(args.dir, "key.json"), encoding="utf-8") as f:
            key = json.load(f)
        results = []
        with open(os.path.join(args.dir, "jury.jsonl"), encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(json.loads(line))
        with open(args.answers, encoding="utf-8") as f:
            owner = parse_owner(f.read(), expected=len(key))
        print(json.dumps(agreement(key, owner, results), ensure_ascii=False))


if __name__ == "__main__":
    main()
