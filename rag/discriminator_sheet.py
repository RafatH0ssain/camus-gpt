#!/usr/bin/env python3
"""
discriminator_sheet.py — builds the blind "him or the machine" test from
docs/HUMAN_EVAL_PROTOCOL.md and scores your answers.

Inputs are two jsonl files: model answers (field `answer`, e.g. eval/probe_scores.jsonl) and
reference passages (field `response`). N of each are sampled (those within the word limits),
shuffled together and written as a numbered, unlabelled markdown sheet. The source of each
number goes to a separate key file, which you must not open until you have answered.

  python rag/discriminator_sheet.py --model eval/probe_scores.jsonl --reference refs.jsonl \\
      --n 10 --seed 1 --out sheet.md --key key.json
  python rag/discriminator_sheet.py --score key.json answers.txt

answers.txt: one line per passage, "<number> <him|machine>" (also h/m, or 1/0 for him/machine).
Passages are reproduced verbatim; the only edit is dropping a leading "Camus:" / "Albert Camus:"
speaker label, which would give the source away.
"""
import argparse, json, random, re, sys

_LABEL = re.compile(r"^\s*(?:albert\s+)?camus\s*:\s*", re.I)


def clean(text):
    return _LABEL.sub("", text.strip(), count=1)


def nwords(text):
    return len(text.split())


def load(path, field):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            v = json.loads(line).get(field)
            if isinstance(v, str) and v.strip():
                out.append(clean(v))
    return out


def eligible(texts, lo, hi):
    seen, out = set(), []
    for t in texts:
        if lo <= nwords(t) <= hi and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def build(model, reference, n, seed, lo, hi):
    """Return (passages, key): passages is a shuffled list of texts, key[i] is 'him'/'machine'."""
    rng = random.Random(seed)
    m, r = eligible(model, lo, hi), eligible(reference, lo, hi)
    if len(m) < n or len(r) < n:
        raise SystemExit(f"need {n} of each within {lo}-{hi} words; "
                         f"found {len(m)} model, {len(r)} reference")
    items = [(t, "machine") for t in rng.sample(m, n)] + [(t, "him") for t in rng.sample(r, n)]
    rng.shuffle(items)
    return [t for t, _ in items], [s for _, s in items]


def render(passages):
    lines = ["# Him or the machine", "",
             "Each passage is either Camus or the model. Write one line per number in "
             "`answers.txt`: `<number> him` or `<number> machine`.", ""]
    for i, t in enumerate(passages, 1):
        lines += [f"## {i}", "", t, ""]
    return "\n".join(lines)


_ALIASES = {"him": "him", "h": "him", "1": "him", "machine": "machine", "m": "machine", "0": "machine"}


def parse_answers(text):
    ans = {}
    for line in text.splitlines():
        parts = line.replace(":", " ").replace(",", " ").split()
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].lower() in _ALIASES:
            ans[int(parts[0])] = _ALIASES[parts[1].lower()]
    return ans


def score(key, answers):
    """Return (correct, answered, total, per-source dict)."""
    per = {"him": [0, 0], "machine": [0, 0]}
    for num, guess in answers.items():
        if not 1 <= num <= len(key):
            continue
        truth = key[num - 1]
        per[truth][1] += 1
        if guess == truth:
            per[truth][0] += 1
    correct = per["him"][0] + per["machine"][0]
    answered = per["him"][1] + per["machine"][1]
    return correct, answered, len(key), per


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="jsonl of model answers (field: answer)")
    ap.add_argument("--reference", help="jsonl of reference passages (field: response)")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-words", type=int, default=15)
    ap.add_argument("--max-words", type=int, default=120)
    ap.add_argument("--out", default="discriminator_sheet.md")
    ap.add_argument("--key", default="key.json")
    ap.add_argument("--score", nargs=2, metavar=("KEY", "ANSWERS"), help="score an answers file")
    a = ap.parse_args(argv)

    if a.score:
        with open(a.score[0], encoding="utf-8") as f:
            key = json.load(f)["key"]
        with open(a.score[1], encoding="utf-8") as f:
            answers = parse_answers(f.read())
        correct, answered, total, per = score(key, answers)
        if not answered:
            raise SystemExit("no valid answers found")
        print(f"accuracy: {correct}/{answered} = {100 * correct / answered:.0f}%"
              + (f"  ({total - answered} unanswered)" if answered < total else ""))
        print(f"  him identified: {per['him'][0]}/{per['him'][1]}   "
              f"machine identified: {per['machine'][0]}/{per['machine'][1]}")
        return

    if not (a.model and a.reference):
        ap.error("--model and --reference are required (or use --score)")
    passages, key = build(load(a.model, "answer"), load(a.reference, "response"),
                          a.n, a.seed, a.min_words, a.max_words)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(render(passages))
    with open(a.key, "w", encoding="utf-8") as f:
        json.dump({"seed": a.seed, "key": key}, f, indent=1)
    print(f"wrote {a.out} ({len(passages)} passages) and {a.key} — do not open the key yet")


if __name__ == "__main__":
    main()
