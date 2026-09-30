#!/usr/bin/env python3
"""
naturalize_corpus.py — strip synthetic conversational tells from the SFT corpus.

Three passes, in this order, all seeded so the corpus is byte-reproducible:

  T1  template openers    "Yes, and " / "No, and " / "Rarely, and " -> sentence break
  T2  "I think" openers   capped at 3% of the conversational rows of camus_sft.jsonl
  T3  trailing questions  capped at 15% per file, by dropping the last sentence

T2 only ever strips the bare "I think ". Rows opening "I think that " are left
alone: there the "that" is the subject of the sentence ("I think that is
precisely the condition"), so removing it would leave "Is precisely ...".

Essayist rows in camus_sft.jsonl are real prose and are never edited. Neither is
any row the transforms leave alone: unchanged lines are copied back as their
original bytes, and only rewritten rows are re-serialized with json.dumps. Row
count, row order and key order are therefore preserved exactly.

    python pipeline/naturalize_corpus.py --dry-run
    python pipeline/naturalize_corpus.py --seed 13
"""
import argparse
import json
import random
import re
import statistics
from collections import Counter
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

FILES = (
    ("camus_sft.jsonl", "conversational"),
    ("camus_conversational.jsonl", None),
    ("camus_refusals.jsonl", None),
    ("camus_epistemic.jsonl", None),
    ("camus_analysis.jsonl", None),
    ("camus_phase3.jsonl", None),
)

I_THINK = "I think"
I_THINK_TARGET = 0.03
QUESTION_TARGET = 0.15

T1_PREFIXES = (
    ("Yes, and ", ""),
    ("No, and ", "No. "),
    ("Rarely, and ", "Rarely. "),
)
T2_PREFIX = "I think "
T2_KEEP_PREFIX = "I think that "

SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+")
T3_TAIL_CHARS = frozenset(".!\u2026\"\u201d'\u2014")
OPENER_LEAD = "\"'\u201c\u2018([{"
OPENER_TRAIL = ",.;:!?\u2014-"


def upper_first(text):
    return text[:1].upper() + text[1:]


def t1_apply(text):
    for prefix, replacement in T1_PREFIXES:
        if text.startswith(prefix):
            return replacement + upper_first(text[len(prefix):])
    return None


def t2_is_candidate(text):
    if not text.startswith(T2_PREFIX):
        return False
    if text.startswith(T2_KEEP_PREFIX):
        return False
    return text[len(I_THINK):len(I_THINK) + 1] != ","


def t2_apply(text):
    if not t2_is_candidate(text):
        return None
    return upper_first(text[len(T2_PREFIX):])


def ends_with_question(text):
    return text.rstrip().endswith("?")


def t3_apply(text):
    if not ends_with_question(text):
        return None
    sentences = SENTENCE_SPLIT.split(text.strip())
    if len(sentences) < 2:
        return None
    remainder = " ".join(sentences[:-1])
    if len(remainder.split()) < 4:
        return None
    if remainder[-1:] not in T3_TAIL_CHARS:
        return None
    return remainder


def word_count(text):
    return len(text.split())


def two_word_opener(text):
    opener = " ".join(text.split()[:2]).lower()
    return opener.lstrip(OPENER_LEAD).rstrip(OPENER_TRAIL)


def run_t2(texts, seed):
    cap = int(len(texts) * I_THINK_TARGET)
    current = sum(1 for text in texts if text.startswith(I_THINK))
    if current <= cap:
        return texts, 0
    candidates = [i for i, text in enumerate(texts) if t2_is_candidate(text)]
    rng = random.Random(seed)
    rng.shuffle(candidates)
    changed = 0
    for index in candidates:
        if current <= cap:
            break
        new = t2_apply(texts[index])
        if new is not None and new.strip() and not new.startswith(I_THINK):
            texts[index] = new
            current -= 1
            changed += 1
    return texts, changed


def run_t3(texts, seed):
    cap = int(len(texts) * QUESTION_TARGET)
    current = sum(1 for text in texts if ends_with_question(text))
    if current <= cap:
        return texts, 0
    candidates = [i for i, text in enumerate(texts) if t3_apply(text) is not None]
    rng = random.Random(seed)
    rng.shuffle(candidates)
    changed = 0
    for index in candidates:
        if current <= cap:
            break
        new = t3_apply(texts[index])
        if new is not None and new.strip() and not ends_with_question(new):
            texts[index] = new
            current -= 1
            changed += 1
    return texts, changed


def process_file(path, seed, type_filter):
    raw = path.read_bytes().decode("utf-8")
    lines = raw.split("\n")
    trailing_newline = bool(lines) and lines[-1] == ""
    if trailing_newline:
        lines.pop()

    objects = []
    texts = []
    for line in lines:
        obj = json.loads(line) if line.strip() else None
        if not isinstance(obj, dict):
            objects.append(None)
            texts.append(None)
            continue
        objects.append(obj)
        texts.append(obj.get("response"))

    def editable(index):
        if objects[index] is None or not isinstance(texts[index], str):
            return False
        return type_filter is None or objects[index].get("type") == type_filter

    t1_idx = [i for i in range(len(lines)) if type_filter is not None and editable(i)]
    t3_idx = [i for i in range(len(lines)) if editable(i)]
    measured_idx = t1_idx if type_filter is not None else t3_idx

    before_texts = {i: texts[i] for i in measured_idx}

    counts = {"T1": 0, "T2": 0, "T3": 0}
    for i in t1_idx:
        new = t1_apply(texts[i])
        if new is not None:
            texts[i] = new
            counts["T1"] += 1

    if t1_idx:
        subset = [texts[i] for i in t1_idx]
        subset, counts["T2"] = run_t2(subset, seed)
        for i, text in zip(t1_idx, subset):
            texts[i] = text

    subset = [texts[i] for i in t3_idx]
    subset, counts["T3"] = run_t3(subset, seed)
    for i, text in zip(t3_idx, subset):
        texts[i] = text

    out = []
    for i, obj in enumerate(objects):
        if obj is None:
            out.append(lines[i])
            continue
        if texts[i] == obj.get("response"):
            out.append(lines[i])
        else:
            obj["response"] = texts[i]
            out.append(json.dumps(obj, ensure_ascii=False))

    return {
        "path": path,
        "rows": len(lines),
        "eligible": len(t3_idx),
        "measured_before": before_texts,
        "measured_after": {i: texts[i] for i in measured_idx},
        "counts": counts,
        "content": "\n".join(out) + ("\n" if trailing_newline else ""),
    }


def report(stats):
    path = stats["path"]
    before = stats["measured_before"]
    after = stats["measured_after"]
    n = len(before)
    order = sorted(before)
    b_list = [before[i] for i in order]
    a_list = [after[i] for i in order]

    counts = stats["counts"]
    changed_rows = sum(1 for i in order if before[i] != after[i])

    print(f"\n{path}")
    print(f"  rows {stats['rows']}   eligible {stats['eligible']}   measured {n}")
    print(f"  changed by transform   T1 {counts['T1']}   T2 {counts['T2']}   "
          f"T3 {counts['T3']}   distinct rows {changed_rows}")

    for label, pred in (("ends with question", ends_with_question),
                        ('starts "I think"', lambda t: t.startswith(I_THINK))):
        b = sum(1 for t in b_list if pred(t))
        a = sum(1 for t in a_list if pred(t))
        print(f"  {label:<20} {b / n:6.2%} -> {a / n:6.2%}   ({b} -> {a} of {n})")

    b_words = [word_count(t) for t in b_list]
    a_words = [word_count(t) for t in a_list]
    b_med, b_sd = statistics.median(b_words), statistics.pstdev(b_words)
    a_med, a_sd = statistics.median(a_words), statistics.pstdev(a_words)
    print(f"  words                 median {b_med:.1f} -> {a_med:.1f}   "
          f"pop sd {b_sd:.1f} -> {a_sd:.1f}")

    b_open = Counter(two_word_opener(t) for t in b_list)
    a_open = Counter(two_word_opener(t) for t in a_list)
    b_open.pop("", None)
    a_open.pop("", None)
    top_b = sorted(b_open.items(), key=lambda kv: (-kv[1], kv[0]))[:10]
    top_a = sorted(a_open.items(), key=lambda kv: (-kv[1], kv[0]))[:10]
    print("  top-10 two-word openers (count, share, opener)")
    print(f"    {'rank':>4}  {'before':<30}  {'after':<30}")
    for rank in range(max(len(top_b), len(top_a))):
        cells = []
        for table in (top_b, top_a):
            if rank < len(table):
                name, count = table[rank]
                cells.append(f"{count:>6}  {count / n:6.2%}  {name[:18]}")
            else:
                cells.append(f"{'':>6}  {'':>6}  {'':<18}")
        print(f"    {rank + 1:>4}  {cells[0]}  {cells[1]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true",
                    help="report the metrics without writing anything")
    ap.add_argument("--seed", type=int, default=13,
                    help="seed for the random subsets (default 13)")
    args = ap.parse_args()

    print(f"naturalize_corpus  seed={args.seed}  mode={'dry-run' if args.dry_run else 'write'}")

    for name, type_filter in FILES:
        path = DATA_DIR / name
        stats = process_file(path, args.seed, type_filter)
        report(stats)
        if not args.dry_run:
            path.write_bytes(stats["content"].encode("utf-8"))


if __name__ == "__main__":
    main()
