#!/usr/bin/env python3
"""
corpus_metrics.py — uniformity metrics for a JSONL corpus.

Reads one or more JSONL files, pulls a string field from each row (default
"response"), optionally filtered with --where key=value, and reports:

  n; word-count buckets 0-9, 10-24, 25-49, 50-99, 100-199, 200-399, 400+
  median / mean / population sd
  the top-10 two-word openers and their combined share
  first-word concentration: most common first word and top-10 share
  question-ending rate
  exact duplicates (lowercase + whitespace normalised)
  near-duplicate pairs (Jaccard >= 0.8 over word 3-shingles, first 3000 texts)

The importable entry point is metrics(texts) -> dict.

    python pipeline/corpus_metrics.py FILE [FILE...] [--field response] [--where key=value]
"""
import argparse
import json
import statistics
from collections import Counter

BUCKETS = (
    ("0-9", 0, 9),
    ("10-24", 10, 24),
    ("25-49", 25, 49),
    ("50-99", 50, 99),
    ("100-199", 100, 199),
    ("200-399", 200, 399),
    ("400+", 400, None),
)

OPENER_LEAD = "\"'\u201c\u2018(["
OPENER_TRAIL = ",.;:!?\u2014-"

NEAR_DUP_THRESHOLD = 0.8
NEAR_DUP_CAP = 3000
SHINGLE_N = 3


def normalize(text):
    return " ".join(text.lower().split())


def opener_of(text):
    return " ".join(text.split()[:2]).lower().lstrip(OPENER_LEAD).rstrip(OPENER_TRAIL)


def first_word_of(text):
    words = text.split()
    if not words:
        return ""
    return words[0].lower().lstrip(OPENER_LEAD).rstrip(OPENER_TRAIL)


def ends_with_question(text):
    return text.rstrip().endswith("?")


def shingles(text, n=SHINGLE_N):
    words = normalize(text).split()
    if not words:
        return set()
    if len(words) < n:
        return {tuple(words)}
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def near_duplicate_count(texts, threshold=NEAR_DUP_THRESHOLD, cap=NEAR_DUP_CAP):
    docs = [shingles(t) for t in texts[:cap]]
    count = 0
    for i in range(len(docs)):
        a = docs[i]
        if not a:
            continue
        for j in range(i + 1, len(docs)):
            b = docs[j]
            if not b:
                continue
            lo, hi = (len(a), len(b)) if len(a) <= len(b) else (len(b), len(a))
            if lo / hi < threshold:
                continue
            inter = len(a & b)
            if inter and inter / (len(a) + len(b) - inter) >= threshold:
                count += 1
    return count


def metrics(texts):
    n = len(texts)
    word_counts = [len(t.split()) for t in texts]

    buckets = {}
    for label, low, high in BUCKETS:
        count = sum(1 for w in word_counts if w >= low and (high is None or w <= high))
        buckets[label] = {"count": count, "pct": (count / n if n else 0.0)}

    if n:
        mean = sum(word_counts) / n
        median = statistics.median(word_counts)
        pstdev = statistics.pstdev(word_counts)
    else:
        mean = median = pstdev = 0.0

    opener_counts = Counter(opener_of(t) for t in texts)
    opener_counts.pop("", None)
    openers_top = sorted(opener_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:10]
    openers_top_share = sum(c for _, c in openers_top) / n if n else 0.0

    first_counts = Counter(first_word_of(t) for t in texts)
    first_counts.pop("", None)
    first_top = sorted(first_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    first_word, first_word_count = first_top[0] if first_top else ("", 0)
    first_word_share = first_word_count / n if n else 0.0
    first_word_top10_share = sum(c for _, c in first_top[:10]) / n if n else 0.0

    question_rate = (sum(1 for t in texts if ends_with_question(t)) / n) if n else 0.0

    distinct = len({normalize(t) for t in texts})
    duplicates = n - distinct

    near = near_duplicate_count(texts)

    return {
        "n": n,
        "buckets": buckets,
        "median": median,
        "mean": mean,
        "pstdev": pstdev,
        "openers_top": [{"opener": name, "count": c, "share": c / n if n else 0.0}
                        for name, c in openers_top],
        "openers_top_share": openers_top_share,
        "first_word": first_word,
        "first_word_share": first_word_share,
        "first_word_top10_share": first_word_top10_share,
        "question_rate": question_rate,
        "duplicates": duplicates,
        "near_duplicates": near,
    }


def parse_where(value):
    if value is None:
        return None
    if "=" not in value:
        raise ValueError("--where must be key=value")
    key, val = value.split("=", 1)
    return key, val


def collect_texts(path, field="response", where=None):
    key, val = where if where else (None, None)
    texts = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            if key is not None and str(row.get(key)) != val:
                continue
            value = row.get(field)
            if isinstance(value, str):
                texts.append(value)
    return texts


def print_metrics(label, m):
    print(f"\n{label}")
    print(f"  n {m['n']}")
    print(f"  words   median {m['median']:.1f}   mean {m['mean']:.1f}   "
          f"pop sd {m['pstdev']:.1f}")
    print("  word-count buckets")
    for name, _, _ in BUCKETS:
        b = m["buckets"][name]
        print(f"    {name:>7}  {b['count']:>6}  {b['pct']:6.2%}")
    print(f"  top-10 two-word openers: share {m['openers_top_share']:.2%}")
    for rank, entry in enumerate(m["openers_top"], 1):
        print(f"    {rank:>2}  {entry['count']:>6}  {entry['share']:6.2%}  {entry['opener'][:30]}")
    print(f"  first word {m['first_word']!r} share {m['first_word_share']:.2%}; "
          f"top-10 first words share {m['first_word_top10_share']:.2%}")
    print(f"  question-ending {m['question_rate']:.2%}")
    print(f"  exact duplicates {m['duplicates']}")
    print(f"  near duplicates {m['near_duplicates']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("files", nargs="+")
    ap.add_argument("--field", default="response")
    ap.add_argument("--where", default=None, help="only rows where key == value")
    args = ap.parse_args()

    where = parse_where(args.where)

    combined = []
    for path in args.files:
        texts = collect_texts(path, args.field, where)
        combined.extend(texts)
        print_metrics(path, metrics(texts))

    if len(args.files) > 1:
        print_metrics("COMBINED", metrics(combined))


if __name__ == "__main__":
    main()
