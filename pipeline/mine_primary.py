#!/usr/bin/env python3
"""
mine_primary.py — primary-text passages from ingested chunks.

Reads data/source_chunks.jsonl (rows {id, source, stream, date, text} from
kb/ingest_sources.py), keeps stream == "voice" chunks, splits them into
passages, removes extraction junk and running headers, deduplicates exact
passages, and writes:

    data/primary_candidates.jsonl   rows {source, chunk_id, passage, words, prompt}

Splitting: on blank lines; a chunk with no blank lines is split on sentence
boundaries into runs of at most --max-words words, never breaking a sentence.

    python pipeline/mine_primary.py [--sources "Name A" "Name B" ...]
        [--min-words 3] [--max-words 150] [--out data/primary_candidates.jsonl]
"""
import argparse
import json
import re
from collections import Counter

from corpus_metrics import metrics

VOICE = "voice"
HEADER_MIN_CHUNKS = 5
JUNK_CHAR_RATIO = 0.10

SENTENCE_SPLIT = re.compile(r"(?<=[.!?\u2026])\s+")
BLANK_LINE = re.compile(r"\n\s*\n")
DIGITS_ONLY = re.compile(r"^[0-9]+$")
ROMAN_STRICT = re.compile(
    r"^M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$", re.I)
ORDINARY_PUNCT = set(",.;:!?'\"()[]{}-/\\|@#$%^&*+=`~\u2014\u2013\u2026"
                     "\u00ab\u00bb\u201c\u201d\u2018\u2019")


def is_page_or_roman(line):
    text = line.strip()
    if not text:
        return False
    if DIGITS_ONLY.fullmatch(text):
        return True
    return len(text) <= 7 and ROMAN_STRICT.fullmatch(text) is not None


def is_junk(passage):
    if not passage:
        return True
    bad = sum(1 for ch in passage
              if not ch.isalpha() and not ch.isspace() and ch not in ORDINARY_PUNCT)
    return bad / len(passage) > JUNK_CHAR_RATIO


def running_headers(rows):
    counts = {}
    for row in rows:
        source = row.get("source", "")
        lines = {line.strip() for line in str(row.get("text", "")).splitlines()
                 if line.strip()}
        for line in lines:
            counts[(source, line)] = counts.get((source, line), 0) + 1
    headers = {}
    for (source, line), count in counts.items():
        if count > HEADER_MIN_CHUNKS:
            headers.setdefault(source, set()).add(line)
    return headers


def split_sentences(text, max_words):
    runs = []
    buffer = []
    total = 0
    for sentence in SENTENCE_SPLIT.split(text.strip()):
        words = len(sentence.split())
        if buffer and total + words > max_words:
            runs.append(" ".join(buffer))
            buffer, total = [], 0
        buffer.append(sentence)
        total += words
    if buffer:
        runs.append(" ".join(buffer))
    return runs


def split_passages(text, max_words):
    if BLANK_LINE.search(text):
        paragraphs = re.split(BLANK_LINE, text)
        return [" ".join(p.split()) for p in paragraphs if p.strip()]
    return [" ".join(run.split()) for run in split_sentences(text, max_words) if run.strip()]


def strip_lines(text, headers):
    kept = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            if is_page_or_roman(stripped) or stripped in headers:
                continue
        kept.append(line)
    return "\n".join(kept)


def mine_rows(rows, sources=None, min_words=3, max_words=150):
    selected = [row for row in rows
                if row.get("stream") == VOICE
                and (sources is None or row.get("source") in sources)]

    headers = running_headers(selected)
    seen = set()
    candidates = []
    per_source = Counter()

    for row in selected:
        source = row.get("source", "")
        clean = strip_lines(str(row.get("text", "")), headers.get(source, set()))
        for passage in split_passages(clean, max_words):
            if len(passage.split()) < min_words:
                continue
            if is_junk(passage):
                continue
            key = " ".join(passage.lower().split())
            if key in seen:
                continue
            seen.add(key)
            candidates.append({
                "source": source,
                "chunk_id": row.get("id"),
                "passage": passage,
                "words": len(passage.split()),
                "prompt": None,
            })
            per_source[source] += 1

    return candidates, per_source


def load_rows(path):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--in", dest="src", default="data/source_chunks.jsonl")
    ap.add_argument("--sources", nargs="+", default=None)
    ap.add_argument("--min-words", type=int, default=3)
    ap.add_argument("--max-words", type=int, default=150)
    ap.add_argument("--out", default="data/primary_candidates.jsonl")
    args = ap.parse_args()

    rows = load_rows(args.src)
    candidates, per_source = mine_rows(rows, args.sources, args.min_words, args.max_words)

    with open(args.out, "w", encoding="utf-8") as handle:
        for row in candidates:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"passages -> {args.out}  ({len(candidates)} total)")
    for source, count in sorted(per_source.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {count:>6}  {source}")

    print("\ncorpus metrics over passages:")
    print(json.dumps(metrics([row["passage"] for row in candidates]),
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
