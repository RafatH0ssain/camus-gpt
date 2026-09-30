#!/usr/bin/env python3
"""
mine_primary.py — primary-text passages from ingested chunks.

Reads data/source_chunks.jsonl (rows {id, source, stream, date, text} from
kb/ingest_sources.py), keeps stream == "voice" chunks, splits them into
passages, removes extraction junk and running headers, drops editorial notes
and front/back matter, deduplicates exact passages, and writes:

    data/primary_candidates.jsonl   rows {source, chunk_id, passage, words, prompt}

Splitting: on blank lines; a chunk with no blank lines is split on sentence
boundaries into runs of at most --max-words words, never breaking a sentence. A
row marked "entry" is one notebook entry: it is kept whole, and only split on
sentence boundaries if it is longer than --max-words.

mine_rows() returns (candidates, per_source_counts, junk_filtered_counts); the
printed report lists the sources and then what each junk filter removed.

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

# ── junk filters ───────────────────────────────────────────────────────────────
# Three kinds of passage that survive the character and running-header filters but
# are not Camus talking. They run in this order and a passage is counted under the
# first one that matches, so the three counts add up to everything dropped here.

EDITORIAL_NAME = re.compile(r"\bCamus\b")
FRONT_MATTER = re.compile(
    r"typography|binding design|isbn|copyright|all rights reserved"
    r"|translated by|translated from|library of congress|printed in"
    r"|first edition|published by|edited by|introduction by", re.I)
APPARATUS = re.compile(r"^(?:See\s|Cf\.|Note:|\[Translator|\[Editor|\d+\.\s)")


def is_editorial_note(passage):
    """The notebooks' editors write about Camus in the third person; he does not
    name himself, so the name marks editorial matter, not his own words."""
    return EDITORIAL_NAME.search(passage) is not None


def is_front_back_matter(passage):
    """Copyright page, colophon, translator's and editor's credits."""
    return FRONT_MATTER.search(passage) is not None


def is_apparatus_line(passage):
    """A footnote, a cross-reference, or a translator/editor's aside left in the
    text flow by the extraction."""
    return APPARATUS.match(passage) is not None


JUNK_FILTERS = (
    ("editorial notes (names Camus)", is_editorial_note),
    ("front/back matter", is_front_back_matter),
    ("footnote/apparatus lines", is_apparatus_line),
)


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


def split_entry_passage(text, max_words):
    """One notebook entry. Kept whole however many paragraphs it holds; split on
    sentence boundaries only when it is longer than max_words."""
    flat = " ".join(text.split())
    if len(flat.split()) <= max_words:
        return [flat]
    return [" ".join(run.split()) for run in split_sentences(flat, max_words)]


def strip_lines(text, headers):
    kept = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            if is_page_or_roman(stripped) or stripped in headers:
                continue
        kept.append(line)
    return "\n".join(kept)


def junk_filter(passage):
    """Label of the first junk filter that rejects the passage, else None."""
    for label, test in JUNK_FILTERS:
        if test(passage):
            return label
    return None


def mine_rows(rows, sources=None, min_words=3, max_words=150):
    selected = [row for row in rows
                if row.get("stream") == VOICE
                and (sources is None or row.get("source") in sources)]

    headers = running_headers(selected)
    seen = set()
    candidates = []
    per_source = Counter()
    filtered = Counter()

    for row in selected:
        source = row.get("source", "")
        clean = strip_lines(str(row.get("text", "")), headers.get(source, set()))
        is_entry = row.get("entry") is True
        passages = (split_entry_passage(clean, max_words) if is_entry
                    else split_passages(clean, max_words))
        for passage in passages:
            if len(passage.split()) < min_words:
                continue
            if is_junk(passage):
                continue
            reason = junk_filter(passage)
            if reason is not None:
                filtered[reason] += 1
                continue
            key = " ".join(passage.lower().split())
            if key in seen:
                continue
            seen.add(key)
            candidate = {
                "source": source,
                "chunk_id": row.get("id"),
                "passage": passage,
                "words": len(passage.split()),
                "prompt": None,
            }
            if is_entry:
                candidate["entry"] = True
            candidates.append(candidate)
            per_source[source] += 1

    return candidates, per_source, filtered


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
    candidates, per_source, filtered = mine_rows(
        rows, args.sources, args.min_words, args.max_words)

    with open(args.out, "w", encoding="utf-8") as handle:
        for row in candidates:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"passages -> {args.out}  ({len(candidates)} total)")
    for source, count in sorted(per_source.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {count:>6}  {source}")

    print("\npassages removed by junk filter:")
    for label, _ in JUNK_FILTERS:
        print(f"  {filtered[label]:>6}  {label}")
    print(f"  {sum(filtered.values()):>6}  total")

    print("\ncorpus metrics over passages:")
    print(json.dumps(metrics([row["passage"] for row in candidates]),
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
