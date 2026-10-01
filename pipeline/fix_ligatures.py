#!/usr/bin/env python3
"""
fix_ligatures.py — repair PDF ligature splits in the corpora.

A PDF text layer now and then emits a ligature as its own characters and then
breaks the line, leaving "fi lth" where the page said "filth" and "suffi
ciently" where it said "sufficiently". The halves are not words, so a
dictionary check repairs them safely. Two shapes are considered:

  A  a token ending in fi / fl / ff / ffi / ffl, then whitespace, then a token
     starting with a lowercase letter                      -> join the two
  B  a standalone fi / fl / ff / ffi / ffl between two tokens -> join all three
     ("di ff erent" -> "different", "in fi nite" -> "infinite")

A join fires only when the joined word, lowercased and stripped of surrounding
punctuation, is in the dictionary AND at least one of the pieces is not. The
second half is what keeps "staff meeting" from collapsing into "staffman" —
both pieces there are words — and "off the", "stuff that" and "cliff edge" need
no such help: none of those joins is a word in the first place.

Punctuation on the surviving ends is preserved: only the whitespace between the
pieces disappears.

Rows that do not change are written back as their original bytes, so only the
rewritten rows are re-serialized with json.dumps. Row count, row order and key
order are preserved exactly, and a second run makes no further edits.

    python pipeline/fix_ligatures.py data/camus_sft.jsonl --field response --dry-run
    python pipeline/fix_ligatures.py camus_kb_full.jsonl --field text --field quote
    python pipeline/fix_ligatures.py data/camus_multiturn.jsonl --field messages

--field messages means "the content of every assistant turn of the messages
list" — the multiturn rows have no response field.
"""
import argparse
import json
import re
import string
import sys
from pathlib import Path

DICT_PATH = "/usr/share/dict/words"

LIGATURES = ("ffi", "ffl", "ff", "fi", "fl")
STANDALONE = frozenset(LIGATURES)

TOKEN_RE = re.compile(r"\S+")
PUNCT = string.punctuation

SAMPLE_LIMIT = 20


def load_words(path=DICT_PATH):
    """The dictionary as a lowercased set."""
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"dictionary not found: {path}")
    with path.open(encoding="utf-8", errors="ignore") as fh:
        return {line.strip().lower() for line in fh if line.strip()}


def bare(token):
    """The token lowercased and stripped of surrounding punctuation."""
    return token.strip(PUNCT).lower()


def ends_with_ligature(token):
    return token.endswith(LIGATURES)


def is_standalone_ligature(token):
    return bare(token) in STANDALONE


def candidates(text):
    """Every join the two shapes offer, as (start, end, replacement, rule, pieces)."""
    tokens = [(m.start(), m.end(), m.group()) for m in TOKEN_RE.finditer(text)]
    out = []
    for i, (s1, e1, t1) in enumerate(tokens):
        if ends_with_ligature(t1) and i + 1 < len(tokens):
            s2, e2, t2 = tokens[i + 1]
            if s2 > e1 and t2[:1].islower():
                out.append((s1, e2, t1 + t2, "A", (t1, t2)))
        if is_standalone_ligature(t1) and 0 < i < len(tokens) - 1:
            s0, _, t0 = tokens[i - 1]
            s2, e2, t2 = tokens[i + 1]
            if s0 < s1 < e1 < s2 and t2[:1].islower():
                out.append((s0, e2, t0 + t1 + t2, "B", (t0, t1, t2)))
    # leftmost first; at one position the three-token join wins over the pair
    out.sort(key=lambda c: (c[0], -(c[1] - c[0])))
    return out


def accept(candidate, words):
    """Turn a candidate into a join, or reject it.

    Two gates: the joined word must be in the dictionary, and at least one of
    the pieces must not be — "staff man" is left alone because both pieces are
    words even though "staffman" is not.
    """
    start, end, replacement, rule, pieces = candidate
    word = bare(replacement)
    if word not in words:
        return None
    if all(bare(piece) in words for piece in pieces):
        return None
    return start, end, replacement, rule, pieces, word


def fix_text(text, words, max_passes=256):
    """Return (fixed text, joins). Each join records its own before -> after."""
    joins = []
    current = text
    for _ in range(max_passes):
        chosen = None
        for candidate in candidates(current):
            chosen = accept(candidate, words)
            if chosen is not None:
                break
        if chosen is None:
            break
        start, end, replacement, rule, pieces, word = chosen
        joins.append(join_record(current, start, end, replacement, rule, pieces, word))
        current = current[:start] + replacement + current[end:]
    return current, joins


def join_record(text, start, end, replacement, rule, pieces, word, width=26):
    """One join, with a little context either side, as a printable before/after."""
    pre = " ".join(text[max(0, start - width):start].split())
    post = " ".join(text[end:end + width].split())
    mid = " ".join(text[start:end].split())

    def line(middle):
        return " ".join(part for part in (pre, middle, post) if part)

    return {
        "rule": rule,
        "word": word,
        "pieces": " ".join(pieces),
        "before": line(mid),
        "after": line(replacement),
    }


def fix_messages(value, words):
    """Fix every assistant turn; returns (new value, joins)."""
    if not isinstance(value, list):
        return value, []
    joins = []
    out = []
    for message in value:
        if (isinstance(message, dict) and message.get("role") == "assistant"
                and isinstance(message.get("content"), str)):
            new, got = fix_text(message["content"], words)
            if got:
                message = dict(message)
                message["content"] = new
                joins.extend(got)
        out.append(message)
    return out, joins


def process_file(path, fields, words):
    """Rewrite the named fields of every row; unchanged lines keep their bytes."""
    path = Path(path)
    raw = path.read_bytes().decode("utf-8")
    lines = raw.split("\n")
    trailing_newline = bool(lines) and lines[-1] == ""
    if trailing_newline:
        lines.pop()

    out = []
    rows_changed = 0
    joins = 0
    samples = []

    for lineno, line in enumerate(lines, 1):
        obj = None
        if line.strip():
            try:
                parsed = json.loads(line)
            except ValueError:
                parsed = None
            if isinstance(parsed, dict):
                obj = parsed

        if obj is None:
            out.append(line)
            continue

        edits = []
        for field in fields:
            if field == "messages":
                if isinstance(obj.get("messages"), list):
                    new, got = fix_messages(obj["messages"], words)
                    if got:
                        obj["messages"] = new
                        edits.extend(got)
                continue
            value = obj.get(field)
            if not isinstance(value, str):
                continue
            new, got = fix_text(value, words)
            if got:
                obj[field] = new
                edits.extend(got)

        if edits:
            rows_changed += 1
            joins += len(edits)
            samples.extend(dict(join, line=lineno) for join in edits)
            out.append(json.dumps(obj, ensure_ascii=False))
        else:
            out.append(line)

    return {
        "path": path,
        "rows": len(lines),
        "rows_changed": rows_changed,
        "joins": joins,
        "samples": samples,
        "content": "\n".join(out) + ("\n" if trailing_newline else ""),
    }


def report(stats, dry_run, limit=SAMPLE_LIMIT):
    path = stats["path"]
    print(f"\n{path}   [{'dry-run' if dry_run else 'write'}]")
    print(f"  rows {stats['rows']}   rows changed {stats['rows_changed']}   "
          f"joins {stats['joins']}")
    samples = stats["samples"]
    if not samples:
        print("  no joins")
        return
    print(f"  sample joins ({min(limit, len(samples))} of {len(samples)})")
    for i, sample in enumerate(samples[:limit], 1):
        print(f"   {i:>2}. line {sample['line']:>6} [{sample['rule']}] "
              f"{sample['before']!r}\n         -> {sample['after']!r}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("file", help="JSONL file to repair")
    parser.add_argument("--field", action="append", required=True, dest="fields",
                        help="field to repair; repeat for several "
                             "('messages' = assistant turns)")
    parser.add_argument("--dry-run", action="store_true",
                        help="report the joins without writing anything")
    parser.add_argument("--dict", default=DICT_PATH,
                        help=f"word list (default {DICT_PATH})")
    parser.add_argument("--samples", type=int, default=SAMPLE_LIMIT,
                        help=f"sample joins to print (default {SAMPLE_LIMIT})")
    args = parser.parse_args(argv)

    words = load_words(args.dict)
    stats = process_file(args.file, args.fields, words)
    report(stats, args.dry_run, args.samples)
    if not args.dry_run:
        Path(stats["path"]).write_bytes(stats["content"].encode("utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())