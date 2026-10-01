#!/usr/bin/env python3
"""
stylometry.py — Burrows' Delta between a reference corpus and one or more
candidates, plus expressive-punctuation rates.

Burrows' Delta, as implemented here
-----------------------------------
Tokenise to lowercase words. The feature set is the N most frequent words of the
REFERENCE corpus (default 150). Each feature's frequency rate is measured in
sub-samples of the reference — chunks of ~1,000 words (CHUNK_WORDS, --chunk) —
and turned into a z-score with that feature's own mean and sample sd across
those chunks:

    z_i(text) = (rate_i(text) - mean_i) / sd_i

The reference's own mean z-score is therefore 0, and

    Delta(candidate, reference) = mean_i | z_i(candidate) - z_i(reference) |
                                 = mean_i |z_i(candidate)|

Why sub-samples: a corpus of a few thousand words cannot estimate a per-word
standard deviation from one aggregate count — the sd would be an artefact of the
corpus length, not of the author's variation. Chunking gives a distribution of
chunk-level rates to take the sd over, so Delta measures how far a candidate sits
from the reference's internal spread. Consequence worth knowing: Delta is
sensitive to CHUNK_WORDS, and a reference shorter than one chunk has no spread at
all (one sub-sample, sd 0). Features with zero variance carry no information and
are dropped; --features can then be raised to keep N.

--self-split is the calibration: the reference's chunks are shuffled with seed
13, cut in half, a profile built from half A, and Delta(half B) reported. That is
the distance between two samples of the SAME author, i.e. the noise floor. Any
candidate scoring near it is indistinguishable from the author at this sample
size.

--mask-names drops capitalised tokens that are not sentence-initial before
counting, which removes proper nouns (names, places, publishers) from the feature
set. The table always reports the masked and unmasked Delta so one run gives the
comparison.

Inputs are "PATH[:field]". For a .jsonl the field defaults to "response"; rows
carrying {"messages": [...]} contribute their assistant turns instead. For a .md
file every line after a "**camus:**" marker is taken as one reply. --where
key=value and --where-prefix key=value filter the reference's rows, as in
corpus_metrics.py; each candidate takes its own --candidate-where /
--candidate-where-prefix, positionally, so a reference filter never empties a
candidate that lacks the same key.

    python pipeline/stylometry.py --reference REF --candidate CAND [--candidate ...]
"""
import argparse
import json
import os
import random
import re
import statistics
import sys
from collections import Counter

CHUNK_WORDS = 1000
DEFAULT_FEATURES = 150
SELF_SPLIT_SEED = 13
ASSISTANT_MARKER = "**camus:**"

# Expressive punctuation, reported per 1,000 words. (column label, character)
PUNCT = (("!", "!"), ("?", "?"), ("\u2026", "\u2026"), ("\u2014", "\u2014"),
         (";", ";"), (":", ":"), ("()", "("))

_WORD_RE = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)*", re.UNICODE)
_OPENERS = "\"'([{“«‘‚‛¿¡"
_CLOSERS = "\"')\\]”’»›‚"
_SENTENCE_PUNCT = ".!?…"
# Quotes, brackets and whitespace carry no information about where a sentence
# starts, and they can sit on either side of the punctuation (a closing quote
# before it: 'end." Next'; an opening quote after it: 'Then "Next ...'). Strip
# them all and look at what is left.
_QUOTES_RE = re.compile("[" + re.escape(_OPENERS + _CLOSERS) + r"\s]+")


def opens_sentence(gap):
    """True when the text between two tokens closes a sentence."""
    residue = _QUOTES_RE.sub("", gap)
    return bool(residue) and all(ch in _SENTENCE_PUNCT for ch in residue)


def word_spans(text):
    """[(token, opens_sentence)] over the alphabetic tokens of text, in order."""
    spans, first = [], True
    pos = 0
    for match in _WORD_RE.finditer(text):
        gap = "" if first else text[pos:match.start()]
        spans.append((match.group(0), first or opens_sentence(gap)))
        first = False
        pos = match.end()
    return spans


def tokenize(text, mask_names=False):
    """Lowercase word tokens. With mask_names, capitalised tokens that do not open
    a sentence are dropped — proper nouns, not the grammar."""
    out = []
    for token, opens in word_spans(text):
        if mask_names and token[:1].isupper() and not opens:
            continue
        out.append(token.lower())
    return out


def chunks(tokens, size=CHUNK_WORDS):
    """Split a token list into ~size-word chunks. The tail, if shorter than half a
    chunk, is merged into the previous one so it does not understate the sd."""
    if not tokens:
        return []
    out = [tokens[i:i + size] for i in range(0, len(tokens), size)]
    if len(out) > 1 and len(out[-1]) < size / 2:
        out[-2].extend(out.pop())
    return out


def rates(token_list, vocab):
    n = len(token_list) or 1
    counts = Counter(token_list)
    return [counts.get(w, 0) / n for w in vocab]


class Profile:
    """Feature vocabulary + per-feature mean and sample sd over sub-samples."""

    def __init__(self, vocab, means, sds, n_chunks, n_tokens, dropped):
        self.vocab = vocab
        self.means = means
        self.sds = sds
        self.n_chunks = n_chunks
        self.n_tokens = n_tokens
        self.dropped = dropped

    def z(self, token_list):
        n = len(token_list) or 1
        counts = Counter(token_list)
        return [(counts.get(w, 0) / n - m) / s
                for w, m, s in zip(self.vocab, self.means, self.sds)]

    def delta(self, token_list):
        """Mean absolute z-score difference against this profile."""
        zs = self.z(token_list)
        if not zs:
            return float("nan")
        return sum(abs(z) for z in zs) / len(zs)

    def as_dict(self):
        return {"vocab": self.vocab, "means": self.means, "sds": self.sds,
                "n_chunks": self.n_chunks, "n_tokens": self.n_tokens,
                "dropped": self.dropped}


def _profile_from_subsamples(subsamples, features):
    """Profile from explicit sub-samples (used by self-split, which supplies its
    own halves instead of chunking)."""
    flat = [t for sub in subsamples for t in sub]
    totals = Counter()
    for sub in subsamples:
        totals.update(sub)
    vocab = [w for w, _ in sorted(totals.items(), key=lambda kv: (-kv[1], kv[0]))][:features]
    matrix = [rates(sub, vocab) for sub in subsamples]
    kept, kept_words = [], set()
    for i, word in enumerate(vocab):
        column = [row[i] for row in matrix]
        sd = statistics.stdev(column)
        if sd > 0:
            kept.append((word, statistics.mean(column), sd))
            kept_words.add(word)
    return Profile([k[0] for k in kept], [k[1] for k in kept], [k[2] for k in kept],
                   len(subsamples), len(flat), [w for w in vocab if w not in kept_words])


def build_profile(samples, features=DEFAULT_FEATURES, chunk=CHUNK_WORDS):
    """samples: list of token lists, one per source text."""
    subsamples = chunks([t for s in samples for t in s], chunk)
    if not subsamples:
        raise ValueError("reference corpus is empty")
    if len(subsamples) == 1:
        n_words = sum(len(s) for s in subsamples)
        raise ValueError(
            f"reference has {n_words} words — fewer than one {chunk}-word chunk, so "
            f"there are no sub-samples and no sd to normalise by. Supply more "
            f"reference text, or lower --chunk.")
    if features < 2:
        raise ValueError("--features must be at least 2")
    profile = _profile_from_subsamples(subsamples, features)
    if not profile.vocab:
        raise ValueError("every feature has zero variance across the reference "
                         "chunks — the reference is too small or too uniform")
    return profile


def self_split(samples, features=DEFAULT_FEATURES, chunk=CHUNK_WORDS,
               seed=SELF_SPLIT_SEED):
    """Shuffle the reference chunks with a fixed seed, cut in half, and report
    Delta(half B | profile built from half A): the distance between two samples of
    the same author."""
    subs = list(chunks([t for s in samples for t in s], chunk))
    rng = random.Random(seed)
    rng.shuffle(subs)
    mid = len(subs) // 2
    a, b = subs[:mid], subs[mid:]
    if not a or not b:
        raise ValueError(f"self-split needs at least two chunks, got {len(subs)}")
    profile = _profile_from_subsamples(a, features)
    if not profile.vocab:
        raise ValueError("self-split half A has no variance across its chunks")
    return {"delta": profile.delta([t for sub in b for t in sub]),
            "chunks_a": len(a), "chunks_b": len(b),
            "words_a": sum(len(s) for s in a), "words_b": sum(len(s) for s in b),
            "seed": seed}


def punctuation_rates(text):
    """Counts of each expressive mark per 1,000 words."""
    n_words = len(_WORD_RE.findall(text))
    out = {}
    for name, ch in PUNCT:
        count = text.count(ch)
        if name == "()":
            count = min(count, text.count(")"))     # closed pairs only
        out[name] = (count / n_words * 1000) if n_words else 0.0
    return out


# ───────────────────────────────────────────────────────── input ────

def parse_spec(spec, default_field="response"):
    """'PATH[:field]' -> (path, field). A spec that names a real file is taken
    whole, so a colon inside a filename cannot split it."""
    if os.path.exists(spec):
        return spec, default_field
    if ":" not in spec:
        raise FileNotFoundError(spec)
    path, field = spec.rsplit(":", 1)
    return path, field


def parse_where(value):
    if value is None:
        return None
    if "=" not in value:
        raise ValueError("--where must be key=value")
    key, val = value.split("=", 1)
    return key, val


def assistant_turns(row):
    """Assistant content from a {"messages": [...]} row, in order."""
    out = []
    for message in row.get("messages") or []:
        if not isinstance(message, dict):
            continue
        if message.get("role") == "assistant" and isinstance(message.get("content"), str):
            out.append(message["content"])
    return out


def jsonl_texts(path, field="response", where=None, where_prefix=None):
    key, val = where if where else (None, None)
    pkey, pval = where_prefix if where_prefix else (None, None)
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
            if pkey is not None and not str(row.get(pkey) or "").startswith(pval):
                continue
            value = row.get(field)
            if isinstance(value, str):
                texts.append(value)
            elif isinstance(row.get("messages"), list):
                texts.extend(assistant_turns(row))
    return texts


_STOP_AT_MD = re.compile(r"^\s*(---|\*\*you:\*\*|#|\*\*system:)")


def md_texts(path):
    """Replies from a transcript whose assistant lines start with '**camus:**'.
    A reply runs from the marker to the next blank line, rule, or new speaker."""
    texts, current = [], None
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            stripped = line.rstrip("\n")
            if stripped.lstrip().startswith(ASSISTANT_MARKER):
                if current is not None:
                    texts.append(current.strip())
                current = stripped.lstrip()[len(ASSISTANT_MARKER):].strip()
                continue
            if current is None:
                continue
            if not stripped.strip() or _STOP_AT_MD.match(stripped):
                texts.append(current.strip())
                current = None
                continue
            current += " " + stripped.strip()
    if current is not None:
        texts.append(current.strip())
    return [t for t in texts if t]


def read_texts(spec, field="response", where=None, where_prefix=None):
    path, field = parse_spec(spec, field)
    if path.lower().endswith(".md"):
        return md_texts(path)
    return jsonl_texts(path, field, where, where_prefix)


# ───────────────────────────────────────────────────────── report ────

def label_for(spec):
    """Readable column label: the spec with any field suffix stripped, keeping
    the last three path components so eval_human session names stay distinct."""
    path = parse_spec(spec)[0]
    parts = path.split("/")
    return "/".join(parts[-3:])


def print_table(rows, punct_keys, self_split_result, notes):
    width = max([len(r["label"]) for r in rows] + [24])
    head = (f"{'candidate':<{width}}  {'words':>7}  {'delta':>6}  "
            f"{'delta_masked':>12}  " +
            "  ".join(f"{k:>5}" for k in punct_keys))
    print("\n" + head)
    print("-" * len(head))
    for r in rows:
        punct = "  ".join(f"{r['punct'][k]:>5.1f}" for k in punct_keys)
        print(f"{r['label']:<{width}}  {r['words']:>7}  {r['delta']:>6.3f}  "
              f"{r['delta_masked']:>12.3f}  {punct}")
    if self_split_result:
        s = self_split_result
        print(f"\nself-split floor (seed {s['seed']}): "
              f"Delta(half B | half A) = {s['delta']:.3f}   "
              f"chunks {s['chunks_a']}/{s['chunks_b']}, "
              f"words {s['words_a']}/{s['words_b']}")
    for note in notes:
        print(f"note: {note}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", required=True,
                    help='"PATH[:field]" — the corpus to compare against')
    ap.add_argument("--candidate", action="append", default=[], required=True,
                    help='"PATH[:field]" — repeatable')
    ap.add_argument("--field", default="response",
                    help="default text field for .jsonl inputs (default: response)")
    ap.add_argument("--where", default=None,
                    help="row filter for the REFERENCE: key == value")
    ap.add_argument("--where-prefix", default=None,
                    help="row filter for the REFERENCE: str(key) starts with value")
    ap.add_argument("--candidate-where", action="append", default=[],
                    help="per-candidate row filter key == value, given in the same "
                         "order as --candidate (shorter lists leave the rest unfiltered)")
    ap.add_argument("--candidate-where-prefix", action="append", default=[],
                    help="per-candidate row filter, str(key) starts with value, "
                         "given in the same order as --candidate")
    ap.add_argument("--features", type=int, default=DEFAULT_FEATURES,
                    help=f"most-frequent words to use (default: {DEFAULT_FEATURES})")
    ap.add_argument("--chunk", type=int, default=CHUNK_WORDS,
                    help=f"sub-sample size in words (default: {CHUNK_WORDS})")
    ap.add_argument("--mask-names", action="store_true",
                    help="drop non-sentence-initial capitalised tokens before counting")
    ap.add_argument("--self-split", action="store_true",
                    help=f"also report the floor: Delta between two halves of the "
                         f"reference (seed {SELF_SPLIT_SEED})")
    ap.add_argument("--list-labels", action="store_true",
                    help="print each input's text count and stop")
    args = ap.parse_args(argv)

    where = parse_where(args.where)
    where_prefix = parse_where(args.where_prefix)

    ref_texts = read_texts(args.reference, args.field, where, where_prefix)
    ref_plain = [tokenize(t) for t in ref_texts]
    ref_masked = [tokenize(t, mask_names=True) for t in ref_texts]

    candidates = []
    for i, spec in enumerate(args.candidate):
        # Candidates are selected by their own flags: the reference's filter is
        # a statement about the reference, and leaking it would empty every
        # candidate that lacks the reference's key.
        cand_where = parse_where(args.candidate_where[i]) if i < len(args.candidate_where) else None
        cand_prefix = (parse_where(args.candidate_where_prefix[i])
                       if i < len(args.candidate_where_prefix) else None)
        texts = read_texts(spec, args.field, cand_where, cand_prefix)
        label = label_for(spec)
        if cand_where or cand_prefix:
            label += "  [" + ",".join(filter(None, [
                "=".join(cand_where) if cand_where else "",
                "starts:" + "=".join(cand_prefix) if cand_prefix else ""])) + "]"
        if not texts:
            print(f"warning: no texts from {spec} — omitted from the table",
                  file=sys.stderr)
        candidates.append((label, texts))

    if args.list_labels:
        print(f"{args.reference}\t{len(ref_texts)}")
        for label, texts in candidates:
            print(f"{label}\t{len(texts)}")
        return 0

    try:
        profile = build_profile(ref_plain, args.features, args.chunk)
        profile_masked = build_profile(ref_masked, args.features, args.chunk)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    split = None
    if args.self_split:
        try:
            split = self_split(ref_plain, args.features, args.chunk)
        except ValueError as exc:
            print(f"warning: self-split unavailable: {exc}", file=sys.stderr)

    notes = [f"reference: {args.reference} — {profile.n_tokens} words in "
             f"{profile.n_chunks} chunks, {len(profile.vocab)} features "
             f"({len(profile.dropped)} dropped for zero variance)"]
    if args.mask_names:
        notes.append("headline delta uses --mask-names tokenisation; "
                     "delta_masked column is the same figure")

    rows = []
    for label, texts in candidates:
        if not texts:
            continue
        joined = "\n\n".join(texts)
        plain = tokenize(joined)
        masked = tokenize(joined, mask_names=True)
        primary = profile_masked if args.mask_names else profile
        rows.append({
            "label": label,
            "n_texts": len(texts),
            "words": len(plain),
            "delta": primary.delta(plain if not args.mask_names else masked),
            "delta_masked": profile_masked.delta(masked),
            "delta_plain": profile.delta(plain),
            "punct": punctuation_rates(joined),
        })

    print_table(rows, [name for name, _ in PUNCT], split, notes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
