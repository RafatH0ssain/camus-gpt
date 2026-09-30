#!/usr/bin/env python3
"""
verify_quotes.py — check web-collected quotes against the full text of the books.

Reads data/primary_quotes_web.jsonl (rows {text, claimed_source, site, url,
web_status}, where web_status is sourced | disputed | misattributed |
popular_unverified) and searches the book text in data/source_chunks.jsonl and
data/primary_chunks.jsonl (rows {source, text}) for every quote. Both corpora
are searched: lines from the novels are his writing too. Writes

    data/primary_quotes.jsonl   the input row plus match_status, score,
                                matched_source, matched_excerpt, final

Per quote:
  * normalise: lowercase, curly quotes and apostrophes and the several dashes
    unified, punctuation stripped, whitespace collapsed;
  * candidates: every chunk holding one of the quote's 3 rarest tokens, taken
    from an inverted index over chunk tokens, examined shortest chunk first so
    the best score rises early (ties in corpus order, so the result is stable);
  * exact: if the normalised quote is a substring of a candidate chunk's
    normalised text the score is 1.0;
  * otherwise a window of the quote's token length ±20% slides over the
    candidate's tokens and is scored with difflib.SequenceMatcher on tokens.
    A window is skipped when the count of the quote's own tokens inside it
    cannot beat the best score so far: a match block needs a token on both
    sides, so the bound is exact, and it turns a five-hour scan into minutes.

match_status: verified at >= 0.85, likely_translation at 0.60-0.85 (same
passage, different translation), not_found below. The emitted score is rounded
to 2 decimals first and the status is read off that rounded score, so the number
in the file and the label in the file never disagree. final folds in the web
status: not_found + sourced -> sourced_not_in_corpus (a work we do not hold),
+ misattributed -> misattributed, + disputed -> disputed, + popular_unverified
-> unverified. An unrecognised web_status lands in unverified.

The report prints the counts of final by web_status, then every row that comes
out misattributed or unverified, one per line, and finally every row the corpus
verifies although the web calls it misattributed — the rows where the corpus
contradicts the web and which matter most.

    python pipeline/verify_quotes.py [--quotes FILE] [--corpus FILE [FILE ...]]
                                     [--out FILE]
"""
import argparse
import difflib
import json
import re
import sys
from collections import Counter, defaultdict, namedtuple
from itertools import accumulate

RARE_TOKENS = 3
WINDOW_SLACK = 0.20
VERIFIED_AT = 0.85
LIKELY_AT = 0.60
EXCERPT_MAX_WORDS = 40
PREVIEW_CHARS = 90

WEB_STATUSES = ("sourced", "disputed", "misattributed", "popular_unverified")
FINALS = ("verified", "likely_translation", "sourced_not_in_corpus",
          "misattributed", "disputed", "unverified")
MATCH_STATUSES = ("verified", "likely_translation", "not_found")
NOT_FOUND_FINAL = {
    "sourced": "sourced_not_in_corpus",
    "disputed": "disputed",
    "misattributed": "misattributed",
    "popular_unverified": "unverified",
}
UNVERIFIED_DEFAULT = "unverified"
FLAGGED_FINALS = ("misattributed", "unverified")
CONTRADICTED_WEB_STATUS = "misattributed"

UNIFY = {}
UNIFY.update({ch: "'" for ch in "\u2018\u2019\u201a\u201b"})
UNIFY.update({ch: '"' for ch in "\u201c\u201d\u201e\u201f"})
UNIFY.update({ch: "-" for ch in "\u2010\u2011\u2012\u2013\u2014\u2015\u2212"})
PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
WORD = re.compile(r"\w+", re.UNICODE)

Match = namedtuple("Match", "score source excerpt")


def normalise(text):
    """Lowercase, unify curly quotes and dashes, strip punctuation, collapse
    whitespace. The token list of the result is exactly the \\w+ runs of the
    text, lowercased."""
    lowered = str(text).lower().translate(UNIFY)
    return " ".join(PUNCT.sub(" ", lowered).split())


def tokenise(text):
    return [match.group().lower() for match in WORD.finditer(str(text))]


def tokenise_spans(text):
    """Tokens plus (start, end) character offsets in the original text, so an
    excerpt can be sliced out with its punctuation intact."""
    tokens = []
    spans = []
    for match in WORD.finditer(str(text)):
        tokens.append(match.group().lower())
        spans.append((match.start(), match.end()))
    return tokens, spans


def window_sizes(n, slack=WINDOW_SLACK):
    """Window lengths tried for a quote of n tokens: n ± slack, at least one."""
    low = max(1, int(n * (1.0 - slack)))
    high = max(low, int(n * (1.0 + slack)) + 1)
    return list(range(low, high))


def excerpt(text, start_token, max_words=EXCERPT_MAX_WORDS):
    """At most max_words words of the original text from token start_token on."""
    _, spans = tokenise_spans(text)
    if start_token >= len(spans):
        return ""
    end = min(len(spans), start_token + max_words)
    return " ".join(text[spans[start_token][0]:spans[end - 1][1]].split())


class Corpus:
    """The book text, normalised once, with an inverted index over its tokens.

    Only the normalised text is held; a candidate's tokens are re-split when it
    is scanned, which keeps the whole corpus out of the token objects.
    """

    def __init__(self, chunks):
        self.sources = [str(chunk.get("source", "")) for chunk in chunks]
        self.texts = [str(chunk.get("text", "")) for chunk in chunks]
        self.norm = [normalise(text) for text in self.texts]
        self.sizes = [len(text.split()) for text in self.norm]
        self.index = defaultdict(list)
        for position, text in enumerate(self.norm):
            for token in set(text.split()):
                self.index[token].append(position)

    def __len__(self):
        return len(self.texts)

    def tokens_at(self, position):
        return self.norm[position].split()

    def df(self, token):
        return len(self.index.get(token, ()))

    def candidates(self, q_tokens, rare=RARE_TOKENS):
        """Chunks holding any of the quote's rarest tokens, shortest first.

        Only tokens the corpus actually holds are ranked: a token with no
        postings cannot select a chunk, and it would otherwise be the rarest
        of all and push the ones that do select out of the way.
        """
        held = [token for token in set(q_tokens) if token in self.index]
        rarest = sorted(held, key=lambda t: (self.df(t), t))[:rare]
        found = set()
        for token in rarest:
            found.update(self.index[token])
        return sorted(found, key=lambda i: (self.sizes[i], i))


def best_match(corpus, text, excerpt_words=EXCERPT_MAX_WORDS):
    """The best passage in the corpus for one quote: (score, source, excerpt)."""
    quote = normalise(text)
    q_tokens = quote.split()
    if not q_tokens:
        return Match(0.0, "", "")
    candidates = corpus.candidates(q_tokens)
    if not candidates:
        return Match(0.0, "", "")

    for position in candidates:
        at = corpus.norm[position].find(quote)
        if at >= 0:
            start = len(corpus.norm[position][:at].split())
            return Match(1.0, corpus.sources[position],
                         excerpt(corpus.texts[position], start, excerpt_words))

    n = len(q_tokens)
    widths = [w for w in window_sizes(n) if w]
    narrowest = widths[0]
    q_counts = Counter(q_tokens)
    matcher = difflib.SequenceMatcher()
    matcher.set_seq1(q_tokens)
    best_score = 0.0
    best = None

    for position in candidates:
        tokens = corpus.tokens_at(position)
        prefix = list(accumulate(map(q_counts.__contains__, tokens), initial=0))
        if 2 * prefix[-1] <= best_score * (n + narrowest):
            continue
        for width in widths:
            if width > len(tokens):
                break
            bound = best_score * (n + width)
            for start in range(0, len(tokens) - width + 1):
                if 2 * (prefix[start + width] - prefix[start]) <= bound:
                    continue
                matcher.set_seq2(tokens[start:start + width])
                score = matcher.ratio()
                if score > best_score:
                    best_score = score
                    best = (position, start)
                    bound = best_score * (n + width)

    if best is None:
        return Match(0.0, "", "")
    position, start = best
    return Match(best_score, corpus.sources[position],
                 excerpt(corpus.texts[position], start, excerpt_words))


def match_status(score):
    if score >= VERIFIED_AT:
        return "verified"
    if score >= LIKELY_AT:
        return "likely_translation"
    return "not_found"


def decide(web_status, status):
    if status == "verified":
        return "verified"
    if status == "likely_translation":
        return "likely_translation"
    return NOT_FOUND_FINAL.get(web_status, UNVERIFIED_DEFAULT)


def verify_row(corpus, row, excerpt_words=EXCERPT_MAX_WORDS):
    match = best_match(corpus, row.get("text", ""), excerpt_words)
    score = round(match.score, 2)
    status = match_status(score)
    verified = dict(row)
    verified.update({
        "match_status": status,
        "score": score,
        "matched_source": match.source,
        "matched_excerpt": match.excerpt,
        "final": decide(str(row.get("web_status", "")), status),
    })
    return verified


def verify_all(corpus, rows, excerpt_words=EXCERPT_MAX_WORDS):
    return [verify_row(corpus, row, excerpt_words) for row in rows]


def load_rows(path):
    """JSONL rows, plus the line numbers that would not parse."""
    rows = []
    skipped = []
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                skipped.append((path, number))
                continue
            if isinstance(row, dict):
                rows.append(row)
            else:
                skipped.append((path, number))
    return rows, skipped


def load_corpus(paths):
    chunks = []
    skipped = []
    for path in paths:
        rows, bad = load_rows(path)
        chunks.extend(rows)
        skipped.extend(bad)
    return chunks, skipped


def statuses(results):
    seen = [status for status in WEB_STATUSES
            if any(row.get("web_status") == status for row in results)]
    extra = sorted({str(row.get("web_status", "")) for row in results}
                   - set(WEB_STATUSES))
    return seen + extra


def count_table(results):
    """final counts keyed by web_status, every web_status and final that occurs
    present, and a "total" row and column."""
    web = statuses(results)
    finals = [final for final in FINALS
              if any(row.get("final") == final for row in results)]
    table = Counter((row.get("web_status", ""), row.get("final", ""))
                    for row in results)
    return web, finals, table


def format_table(results):
    web, finals, table = count_table(results)
    headers = ["web_status"] + finals + ["total"]
    rows = []
    for status in web:
        cells = [table.get((status, final), 0) for final in finals]
        rows.append([str(status)] + [str(cell) for cell in cells]
                    + [str(sum(cells))])
    rows.append(["total"]
                + [str(sum(table.get((status, final), 0) for status in web))
                   for final in finals]
                + [str(len(results))])
    widths = [max([len(headers[column])]
                  + [len(row[column]) for row in rows])
              for column in range(len(headers))]
    return "\n".join(
        "  ".join(cell.ljust(widths[column]) for column, cell in enumerate(row))
        for row in [headers] + rows)


def preview(text, limit=PREVIEW_CHARS):
    return " ".join(str(text).split())[:limit]


def report(results, label=None):
    contradictions = [row for row in results
                      if row.get("web_status") == CONTRADICTED_WEB_STATUS
                      and row.get("match_status") == "verified"]
    flagged = [row for row in results if row.get("final") in FLAGGED_FINALS]
    lines = []
    if label:
        lines.append(label)
    lines.append("")
    lines.append("final by web_status")
    lines.append(format_table(results))
    lines.append("")
    lines.append(f"misattributed / unverified  ({len(flagged)} of {len(results)})")
    for row in flagged:
        lines.append("  [{status}] {score:.2f}  {source}  {text}".format(
            status=row.get("web_status", ""), score=row.get("score", 0.0),
            source=row.get("claimed_source") or "-",
            text=preview(row.get("text", ""))))
    lines.append("")
    if contradictions:
        lines.append(f"web says misattributed, corpus verifies the words  "
                     f"({len(contradictions)})")
        for row in contradictions:
            lines.append("  {score:.2f}  {source}  {text}".format(
                score=row.get("score", 0.0),
                source=row.get("matched_source") or "-",
                text=preview(row.get("text", ""))))
    else:
        lines.append("web says misattributed, corpus verifies the words  (0)")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--quotes", default="data/primary_quotes_web.jsonl")
    ap.add_argument("--corpus", nargs="+",
                    default=["data/source_chunks.jsonl",
                             "data/primary_chunks.jsonl"])
    ap.add_argument("--out", default="data/primary_quotes.jsonl")
    args = ap.parse_args()

    quotes, bad_quotes = load_rows(args.quotes)
    chunks, bad_chunks = load_corpus(args.corpus)
    corpus = Corpus(chunks)
    results = verify_all(corpus, quotes)

    with open(args.out, "w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts = Counter(row["match_status"] for row in results)
    print(report(results, label="verify_quotes  quotes {n}  corpus chunks {c}  "
                                "->  {out}".format(n=len(quotes), c=len(corpus),
                                                   out=args.out)))
    print()
    print("match_status  " + "   ".join(
        f"{status} {counts[status]}" for status in MATCH_STATUSES))
    if bad_quotes or bad_chunks:
        print("skipped malformed lines  "
              f"quotes {len(bad_quotes)}  corpus {len(bad_chunks)}", file=sys.stderr)


if __name__ == "__main__":
    main()
