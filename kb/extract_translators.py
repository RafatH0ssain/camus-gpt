#!/usr/bin/env python3
"""
extract_translators.py — read the front matter of every PDF in
data/primary_inventory.jsonl and record who translated it, when, and for whom.

Metadata only: the output (kb/translators.json) holds names, dates, publishers
and a page reference. No book text is ever written to disk.

The front matter is too inconsistent for a single regex — covers are scanned
images, title pages credit the editor rather than the translator, essay
collections credit several translators in different places, and one scan is
OCR'd badly enough that "TRANSLATED FROM THE FRENCH" comes out as
"TRANS1,ATE, 1) FROM TIIE FR ENCE1". So the evidence is curated below and this
script's job is to VERIFY it: every translator name is re-read from the page it
is attributed to (comparison ignores case, spacing and punctuation, which is
what defeats the bad OCR). A name that does not appear on the cited page is a
hard error, so the file cannot drift away from the PDFs.

Anything not found is recorded as "unknown". Nothing is inferred from outside
knowledge of who usually translated Camus.

    python kb/extract_translators.py            # verify, then write kb/translators.json
    python kb/extract_translators.py --check    # verify only
"""
import argparse
import json
import os
import re
import sys

INVENTORY = "data/primary_inventory.jsonl"
OUT = "kb/translators.json"
FRONT_MATTER_PAGES = 8

UNKNOWN = "unknown"

# file -> provenance, as read off the front matter. "evidence" is the PDF page
# (1-based) the attribution was read from; "where" says which page each element
# came from when they differ.
CURATED = {
    "raw_books/1935_1942_notebooks.pdf": {
        "translators": ["Philip Thody"],
        "year": "1963",
        "publisher": "Alfred A. Knopf (Borzoi); Hamish Hamilton Ltd. (UK)",
        "evidence": "p. 9",
    },
    "raw_books/1942_1951_notebooks.pdf": {
        "translators": ["Justin O'Brien"],
        "year": "1965",
        "publisher": "Alfred A. Knopf (Borzoi)",
        "evidence": "p. 9",
    },
    "raw_books/1951_1959_notebooks.pdf": {
        # Title page credits Bloom; the volume carries no copyright page, so no
        # year or publisher is printed anywhere in the file.
        "translators": ["Ryan Bloom"],
        "year": UNKNOWN,
        "publisher": UNKNOWN,
        "evidence": "p. 5",
    },
    "raw_books/1937_1957_lyrical_and_critical_essays.pdf": {
        # Three, on the title page and in the copyright page: Kennedy translated
        # the book, Thody edited it and made the GB edition, O'Brien is credited
        # for the four essays reprinted from The Myth of Sisyphus.
        "translators": ["Ellen Conroy Kennedy", "Philip Thody", "Justin O'Brien"],
        "year": "1968",
        "publisher": "Alfred A. Knopf (Vintage Books edition)",
        "evidence": "p. 4 (Kennedy, Thody); p. 5 (O'Brien)",
    },
    "raw_books/1939_1958_algerianChronicles_writing.pdf": {
        "translators": ["Arthur Goldhammer"],
        "year": "2013",
        "publisher": "The Belknap Press of Harvard University Press",
        "evidence": "p. 3 (translator); p. 4 (year)",
    },
    "raw_books/1942_theMythOfSisyphus_novel.pdf": {
        # The copyright page names Knopf and 1955 but no translator.
        "translators": [UNKNOWN],
        "year": "1955",
        "publisher": "Alfred A. Knopf (Vintage Books edition)",
        "evidence": "p. 5 (year, publisher; no translator credited)",
    },
    "raw_books/1942_theStranger_book.pdf": {
        # Page 1 is an image-only cover; the text layer starts at Part One on
        # page 2, so no front matter survived the conversion.
        "translators": [UNKNOWN],
        "year": UNKNOWN,
        "publisher": UNKNOWN,
        "evidence": "not stated in pp. 1-8",
    },
    "raw_books/1944_1947_camusAtCombat_journal.pdf": {
        "translators": ["Arthur Goldhammer"],
        "year": "2006",
        "publisher": "Princeton University Press",
        "evidence": "p. 7 (translator, publisher); p. 8 (year)",
    },
    "raw_books/1945_1951_1959_interviews.pdf": {
        # 9-page LibreOffice export that opens straight into the text.
        "translators": [UNKNOWN],
        "year": UNKNOWN,
        "publisher": UNKNOWN,
        "evidence": "not stated in pp. 1-8",
    },
    "raw_books/1946_1949_americanJournals.pdf": {
        "translators": ["Hugh Levick"],
        "year": "1987",
        "publisher": "Paragon House Publishers",
        "evidence": "p. 5 (translator, publisher); p. 6 (year)",
    },
    "raw_books/1947_thePlague_book.pdf": {
        "translators": ["Stuart Gilbert"],
        "year": "1948",
        "publisher": "Random House (The Modern Library)",
        "evidence": "p. 1",
    },
    "raw_books/1949_theJustAssassins_play.pdf": {
        "translators": ["Suzanne M. Saunders"],
        "year": UNKNOWN,
        "publisher": UNKNOWN,
        "evidence": "p. 1",
    },
    "raw_books/1951_theRebel_book.pdf": {
        "translators": ["Anthony Bower"],
        "year": UNKNOWN,
        "publisher": UNKNOWN,
        "evidence": "p. 1",
    },
    "raw_books/1956_theFall_book.pdf": {
        "translators": [UNKNOWN],
        "year": UNKNOWN,
        "publisher": UNKNOWN,
        "evidence": "not stated in pp. 1-8",
    },
    "raw_books/1957_exileAndTheKingdom_short_stories.pdf": {
        "translators": ["Justin O'Brien"],
        "year": "1957",
        "publisher": "Alfred A. Knopf (Vintage Books edition)",
        "evidence": "p. 1",
    },
    "raw_books/1958_theInvincibleSummer_book.pdf": {
        "translators": ["Herma Briffault"],
        "year": "1958",
        "publisher": "George Braziller, Inc.",
        "evidence": "p. 2",
    },
    "raw_books/1971_aHappyDeath_book.pdf": {
        "translators": [UNKNOWN],
        "year": "1972",
        "publisher": "Alfred A. Knopf",
        "evidence": "p. 3 (year, publisher; no translator credited)",
    },
    "raw_books/1994_theFirstMan_novel.pdf": {
        "translators": ["David Hapgood"],
        "year": "1995",
        "publisher": "Alfred A. Knopf",
        "evidence": "p. 6",
    },
    "raw_books/lectures_speeches_letters.pdf": {
        # No "translated by" line and no CIP translator entry; the introduction
        # is signed by Justin O'Brien, who signs most of his Camus translations
        # the same way. Recorded as found rather than inferred from that.
        "translators": ["Justin O'Brien"],
        "year": "1960",
        "publisher": "Alfred A. Knopf",
        "evidence": "p. 6 (introduction signed JUSTIN O'BRIEN); p. 3 (year, publisher)",
    },
    "sources/algerian_chronicles.pdf": {
        "translators": ["Arthur Goldhammer"],
        "year": "2013",
        "publisher": "The Belknap Press of Harvard University Press",
        "evidence": "p. 4 (translator, publisher); p. 5 (year)",
    },
    "sources/camus_at_combat.pdf": {
        "translators": ["Arthur Goldhammer"],
        "year": "2006",
        "publisher": "Princeton University Press",
        "evidence": "p. 7 (translator, publisher); p. 8 (year)",
    },
    "sources/camus_kb.jsonl": {
        "translators": [UNKNOWN],
        "year": UNKNOWN,
        "publisher": UNKNOWN,
        "evidence": "not a PDF (derived KB, no front matter)",
    },
    "sources/lottman_biography.pdf": {
        # A biography ABOUT Camus, not a translation of him; no translator credited.
        "translators": [UNKNOWN],
        "year": "1979",
        "publisher": "Gingko Press Inc. (first published by Doubleday & Company)",
        "evidence": "p. 7 (year, publisher); no translator credited",
    },
    "sources/lyrical_critical_essays.pdf": {
        "translators": ["Ellen Conroy Kennedy", "Philip Thody", "Justin O'Brien"],
        "year": "1968",
        "publisher": "Alfred A. Knopf (Vintage Books edition)",
        "evidence": "p. 4 (Kennedy, Thody); p. 5 (O'Brien)",
    },
    "sources/notebooks_1933_1959.pdf": {
        "translators": ["Ryan Bloom"],
        "year": "2025",
        "publisher": "University of Chicago Press",
        "evidence": "p. 6 (translator, publisher); p. 7 (year)",
    },
    "sources/resistance_rebellion_death.pdf": {
        "translators": ["Justin O'Brien"],
        "year": "1960",
        "publisher": "Alfred A. Knopf",
        "evidence": "p. 6 (introduction signed JUSTIN O'BRIEN); p. 3 (year, publisher)",
    },
    "sources/todd_a_life.pdf": {
        "translators": ["Benjamin Ivry"],
        "year": "1998",
        "publisher": "Vintage (Random House)",
        "evidence": "p. 5 (translator); p. 6 (year, publisher)",
    },
}


def norm(s):
    """Letters and digits only, lowercased. Kills spacing, punctuation and the
    substitutions a bad OCR makes inside a name."""
    return re.sub(r"[^a-z0-9]", "", s.lower())


def page_text(doc, page_no):
    if page_no < 1 or page_no > len(doc):
        return ""
    return doc[page_no - 1].get_text()


def evidence_pages(evidence):
    """Page numbers cited in an evidence string, e.g. 'p. 4 (Kennedy); p. 5'."""
    return [int(n) for n in re.findall(r"p\.\s*(\d+)", evidence)]


def verify(path, entry, doc=None):
    """Return a list of complaints; empty means the entry is supported."""
    problems = []
    if not path.lower().endswith(".pdf"):
        return problems

    import pymupdf

    close = doc is None
    if close:
        doc = pymupdf.open(path)

    try:
        pages = evidence_pages(entry["evidence"])
        if not pages:
            return problems
        # Bad OCR turns a name into fragments, so compare against the whole front
        # matter span as well as the exact cited page: the cited page is where it
        # was read, the span is the tolerance that lets a garbled name through.
        span = "\n".join(page_text(doc, p) for p in range(1, FRONT_MATTER_PAGES + 1))
        span_n = norm(span)
        for name in entry["translators"]:
            if name == UNKNOWN:
                continue
            if any(norm(name) in norm(page_text(doc, p)) for p in pages):
                continue
            if norm(name) in span_n:
                continue
            problems.append(f"{path}: {name!r} not found on cited page(s) "
                            f"{pages} or in pp. 1-{FRONT_MATTER_PAGES}")
    finally:
        if close:
            doc.close()
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--check", action="store_true", help="verify only, write nothing")
    args = ap.parse_args()

    import pymupdf

    rows = [json.loads(line) for line in open(INVENTORY, encoding="utf-8") if line.strip()]

    missing = [r["file"] for r in rows if r["file"] not in CURATED]
    if missing:
        print("no curated entry for:", *missing, sep="\n  ", file=sys.stderr)
        return 1

    records, problems, verified = {}, [], 0
    for row in rows:
        path, source = row["file"], row["title"]
        entry = CURATED[path]
        problems.extend(verify(path, entry))
        if path.lower().endswith(".pdf"):
            verified += 1
        records[source] = {
            "file": path,
            "translators": list(entry["translators"]),
            "year": entry["year"],
            "publisher": entry["publisher"],
            "evidence": entry["evidence"],
        }

    if problems:
        print("EVIDENCE NOT CONFIRMED:", *problems, sep="\n  ", file=sys.stderr)
        return 1
    print(f"verified {verified} PDFs against their cited pages; "
          f"{len(records)} manifest rows")

    if not args.check:
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        with open(OUT, "w", encoding="utf-8") as fh:
            json.dump(records, fh, indent=2, ensure_ascii=False, sort_keys=True)
            fh.write("\n")
        print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
