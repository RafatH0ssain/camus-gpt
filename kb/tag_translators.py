#!/usr/bin/env python3
"""
tag_translators.py — add the translator(s) of each source volume to the mined
primary-text rows.

kb/translators.json is keyed by the manifest title; the mined rows are keyed by
the source label used in kb/sources_manifest.json. Those two vocabularies do not
match ("The Rebel" vs "The Rebel: An Essay on Man in Revolt"), so the join goes
through sources_manifest.json, which maps every PDF filename to both.

Rows are never dropped and no existing field is touched: the tags are appended
in file order, with the original key order preserved.

    python kb/tag_translators.py            # tag candidates in place, rows -> .tagged
    python kb/tag_translators.py --dry-run  # report the join, write nothing

Exits non-zero if any row's source cannot be joined, so an unmapped volume is
never silently tagged "unknown".
"""
import argparse
import json
import os
import sys
import tempfile
from collections import Counter

TRANSLATORS = "kb/translators.json"
INVENTORY = "data/primary_inventory.jsonl"
SOURCES = "kb/sources_manifest.json"
CANDIDATES = "data/primary_candidates.jsonl"
ROWS = "data/primary_rows.jsonl"
ROWS_TAGGED = "data/primary_rows.tagged.jsonl"


def load_join():
    """source label -> (translators, evidence), resolved through the PDF filename.

    translators.json is keyed by manifest title, sources_manifest.json by a short
    label, and the two vocabularies differ ("The Rebel" vs "The Rebel: An Essay
    on Man in Revolt"). The filename is the one identifier all three agree on, so
    the join runs filename -> inventory title -> translators record.

    One collision to know about: the inventory lists raw_books/lectures_
    speeches_letters.pdf and sources/resistance_rebellion_death.pdf under the
    same title, so the title key can hold only one. Their front matter is
    identical (Knopf, O'Brien), so nothing is lost, but the second file inherits
    the first's record rather than keeping its own.
    """
    by_title = json.load(open(TRANSLATORS, encoding="utf-8"))
    inventory = [json.loads(l) for l in open(INVENTORY, encoding="utf-8") if l.strip()]
    titles_seen = Counter(row["title"] for row in inventory)
    by_file = {}
    for row in inventory:
        record = by_title.get(row["title"])
        if record is not None:
            by_file[os.path.basename(row["file"])] = (record["translators"],
                                                      record["evidence"])
    manifest = json.load(open(SOURCES, encoding="utf-8"))
    join, orphan_files = {}, []
    for entry in manifest["sources"]:
        record = by_file.get(os.path.basename(entry["file"]))
        if record is None:
            orphan_files.append(entry["file"])
            continue
        join[entry["source"]] = record
    collisions = [t for t, n in titles_seen.items() if n > 1]
    return join, orphan_files, collisions


def tag(path, join, out_path=None):
    counts, unmapped = Counter(), Counter()
    src = path if out_path is None else path
    handle_in = open(src, encoding="utf-8")
    tmp = None
    handle_out = handle_in
    if out_path is not None:
        tmp = tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False,
                                          dir=os.path.dirname(out_path) or ".")
        handle_out = tmp
    try:
        for line in handle_in:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            source = row.get("source")
            hit = join.get(source)
            if hit is None:
                unmapped[source] += 1
                row["translators"] = None
            else:
                row["translators"] = hit[0]
                counts[source] += 1
            handle_out.write(json.dumps(row, ensure_ascii=False) + "\n")
    finally:
        handle_in.close()
        if tmp is not None:
            tmp.close()
            if out_path == path:            # in-place: swap only once complete
                os.replace(tmp.name, path)
            else:
                os.replace(tmp.name, out_path)
    return counts, unmapped


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    join, orphans, collisions = load_join()
    if orphans:
        print("sources_manifest entries with no translators.json record:", *orphans,
              sep="\n  ", file=sys.stderr)
    if collisions:
        print("inventory titles covering more than one PDF (one record serves both):",
              *collisions, sep="\n  ", file=sys.stderr)

    rc = 0
    for path, out_path in ((CANDIDATES, CANDIDATES), (ROWS, ROWS_TAGGED)):
        if args.dry_run:
            counts, unmapped = tag(path, join, tempfile.mkstemp()[1])
            print(f"{path} -> (dry run) {sum(counts.values())} rows joined")
        else:
            counts, unmapped = tag(path, join, out_path)
            print(f"{path} -> {out_path}  {sum(counts.values())} rows tagged")
        for source, n in sorted(counts.items()):
            print(f"    {n:>5}  {source}  ->  {', '.join(join[source][0])}")
        if unmapped:
            rc = 1
            print(f"    UNMAPPED ({sum(unmapped.values())} rows):", file=sys.stderr)
            for source, n in sorted(unmapped.items()):
                print(f"      {n:>5}  {source!r}", file=sys.stderr)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
