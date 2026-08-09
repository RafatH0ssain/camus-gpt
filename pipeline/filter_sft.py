#!/usr/bin/env python3
"""
filter_sft.py — remove bad sources / corrupted rows from camus_sft.jsonl
WITHOUT re-running any batch. Use after audit_sources.py tells you which book
files are OCR garbage or academic commentary.

Drops, from BOTH camus_sft.jsonl and chunks.jsonl (kept consistent):
  * every row whose `source` you name in --drop-source
  * (unless --keep-bad-rows) straggler chunks in otherwise-ok sources whose
    essayist text still trips the OCR/academic detectors

Backs up the originals to *.prefilter.jsonl first.

    python audit_sources.py --show 3                 # see which sources are bad
    python filter_sft.py --drop-source "caligula play" "camus companion"
    python verify_sft.py                             # re-confirm, then run style
"""
import argparse
import json
import shutil
import config as C
from audit_sources import is_ocr, is_acad


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drop-source", nargs="*", default=[],
                    help="exact source names to remove entirely (quote them)")
    ap.add_argument("--keep-bad-rows", action="store_true",
                    help="do NOT auto-remove straggler OCR/academic rows from kept sources")
    args = ap.parse_args()
    drop = set(args.drop_source)

    sft = load(C.SFT_OUT)
    # straggler corrupted chunks in sources we're otherwise keeping
    bad_ids = set()
    if not args.keep_bad_rows:
        for r in sft:
            if r["source"] in drop:
                continue
            if r["type"] == "essayist" and (is_ocr(r["response"]) or is_acad(r["response"])):
                bad_ids.add(r["src_id"])

    def keep(r):
        return r.get("source") not in drop and r.get("src_id", r.get("id")) not in bad_ids

    sft_kept = [r for r in sft if keep(r)]
    chunks = load(C.CHUNKS)
    chunks_kept = [c for c in chunks
                   if c.get("source") not in drop and c["id"] not in bad_ids]

    shutil.copy(C.SFT_OUT, C.SFT_OUT.replace(".jsonl", ".prefilter.jsonl"))
    shutil.copy(C.CHUNKS, C.CHUNKS.replace(".jsonl", ".prefilter.jsonl"))
    C.write_jsonl(C.SFT_OUT, sft_kept)
    C.write_jsonl(C.CHUNKS, chunks_kept)

    print(f"SFT:    {len(sft)} -> {len(sft_kept)}  (removed {len(sft) - len(sft_kept)})")
    print(f"chunks: {len(chunks)} -> {len(chunks_kept)}  (removed {len(chunks) - len(chunks_kept)})")
    print(f"  dropped sources: {sorted(drop) or 'none'}")
    print(f"  straggler chunks dropped: {len(bad_ids)}")
    print(f"  backups: {C.SFT_OUT.replace('.jsonl', '.prefilter.jsonl')}, "
          f"{C.CHUNKS.replace('.jsonl', '.prefilter.jsonl')}")
    print("  next: python verify_sft.py  &&  python inspect_data.py ./data/camus_sft.jsonl")


if __name__ == "__main__":
    main()
