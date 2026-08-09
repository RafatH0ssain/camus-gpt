#!/usr/bin/env python3
"""
audit_sources.py — find OCR-corrupted and non-Camus (academic) source material.

verify_sft.py proves essayist targets are verbatim copies of chunks. It CANNOT
tell you a chunk is itself garbage. This scans chunks.jsonl per source and flags:

  OCR     mangled extraction (mid-word caps "caLicuLa", stray single letters
          "non i sense", " = " artifacts)
  ACAD    academic apparatus that isn't Camus's prose (citations like
          "(Sartre 1974a: 188)", "ibid", "et al.", "1968: 160")

Output is a per-source table so you can see which book FILES to drop or replace.

    python audit_sources.py
    python audit_sources.py --show 3      # print sample flagged chunks per source
"""
import argparse
import re
import json
from collections import defaultdict
import config as C

MIDCAP   = re.compile(r"\b\w*[a-z][A-Z]\w*")          # caLicuLa, imAgine
LONE     = re.compile(r"(?<=\s)[b-hj-z](?=\s)")        # single letters (not a / i)
STRAYEQ  = re.compile(r"\s=\s|[^\s]=[^\s]")
ACAD     = re.compile(r"\([A-Z][a-z]+\s+\d{4}[a-z]?[:,]\s*\d+"     # (Sartre 1974a: 188
                      r"|\b\d{4}:\s*\d+\b"                          # 1968: 160
                      r"|\b(ibid|op\.?\s?cit|et al\.?|cf\.)\b", re.I)


def is_ocr(t):
    words = max(1, len(t.split()))
    score = (len(MIDCAP.findall(t)) * 2 + len(LONE.findall(t)) + len(STRAYEQ.findall(t)))
    return score / words * 100 >= 4.0     # >=4 artifacts per 100 words

def is_acad(t):
    return bool(ACAD.search(t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", type=int, default=0, help="print N sample flagged chunks per bad source")
    args = ap.parse_args()

    chunks = [json.loads(l) for l in open(C.CHUNKS, encoding="utf-8") if l.strip()]
    by_src = defaultdict(lambda: {"n": 0, "ocr": 0, "acad": 0, "ex_ocr": [], "ex_acad": []})
    for c in chunks:
        s = by_src[c.get("source", "?")]
        s["n"] += 1
        t = c["text"]
        if is_ocr(t):
            s["ocr"] += 1
            if len(s["ex_ocr"]) < args.show: s["ex_ocr"].append(t[:160])
        if is_acad(t):
            s["acad"] += 1
            if len(s["ex_acad"]) < args.show: s["ex_acad"].append(t[:160])

    print(f"\n{'SOURCE':<34}{'chunks':>7}{'OCR%':>7}{'ACAD%':>7}   verdict")
    print("-" * 72)
    rows = sorted(by_src.items(), key=lambda kv: -(kv[1]["ocr"] + kv[1]["acad"]) / max(1, kv[1]["n"]))
    for src, s in rows:
        ocrp = s["ocr"] / s["n"] * 100
        acadp = s["acad"] / s["n"] * 100
        if ocrp >= 20 or acadp >= 20:
            v = "❌ DROP/REPLACE"
        elif ocrp >= 5 or acadp >= 5:
            v = "⚠️  inspect"
        else:
            v = "✅ ok"
        print(f"{src[:33]:<34}{s['n']:>7}{ocrp:>6.0f}%{acadp:>6.0f}%   {v}")
        for e in s["ex_ocr"]:  print(f"      OCR : {e!r}")
        for e in s["ex_acad"]: print(f"      ACAD: {e!r}")

    bad = [src for src, s in by_src.items()
           if (s["ocr"] + s["acad"]) / max(1, s["n"]) >= 0.20]
    print("\nSuggested to drop entirely:", bad or "none")
    print("Re-run with --show 3 to see example flagged chunks per source.")


if __name__ == "__main__":
    main()
