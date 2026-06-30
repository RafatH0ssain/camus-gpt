#!/usr/bin/env python3
"""
merge_kb.py — unify the hand-verified KB + extracted entries into one KB.

Inputs:
  --curated   camus_kb.jsonl        (your 62 hand-verified facts: {id,topic,text})
  --extracted data/kb_extracted.jsonl (from extract_kb.py: {type,text,quote,source,date,chunk_id})
Output:
  camus_kb_full.jsonl   unified rows {id,type,text,source,date,quote}

Dedup: normalized-text exact match (lowercase, punctuation/space-stripped). The curated
facts win ties and are tagged source='curated (verified)'. Semantic near-dupes may remain;
retrieval top-k tolerates them. Delete camus_kb_vectors.npy after merging to force re-index.
"""
import argparse, json, os, re

def norm(t): return re.sub(r"[^a-z0-9 ]","", t.lower()); 
def key(t):  return re.sub(r"\s+"," ", norm(t)).strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curated", default="./camus_kb.jsonl")
    ap.add_argument("--extracted", default="./data/kb_extracted.jsonl")
    ap.add_argument("--out", default="./camus_kb_full.jsonl")
    args = ap.parse_args()

    rows, seen = [], set()
    def add(r):
        k = key(r["text"])
        if len(k) < 8 or k in seen: return False
        seen.add(k); rows.append(r); return True

    n_cur = 0
    if os.path.exists(args.curated):
        for l in open(args.curated, encoding="utf-8"):
            if not l.strip(): continue
            d = json.loads(l)
            if add({"type":"fact","text":d["text"],"source":"curated (verified)","date":"","quote":None}):
                n_cur += 1

    n_ext = 0; by_src = {}
    if os.path.exists(args.extracted):
        for l in open(args.extracted, encoding="utf-8"):
            if not l.strip(): continue
            d = json.loads(l)
            if add({"type":d.get("type","fact"),"text":d["text"],"source":d.get("source","?"),
                    "date":d.get("date",""),"quote":d.get("quote")}):
                n_ext += 1; by_src[d.get("source","?")] = by_src.get(d.get("source","?"),0)+1

    for i,r in enumerate(rows): r["id"] = i
    with open(args.out,"w",encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r,ensure_ascii=False)+"\n")

    nf = sum(1 for r in rows if r["type"]=="fact"); nv = sum(1 for r in rows if r["type"]=="view")
    print(f"✅ {len(rows)} unified entries -> {args.out}")
    print(f"   curated kept: {n_cur}   extracted kept: {n_ext}   (dupes dropped on the way)")
    print(f"   by type: fact={nf}  view={nv}")
    if by_src:
        print("   extracted by source:")
        for s,c in sorted(by_src.items(), key=lambda x:-x[1]): print(f"     {c:5d}  {s}")
    print("\n   NEXT: delete camus_kb_vectors.npy, then run camus_rag.py to re-index.")

if __name__=="__main__": main()
