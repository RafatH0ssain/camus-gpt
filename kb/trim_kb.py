#!/usr/bin/env python3
"""
trim_kb.py — shrink the KB by removing near-duplicate / redundant extracted facts,
keeping ALL hand-verified curated facts. Reuses the existing vectors, so NO re-embedding
is needed for the trimmed LOCAL index — it just selects the surviving rows.

  python trim_kb.py                 # default dedup cosine 0.88
  python trim_kb.py --dedup 0.85    # more aggressive (smaller KB)
  python trim_kb.py --dedup 0.92    # gentler (larger KB)

Outputs:
  camus_kb_trimmed.jsonl
  camus_kb_vectors_trimmed.npy      (matching local index; for quick local testing)

To test locally: back up camus_kb_full.jsonl + camus_kb_vectors.npy, copy the two trimmed
files over those canonical names, then run `camus --debug`. To deploy: rebuild the Space's
vectors from the trimmed jsonl with embed_kb_llamacpp.py and upload both to the dataset repo.
"""
import argparse, json
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb",  default="camus_kb_full.jsonl")
    ap.add_argument("--vec", default="camus_kb_vectors.npy")
    ap.add_argument("--out-kb",  default="camus_kb_trimmed.jsonl")
    ap.add_argument("--out-vec", default="camus_kb_vectors_trimmed.npy")
    ap.add_argument("--dedup", type=float, default=0.88, help="drop an extracted fact this close to a kept one")
    ap.add_argument("--min-len", type=int, default=40, help="drop extracted facts shorter than this")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.kb, encoding="utf-8") if l.strip()]
    V = np.load(args.vec).astype(np.float32)
    if len(rows) != len(V):
        raise SystemExit(f"KB ({len(rows)}) and vectors ({len(V)}) out of sync — rebuild the index first.")
    N, dim = V.shape
    is_cur = np.array([r.get("source") == "curated (verified)" for r in rows])
    print(f"Loaded {N} entries  ({is_cur.sum()} curated, {N-is_cur.sum()} extracted)")

    # process curated first (they anchor and absorb extracted near-dups), then extracted longest-first
    ext = [i for i in range(N) if not is_cur[i]]
    ext.sort(key=lambda i: -len(rows[i]["text"]))
    order = [i for i in range(N) if is_cur[i]] + ext

    kept_mat = np.zeros((N, dim), np.float32)   # preallocated kept vectors
    kept_idx = []
    nkept = 0
    dropped_dup = dropped_short = 0
    for c, i in enumerate(order):
        r = rows[i]
        if not is_cur[i]:
            if len(r["text"]) < args.min_len:
                dropped_short += 1; continue
            if nkept:
                sims = kept_mat[:nkept] @ V[i]
                if float(sims.max()) >= args.dedup:
                    dropped_dup += 1; continue
        kept_mat[nkept] = V[i]; kept_idx.append(i); nkept += 1
        if c % 2000 == 0:
            print(f"  scanned {c}/{N}  kept {nkept}")

    # preserve original file order among survivors, reassign ids
    kept_idx.sort()
    out_rows = []
    for newid, i in enumerate(kept_idx):
        r = dict(rows[i]); r["id"] = newid; out_rows.append(r)
    out_vec = V[kept_idx]

    with open(args.out_kb, "w", encoding="utf-8") as f:
        for r in out_rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    np.save(args.out_vec, out_vec)

    print(f"\n✅ kept {len(kept_idx)} / {N}  ({len(kept_idx)/N:.0%})")
    print(f"   dropped: {dropped_dup} near-duplicates, {dropped_short} too-short")
    print(f"   curated kept: {sum(is_cur[i] for i in kept_idx)} (all curated retained)")
    print(f"   -> {args.out_kb}  +  {args.out_vec}")

if __name__ == "__main__":
    main()
