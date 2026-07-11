#!/usr/bin/env python3
"""
diagnose_retrieval.py v2 — measure how a query scores across every retrieval channel.

For the final top-N candidates (and any --contains-matched anchor) it reports: dense
cosine (the confidence/gate basis), BM25 score, per-channel rank, fused (RRF) rank,
cross-encoder score, and the anchor's stored-vs-fresh vector alignment — so
"why didn't my anchor win?" is answered with numbers, not guesses.

Imports retrieval pieces from camus_rag.py so the diagnosis and the chat share ONE
implementation (constants, tokenizer, fusion, reranker).

Run from repo root:
  python rag/diagnose_retrieval.py --query "name your works" --contains "Name your works"
  python rag/diagnose_retrieval.py --query "do you have a cat" --top 20
  python rag/diagnose_retrieval.py --query "..." --no-rerank      (fused order only)
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import camus_rag as cr


def channel_ranks(query, facts, vecs, bm25):
    cos = vecs @ cr.embed(query, prefix="search_query: ")
    n = len(facts)
    basis = cos + np.array([cr.CURATED_BOOST if f.get("source") == "curated (verified)" else 0.0
                            for f in facts], dtype=np.float32)
    d_rank = np.empty(n, dtype=np.int64); d_rank[np.argsort(-basis)] = np.arange(n)
    if bm25 is not None:
        bs = np.asarray(bm25.get_scores(cr._tok(query)), dtype=np.float32)
        b_rank = np.empty(n, dtype=np.int64); b_rank[np.argsort(-bs)] = np.arange(n)
        rrf = 1.0 / (cr.RRF_K + d_rank) + 1.0 / (cr.RRF_K + b_rank)
    else:
        bs = b_rank = None
        rrf = 1.0 / (cr.RRF_K + d_rank)
    f_rank = np.empty(n, dtype=np.int64); f_rank[np.argsort(-rrf)] = np.arange(n)
    return cos, bs, d_rank, b_rank, rrf, f_rank


def row(i, cos, bs, d_rank, b_rank, ce_by_idx, final_pos, facts, floor_set, _fscore):
    star = "*" if facts[i].get("source") == "curated (verified)" else " "
    fin = f"#{final_pos[i]+1:<3}" if i in final_pos else "  - "
    ce = f"{ce_by_idx[i]:+6.2f}" if i in ce_by_idx else "    --"
    bm = f"{bs[i]:6.1f}(r{b_rank[i]:<5})" if bs is not None else "      --     "
    fl = "F" if i in floor_set else " "
    f  = _fscore(i) if i in final_pos else float(cos[i])
    return (f"  {fin}{star}{fl} ce={ce} cos={cos[i]:.3f}(r{d_rank[i]:<5}) bm25={bm} f={f:.3f} "
            f"[{facts[i].get('type','?'):4s}|{facts[i].get('source','?')[:20]}] {facts[i]['text'][:54]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True)
    ap.add_argument("--contains", help="substring locating anchor entries to track")
    ap.add_argument("--top", type=int, default=15, help="final candidates to print")
    ap.add_argument("--no-rerank", action="store_true")
    args = ap.parse_args()

    facts, vecs = cr.load_kb()
    bm25 = cr.build_bm25(facts)
    ce = None if args.no_rerank else cr.load_reranker()

    cos, bs, d_rank, b_rank, rrf, f_rank = channel_ranks(args.query, facts, vecs, bm25)
    d_order = np.argsort(-(cos + np.array(
        [cr.CURATED_BOOST if f.get("source") == "curated (verified)" else 0.0 for f in facts],
        dtype=np.float32)))
    pool = list(dict.fromkeys(list(np.argsort(-rrf)[:cr.FUSE_K]) + list(d_order[:cr.DENSE_FLOOR])))
    pool = [int(i) for i in pool]
    floor_set = set(int(i) for i in d_order[:cr.DENSE_FLOOR])
    if ce is not None:
        scores = np.asarray(ce.predict([(args.query, facts[i]["text"]) for i in pool]),
                            dtype=np.float32)
        ce_by_idx = {int(i): float(s) for i, s in zip(pool, scores)}
    else:
        ce_by_idx = {}
    def _fscore(i):
        c = float(cos[i])
        return c + (cr.CE_WEIGHT / (1.0 + np.exp(-ce_by_idx[i])) if i in ce_by_idx else 0.0)
    ranked = sorted(pool, key=lambda i: -_fscore(i))
    final_pos = {int(i): p for p, i in enumerate(ranked)}

    print(f'\nquery: "{args.query}"   (boost={cr.CURATED_BOOST}, fuse={cr.FUSE_K}, '
          f'top_k={cr.TOP_K}, rerank={"on" if ce is not None else "off"})')
    print(f"\n=== final order (top {args.top}; '#n' final, F=dense-floor, gate: cos>={cr.THRESHOLD} "
          f"or ce>={cr.THRESHOLD} floor)")
    for i in ranked[:args.top]:
        print(row(int(i), cos, bs, d_rank, b_rank, ce_by_idx, final_pos, facts, floor_set, _fscore))

    if args.contains:
        matches = [j for j, f in enumerate(facts) if args.contains.lower() in f["text"].lower()]
        print(f"\n=== anchors matching --contains ({len(matches)}) ===")
        for j in matches[:10]:
            print(row(j, cos, bs, d_rank, b_rank, ce_by_idx, final_pos, facts, floor_set, _fscore))
            fresh = cr.embed(facts[j]["text"], prefix="search_document: ")
            align = float(fresh @ vecs[j])
            fused = f_rank[j]
            in_pool = "IN rerank pool" if j in final_pos else f"NOT in pool (fused rank {fused+1} > {cr.FUSE_K})"
            in_topk = "IN final TOP_K" if j in final_pos and final_pos[j] < cr.TOP_K else "not in final TOP_K"
            print(f"        stored-vs-fresh alignment {align:.3f} "
                  f"({'ALIGNED' if align > 0.98 else 'STALE INDEX? rebuild'}); {in_pool}; {in_topk}")

if __name__ == "__main__":
    main()
