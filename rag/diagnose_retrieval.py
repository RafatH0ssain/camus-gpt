#!/usr/bin/env python3
"""
diagnose_retrieval.py — measure raw score, rank, and stored-vs-fresh embedding
alignment for the curated works anchor against the query "name your works".

Run from the project root with Ollama + nomic-embed-text running:
    python rag/diagnose_retrieval.py
"""
import json, numpy as np, requests

KB  = "camus_kb_full.jsonl"
VEC = "camus_kb_vectors.npy"

rows = [json.loads(l) for l in open(KB, encoding="utf-8") if l.strip()]
V = np.load(VEC)
print(f"KB rows: {len(rows)}   vectors: {V.shape}")
if len(rows) != len(V):
    print("\n*** MISALIGNED: row count != vector count. The index does not match the KB. ***")
    print("    -> rebuild: del camus_kb_vectors.npy ; python build_index.py")
    raise SystemExit

def embed(q):
    r = requests.post("http://localhost:11434/api/embed",
                      json={"model": "nomic-embed-text", "input": "search_query: " + q})
    r.raise_for_status()
    a = np.array(r.json()["embeddings"][0], dtype=np.float32)
    return a / (np.linalg.norm(a) + 1e-9)

q = embed("name your works")
sims = V @ q
order = np.argsort(-sims)

# locate the curated works anchor(s)
anchor_ids = [i for i, r in enumerate(rows) if "best-known works" in r["text"]]
print(f"\nanchor row index(es) in KB: {anchor_ids}")
for aid in anchor_ids:
    rank = int(np.where(order == aid)[0][0]) + 1
    print(f"  id {aid}: score {sims[aid]:.3f}   rank {rank} of {len(rows)}")
    print(f"     KB text : {rows[aid]['text'][:75]}")
    # SANITY: does vector[aid] actually correspond to this text? compare to a fresh embed
    doc = requests.post("http://localhost:11434/api/embed",
                        json={"model":"nomic-embed-text","input":"search_document: "+rows[aid]['text']})
    dv = np.array(doc.json()["embeddings"][0], dtype=np.float32)
    dv = dv/(np.linalg.norm(dv)+1e-9)
    align = float(V[aid] @ dv)
    print(f"     stored-vs-fresh embedding cosine: {align:.3f}  ({'ALIGNED' if align>0.98 else 'MISALIGNED!'})")

print("\nTop 10 for 'name your works':")
for i in order[:10]:
    print(f"  {sims[i]:.3f}  [{str(rows[i].get('source','?'))[:18]}] {rows[i]['text'][:64]}")
