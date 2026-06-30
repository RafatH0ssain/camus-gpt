#!/usr/bin/env python3
"""
build_index.py — robust one-time embedding of camus_kb_full.jsonl -> camus_kb_vectors.npy

Fixes the "ran 4 hours with no output" problem:
  • BATCHED via Ollama /api/embed (many inputs per request, not one HTTP call each)
  • keep_alive so the embed model stays loaded (no load/unload thrash between calls)
  • PROGRESS line every batch (count, rate, ETA) so 'slow' is distinguishable from 'stuck'
  • RESUMABLE: partial vectors cached every flush; re-run continues where it stopped
  • per-request timeout + retries so a stutter doesn't kill the run

SETUP  ollama pull nomic-embed-text ; pip install numpy requests
RUN    python build_index.py
       python build_index.py --batch 64 --kb ./camus_kb_full.jsonl
"""
import argparse, json, os, time, hashlib
import numpy as np
import requests

OLLAMA      = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
KEEP_ALIVE  = "15m"          # keep the model resident -> no reload thrash
TIMEOUT     = 120
RETRIES     = 4

def embed_batch(texts, prefix="search_document: "):
    payload = {"model": EMBED_MODEL, "input": [prefix + t for t in texts], "keep_alive": KEEP_ALIVE}
    last = None
    for attempt in range(RETRIES):
        try:
            r = requests.post(f"{OLLAMA}/api/embed", json=payload, timeout=TIMEOUT)
            r.raise_for_status()
            embs = r.json().get("embeddings")
            if not embs or len(embs) != len(texts):
                raise ValueError(f"expected {len(texts)} vectors, got {len(embs) if embs else 0}")
            a = np.array(embs, dtype=np.float32)
            return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
        except Exception as e:
            last = e
            wait = 2 ** attempt
            print(f"    [retry {attempt+1}/{RETRIES} in {wait}s: {type(e).__name__}: {str(e)[:80]}]")
            time.sleep(wait)
    raise RuntimeError(f"embed_batch failed after {RETRIES} tries: {last}")

def fingerprint(rows):
    h = hashlib.md5()
    h.update(str(len(rows)).encode())
    if rows:
        h.update(rows[0]["text"].encode("utf-8", "ignore"))
        h.update(rows[-1]["text"].encode("utf-8", "ignore"))
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", default="./camus_kb_full.jsonl")
    ap.add_argument("--out", default="./camus_kb_vectors.npy")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.kb, encoding="utf-8") if l.strip()]
    N = len(rows)
    fp = fingerprint(rows)
    part = args.out + ".partial.npy"
    meta = args.out + ".partial.json"
    print(f"KB entries: {N}")

    # resume?
    done = 0; vecs = None
    if os.path.exists(part) and os.path.exists(meta):
        m = json.load(open(meta))
        if m.get("fp") == fp and m.get("N") == N:
            vecs = np.load(part); done = len(vecs)
            print(f"Resuming from cached {done}/{N} vectors.")
        else:
            print("KB changed since partial cache — starting fresh.")
    if vecs is None:
        vecs = np.zeros((0, 0), dtype=np.float32)

    t0 = time.time()
    buf = list(vecs) if done else []
    i = done
    while i < N:
        batch = [r["text"] for r in rows[i:i+args.batch]]
        out = embed_batch(batch)
        buf.extend(out)
        i += len(batch)
        # flush partial every batch (cheap insurance)
        arr = np.vstack(buf).astype(np.float32)
        np.save(part, arr)
        json.dump({"fp": fp, "N": N}, open(meta, "w"))
        rate = i / max(time.time() - t0, 1e-6)
        eta  = (N - i) / max(rate, 1e-6)
        print(f"  {i}/{N} ({i/N:.0%})  {rate:.1f} entries/s  ETA {eta/60:.1f} min")

    np.save(args.out, np.vstack(buf).astype(np.float32))
    os.remove(part); os.remove(meta)
    print(f"\n✅ saved {N} vectors -> {args.out}  in {(time.time()-t0)/60:.1f} min")
    print("   now run: python camus_rag.py")

if __name__ == "__main__":
    main()
