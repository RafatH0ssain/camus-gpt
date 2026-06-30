#!/usr/bin/env python3
"""
Build camus_kb_vectors.npy LOCALLY with the SAME nomic GGUF the Space queries with,
so the uploaded index matches the Space's query embedder exactly (no retrieval drift).

  pip install llama-cpp-python huggingface_hub numpy
  python embed_kb_llamacpp.py
  # then upload BOTH files together (same order!):
  huggingface-cli upload your-username/camus-kb camus_kb_full.jsonl   --repo-type dataset
  huggingface-cli upload your-username/camus-kb camus_kb_vectors.npy  --repo-type dataset

Must match the Space's EMBED_REPO / EMBED_FILE variables.
"""
import json, numpy as np
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

EMBED_REPO = "nomic-ai/nomic-embed-text-v1.5-GGUF"
EMBED_FILE = "nomic-embed-text-v1.5.f16.gguf"      # must equal the Space's EMBED_FILE
KB, OUT = "../CamusGPT/camus_kb_full.jsonl", "../CamusGPT/camus_kb_vectors.npy"

m = Llama(model_path=hf_hub_download(EMBED_REPO, EMBED_FILE), embedding=True, n_ctx=2048, verbose=False)
def emb(t):
    v = np.array(m.create_embedding("search_document: " + t)["data"][0]["embedding"], dtype=np.float32)
    if v.ndim == 2: v = v.mean(axis=0)
    return v / (np.linalg.norm(v) + 1e-9)

rows = [json.loads(l) for l in open(KB, encoding="utf-8") if l.strip()]
vecs = []
for i, r in enumerate(rows):
    vecs.append(emb(r["text"]))
    if (i + 1) % 1000 == 0: print(f"  {i+1}/{len(rows)}")
np.save(OUT, np.vstack(vecs).astype(np.float32))
print(f"saved {len(rows)} vectors -> {OUT}  (built from {KB}; upload both, unchanged)")
