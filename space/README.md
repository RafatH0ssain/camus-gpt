---
title: CamusGPT
emoji: 🖊️
colorFrom: gray
colorTo: blue
sdk: gradio
app_file: app.py
pinned: false
short_description: A fictional, RAG-grounded AI persona of Albert Camus.
---

# CamusGPT (Space)

A fine-tuned Llama-3.1-8B persona of Albert Camus, served on **ZeroGPU** via transformers,
grounded in his life and writing by a retrieval (RAG) layer that runs on CPU.

**This is a fictional AI persona for education and conversation — not the real Albert Camus,
and not professional advice.** If you are in crisis, call or text **988** (US & Canada) or
find a local helpline at **findahelpline.com**.

## Configuration
- **Hardware:** ZeroGPU, size `large`.
- **Variables** (Settings → Variables): `MODEL_REPO` (safetensors model repo),
  `KB_REPO` (dataset repo with `camus_kb_full.jsonl` + `camus_kb_vectors.npy`).
  Optional tuning: `CURATED_BOOST`, `TOP_K`, `MAX_TOKENS`, `MAX_TURNS`, `CONFIDENT`, `RELEVANT`.
- The index (`camus_kb_vectors.npy`) must be built with the **llama.cpp** nomic embedder
  (`kb/embed_kb_llamacpp.py`) so it matches the query embedder used here.

See the project repository's `docs/` for the full pipeline and deployment write-ups.
