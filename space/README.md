---
title: CamusGPT
emoji: 🖊️
colorFrom: gray
colorTo: blue
sdk: gradio
app_file: app.py
pinned: false
hf_oauth: true
short_description: A fictional, RAG-grounded AI persona of Albert Camus.
---

# CamusGPT (Space)

A fine-tuned Llama-3.1-8B persona of Albert Camus, served on **ZeroGPU** via transformers and
grounded in his life and writing by a CPU-side retrieval (RAG) layer.

**Fictional AI persona for education and conversation — not the real Albert Camus, and not
professional advice.** In crisis? Call or text **988** (US & Canada) or visit **findahelpline.com**.
Conversations and basic metadata may be logged for safety and to improve the project.

## Configuration
- **Hardware:** ZeroGPU, size `large`.
- **`hf_oauth: true`** (above) enables the "Sign in with Hugging Face" button so the app can
  tell signed-in from anonymous visitors and tailor the GPU-quota message. Remove it to drop
  the button (everyone then sees the anonymous message).
- **Variables:** `MODEL_REPO`, `KB_REPO`. Optional tuning: `CURATED_BOOST`, `TOP_K`,
  `MAX_TOKENS`, `MAX_TURNS`, `CONFIDENT`, `RELEVANT`, `QUOTA_MSG_IN`, `QUOTA_MSG_OUT`.
- **Logging (optional) — Secrets:** `LOG_SHEET_ID` (the Google Sheet's id from its URL) and
  `GCP_SERVICE_ACCOUNT` (the full service-account JSON). Optional: `LOG_HASH_IP=1` to store a
  salted hash instead of the raw IP, `LOG_IP_SALT`. Share the Sheet with the service account's
  email (Editor). Logging is skipped entirely if these aren't set.
- The index (`camus_kb_vectors.npy`) must be built with the **llama.cpp** nomic embedder so it
  matches the query embedder used here.
