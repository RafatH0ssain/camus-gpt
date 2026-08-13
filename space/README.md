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

> **Status: the public deployment is inactive.** The hosted Space is no longer
> running. The code here is still maintained, but it is **untested against the
> current model** — the last verified deployment served the previous build.
>
> `app.py` still targets the **previous 8B model**: `MODEL_REPO` defaults to the
> v1 weights repo, and it loads them with `AutoModelForCausalLM`. Pointing it at
> the current 12B build would need at least three changes first:
>
> - `apply_chat_template(..., return_tensors="pt", return_dict=True)` returns a
>   plain `str` on the current architecture, so `.to(model.device)` raises
>   `AttributeError`. Render to text first, then tokenize.
> - Messages are assembled with a `{"role": "system", ...}` turn. The current
>   base has no system role; the prompt has to be folded into the first user turn.
> - Chat markup and stop tokens differ between the two builds.
>
> Local use via `rag/camus_rag.py` is unaffected — it targets the current model
> and is the maintained path.


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
- **Logging (optional) — Secrets:** `LOG_SHEET_ID` (the Google Sheet's id from its URL),
  `GCP_SERVICE_ACCOUNT` (the full service-account JSON), and **`LOG_IP_SALT`** (a private
  random value). Share the Sheet with the service account's email (Editor). Logging is
  skipped entirely if these aren't set.

  Client IPs are hashed before they reach the sheet (`LOG_HASH_IP` defaults to `1`). The
  salt has a placeholder default that is public in the source, so **logging refuses to
  start until `LOG_IP_SALT` is set to a private value** — a hash salted with a published
  constant is reversible by enumerating the address space. Setting `LOG_HASH_IP=0` stores
  raw IPs instead and prints a warning. The per-IP daily cap is unaffected either way: it
  keys on the same value and never leaves memory.
- The index (`camus_kb_vectors.npy`) must be built with the **llama.cpp** nomic embedder so it
  matches the query embedder used here.
