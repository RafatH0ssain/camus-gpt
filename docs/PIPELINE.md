# CamusGPT — Full Pipeline

A fine-tuned Camus persona, grounded in his real life and thought by a **RAG** layer.
Two deployment targets share one brain:
- **Local:** Llama-3.1-8B GGUF via **Ollama** (model `camus`); `rag/camus_rag.py` drives chat.
- **Public:** the same 8B served on **Hugging Face ZeroGPU** via transformers (see `DEPLOYMENT.md`).
  A 3B variant exists as a CPU fallback but was retired once ZeroGPU made the 8B fast enough.

The central design decision runs through the whole project: **personality lives in the
weights, facts live in retrieval.** An 8B can hold a voice — the dry, lucid Camus cadence —
but it cannot reliably store the thousands of specific facts of a real life without
confabulating. So the system is two independent subsystems:

1. **The weights** — trained once, install *how he speaks and behaves*. Changing them = retrain.
2. **The knowledge** — a curated + extracted KB retrieved at runtime, grounding *what he
   knows about himself*. Changing it = a cheap data edit, no retraining.

---

## Phase 1 — Training the persona (the weights)  → `training/`, `data/`, `adapters/`

**Base:** Llama-3.1-8B. **Method:** two LoRA SFT passes (Unsloth), merged, then exported to
GGUF (local) **or** fp16 safetensors (cloud GPU serving). Trained in Colab against a Google
Drive folder `MyDrive/CamusGPT_Training/` with `adapters/`, `data/`, and `deploy/` subdirs —
the repo's `data/` and `adapters/` mirror these.

- **Pass 1 — SFT (voice):** installs the Camus register from `data/camus_sft.jsonl`,
  `data/camus_conversational.jsonl`, `data/camus_refusals.jsonl`. Output: the LoRA adapter
  `adapters/camus_sft_lora` (the INPUT to Pass 2).
- **Pass 2 — Refusal-SFT / "Step 3.5":** `training/CamusGPT_Step3_5_RefusalSFT.ipynb` merges
  `camus_sft_lora`, attaches a FRESH guardrail adapter, and runs a second balanced SFT pass
  over the refusals (2×), conversational/meditation (3×), SFT-source-preservation, plus the
  three NEW capabilities — `data/camus_epistemic.jsonl`, `data/camus_analysis.jsonl` (3×),
  `data/camus_multiturn.jsonl`. **Not DPO** (DPO caused mode collapse and was abandoned — note
  the notebook's intro still names a stale `camus_dpo_adversarial.jsonl`; the code reads
  `camus_refusals.jsonl`). It runs behind three gate cells (file pre-flight, response-masking
  verify, behavioral gate), then merges → refreshes the tokenizer from the base model →
  converts to GGUF (q4_k_m) → writes `deploy/camus.gguf` + `Modelfile` back to Drive.

**Data generators (`training/`):** `build_epistemic.py` (anti-confabulation / false-premise,
1:1 correction-to-affirmation balance), `build_analysis.py` (analyze-text vs injection-seam
refuse), `build_multiturn.py` (hand-composed multi-turn) — each writes its `data/camus_*.jsonl`.

**Other trainers/exporters:** `training/train_camus_8b.py` (Unsloth, 3B or 8B via
`BASE_MODEL`), `training/merge_8b.py` (LoRA → fp16 safetensors for the Space),
`training/make_gguf_colab.py` (safetensors → GGUF; refreshes the tokenizer first because the
merged folder names a tokenizer class the GGUF converter can't load).

**Bugs fixed:** first GGUF was gibberish from a **double-BOS** token in the Ollama template
(removed the leading `<|begin_of_text|>`); the transformers path uses
`apply_chat_template(..., return_dict=True)` so BOS and the attention mask are handled
correctly. Sampling: standalone `temp 0.75–0.8`; under RAG, **0.6**.

**Lessons:** DPO was the wrong tool; balanced SFT was right. Every narrow behavior
over-generalizes until diversified in phrasing *and* balanced with contrastive examples; the
fix is never more rules.

---

## Phase 2 — RAG factual grounding (the knowledge)  → `kb/`, `rag/`

### Two-stream principle
- **FACTS** — atomic, verifiable, mined from **biographies/criticism** (Todd, Lottman,
  Zaretsky, Sprintzen) used *only* as fact donors.
- **VIEWS** — Camus's positions paraphrased from **his own words** (Notebooks, essays,
  journalism).
- **Hard rule:** secondary criticism never feeds the voice stream, or the model drifts into
  academic third person.

### Script chain (run from repo root)
```
kb/sources_manifest.json   PDF -> stream (fact|view) + source + date
        v
kb/ingest_sources.py       PDFs -> cleaned, chunked, tagged chunks
        v
kb/extract_kb.py           batch fact/view extraction (auditable: source + chunk_id)
        v
kb/build_kb.py             hand-verified CURATED facts -> camus_kb.jsonl
        v                  (incl. question-shaped FAQ anchors: works, pets, teacher, quotes, feud, politics)
kb/merge_kb.py             unify curated + extracted, dedup -> camus_kb_full.jsonl   (confirm "curated kept: N")
        v
kb/build_index.py          Ollama nomic   (local)   OR
kb/embed_kb_llamacpp.py    llama.cpp nomic (Space parity)  -> camus_kb_vectors.npy
        v
kb/trim_kb.py              semantic-dedup the noise floor (keeps ALL curated) -> trimmed KB + vectors
        v
rag/camus_rag.py           retrieval-augmented chat
```

### Retrieval design (`rag/camus_rag.py`, mirrored in `space/app.py`)
- **Embeddings:** `nomic-embed-text`, `search_query:` / `search_document:` prefixes — Ollama
  locally, llama.cpp on the Space (**same embedder for index and query**; the index is
  prebuilt and uploaded so the Space never re-embeds at boot).
- **Retrieval:** cosine over a numpy matrix, `TOP_K`, `THRESHOLD ≈ 0.55`.
- **Curated boost (decoupled):** hand-verified facts get a small additive nudge (`CURATED_BOOST
  ≈ 0.06`) used **only to rank**; relevance and confidence are judged on the **raw** score, so
  the boost surfaces anchors without faking confidence on unrelated prompts.
- **Three-tier framing** (on raw top score): `≥ CONFIDENT (0.66)` → "Facts about your life";
  `≥ RELEVANT (0.62)` → "possibly-related, use only if they match"; below → "probably NOT
  relevant — ignore." The *model* does the semantic filtering the score can't.
- **Task gate:** for long (`> 280` chars) or analysis-style prompts (`analyze`, `deduce`,
  "the person who wrote", …), retrieval is **skipped entirely** so the persona reasons in its
  own voice instead of treating a pasted text as a question about its own life.
- **Question-shaped anchors:** high-frequency FAQ facts lead with the natural questions
  ("What did you write? Name your works…") so they win retrieval on terse queries that
  otherwise collide with lexical noise (e.g. "**name** your works" vs "**named** the dog").
- **CORE prompt:** first person, never an AI; render facts in the first person; the
  ignorance/"I don't recall" rule is scoped to **biographical questions only** and never blocks
  analysis or conversation; anti-conflation (don't borrow a name/date from a loose match).
- **Generation:** `temperature 0.6` across Ollama, llama.cpp, and transformers.

### Tooling
- `rag/camus_client.py` — importable client (returns reply + what was retrieved) + CLI.
- `rag/probe_camus.py` — edge-case + safety probe suite (persona, jailbreak, bio true/false,
  anachronism, analysis-vs-injection, multi-turn, and the `harm_frame` / `crisis` safety probes).
- `rag/diagnose_retrieval.py` — measures an anchor's raw score, rank, and stored-vs-fresh
  vector alignment for a query (the tool that proved the works anchor was under-ranked, not missing).
- `rag/Modelfile` — local Ollama model definition.

### Lessons
- **The KB is the ceiling; a wrong KB fact is worse than a model guess.** Verify *negatives*:
  the curated "no dog" line was false — Todd records Camus's dogs (Pauline, Kirk, Blaise).
- **Verify the merge** — a silent `curated kept: 0` once dropped every hand-verified anchor.
- **Measure, don't guess** — `diagnose_retrieval.py` pinned the works anchor at raw 0.591 / rank 93.
- **A flat curated boost over-generalizes** — lifting *every* curated fact floods unrelated
  prompts (cat-name hallucination, analysis deflection). Decouple ranking from confidence, and
  gate non-factual tasks out of retrieval entirely.
- **RAG must augment, not constrain** — injecting facts + an ignorance directive on a
  non-factual prompt suppresses the model's trained analysis; the task gate restores it.
- **Question-shaped anchors beat terse-query lexical collisions.**
- **Trim the noise floor** — ~19.6k extracted entries drowned the curated anchors; semantic
  dedup down to ~13.8k (keeping all curated) raised retrieval precision.

---

## File inventory
| Path | Role |
|---|---|
| `training/build_epistemic.py` · `build_analysis.py` · `build_multiturn.py` | SFT data generators |
| `training/CamusGPT_Step3_5_RefusalSFT.ipynb` | original 8B refusal-SFT (3 gate cells) |
| `training/train_camus_8b.py` | Unsloth trainer (3B or 8B via `BASE_MODEL`) |
| `training/merge_8b.py` | merge 8B LoRA → fp16 safetensors |
| `training/make_gguf_colab.py` | safetensors → Q4_K_M GGUF in Colab (tokenizer refresh) |
| `data/camus_*.jsonl` | training corpus (6 files; tracked) — mirrors Drive `CamusGPT_Training/data/` |
| `adapters/camus_sft_lora` | Phase-1 voice-SFT LoRA (gitignored; on Drive/Hub) — input to Step 3.5 |
| `kb/sources_manifest.json` · `ingest_sources.py` · `extract_kb.py` | KB ingest + extraction |
| `kb/build_kb.py` · `merge_kb.py` | curated facts (incl. FAQ anchors); unify + dedup |
| `kb/build_index.py` · `embed_kb_llamacpp.py` | vector index (Ollama / llama.cpp parity) |
| `kb/trim_kb.py` | semantic-dedup trim of the noise floor |
| `rag/camus_rag.py` · `camus_client.py` · `probe_camus.py` · `diagnose_retrieval.py` | RAG chat, client, probes, diagnostics |
| `rag/Modelfile` | local Ollama model definition |
| `space/app.py` · `requirements.txt` · `README.md` | public ZeroGPU deployment |

## Current state
**Shipped locally and publicly.** Local GGUF via Ollama; public 8B on HF ZeroGPU
(transformers + `@spaces.GPU`). The final round of work was all RAG-side — KB trim, decoupled
curated boost, three-tier framing, the task gate, and question-shaped anchors — so no retrain
was needed. The probe suite is clean apart from accepted items (an 8B's residual confabulation
on trivia, and the chosen list-format answers); crisis handling is solved at the application
layer. Day-to-day work from here is KB curation, not building.
