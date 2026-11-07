# CamusGPT — pipeline

A fine-tuned Camus persona, grounded in his life and thought by a retrieval layer.
The design split: **personality lives in the weights, facts live in retrieval.**
The weights install how he speaks and behaves — changing them means a retrain. The
knowledge base holds what he knows about himself — changing it is a data edit.

Training runs in Colab, serving runs through Ollama, everything else is local.
Retrieval dependencies: `pip install -r requirements.txt` (both hybrid-retrieval
extras degrade to dense-only retrieval if they are missing).

## Weights — `training/`

Base model `unsloth/gemma-4-12b-it`, two SFT passes and an optional DPO pass:

1. **Phase 1 — voice.** `training/CamusGPT_Phase1_VoiceSFT_Gemma4.ipynb` trains a
   voice LoRA on the assembled corpus.
2. **Phase 2 — behaviour.** `training/CamusGPT_Phase2_SFT_Gemma4.ipynb` folds the
   Phase 1 LoRA into the base and trains a fresh behaviour LoRA on top. The
   "Phase 2b" cell runs an optional DPO pass on the model's own failing replies
   (chosen = the correction, rejected = the failing reply, reference = the SFT
   weights loaded frozen); set `DPO_SET = None` to skip it.
3. **Export.** `training/CamusGPT_Export_GGUF_Gemma4.ipynb` merges both LoRAs into
   the base and quantizes to a q4_k_m GGUF with a matching Modelfile. Register the
   result as the Ollama model `camus4`:

       cd rag && ollama create camus4 -f Modelfile.camus4

The corpus is assembled by `pipeline/assemble_training.py` (see `data/README.md`).

## Knowledge — `kb/`

Facts come from biographies and criticism; views come from his own words.
Secondary criticism never feeds the voice stream. Build chain, from the repo root:

    kb/sources_manifest.json -> kb/ingest_sources.py -> data/source_chunks.jsonl
    kb/extract_kb.py  -> data/kb_extracted.jsonl    (fact/view extraction, batched)
    kb/build_kb.py    -> camus_kb.jsonl             (hand-verified curated facts)
    kb/merge_kb.py    -> camus_kb_full.jsonl        (unify + dedup, curated wins)
    kb/build_index.py -> camus_kb_vectors.npy       (embedding index)
    kb/trim_kb.py     -> optional semantic dedup of the extracted rows

Sources are PDFs under `sources/`, tagged fact or view in `kb/sources_manifest.json`.
`camus_kb_full.jsonl` and `camus_kb_vectors.npy` sit at the repo root, are
gitignored, and are a matched pair.

## Chat + retrieval — `rag/`

`python rag/camus_rag.py` (from the repo root; needs Ollama running with the
`camus4` model and `nomic-embed-text`):

- **Retrieval** is hybrid: dense cosine over the KB vectors plus BM25, fused by
  reciprocal rank and reranked by a cross-encoder. Confidence tiers and the 0.55
  hit floor are judged on the raw cosine — ≥ 0.66 frames hits as facts about his
  life, ≥ 0.62 as possibly related, below that as ignorable. A fixed identity card
  of core facts is always in the prompt, independent of retrieval.
- **Task gate:** messages over 280 characters or analysis-shaped ("analyze",
  "the person who wrote", …) skip retrieval and memory entirely, so the persona
  reasons in its own voice on a pasted text instead of treating it as a question
  about his life.
- **Prompt:** first person, never an AI; retrieved facts are spoken as his own
  memories; "I don't recall" applies only to specific biographical questions.
- **Conversation:** the last 32 messages stay in context; turns evicted from it
  fold into a rolling summary (`fold_evicted` in `rag/camus_rag.py`).
- **Sampling:** temperature 0.45 on factual turns, 0.6 on task turns.
- **Memory:** an optional per-user layer behind `--memory` — see `docs/MEMORY.md`.

`rag/camus_client.py` wraps the same pipeline as an importable client and CLI.
`rag/eval_camus.py` and `rag/eval_memory.py` are the evaluation harnesses.
