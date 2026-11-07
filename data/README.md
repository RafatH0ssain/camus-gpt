# data/ — training corpus and KB intermediates

Nothing here is tracked but this README; the corpus and intermediates are
gitignored and rebuilt or re-supplied locally.

`pipeline/assemble_training.py` reads `data/*.jsonl` (the voice SFT,
conversational, short, refusals, epistemic, analysis, multiturn and phase3 sets),
`data/cloud/*`, and `data/drafts/raft_v6.jsonl`; `data/drafts/defects_*.jsonl`
join only when `data/drafts/DEFECTS_REVIEWED` exists. It assembles them into
`build/training_v3/` with a `MANIFEST.json` — the sets the training notebooks
consume.

Rows come in two shapes. Single-turn: `{"prompt": ..., "response": ..., "kind": ...}`.
Multi-turn: `{"messages": [{"role": "system"|"user"|"assistant", "content": ...}, ...], "kind": ...}`;
a leading system message is used as-is, otherwise the notebook adds the persona prompt.
Replies are always in his voice: first person, no lists, no service tone.

The voice corpus `data/camus_sft.jsonl` is built with the batch pipeline:
`pipeline/camus_pdf_pipeline.py` chunks the books, `pipeline/submit_batch.py` submits the
batch, and `pipeline/fetch_batch.py sft` writes the file.

Knowledge-base intermediates live here too: `data/source_chunks.jsonl`
(`kb/ingest_sources.py`) and `data/kb_extracted.jsonl` (`kb/extract_kb.py`). Both
are regenerable from copyrighted sources and stay local.
