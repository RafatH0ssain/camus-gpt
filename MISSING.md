# MISSING.md — CamusGPT shopping list (Mac, M5 / 24 GB)

Generated 2026-08-09 against clone at `~/Projects/camus-gpt` @ `c3c299f`.
Nothing in this file has been downloaded or run. Commands are for later.

Verified facts:
- Both Hugging Face repos exist and are **public** (no token needed). Sizes below are from the HF API.
- The code was **not modified**. Path mismatches between the download commands and what the
  scripts actually expect are flagged under "Path reality check" — resolve by moving files or
  passing flags, not by editing code.

---

## Path reality check (read before downloading)

`rag/camus_rag.py` and `kb/build_index.py` both hardcode **CWD-relative** paths:

| Script | Constant | Value |
|---|---|---|
| `rag/camus_rag.py:34` | `KB_PATH` | `./camus_kb_full.jsonl` |
| `rag/camus_rag.py:35` | `VEC_PATH` | `./camus_kb_vectors.npy` |
| `rag/camus_rag.py:32` | `GEN_MODEL` | `camus` (Ollama model name) |
| `rag/camus_rag.py:33` | `EMBED_MODEL` | `nomic-embed-text` |
| `kb/build_index.py` | `--kb` / `--out` | `./camus_kb_full.jsonl` → `./camus_kb_vectors.npy` |

`rag/Modelfile` uses `FROM ./camus-8b.Q4_K_M.gguf` — the GGUF must sit **beside the Modelfile**
(i.e. in `rag/`), or `ollama create` is run from wherever the GGUF actually lives.

So the `--local-dir models/` and `--local-dir kb_data/` below will land files where the scripts
do **not** look. Either symlink/copy afterwards, or run each script from the directory holding its
data. `build_index.py` accepts `--kb`/`--out` flags, so it is the flexible one.

---

## 1. From Hugging Face (public — DO NOT RUN YET, multi-GB)

### `rafatho/camus-gguf` (model repo) — verified present

| File | Size |
|---|---|
| `camus-8b.Q4_K_M.gguf` | **4920.7 MB** (~4.9 GB) |
| `model-00001-of-00004.safetensors` | 4976.7 MB |
| `model-00002-of-00004.safetensors` | 4999.8 MB |
| `model-00003-of-00004.safetensors` | 4915.9 MB |
| `model-00004-of-00004.safetensors` | 1168.1 MB |
| `model.safetensors.index.json`, `config.json`, `generation_config.json`, `chat_template.jinja` | < 1 MB each |
| `tokenizer.json` | 17.2 MB |
| `tokenizer_config.json` | 0.1 MB |

The full-precision `safetensors` set (~16.1 GB) is the merged FP16 model — only needed for
re-quantizing or further training, **not** for running the RAG. Pull the GGUF alone.

```bash
hf download rafatho/camus-gguf camus-8b.Q4_K_M.gguf --local-dir models/
```

### `rafatho/camus-kb` (dataset repo) — verified present

| File | Size |
|---|---|
| `camus_kb_full.jsonl` | **3.3 MB** |
| `camus_kb_vectors.npy` | **42.4 MB** |

```bash
hf download rafatho/camus-kb camus_kb_full.jsonl --repo-type dataset --local-dir kb_data/
```

**Note on the `.npy`:** the vectors on that dataset repo were built with **llama.cpp** for the
HF Space. The **local Ollama-built** vectors are a different artifact — they come from the Windows
machine, or get rebuilt here with `kb/build_index.py`. The `.npy` is only 42.4 MB (under the
session's 200 MB ceiling) so it *can* be fetched cheaply as a reference, but do not assume it is
drop-in interchangeable with Ollama-generated embeddings — that has not been verified here.

Total for the two files actually needed: **~4.92 GB**. Free space on the volume: 304 GB. Fine.

### After the GGUF lands (also not run yet)

```bash
ollama create camus -f rag/Modelfile
```

Requires `camus-8b.Q4_K_M.gguf` to be next to `rag/Modelfile` (see Path reality check).

---

## 2. From the Windows machine (`mac_transfer.zip`)

Everything below is absent from the clone. Items marked **gitignored** were deliberately never
pushed; items marked **never committed** have zero commits touching them in the entire history
(`git log --all`), i.e. they were not merely ignored — they were never added.

| Item | Why it's absent | Notes |
|---|---|---|
| `camus_kb_vectors.npy` (Ollama-built) | gitignored (`*.npy`) | **plan of record** — the local-inference vectors; the HF one is the llama.cpp variant |
| `camus_kb_full.jsonl` (matching build) | gitignored (`/camus_kb*.jsonl`, `data/camus_kb_full.jsonl`) | must be **the same build** the vectors were embedded from, or retrieval silently degrades |
| `eval/` (whole directory) | gitignored (`eval/`) | run history / judge outputs from `rag/eval_camus.py` |
| `docs/ROADMAP.md` | gitignored (`docs/ROADMAP.md`) + never committed | no other file in the repo references it |
| `training/build_phase3.py` | **never committed** | not gitignored — simply never pushed |
| `data/camus_phase3.jsonl` | **never committed** | expected 69 lines; not gitignored |
| `adapters/*` (LoRA binaries) | gitignored (`adapters/*`, README kept) | only needed to re-merge; the GGUF already contains the merge |
| anything else in the zip's manifest | — | reconcile against the manifest on arrival |

**Not missing** — the six training corpora are all present and tracked:

| File | Lines |
|---|---|
| `data/camus_sft.jsonl` | 8814 |
| `data/camus_refusals.jsonl` | 1049 |
| `data/camus_epistemic.jsonl` | 327 |
| `data/camus_conversational.jsonl` | 104 |
| `data/camus_analysis.jsonl` | 28 |
| `data/camus_multiturn.jsonl` | 8 |

---

## 3. Rebuildable here (fallback — decision: use only if the zip lacks the vectors)

**Plan of record is the Windows zip** (section 2). The vectors there are guaranteed to match the
KB build they were embedded from; a local rebuild is the contingency, not the default.

Vectors can be regenerated locally once the KB jsonl is in hand:

```bash
# prerequisites: ollama serving + nomic-embed-text pulled + camus_kb_full.jsonl present
python kb/build_index.py --kb ./camus_kb_full.jsonl --out ./camus_kb_vectors.npy --batch 64
```

This produces **Ollama-built** vectors — the correct variety for local `rag/camus_rag.py`.
It writes a `.partial.npy` checkpoint as it goes, so an interrupted run resumes.

The KB jsonl itself is rebuildable in principle via `kb/ingest_sources.py` → `kb/extract_kb.py`
→ `kb/build_kb.py` → `kb/merge_kb.py` → `kb/trim_kb.py`, but that path needs the **source PDFs**
(gitignored, `*.pdf`) listed in `kb/sources_manifest.json`. Downloading the 3.3 MB jsonl from HF
is far cheaper than reconstructing it.

`kb/embed_kb_llamacpp.py` is the llama.cpp-side equivalent of `build_index.py` — that is the one
that produced the `.npy` on the HF dataset repo. Use it only for Space parity, not local runs.

---

## Marker verification (all passed)

`rag/camus_rag.py` — **canonical**, all five markers True:
`"What you know cold"`, `DENSE_FLOOR`, `CE_WEIGHT`, `TEMP_FACTUAL`, `"published after your death"`.

`rag/eval_camus.py` — `PROBES` True, `judge_anthropic` True.
`rag/diagnose_retrieval.py` — `DENSE_FLOOR` True.

Note: `eval_camus.py` and `diagnose_retrieval.py` live in **`rag/`**, not the repo root.

---

## Environment already prepared (2026-08-09) — no action needed

| Component | State |
|---|---|
| Ollama | v0.32.5 (Homebrew), `ollama serve` running on :11434 |
| `nomic-embed-text` | **pulled**, 274 MB; smoke-tested, returns 768-dim embeddings |
| `.venv` | repo root, Python 3.14.6 arm64 |
| torch | 2.13.0, `mps.is_available() == True`, MPS matmul verified |
| also installed | numpy 2.5.1, requests 2.34.2, rank-bm25 0.2.2, sentence-transformers 5.7.0, huggingface_hub 1.27.0 (+ transformers 5.14.1, scikit-learn, scipy) |

All wheels were prebuilt `cp314` / `macosx_arm64` — no source builds, no Apple-Silicon issues.

The `hf` CLI is **only inside the venv** (`.venv/bin/hf`), not on the global PATH. Activate first:

```bash
cd ~/Projects/camus-gpt && source .venv/bin/activate
```

Also note: Homebrew is installed at `/opt/homebrew` but its `shellenv` is in `~/.zprofile` only,
so it is absent from non-login shells (scripts, some CI, some editor terminals). If a script can't
find `ollama` or `python3`, that is why.

Still to do, in order, once artifacts arrive:

```bash
hf download rafatho/camus-gguf camus-8b.Q4_K_M.gguf --local-dir models/
hf download rafatho/camus-kb camus_kb_full.jsonl --repo-type dataset --local-dir kb_data/
# place the GGUF beside rag/Modelfile, then:
ollama create camus -f rag/Modelfile
```
