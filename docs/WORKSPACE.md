# Workspace layout

What lives in git, what lives only on disk, and where each piece has to sit for
the scripts to find it. All paths are relative to the repository root.

## Run from the repository root

`rag/camus_rag.py`, `kb/build_index.py`, and the evaluation scripts resolve
their data paths relative to the **current working directory**, not to the
script. Run them from the repository root:

    python rag/camus_rag.py
    python rag/eval_camus.py
    python training/build_phase3.py

Running them from inside `rag/` or `training/` will not find the knowledge base.

## Untracked artifacts at the repository root

Two files must be present at the root and are never committed:

| File | What it is |
|---|---|
| `camus_kb_full.jsonl` | the merged knowledge base, one record per line |
| `camus_kb_vectors.npy` | embedding vectors, one row per KB line |

They are a **matched pair**. The vector file's row count must equal the KB's
line count, and the vectors must have been produced by the same embedding model
the retrieval code queries at runtime. Mixing a vector file from one embedding
backend with a KB from another silently degrades retrieval instead of failing
loudly — regenerate with `kb/build_index.py` if the pair is ever in doubt.

A second vector file in a different embedding family may sit alongside for use
by the hosted Space. It is ignored by name and must never be loaded locally.

## Model weights

Quantized weights live in `rag/`, beside `rag/Modelfile`, which refers to them
by a relative `FROM ./<weights>.gguf`. Register the local model with:

    cd rag && ollama create camus -f Modelfile

Weights are ignored by extension (`*.gguf`, `*.safetensors`).

## `archive/` and `eval/archive/`

`archive/` holds local reference material that is not part of the project's
history — superseded datasets, earlier vector builds, batch job inputs. Nothing
in it is required to run or rebuild anything.

`eval/archive/` holds superseded evaluation runs. The current baseline
(`eval/eval_history.csv`, `eval/probe_scores.jsonl`, `eval/probe_report.md`) is
tracked; `eval/eval_history.csv` is **append-only** — add rows, never edit them.

## Heavy artifacts move through the Hub

Model weights, the knowledge base, and vector files are distributed through the
project's Hugging Face repositories, never through git. Git carries source,
configuration, the training corpora, and the evaluation baseline. Anything
large enough to need Git LFS belongs on the Hub instead.

## The invariant

`git status` should be clean at all times. Anything untracked is either about
to be committed or belongs under an ignored path.
