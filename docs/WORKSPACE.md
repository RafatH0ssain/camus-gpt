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

## Selecting the model

`GEN_MODEL` overrides which registered model answers; unset falls back to the
previous build. The evaluation harness inherits it through its `camus_rag`
import and records the resolved name in the `gen_model` column of
`eval/eval_history.csv`.

    GEN_MODEL=camus2 python rag/camus_rag.py

**`rag/eval_camus.py` writes to `eval/` by default and will overwrite the
tracked baseline.** For any exploratory run, redirect it:

    python rag/eval_camus.py --out-dir /tmp/eval_scratch

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

**The tracked `rag/Modelfile` builds the 8B v1 only** — its `FROM` names the 8B GGUF.
The current 12B build was registered from a GGUF that is not in the repository, so
`camus2` cannot be rebuilt from a clone; pull it from the Hub, or recreate a Modelfile
against the 12B weights. `ollama show camus2 --modelfile` prints the parameters it was
registered with.

## Model memory footprint

The current model is a 12B build and holds **~8 GB resident** in unified memory
while loaded (the previous 8B build held ~4.9 GB). That memory is GPU-wired and
cannot be paged out, so on a 24 GB machine with a browser and editor open it is
enough to push everything else into swap.

The runtime keeps a model loaded after the last request so the next one is fast.
To release it immediately:

    ollama stop camus2

To shorten the automatic retention, set `OLLAMA_KEEP_ALIVE` (e.g. `2m`) **in the
environment of the server process** — it is read by `ollama serve`, not by the
client, so an already-running server keeps its old value until restarted.

Note that `rag/camus_rag.py` separately requests a 15-minute retention for the
embedding model; that one is small (~370 MB) and not worth shortening.

The previous model is kept installed as a rollback. Do not remove it.

## `archive/` and `eval/archive/`

`archive/` holds local reference material that is not part of the project's
history — superseded datasets, earlier vector builds, batch job inputs. Nothing
in it is required to run or rebuild anything.

`eval/archive/` holds superseded evaluation runs. Two subtrees are tracked
deliberately — `base_selection/` (evidence behind the base-model choice) and
`v2_*/` (the scored runs backing a released build); the rest stays local. The current baseline
(`eval/eval_history.csv`, `eval/probe_scores.jsonl`, `eval/probe_report.md`) is
tracked; `eval/eval_history.csv` is **append-only** — add rows, never edit them.

## Heavy artifacts move through the Hub

Model weights, the knowledge base, and vector files are distributed through the
project's Hugging Face repositories, never through git. Git carries source,
configuration, the training corpora, and the evaluation baseline. Anything
large enough to need Git LFS belongs on the Hub instead.

## The commit guard

`pipeline/hooks/pre-commit` refuses any commit that stages:

- a file larger than **5 MB**,
- anything under the root `archive/`,
- `*.npy`, `*.gguf`, or `*.safetensors`,
- any `*.jsonl` other than `data/camus_*.jsonl` and the tracked eval probe scores,
- `profile.md` and `memory*.jsonl` — local conversation memory and the per-user
  profile. `.gitignore` covers these too; the hook is a second layer because a
  `.gitignore` edit would otherwise be the only thing standing between chat
  content and a public history. A `profile.example.md` template is still
  committable.

It prints what it blocked and why. Override deliberately with
`git commit --no-verify`.

Install it in each clone — **git does not version-control hooks**, so cloning
the repository does not bring it along:

    cp pipeline/hooks/pre-commit .git/hooks/pre-commit
    chmod +x .git/hooks/pre-commit

## Gotchas

**`util_chat()` is not the chat path.** It routes to `MEM_UTIL_MODEL` (an
instruction-follower), not to the persona model, and exists for summarisation and
memory extraction. Anything generated through it will not be in voice. User-facing
replies go through `stream_chat()` / `GEN_MODEL`. This has already caught one test
harness out: replies came back in the wrong voice because the harness reached for the
utility helper.

**Memory extraction loads a second model.** With `--memory`, session-end extraction
calls `MEM_UTIL_MODEL` (default `gemma3:12b`, ~8 GB) while the persona model may still
be resident (~8 GB). On a 24 GB machine that can briefly want both. `OLLAMA_KEEP_ALIVE`
keeps the overlap short; `ollama stop <model>` frees one immediately. Set
`MEM_UTIL_MODEL` to the persona model to avoid the second load entirely, at some cost
to extraction quality.

**Enabling memory makes eval runs non-comparable** to the tracked 34-probe baseline —
see `docs/MEMORY.md`.

## The memory layer

Off by default, behind `--memory`. See **`docs/MEMORY.md`** for the two layers, the
four gitignored files, the CLI flags, dedup and probation, and the tunables.

## The invariant

`git status` should be clean at all times. Anything untracked is either about
to be committed or belongs under an ignored path.
