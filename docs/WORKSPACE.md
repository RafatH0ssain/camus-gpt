# Workspace layout

What lives in git, what lives only on disk, and where each piece has to sit for
the scripts to find it. All paths are relative to the repository root.

## Running

`rag/camus_rag.py` and the `kb/` scripts resolve their data paths relative to the
**current working directory**, not to the script — run them from the repository
root (`python rag/camus_rag.py`, `python kb/build_index.py`).

`GEN_MODEL` names the Ollama model that answers; unset falls back to `camus4`.

## Untracked artifacts at the repository root

Two files must be present at the root and are never committed:

| File | What it is |
|---|---|
| `camus_kb_full.jsonl` | the merged knowledge base, one record per line |
| `camus_kb_vectors.npy` | embedding vectors, one row per KB line |

They are a **matched pair**: the vector file's row count must equal the KB's line
count, and the vectors must come from the same embedding model the retrieval code
queries at runtime. If in doubt, regenerate both with `kb/build_index.py`.

## Model weights

Quantized weights are ignored by extension (`*.gguf`) and live in `rag/`, beside
the model definition: `rag/Modelfile.camus4` loads `rag/camus4-12b.Q4_K_M.gguf`.
Put the exported GGUF there, check it against its `.sha256`, and register it:

    cd rag && ollama create camus4 -f Modelfile.camus4

Retrieval also needs the embedding model and the KB index:

    ollama pull nomic-embed-text
    python kb/build_index.py

## Memory footprint

The quantized 12B holds about **8 GB resident** in unified memory while loaded;
that memory is GPU-wired and cannot be paged out. The runtime keeps a model loaded
after the last request — release it with `ollama stop camus4`, or set
`OLLAMA_KEEP_ALIVE` (e.g. `2m`) in the environment of the server process, which
keeps its old value until restarted. `rag/camus_rag.py` requests 15-minute
retention for the small embedding model; not worth shortening.

## The commit guard

`pipeline/hooks/pre-commit` refuses any commit that stages a file larger than
**5 MB**, assembled bundles under `build/`, `*.npy` / `*.gguf` / `*.safetensors`,
`*.jsonl` outside the allowed corpus paths, or `profile.md` and `memory*.jsonl`
(local conversation memory — covered by `.gitignore` too, guarded here as a
second layer). It prints what it blocked; override with `git commit --no-verify`.
Hooks are not version-controlled — install it in each clone:

    cp pipeline/hooks/pre-commit .git/hooks/pre-commit
    chmod +x .git/hooks/pre-commit

## Gotchas

**`util_chat()` is not the chat path.** It routes to `MEM_UTIL_MODEL`, an
instruction-follower used only for summarisation and memory extraction;
user-facing replies go through `stream_chat()` / `GEN_MODEL` and are not in voice
if they come from the utility helper.

**Long chats load a second model.** Once a chat passes 16 exchanges, the turns that fall out
of the window are summarised by `MEM_UTIL_MODEL` (default `gemma3:12b`), with or without
`--memory`, and memory extraction uses the same model. Set `MEM_UTIL_MODEL=camus4` to avoid the
double load, at some cost to summary quality. Enabling memory also changes the system prompt
every turn, so runs with and without it are not comparable.

## The memory layer

Off by default, behind `--memory`. See `docs/MEMORY.md`.

## The invariant

`git status` should be clean at all times. Anything untracked is either about to
be committed or belongs under an ignored path.
