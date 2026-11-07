# The memory layer

Off by default. Everything below happens only when `rag/camus_rag.py` is run with
`--memory`; without the flag no profile or memories are read.

## Two layers, deliberately separate

| | written by | injected | retrieved |
|---|---|---|---|
| `profile.md` | you, by hand | verbatim, every non-task turn | never |
| `memory.jsonl` | extracted at session end | top matches only | by embedding similarity |

The profile is the "knows me from the first message" layer at a fixed token cost;
learned memory is the "remembers what we discussed" layer. Memory is retrieved
like the knowledge base but through a **separate index**: the knowledge base is
Camus's life, memory is the user's, and a shared index would let one stream
surface on the other's queries through embedding proximity alone. Memory
therefore never enters the fact/view pool or `hits`, never affects the confidence
tiers or the `THRESHOLD` gate (both judged on the knowledge base alone), appends
its own labelled section to the system prompt and nothing else, and is skipped on
the task gate exactly like facts.

## Files (all gitignored, all local)

| file | what it holds |
|---|---|
| `profile.md` | your hand-written profile; copy it from `profile.example.md` |
| `memory.jsonl` | learned memories, one JSON object per line |
| `memory_vectors.npy` | one embedding row per memory, same order |
| `memory_pending.jsonl` | the probation buffer — candidates not yet promoted |

`profile.example.md` is the tracked template; `profile.md` and `memory*.jsonl`
hold conversation content and are guarded by `.gitignore` and the pre-commit hook.

## Setup

    cp profile.example.md profile.md
    $EDITOR profile.md
    python rag/camus_rag.py --memory

Write the profile in the third person, as facts about "the user"; ten to twenty
lines is plenty, since it is injected every turn.

## Command line

| flag | effect |
|---|---|
| `--memory` | enable the layer for this session (default off) |
| `--memories` | list stored memories with kind, hits, last seen, cue, id — then exit |
| `--forget SUBSTRING` | delete every memory containing SUBSTRING, report the count, exit |
| `--reindex-memory` | re-embed every stored memory with the current scheme, then exit |
| `--debug --memory` | show retrieved memories and their similarities, separately from the retrieval table |

## How a memory is written

Extraction runs at session end and every 40 exchanges, only after at least four
exchanges, and is best-effort — a failure prints a warning and the chat exits
cleanly. Candidates land in the probation buffer and are promoted only on a
**second sighting**: a wrong memory is a bad answer forever, so writes are made
deliberately slow. Consolidation is two-tier on cosine against the store —
≥ 0.90 discards the candidate and bumps the original's hit count, 0.85–0.90
revises the existing row in place, below 0.85 appends. The store is capped at
`MAX_MEMORIES`, evicting fewest-hits-then-oldest first.

Each memory may carry a short first-person `cue` of how you would ask about it;
the embedded string is `text | cue`, only `text` is shown in the prompt. Memories
are stored in the third person but queried in the first, and that gap alone loses
matches — cues close it; `--reindex-memory` moves old rows onto the current scheme.

## Tunables

All environment variables, all with working defaults:

| variable | default | what it does |
|---|---|---|
| `MEM_THRESHOLD` | `0.52` | similarity floor for recall, independent of the knowledge base's `0.55` |
| `MEM_TOP_K` | `3` | how many memories may enter a turn — memory augments, never dominates |
| `MEM_UTIL_MODEL` | see `rag/camus_rag.py` | model for extraction and summarisation; falls back to the persona model |
| `MAX_MEMORIES` | `500` | store cap before eviction |
| `PROFILE_PATH`, `MEM_PATH`, `MEM_VEC_PATH`, `PENDING_PATH` | repo root | file locations; handy for testing against a scratch store |

Raising `MEM_THRESHOLD` trades recall for precision; prefer erring high, since a
false recall asserts something about the person rather than about Camus.
