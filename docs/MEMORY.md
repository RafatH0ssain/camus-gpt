# The memory layer

Off by default. Everything below happens only when `rag/camus_rag.py` is run with
`--memory`; without that flag the assembled prompt is byte-for-byte what it was
before this layer existed.

## Two layers, deliberately separate

| | written by | injected | retrieved |
|---|---|---|---|
| `profile.md` | you, by hand | verbatim, every non-task turn | never — it is not scored or ranked |
| `memory.jsonl` | extracted at session end | top matches only | by embedding similarity |

The profile is the "knows me from the first message" layer and costs a fixed number
of tokens. Learned memory is the "remembers what we discussed" layer and is retrieved
like the knowledge base — but through a **separate index**.

That separation is the point. The knowledge base is Camus's life; memory is the
user's. A shared index lets one stream surface on the other's queries through
embedding proximity alone, which is exactly the class of failure this project spent
months removing from retrieval. So memory:

- never enters the fact/view pool and never appears in `hits`,
- never affects the confidence tiers or the `THRESHOLD` gate, which are computed on
  knowledge-base scores alone,
- appends its own clearly-labelled section to the system prompt and nothing else,
- is **skipped on the task gate**, exactly like facts: if `is_task(msg)` is true there
  is no memory retrieval and no memory block, so analysis prompts reason unencumbered.

## Files (all gitignored, all local)

| file | what it holds |
|---|---|
| `profile.md` | your hand-written profile; copy it from `profile.example.md` |
| `memory.jsonl` | learned memories, one JSON object per line |
| `memory_vectors.npy` | one embedding row per memory, same order |
| `memory_pending.jsonl` | the probation buffer — candidates not yet promoted |

`profile.example.md` is the tracked template. `memory.jsonl` and `profile.md` hold
conversation content, so they are covered twice: by `.gitignore` and by the
pre-commit hook, because a `.gitignore` edit should not be the only thing standing
between chat content and a public history.

## Setup

    cp profile.example.md profile.md
    $EDITOR profile.md
    python rag/camus_rag.py --memory

Write the profile in the third person, as facts about "the user". Ten to twenty
lines is plenty — it is injected on every turn, so length is a running cost.

## Command line

| flag | effect |
|---|---|
| `--memory` | enable the layer for this session (default off) |
| `--memories` | list stored memories with kind, hits, last seen, cue, id — then exit |
| `--forget SUBSTRING` | delete every memory containing SUBSTRING, report the count, exit |
| `--reindex-memory` | re-embed every stored memory with the current scheme, then exit |
| `--debug --memory` | show retrieved memories and their scores, separate from the retrieval table |

`--reindex-memory` is the upgrade path. Memories written before the cue field existed
were embedded from their text alone; once cued rows sit beside them the two are scored
on different bases and the older ones retrieve worse for no visible reason. Reindexing
puts every row on the current scheme.

## How a memory is written

Extraction runs at session end (and every 40 exchanges, so a crash loses little), and
only when the session ran to at least four exchanges. It is best-effort throughout: a
failure prints a warning and the chat exits cleanly.

Candidates do not go straight into the store. They land in the probation buffer and are
promoted only on a **second sighting**. A wrong retrieval is one bad answer; a wrong
memory is a bad answer forever, so writes are the dangerous operation and are made
deliberately slow.

Consolidation is two-tier, on cosine against the existing store:

- **≥ 0.90** — the same memory said again: discard the candidate, bump the original's
  hit count and last-seen date.
- **0.85 – 0.90** — a near-twin: revise the existing row in place rather than appending
  a near-duplicate, preferring the fuller statement.
- **below 0.85** — genuinely new: append.

Unbounded near-duplicates would flood retrieval the way the untrimmed knowledge base
once did. The store is capped at `MAX_MEMORIES`, evicting fewest-hits-then-oldest first.

### Cues

Each memory may carry a short first-person `cue` describing how you would *ask* about
it — `"The user is named Ada."` gets `"my name; who I am"`. The embedded string is
`text | cue`; only `text` is shown in the prompt.

This exists because memories are stored in the third person but queried in the first,
and that gap alone costs roughly 0.06–0.15 of cosine — enough that `"what is my name?"`
would not retrieve `"The user is named Ada."` Cues close it. Rows without a cue still
work; they just match less well, which is what `--reindex-memory` is for.

## Tunables

All environment variables, all with working defaults:

| variable | default | what it does |
|---|---|---|
| `MEM_THRESHOLD` | `0.52` | similarity floor for recall. Independent of the knowledge base's `0.55` |
| `MEM_TOP_K` | `3` | how many memories may enter a turn. Small on purpose — memory augments, never dominates |
| `MEM_UTIL_MODEL` | `gemma3:12b` | model for extraction and summarisation; falls back to the persona model if not installed |
| `MAX_MEMORIES` | `500` | store cap before eviction |
| `PROFILE_PATH`, `MEM_PATH`, `MEM_VEC_PATH`, `PENDING_PATH` | repo root | file locations; handy for testing against a scratch store |

Raising `MEM_THRESHOLD` trades recall for precision. Prefer erring high: a false recall
is worse for memory than for knowledge-base facts, because it asserts something about
the person rather than about Camus.

## Evaluation

**Enabling memory makes an evaluation run non-comparable to the 34-probe baseline.**
The recorded baseline in `eval/eval_history.csv` was produced with memory off, and the
probes assume a fixed system prompt; injecting a profile and recalled memories changes
that prompt on every non-task turn. `rag/eval_camus.py` never enables memory, and it
should stay that way — if you want to measure the memory layer, build a separate probe
set for it rather than reusing this one.
