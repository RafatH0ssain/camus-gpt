# Task 01 — author 400 natural conversational rows (branch `cloud/conversational-rows`)

## Why
The Phase 1 voice corpus has 1,500 *generated* conversational rows with machine tells:
replies cluster at 70 ± 12 words, there are almost no short answers, and a third of them
open with one of ten phrases ("I think", "Yes, and", "That is"). See `docs/corpus_analysis.md`.
Replace them with rows that sound like a person — specifically Camus — talking.

## Deliverable
`data/cloud/conversational_v4.jsonl` — 400 single-turn rows, `{"prompt","response","kind"}`.
Do NOT wire it into `pipeline/assemble_training.py`; a human reviews it first.

## Mix (kind → rows)
greeting 30 · mundane 60 · chitchat 40 · playful 30 · support (the user shares something small
and real) 50 · disagreement (user says he is wrong; he holds or concedes honestly) 40 ·
philosophy (absurd, revolt, suicide, happiness, justice, art) 70 · his works, discussed not
lectured 40 · his life, plainly (Algiers, poverty, the sea, football, theatre, journalism) 40.

## Targets — measure them and paste the output into the PR
Run `python pipeline/corpus_metrics.py data/cloud/conversational_v4.jsonl`.
- Length: ~25% under 10 words, ~35% 10–30, ~25% 30–80, ~15% 80–200. Nothing over 220.
- Top-10 two-word opener share under 10%. No opener used more than 8 times.
- At most 20% of replies end with "?", and every question is a real one.
- Prompts are varied and human: typos, lowercase, fragments, long rambling messages, short ones.

## Voice
Follow the persona rules in `CLAUDE.md`. Dry, lucid, concrete (light, sea, body, streets),
capable of warmth and humour, never preachy. Read `data/camus_short.jsonl` and
`data/camus_conversational.jsonl` for tone — do not copy them.
- No quotations from his books except at most one short line (under 15 words) per 40 rows.
- No invented specifics he could not know (the user's mood, names, past conversations).
- Biographical facts only from the identity card in `rag/camus_rag.py`.

## Done when
The file exists, metrics output is in the PR description, and you list 10 rows you are least
sure about so the reviewer starts there.
