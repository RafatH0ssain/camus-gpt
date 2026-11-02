# Task 06 — 120 natural multi-turn conversations (branch `cloud/multiturn-rows`)

## Why
Task 01 writes single-turn rows. The persona's weakest live behaviour is across turns: it
forgets what the user said, riffs instead of continuing a vague follow-up, folds when pushed,
and treats every message as a cue for a full answer.

## Deliverable
`data/cloud/multiturn_v4.jsonl` — 120 conversations,
`{"messages":[{"role":"user"|"assistant","content":...}, ...], "kind": ...}`, 3–6 user turns
each. Do NOT wire it into the assembler.

## Mix (kind → conversations)
recall (the user mentions a fact early; later he uses it plainly when it matters, never
volunteers it) 25 · topic_change (an abrupt swerve; he follows without ceremony) 20 ·
vague_followup ("why?", "go on", "meaning?" — he continues the thread, not a new riff) 20 ·
pushback (the user insists he is wrong; he holds where he is right, concedes where he is not)
20 · drift_to_small (philosophy that turns into lunch plans or the weather) 15 ·
disclosure (something real but small; warmth without therapy-speak) 20.

## Targets — measure on assistant turns, paste results in the PR
- Median assistant turn under 25 words; at least 30% under 10 words; some over 60.
- At most 20% of assistant turns end with "?".
- No two conversations open with the same user line.

## Rules
Persona rules in `CLAUDE.md`. Never invent what the user did not say. Facts about his life
only from `_IDENTITY_CARD` in `rag/camus_rag.py`. No book quotations beyond one short line
per 20 conversations. Do not reuse prompts from task 01's brief examples.
