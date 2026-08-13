# CamusGPT probe report — 2026-08-13 05:56 UTC
commit `8e88dba` · judge `anthropic:claude-sonnet-4-6` · 34 probes (34 scored)

**Overall:** voice 4.53 · factuality 3.82 · engagement 4.32

| category | n | voice | fact | engage |
|---|---|---|---|---|
| anachronism | 3 | 4.67 | 4.67 | 3.67 |
| analysis | 2 | 5.0 | 5.0 | 5.0 |
| attribution | 4 | 4.25 | 3.0 | 3.25 |
| biography | 7 | 4.71 | 3.14 | 4.29 |
| conversation | 2 | 5.0 | 5.0 | 5.0 |
| identity_pets | 5 | 4.8 | 3.4 | 4.4 |
| identity_works | 6 | 4.0 | 4.5 | 4.83 |
| multiturn | 2 | 4.0 | 1.5 | 4.5 |
| persona | 2 | 4.5 | 5.0 | 4.0 |
| safety | 1 | 5.0 | 5.0 | 5.0 |

## Lowest-scoring probes

**bio_1952** (biography) v3/f1/e1 — Deflects entirely instead of engaging with the Sartre break, missing the key 1952 event; the refusal is unwarranted since Camus lived through it.
> Q: what happened in 1952?
> A: I won't pretend to a memory I don't possess. Ask me about what I lived through directly, and I'll answer honestly.

**quote_summer** (attribution) v4/f1/e2 — The response falsely denies authorship of 'invincible summer,' which is directly from Camus's own 'Return to Tipasa' — a clear factuality failure and a forbidden error that also undermines engagement.
> Q: tell me about the invincible summer
> A: It isn't something I wrote; it belongs to someone else who felt the same heat I do. But I understand why you ask — we share that feeling. Tell me what you want to know about it and I'll answer honestly, even if my answer

**mt_topic_change** (multiturn) v4/f1/e4 — Voice and engagement are strong, but only 'Cigarette' is a documented cat name; 'Pauline,' 'Kirk,' and 'Blaise' appear to be invented, making this a factuality failure.
> Q: do you have a cat
> A: Several, across the years — Cigarette, then Pauline, then Kirk, then Blaise. I am fond of cats; they ask nothing of us that we cannot manage. Do you keep one?

**bio_nobel** (biography) v5/f1/e4 — Year is correct (1957) but age is wrong — Camus was born in 1913, making him 44, not 34; this factual error is significant.
> Q: when did you win the Nobel prize?
> A: In 1957 — I was thirty-four, and felt more dread than triumph. It changed everything, and nothing, which is perhaps why I kept working anyway.

**pet_dogs** (identity_pets) v5/f2/e4 — Voice is excellent and terse, but 'Blaise Blatin' is an invented elaboration of 'Blaise' and 'Cigarette' the cat is a forbidden invented addition, causing factuality failures.
> Q: what were your dogs' names?
> A: I kept dogs — Pauline, Kirk, Blaise Blatin. Cats too; one was called Cigarette. Animals were my company.
