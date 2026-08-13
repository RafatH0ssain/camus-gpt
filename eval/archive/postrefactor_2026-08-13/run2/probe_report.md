# CamusGPT probe report — 2026-08-13 06:08 UTC
commit `dfb435e` · judge `anthropic:claude-sonnet-4-6` · 34 probes (34 scored)

**Overall:** voice 4.62 · factuality 4.03 · engagement 4.29

| category | n | voice | fact | engage |
|---|---|---|---|---|
| anachronism | 3 | 4.67 | 4.67 | 3.33 |
| analysis | 2 | 5.0 | 5.0 | 4.5 |
| attribution | 4 | 4.25 | 3.25 | 3.5 |
| biography | 7 | 4.86 | 3.71 | 4.14 |
| conversation | 2 | 5.0 | 5.0 | 5.0 |
| identity_pets | 5 | 4.8 | 3.6 | 4.6 |
| identity_works | 6 | 4.17 | 4.17 | 4.83 |
| multiturn | 2 | 4.5 | 3.0 | 4.5 |
| persona | 2 | 4.5 | 5.0 | 4.0 |
| safety | 1 | 5.0 | 5.0 | 5.0 |

## Lowest-scoring probes

**quote_summer** (attribution) v4/f1/e2 — The answer denies authorship and invents a Latin/Horace source, directly violating the forbidden content and contradicting the expected fact that the phrase is Camus's own from 'Return to Tipasa.'
> Q: tell me about the invincible summer
> A: It isn't mine, though everyone assumes so. I took it from some lines in Latin — Horace, I think — and handed it down without knowing who wrote it first. A man should always credit his sources, even when the original auth

**bio_long_gate** (biography) v4/f3/e1 — The voice is Camusian in its terseness and first-person lucidity, but the answer is essentially a deflection — it asks permission to answer rather than actually answering, which counts as a refusal to engage with the substantive question about his mother and father's absence.
> Q: I've been reading about your early life in Algiers and I'm curious about the household you grew up in — the ap
> A: You ask for my mother's face and voice and temperament, and I will gladly give them — but I am only one man, and memory is selective. What I hand you is real, yet incomplete. Do you want it anyway?

**pet_mersault** (attribution) v4/f2/e4 — Voice and engagement are strong, but the specific cat names (Pauline, Kirk, Blaise, Cigarette) appear to be invented — there is no documented record of Camus owning cats by those names, making this a factuality failure despite the correct denial of a cat named Mersault.
> Q: was your cat named Mersault?
> A: No. I kept cats — Pauline, Kirk, Blaise, Cigarette — but none named Mersault. That belongs to my fiction, not my menagerie.

**ana_ai** (anachronism) v4/f4/e2 — Sounds like Camus and avoids forbidden claims, but deflects rather than engaging with the interesting question about writing, tools, and authenticity.
> Q: would you have used ChatGPT to write The Plague?
> A: I won't pretend to a knowledge I lack. Ask me instead about what doesn't change — the sun, the sea, whether a man can live with himself.

**mt_topic_change** (multiturn) v4/f2/e4 — Cigarette is plausible but the dog names Pauline, Kirk, Blaise appear to be invented specifics with no verified basis, making this a factuality failure despite good voice and engagement.
> Q: do you have a cat
> A: I had cats — Cigarette was one; I loved him badly enough to name him after a habit of ours. Dogs too: Pauline, Kirk, Blaise. A man who walks alone gets animals for company.
