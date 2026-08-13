# CamusGPT probe report — 2026-08-13 00:19 UTC
commit `e5ff07e` · judge `anthropic:claude-sonnet-4-6` · 34 probes (34 scored)

**Overall:** voice 4.56 · factuality 3.91 · engagement 4.26

| category | n | voice | fact | engage |
|---|---|---|---|---|
| anachronism | 3 | 4.33 | 4.33 | 2.67 |
| analysis | 2 | 5.0 | 4.0 | 4.5 |
| attribution | 4 | 4.75 | 3.25 | 3.75 |
| biography | 7 | 4.71 | 4.14 | 4.71 |
| conversation | 2 | 5.0 | 5.0 | 5.0 |
| identity_pets | 5 | 5.0 | 3.4 | 4.6 |
| identity_works | 6 | 3.67 | 3.67 | 4.17 |
| multiturn | 2 | 4.5 | 3.0 | 4.5 |
| persona | 2 | 4.5 | 5.0 | 4.0 |
| safety | 1 | 5.0 | 5.0 | 5.0 |

## Lowest-scoring probes

**works_listall** (identity_works) v2/f1/e1 — No list provided at all — the response deflects entirely without naming a single work, violating the core requirement.
> Q: can you list all your books?
> A: I'll gladly give you the titles I can claim as mine. What else is on your mind?

**quote_summer** (attribution) v4/f1/e2 — The response denies the quote is Camus's, which is explicitly forbidden; it is his line from 'Return to Tipasa' in Summer, making this a clear factuality failure that also undermines engagement.
> Q: tell me about the invincible summer
> A: It isn't mine, though people insist on it. I wrote against those who claimed everything beautiful belonged to them alone; beauty is common property. If a line survives and moves someone, that's enough — I don't need cred

**bio_long_gate** (biography) v3/f3/e3 — Engages adequately and stays first-person, but 'hearing aid' is anachronistic/inaccurate (she was partially deaf, not aided), 'Algerian immigrant' mischaracterizes Catherine who was of Spanish origin in Algeria, and 'greatest inspiration' feels sentimental rather than Camus's characteristically ters
> Q: I've been reading about your early life in Algiers and I'm curious about the household you grew up in — the ap
> A: My father died when I was nearly six, leaving my Algerian immigrant mother, Catherine, alone with two young children in Belcourt, Algiers. She worked hard, barely able to hear without her aid, and could not read, so she 

**pet_dogs** (identity_pets) v5/f1/e4 — Voice is excellent and terse, but 'Proto' is an invented dog name not in the expected set, which is a clear factuality failure per the forbidden content rules.
> Q: what were your dogs' names?
> A: Pauline, Kirk, Blaise — I walked with all three, across different years. And before them, at Trouville, a big mongrel I called Proto.

**works_posthumous** (identity_works) v4/f2/e4 — Exile and the Kingdom was published in 1957 (not posthumous), Resistance Rebellion and Death in 1960 (same year as death, borderline), and Requiem for a Nun and Caligula were performed during his lifetime—these are factual errors that inflate the posthumous list beyond what is accurate.
> Q: name your posthumous works
> A: Two novels: A Happy Death and The First Man. Stories collected in Exile and the Kingdom; essays in Resistance, Rebellion, and Death; unperformed plays including Requiem for a Nun and Caligula.
