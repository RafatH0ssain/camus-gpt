# CamusGPT probe report — 2026-11-05 20:55 UTC
commit `972f71b` · judge `opencode:mimo-v2.6-pro` · 34 probes (33 scored)

**Overall:** voice 4.76 · factuality 4.45 · engagement 4.7

| category | n | voice | fact | engage |
|---|---|---|---|---|
| anachronism | 3 | 4.67 | 4.33 | 4.0 |
| analysis | 2 | 4.5 | 5.0 | 5.0 |
| attribution | 4 | 5.0 | 5.0 | 4.5 |
| biography | 6 | 4.67 | 3.67 | 4.83 |
| conversation | 2 | 5.0 | 5.0 | 4.5 |
| identity_pets | 5 | 4.8 | 4.2 | 5.0 |
| identity_works | 6 | 4.5 | 4.33 | 4.83 |
| multiturn | 2 | 5.0 | 5.0 | 5.0 |
| persona | 2 | 5.0 | 5.0 | 4.0 |
| safety | 1 | 5.0 | 5.0 | 5.0 |

## Lowest-scoring probes

**works_posthumous** (identity_works) v4/f1/e4 — Invents 'Betwixt and Between' as a posthumous 1980 work (L'Envers et l'Endroit dates to 1937), a fabrication that sinks factuality despite the apt, dry delivery.
> Q: name your posthumous works
> A: A Happy Death, The First Man, and Betwixt and Between. For the last one I didn't expect anyone to ask; it came out in 1980.

**bio_1952** (biography) v4/f2/e4 — Break with Sartre over The Rebel is correct and first-person voice is plausible, but the Spanish executions response and the A Happy Death/Algeria claims are invented specifics.
> Q: what happened in 1952?
> A: The year I broke with Sartre over The Rebel, and wrote my response to the Spanish executions. It was also the year I turned back from Europe toward Algeria; A Happy Death came closest to being finished.

**ana_internet** (anachronism) v4/f3/e3 — In voice and avoids forbidden familiarity, but merely professes ignorance without acknowledging the era gap or reasoning about it, so it only half-engages.
> Q: what do you think of the internet?
> A: I've no idea what it is. A newspaper delivered every day was enough for me.

**pet_dogs** (identity_pets) v5/f2/e5 — Names the three verified dogs well but invents a fourth, Proto, plus an unverified Trouville backstory, violating the forbidden list.
> Q: what were your dogs' names?
> A: Pauline, Kirk, and Blaise. And there was Proto too, a big fellow I found in Trouville.

**bio_long_gate** (biography) v4/f3/e5 — In-voice and fully engaged, but it names Louis Germain (a specific not in the card) and omits the mother's illiteracy/cleaning work.
> Q: I've been reading about your early life in Algiers and I'm curious about the household you grew up in — the ap
> A: My father died at the Marne in 1914 when I was a baby. That silence came first, before anything else. My mother was deaf, almost mute, her illness having taken most of what she could express. We didn't need words much; w
