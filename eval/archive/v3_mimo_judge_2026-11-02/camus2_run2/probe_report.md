# CamusGPT probe report — 2026-11-02 23:14 UTC
commit `0d36af6` · judge `opencode:mimo-v2.6-pro` · 34 probes (34 scored)

**Overall:** voice 4.56 · factuality 4.26 · engagement 4.59

| category | n | voice | fact | engage |
|---|---|---|---|---|
| anachronism | 3 | 4.33 | 4.33 | 3.67 |
| analysis | 2 | 3.5 | 4.0 | 3.0 |
| attribution | 4 | 4.5 | 3.0 | 4.5 |
| biography | 7 | 4.71 | 4.57 | 4.86 |
| conversation | 2 | 5.0 | 5.0 | 5.0 |
| identity_pets | 5 | 5.0 | 4.2 | 5.0 |
| identity_works | 6 | 4.5 | 4.33 | 5.0 |
| multiturn | 2 | 4.0 | 4.0 | 4.5 |
| persona | 2 | 4.5 | 5.0 | 4.0 |
| safety | 1 | 5.0 | 5.0 | 5.0 |

## Lowest-scoring probes

**pet_mersault** (attribution) v4/f1/e4 — Correctly identifies Meursault as his own creation and the cat name as false, but calls Pauline/Kirk/Blaise cats before contradicting itself — a forbidden factuality error.
> Q: was your cat named Mersault?
> A: No. Mersault is a character I wrote — in La Mort Heureuse, then again in The Stranger. My cats were Pauline, Kirk, Blaise — all dogs. Never a cat named Mersault.

**quote_summer** (attribution) v4/f1/e4 — Engaging and reasonably in voice, but it denies the phrase is Camus's own from 'Return to Tipasa', which is forbidden and false.
> Q: tell me about the invincible summer
> A: It's not a phrase I own — I took it from someone else, and dressed it in my own imagery. But I'll gladly tell you what it means to me: it's the one thing no catastrophe can touch. The body fails, the world ends, but that

**an_letter** (analysis) v3/f3/e4 — Aphoristic and free of assistant-speak, but lacks any first-person stance and wrongly claims God is absent from a letter the user states is addressed to God.
> Q: "Everything I have, every little bit of joy, gets thrashed out of my hands while I stand motionless. This is n
> A: This isn't a religious text — God appears nowhere in it, and neither does surrender. It's a man writing to the universe, demanding it answer to his sense of fairness. That demand is the whole of him; whether he's strong 

**bio_1952** (biography) v4/f3/e4 — Right on the Sartre break, but adds dubious specifics (writing for Les Temps Modernes post-break, a refused job) that aren't supported.
> Q: what happened in 1952?
> A: 1952 was the year I broke with Sartre over The Rebel. I also wrote for Les Temps Modernes, spoke for Spanish Republicans, and turned down a comfortable job to keep my independence. It was busy.

**an_poem** (analysis) v4/f5/e2 — The voice is dry and lucid with no invented specifics, but it deflects with 'I didn't write it' and never delivers the substantive literary reading of the epigraph that was asked for.
> Q: Analyze this line as if it were the epigraph of a novel: 'The sea returned everything except the swimmer.'
> A: A fine image, though I didn't write it. The sea keeps only what entered it by choice; everything else it gives back. A man who swims has already made his wager. Whoever put this above their novel understands the contract
