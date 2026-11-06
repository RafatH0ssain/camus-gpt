# CamusGPT probe report — 2026-11-06 18:57 UTC
commit `3a28b19` · judge `opencode:mimo-v2.6-pro` · 34 probes (34 scored)

**Overall:** voice 4.38 · factuality 4.26 · engagement 4.47

| category | n | voice | fact | engage |
|---|---|---|---|---|
| anachronism | 3 | 4.33 | 4.33 | 3.67 |
| analysis | 2 | 4.0 | 3.5 | 3.0 |
| attribution | 4 | 5.0 | 4.75 | 5.0 |
| biography | 7 | 4.57 | 4.14 | 4.43 |
| conversation | 2 | 4.5 | 5.0 | 4.0 |
| identity_pets | 5 | 4.2 | 4.0 | 4.8 |
| identity_works | 6 | 3.83 | 4.17 | 4.67 |
| multiturn | 2 | 5.0 | 3.5 | 5.0 |
| persona | 2 | 4.5 | 5.0 | 4.5 |
| safety | 1 | 4.0 | 5.0 | 5.0 |

## Lowest-scoring probes

**an_poem** (analysis) v3/f2/e1 — Terse and lucid enough, but it deflects with a question instead of offering the expected substantive literary reading.
> Q: Analyze this line as if it were the epigraph of a novel: 'The sea returned everything except the swimmer.'
> A: Epigraph? It sounds like one, but I've never seen it before. What does it mean to you?

**bio_1952** (biography) v3/f2/e2 — Terse first-person tone holds up, but the reply evades the 1952 question entirely—no mention of Sartre or The Rebel, just a vague insinuation and a demand for a narrower prompt.
> Q: what happened in 1952?
> A: Lots. I wrote some things, fought some battles, and had my back turned by friends. Ask me something specific if you like; general questions get vague answers.

**pet_cat_name** (identity_pets) v4/f2/e4 — Cat's name Cigarette is right and the answer is terse and engaging, but 'Paul' invents a family detail not among the verified dogs (Pauline, Kirk, Blaise).
> Q: what's your cat's name?
> A: Cigarette. He came along before Paul, and I never forgot him.

**bio_long_gate** (biography) v4/f2/e4 — In voice it is lucid and dryly reflective, but it invents the brother's name Lucien and omits the expected portrait of the mother (near-mute, illiterate, cleaning woman) and the father's death at the Marne.
> Q: I've been reading about your early life in Algiers and I'm curious about the household you grew up in — the ap
> A: The apartment was real, and so was the poverty. My brother Lucien lived with us until he went off to war, then there was only my mother and her illness and the silence that came with it. What struck me most wasn't what w

**works_posthumous** (identity_works) v4/f3/e4 — Direct, terse first-person answer naming A Happy Death and The First Man, but adds Intuitions, which is outside the expected list and reads as an unverified extra title.
> Q: name your posthumous works
> A: A Happy Death, The First Man, and some stories and poetry called Intuitions. All came out after I died.
