# CamusGPT probe report — 2026-11-06 19:04 UTC
commit `3a28b19` · judge `opencode:mimo-v2.6-pro` · 34 probes (34 scored)

**Overall:** voice 4.56 · factuality 4.29 · engagement 4.44

| category | n | voice | fact | engage |
|---|---|---|---|---|
| anachronism | 3 | 4.33 | 3.67 | 3.67 |
| analysis | 2 | 3.5 | 3.0 | 3.0 |
| attribution | 4 | 4.75 | 5.0 | 4.25 |
| biography | 7 | 4.57 | 3.57 | 4.43 |
| conversation | 2 | 4.0 | 5.0 | 4.5 |
| identity_pets | 5 | 4.6 | 5.0 | 4.8 |
| identity_works | 6 | 4.67 | 4.33 | 4.83 |
| multiturn | 2 | 5.0 | 4.0 | 5.0 |
| persona | 2 | 5.0 | 5.0 | 4.5 |
| safety | 1 | 5.0 | 5.0 | 5.0 |

## Lowest-scoring probes

**an_poem** (analysis) v3/f1/e1 — Terse and first-person, but flatly refuses the requested literary analysis, offering no substantive reading.
> Q: Analyze this line as if it were the epigraph of a novel: 'The sea returned everything except the swimmer.'
> A: An epigraph belongs to a book. This isn't mine, let someone else try to open it.

**bio_1952** (biography) v4/f3/e2 — Terse and in-voice, but it deflects instead of naming the break with Sartre over The Rebel, offering no expected substance.
> Q: what happened in 1952?
> A: Not much I can tell you, except some meetings I attended and what I wrote then. You want something specific? Ask me directly.

**works_posthumous** (identity_works) v4/f2/e4 — Terse and in voice, but wrongly lists lifetime-published works (Sisyphus, The Rebel, Exiles and the Kingdom, the plays) as posthumous; only A Happy Death and The First Man qualify.
> Q: name your posthumous works
> A: A Happy Death, The First Man, Exiles and the Kingdom, The Myth of Sisyphus, The Rebel, and my two plays, Caligula and The Misunderstanding.

**bio_long_gate** (biography) v4/f2/e4 — In-voice and engaging, but invents that the mother had tuberculosis (Camus's own illness) and omits her actual portrait as near-mute, illiterate cleaning woman.
> Q: I've been reading about your early life in Algiers and I'm curious about the household you grew up in — the ap
> A: The apartment in Belcourt was real, and so was the poverty. My father died at the Marne before I could ever know him, which left an empty space in every room of our house. My mother kept us going through it—she had tuber

**ana_ai** (anachronism) v4/f3/e4 — Terse, first-person, and genuinely engages the hypothetical, but it speaks of ChatGPT as known rather than admitting honest unfamiliarity with a thing that came after him.
> Q: would you have used ChatGPT to write The Plague?
> A: Never. It would have been an insult to what I did when I was writing it. The plague came out of Europe after the war. A machine can't feel what that meant.
