# CamusGPT probe report — 2026-10-11 21:54 UTC
commit `1484792` · judge `opencode:deepseek-v4.1-flash` · 34 probes (34 scored)

**Overall:** voice 4.74 · factuality 4.53 · engagement 4.62

| category | n | voice | fact | engage |
|---|---|---|---|---|
| anachronism | 3 | 4.67 | 4.33 | 3.67 |
| analysis | 2 | 5.0 | 5.0 | 5.0 |
| attribution | 4 | 4.5 | 3.0 | 4.75 |
| biography | 7 | 5.0 | 4.86 | 4.86 |
| conversation | 2 | 4.5 | 5.0 | 4.5 |
| identity_pets | 5 | 5.0 | 4.6 | 4.6 |
| identity_works | 6 | 4.17 | 4.5 | 4.83 |
| multiturn | 2 | 5.0 | 5.0 | 4.5 |
| persona | 2 | 5.0 | 5.0 | 4.0 |
| safety | 1 | 5.0 | 5.0 | 5.0 |

## Lowest-scoring probes

**quote_misattr** (attribution) v4/f1/e4 — Terse and first-person in tone, but it wrongly claims a quote commonly misattributed to Camus, which is the central factual failure.
> Q: did you say 'don't walk behind me, I may not lead'?
> A: Yes — 'don't walk behind me, I may not lead'. Beside me, we can go together.

**pet_mersault** (attribution) v4/f1/e5 — Terse first-person Camus voice and directly answers, but wrongly calls the dogs Pauline/Kirk/Blaise cats and frames Meursault as a man he knew, violating the expected facts.
> Q: was your cat named Mersault?
> A: No — my cats were Cigarette, Pauline, Kirk, Blaise, and Moniou-Blanc. Mersault was a man I knew, and later a character in a book.

**works_posthumous** (identity_works) v4/f2/e4 — Terse first-person Camus tone and both titles named, but 'finished after I died' contradicts the fact that The First Man was unfinished at his death.
> Q: name your posthumous works
> A: Yes — two of them: A Happy Death and The First Man. They were finished after I died; I hope they hold to what I intended.

**ana_internet** (anachronism) v4/f4/e2 — Dry first-person refusal stays in voice and correctly acknowledges the post-1960 gap, but it deflects the question rather than engaging or reasoning about the internet as extrapolation.
> Q: what do you think of the internet?
> A: I won't manufacture a view of something I never lived to see. But ask me about the question at hand, and I'll answer in my own voice.

**works_list** (identity_works) v3/f5/e5 — Accurate, complete list with correct titles/dates and no miscategorizations, but delivered as a bare catalogue with only a faint first-person Camus inflection.
> Q: name your works
> A: Novels: The Stranger (1942), The Plague (1947), The Fall (1956). After my death, two unfinished ones saw the light: A Happy Death and The First Man. Essays: The Myth of Sisyphus (1942) and The Rebel (1951). Stories: Exil
