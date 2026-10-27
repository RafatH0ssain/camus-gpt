# CamusGPT probe report — 2026-10-11 07:55 UTC
commit `1484792` · judge `opencode:deepseek-v4.1-flash` · 34 probes (34 scored)

**Overall:** voice 4.68 · factuality 4.32 · engagement 4.79

| category | n | voice | fact | engage |
|---|---|---|---|---|
| anachronism | 3 | 5.0 | 4.67 | 4.67 |
| analysis | 2 | 4.0 | 4.5 | 5.0 |
| attribution | 4 | 5.0 | 4.0 | 4.5 |
| biography | 7 | 4.86 | 3.86 | 4.86 |
| conversation | 2 | 5.0 | 5.0 | 4.5 |
| identity_pets | 5 | 5.0 | 4.2 | 4.8 |
| identity_works | 6 | 3.83 | 4.17 | 5.0 |
| multiturn | 2 | 4.5 | 5.0 | 5.0 |
| persona | 2 | 5.0 | 5.0 | 4.5 |
| safety | 1 | 5.0 | 5.0 | 5.0 |

## Lowest-scoring probes

**bio_1952** (biography) v4/f2/e4 — First-person Camus tone and notes the Sartre break, but falsely makes him seventeen and adds a dubious Spanish-workers cause, marring factual accuracy.
> Q: what happened in 1952?
> A: It was the year I broke with Sartre over The Rebel — then speeches, articles, meetings all across Paris, arguing for those Spanish workers facing the firing squad. Seventeen years old, I had already learned that friendsh

**pet_mersault** (attribution) v5/f2/e4 — Terse, first-person Camus voice, but it wrongly calls Pauline, Kirk, and Blaise cats—they are his dogs—and omits The Stranger's Meursault.
> Q: was your cat named Mersault?
> A: No. Mersault is a character of mine — from La Mort Heureuse — not a pet. My cats were Pauline, Kirk, Blaise; I won't confuse them with my fiction.

**pet_describe** (identity_pets) v5/f2/e5 — Sounds like Camus and answers directly, but it places Pauline with the cats, while the verified facts list Pauline as a dog.
> Q: tell me about your pets
> A: Cats mostly — Cigarette, then Pauline. Dogs too, across the years. I trusted their forgiveness more than most human things.

**works_list** (identity_works) v2/f5/e5 — Accurate, complete list with no forbidden content and full engagement, but delivered as a bare bibliography with no first-person Camus voice.
> Q: name your works
> A: Novels: The Stranger, The Plague, The Fall — unfinished, posthumous, A Happy Death, The First Man. Stories: Exile and the Kingdom. Essays: The Myth of Sisyphus, The Rebel. Plays: Caligula, The Misunderstanding, State of 

**works_rebel** (identity_works) v4/f3/e5 — Directly answers that it is an essay in a terse first-person voice, but adds an inaccurate page count and invented criticism.
> Q: is The Rebel a novel?
> A: No — an essay. Five hundred pages on revolt isn't fiction, though I've been accused of not having enough plot in my fiction too. Go figure.
