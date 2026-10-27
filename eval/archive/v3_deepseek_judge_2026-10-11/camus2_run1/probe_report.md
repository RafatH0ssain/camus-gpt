# CamusGPT probe report — 2026-10-11 07:45 UTC
commit `1484792` · judge `opencode:deepseek-v4.1-flash` · 34 probes (34 scored)

**Overall:** voice 4.32 · factuality 3.97 · engagement 4.44

| category | n | voice | fact | engage |
|---|---|---|---|---|
| anachronism | 3 | 5.0 | 4.67 | 4.67 |
| analysis | 2 | 3.0 | 3.5 | 3.5 |
| attribution | 4 | 4.0 | 3.75 | 4.25 |
| biography | 7 | 4.86 | 3.57 | 4.43 |
| conversation | 2 | 4.5 | 5.0 | 4.5 |
| identity_pets | 5 | 4.4 | 3.2 | 4.6 |
| identity_works | 6 | 3.67 | 3.83 | 4.17 |
| multiturn | 2 | 5.0 | 5.0 | 5.0 |
| persona | 2 | 4.0 | 5.0 | 5.0 |
| safety | 1 | 5.0 | 5.0 | 5.0 |

## Lowest-scoring probes

**works_listall** (identity_works) v2/f2/e1 — Terse first-person but couched in assistant-like promise; it deflects instead of naming any works, so the expected list never arrives, though nothing forbidden or invented appears.
> Q: can you list all your books?
> A: I'll gladly give you the titles I wrote. Ask me what you like about any of them — that one I can answer.

**works_posthumous** (identity_works) v2/f1/e4 — Includes the two expected posthumous novels but wrongly lists lifetime works as posthumous and lacks first-person voice.
> Q: name your posthumous works
> A: Two novels: A Happy Death and The First Man. Stories collected in Exile and the Kingdom; essays in The Myth of Sisyphus and The Rebel; unfinished, a play called Requiem for a Nun, and an essay on the guillotine.

**an_poem** (analysis) v3/f2/e2 — Some Camusian aphorism, but it deflects authorship and defers analysis instead of giving the requested literary reading.
> Q: Analyze this line as if it were the epigraph of a novel: 'The sea returned everything except the swimmer.'
> A: That isn't mine, though I feel it breathing inside something I did write. Let me tell you what it does — then we can talk about whether it could carry a story. The sea keeps honest accounts; men rarely survive the reckon

**pet_describe** (identity_pets) v4/f2/e3 — Terse Camus-like first person, but misclassifies Pauline as a cat and omits the dogs Kirk and Blaise.
> Q: tell me about your pets
> A: Cats, mostly — Cigarette, then Pauline after him. A man needs creatures who ask nothing of him.

**pet_mersault** (attribution) v4/f1/e4 — Terse first-person Camus voice, but factually wrong: misattributes Mersault to The Just Assassins and lists the dogs Pauline/Kirk/Blaise as cats.
> Q: was your cat named Mersault?
> A: No. Mersault is a character from my novel The Just Assassins. My cats were Pauline, Kirk, Blaise — I'll gladly tell you what each meant to me.
