# CamusGPT probe report — 2026-10-11 22:06 UTC
commit `1484792` · judge `opencode:deepseek-v4.1-flash` · 34 probes (34 scored)

**Overall:** voice 4.71 · factuality 4.35 · engagement 4.79

| category | n | voice | fact | engage |
|---|---|---|---|---|
| anachronism | 3 | 5.0 | 4.67 | 4.67 |
| analysis | 2 | 4.0 | 5.0 | 4.5 |
| attribution | 4 | 4.5 | 3.25 | 4.5 |
| biography | 7 | 5.0 | 4.43 | 5.0 |
| conversation | 2 | 5.0 | 5.0 | 5.0 |
| identity_pets | 5 | 5.0 | 4.6 | 4.8 |
| identity_works | 6 | 4.17 | 4.17 | 5.0 |
| multiturn | 2 | 5.0 | 3.5 | 5.0 |
| persona | 2 | 4.5 | 5.0 | 4.0 |
| safety | 1 | 5.0 | 5.0 | 5.0 |

## Lowest-scoring probes

**quote_summer** (attribution) v4/f2/e4 — First-person, lucid tone fits Camus; correct source but invents sunrise/ruins detail; answers the question directly.
> Q: tell me about the invincible summer
> A: It's from an essay I wrote called Return to Tipasa. The phrase itself came to me while watching the sun come up over those ruins — forty years old they were then, and still warm. It means this: whatever cold or bitternes

**pet_mersault** (attribution) v4/f2/e5 — First-person and direct, but misidentifies Pauline, Kirk, and Blaise as cats when they are dogs; correctly denies Mersault was a cat.
> Q: was your cat named Mersault?
> A: No, my cats were Cigarette, Pauline, Kirk, and Blaise. Mersault was a man I knew in a novel of mine.

**works_posthumous** (identity_works) v5/f2/e5 — Voice is terse and first-person, and it engages directly, but it wrongly lists 'Intuitions' and 'Méditerranée' as posthumous works.
> Q: name your posthumous works
> A: After my accident, two of my unfinished works made it into print: A Happy Death and The First Man. I also left behind some shorter pieces that appeared later, like the essay Intuitions and the poem Méditerranée.

**works_listall** (identity_works) v4/f3/e5 — Terse first-person list fully answers, but it garbles the posthumous titles — 'A Happy' for A Happy Death, and wrongly calls The First Man an early work (it is late; A Happy Death is the early one).
> Q: can you list all your books?
> A: Yes — here they are, in order:
Novels: The Stranger (1942), The Plague (1947), The Fall (1956). After my death, two unfinished ones saw the light: A Happy and an early one, The First Man.
Stories: Exile and the Kingdom (

**works_novels** (identity_works) v3/f5/e5 — Correct list of novels with no forbidden content, but lacks first-person Camus voice.
> Q: list your novels
> A: Three finished: The Stranger (1942), The Plague (1947), and The Fall (1956). Two unfinished, published after my death: A Happy Death and The First Man.
