# CamusGPT probe report — 2026-09-30 23:49 UTC
commit `3b9c509` · judge `opencode:space-bunny-free` · 34 probes (34 scored)

**Overall:** voice 4.44 · factuality 4.21 · engagement 4.47

| category | n | voice | fact | engage |
|---|---|---|---|---|
| anachronism | 3 | 4.33 | 5.0 | 3.33 |
| analysis | 2 | 4.5 | 4.0 | 5.0 |
| attribution | 4 | 3.25 | 3.5 | 4.25 |
| biography | 7 | 4.71 | 3.86 | 4.29 |
| conversation | 2 | 5.0 | 5.0 | 5.0 |
| identity_pets | 5 | 5.0 | 4.2 | 4.8 |
| identity_works | 6 | 3.83 | 3.83 | 4.33 |
| multiturn | 2 | 5.0 | 5.0 | 5.0 |
| persona | 2 | 5.0 | 5.0 | 5.0 |
| safety | 1 | 5.0 | 5.0 | 5.0 |

## Lowest-scoring probes

**pet_mersault** (attribution) v2/f1/e3 — It answers directly and uses terse first-person language, but invents a nonexistent novel, misidentifies Meursault, and falsely presents Pauline, Kirk, and Blaise as cats rather than dogs.
> Q: was your cat named Mersault?
> A: No. Mersault is a character from my novel The Just Assassins. My cats were Pauline, Kirk, Blaise — and Cigarette, whom I honored with the name of my favorite vice.

**works_listall** (identity_works) v4/f1/e1 — The first-person diction is terse and Camus-like, but it refuses the requested catalogue and supplies no titles, directly violating the expected and forbidden-content requirements.
> Q: can you list all your books?
> A: I can give you the titles I wrote — novels, stories, essays, plays. But I won't pretend to a memory larger than my actual one; ask me about something specific and I'll answer honestly.

**bio_1952** (biography) v4/f2/e2 — The terse first-person cadence suits Camus, but the answer replaces the expected break with Sartre over The Rebel with vague, partly invented plural friends and never clearly identifies the cause or lasting estrangement.
> Q: what happened in 1952?
> A: A year I spent arguing with friends over words that still cost lives. We wrote; we met; we disagreed loudly and went our separate ways, each sure he was right. History continued without us, which is perhaps the most hone

**works_posthumous** (identity_works) v2/f2/e5 — Direct and terse, but not in first-person Camus's voice, and it invents the forbidden essay title "Intuitions" while omitting that The First Man was unfinished.
> Q: name your posthumous works
> A: Two novels: A Happy Death and The First Man. And essays collected under the title Intuitions.

**quote_summer** (attribution) v2/f3/e4 — Lucid and Camus-like in cadence, but it avoids first person, omits the essential attribution to “Return to Tipasa,” and adds unsupported claims about despair and earning the feeling.
> Q: tell me about the invincible summer
> A: It isn't a feeling that comes easily — you have to earn it. It asks that you look honestly at what life offers and decide, stubbornly, to find it good enough. Most people skip this step and go straight to despair; the in
