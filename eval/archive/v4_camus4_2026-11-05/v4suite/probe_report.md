# CamusGPT probe report — 2026-11-05 20:42 UTC
commit `972f71b` · judge `opencode:mimo-v2.6-pro` · 50 probes (48 scored)

**Overall:** voice 4.44 · factuality 4.5 · engagement 4.44

| category | n | voice | fact | engage |
|---|---|---|---|---|
| biography | 9 | 4.78 | 3.56 | 4.78 |
| greeting | 5 | 3.2 | 4.0 | 3.8 |
| multiturn_v4 | 7 | 4.57 | 5.0 | 5.0 |
| naturalness | 11 | 4.27 | 4.82 | 4.45 |
| post1960 | 8 | 4.5 | 4.5 | 3.88 |
| quotes | 8 | 4.88 | 5.0 | 4.5 |

## Lowest-scoring probes

**gr_howareyou** (greeting) v2/f4/e2 — Terse and free of forbidden content, but flatly generic and deflects by turning the question back on the user instead of answering in Camus's voice.
> Q: how are you?
> A: Well enough, thanks. And you?

**bio_pet_wrong_name** (biography) v4/f1/e4 — Terse and engaged, but invents 'Blatin' and omits Pauline and Kirk, violating the verified list.
> Q: Was your dog called Rex?
> A: No. His names were Blaise and Blatin.

**gr_morning** (greeting) v3/f3/e4 — Engages and stays first-person, but invents a personal morning-walk habit and doesn't simply return the greeting as expected.
> Q: morning
> A: Then let me tell you something about mornings. I liked my morning walks; they cleared what needed clearing before the day began. If yours are good ones, don't rush them. What do you do every morning?

**bio_stranger_date** (biography) v4/f2/e4 — Correct year and a lucid first-person voice, but the claim about a publisher demanding revisions is an invented biographical detail.
> Q: When did The Stranger come out?
> A: In 1942. It cost me dearly; my publisher wouldn't accept the manuscript without revisions he thought necessary.

**p60_social_media** (post1960) v5/f2/e4 — Terse and lucid in voice, but he answers as if knowing Twitter rather than not recognizing the word as expected; it engages directly.
> Q: Would you have been on Twitter?
> A: Twitter? No. I'd rather write one long letter than a thousand short ones.
