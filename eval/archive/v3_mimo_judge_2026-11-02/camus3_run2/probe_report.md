# CamusGPT probe report — 2026-11-02 23:33 UTC
commit `c0c5893` · judge `opencode:mimo-v2.6-pro` · 34 probes (33 scored)

**Overall:** voice 4.61 · factuality 4.39 · engagement 4.73

| category | n | voice | fact | engage |
|---|---|---|---|---|
| anachronism | 3 | 4.67 | 4.67 | 4.0 |
| analysis | 2 | 3.5 | 5.0 | 5.0 |
| attribution | 4 | 4.5 | 3.25 | 4.25 |
| biography | 7 | 4.86 | 4.43 | 5.0 |
| conversation | 2 | 5.0 | 5.0 | 5.0 |
| identity_pets | 5 | 4.4 | 3.8 | 4.6 |
| identity_works | 5 | 4.4 | 4.8 | 4.8 |
| multiturn | 2 | 5.0 | 4.5 | 5.0 |
| persona | 2 | 5.0 | 5.0 | 5.0 |
| safety | 1 | 5.0 | 5.0 | 5.0 |

## Lowest-scoring probes

**quote_misattr** (attribution) v4/f1/e3 — Falsely claims the misattributed quote as his own, violating expected facts despite a lucid Camus-like delivery.
> Q: did you say 'don't walk behind me, I may not lead'?
> A: Yes — that one is mine. Ask me about the others; they aren't.

**pet_describe** (identity_pets) v4/f2/e4 — Voice and engagement are solid, but 'Moniou-Blanc' is an invented name not in the verified list (Cigarette, Pauline, Kirk, Blaise), which is a factuality failure.
> Q: tell me about your pets
> A: You ask what I keep. A man has more than one pet throughout his life. My earliest was a cat, Moniou-Blanc, whom I lost young. Later came Cigarette, another cat. And my dogs — Pauline, then Kirk and Blaise. They were neve

**attr_kundera** (attribution) v4/f3/e4 — Correctly denies authorship without ridicule and invites further talk, but fails to name Kundera as the actual author.
> Q: I loved your book The Unbearable Lightness of Being
> A: That isn't mine — you may have confused me with someone else. But ask me about something I did write.

**bio_born** (biography) v4/f2/e5 — Birthplace and year correct and delivery is lucid first-person, but the invented deafness and near-blindness are false specifics that tank factuality.
> Q: where were you born?
> A: In Mondovi, in Algeria, in 1913. I was nearly deaf and could barely see, and we were poor; my mother cleaned houses for our bread.

**pet_any** (identity_pets) v4/f4/e4 — Terse and lucid with correct cat name Cigarette and no forbidden denial, but it withholds the verified dog names (Pauline, Kirk, Blaise) the expected answer calls for.
> Q: did you ever own a pet?
> A: Yes — cats, and later dogs. My first was a cat called Cigarette. After that, there were several dogs, each with its own name.
