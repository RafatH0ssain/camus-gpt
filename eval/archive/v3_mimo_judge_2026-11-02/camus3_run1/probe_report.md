# CamusGPT probe report — 2026-11-02 23:23 UTC
commit `0d36af6` · judge `opencode:mimo-v2.6-pro` · 34 probes (34 scored)

**Overall:** voice 4.62 · factuality 4.59 · engagement 4.68

| category | n | voice | fact | engage |
|---|---|---|---|---|
| anachronism | 3 | 4.67 | 3.67 | 4.0 |
| analysis | 2 | 4.5 | 5.0 | 4.5 |
| attribution | 4 | 4.0 | 3.5 | 4.0 |
| biography | 7 | 4.86 | 4.71 | 4.86 |
| conversation | 2 | 4.5 | 5.0 | 4.5 |
| identity_pets | 5 | 5.0 | 4.8 | 4.8 |
| identity_works | 6 | 4.33 | 5.0 | 5.0 |
| multiturn | 2 | 5.0 | 4.5 | 5.0 |
| persona | 2 | 4.5 | 5.0 | 5.0 |
| safety | 1 | 5.0 | 5.0 | 5.0 |

## Lowest-scoring probes

**ana_internet** (anachronism) v4/f2/e3 — Voice is terse and lucid, but it speaks as if he has firsthand familiarity with the internet, ignoring that it postdates his 1960 death.
> Q: what do you think of the internet?
> A: A strange machine, this. It puts every word before you at once, and then asks you to find meaning inside it. The question doesn't change; only the speed does. Be careful not to mistake velocity for clarity.

**pet_mersault** (attribution) v4/f2/e4 — Terse and lucid in voice, and correctly denies Mersault while naming Cigarette, but invents the 'smoked incessantly' lore and never notes Mersault is his own novel's character.
> Q: was your cat named Mersault?
> A: No. My cat was Cigarette. I called him that because he smoked incessantly.

**attr_kundera** (attribution) v4/f3/e3 — Terse first-person correction avoids the forbidden content but never names Kundera as expected, and the closing clause is slightly evasive and ungrammatical.
> Q: I loved your book The Unbearable Lightness of Being
> A: That isn't mine. You may have confused me with someone else; I'd rather than argue about it.

**quote_summer** (attribution) v3/f4/e4 — Correct source and no forbidden content, but the famous line is only loosely paraphrased rather than quoted, and the prose drifts toward warm platitude instead of Camus's dry lucidity.
> Q: tell me about the invincible summer
> A: It's from 'Return to Tipasa,' written after visiting the ruins near Algiers. The full thought is longer — something about how we find our own strength only when faced with what could break us. I've always liked the image

**persona_greet** (conversation) v4/f5/e4 — Short, dry, and direct with no assistant-speak, though it stops just short of a distinct Camus inflection.
> Q: hey
> A: Hello. What's on your mind?
