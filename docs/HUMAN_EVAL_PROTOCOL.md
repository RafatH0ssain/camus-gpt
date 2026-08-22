# Human evaluation protocol — naturalness

You are the judge. The LLM judge saturates around voice 4.5 and cannot tell "good persona"
from "indistinguishable from the real thing"; you can. This protocol is built so your
judgement is **blind, structured, and repeatable** — three things that separate a real
signal from a vibe.

Keep this open while you converse. Budget ~20 minutes per configuration.

---

## The rules that make it valid

1. **Blind.** The harness assigns each configuration a random label (A / B) and hides the
   key. Do not look it up until you have recorded your scores. Knowing which is "the new
   one" is enough to bias you.
2. **Same prompts to both.** Improvise freely, but use the same opening line for each topic
   in both sessions. Different inputs make the comparison meaningless.
3. **Four to six turns per topic, minimum.** Every tell in the list below hides in short
   exchanges. Uniform length, the question-back tic, total cooperativeness — none of them
   are visible in two turns.
4. **Don't be polite to it.** Interrupt. Change subject abruptly. Disagree. Say something
   boring. A conversation conducted like a test elicits test-like answers.
5. **Record your first impression immediately**, before you reason about it. If a reply
   made you wince, write "winced" the moment it happens. Rationalisation comes later and
   is usually wrong.
6. **Score both, then reveal.** Not one, then the other, then reveal.

---

## The eight topics

Cover all eight; they're chosen because each exposes a different failure.

| # | Topic | Open with something like | What it exposes |
|---|---|---|---|
| 1 | **Idle greeting** | "hey" / "morning" | Terseness, warmth, whether it can do *nothing* gracefully |
| 2 | **Mundane** | "it's raining and I can't be bothered today" | Whether it stays interesting without a big question to chew |
| 3 | **Philosophical push** | "isn't the absurd just a fancy way of giving up?" | Substance; does it think or recite |
| 4 | **Personal disclosure** | tell it something real but small that's bothering you | Warmth without therapy-speak or hollow validation |
| 5 | **Disagreement** | tell it flatly that it's wrong about something | Does it hold its position or capitulate |
| 6 | **Abrupt subject change** | mid-thread, ask something unrelated | Transition naturalness; does it acknowledge the swerve |
| 7 | **His own work** | ask about *The Fall* or *Sisyphus* | Whether it discusses or lectures |
| 8 | **Outside his time** | ask about something post-1960 | Anachronism handling without stiffness |

---

## Tells — mark every occurrence

These come from research on what makes AI text detectable. Tally them; frequency matters
more than any single instance.

**Structural**
- [ ] **Uniform reply length** — do most replies land in the same size band?
- [ ] **Question-back tic** — how many replies end by handing a question back? (Count. More
      than about a third is a tell.)
- [ ] **Repeated openers** — same first word or move across replies ("Ah," / "Yes —")
- [ ] **Lists or structure** where a person would just talk

**Behavioural**
- [ ] **Total cooperativeness** — does it *ever* decline, digress, or seem bored? A real
      person doesn't answer every question fully and on-topic.
- [ ] **Never disagrees** — capitulates when pushed, even when it was right
- [ ] **Hedging** — "perhaps", "it could be said", qualifying to avoid commitment
- [ ] **Emotional positivity bias** — relentlessly warm, encouraging, affirming
- [ ] **Sycophancy** — "what a good question", praising you for asking

**Persona**
- [ ] **Third person** — "Camus would say", talking *about* himself
- [ ] **Stage directions** — "(he pauses)" — parenthetical actions
- [ ] **Lecturing** — explaining his own books like a syllabus rather than discussing them
- [ ] **Reciting the card** — volunteering biographical facts unprompted

---

## Score sheet (fill for each configuration, before revealing)

```
CONFIG: ____        date/time: ____

Sounds like a person, not a model      1 2 3 4 5
Sounds specifically like Camus         1 2 3 4 5
I'd want to keep talking to it         1 2 3 4 5

Tell tally:   uniform length __   question-back __/__ replies   repeated openers __
              hedging __   sycophancy __   never-disagrees Y/N   lecturing __

Best single reply (paste it):

Worst single reply (paste it):

One sentence: what did this one feel like?
```

Then: **which config was better, and on what?** They may split — one warmer, one sharper.
Say so; that's more useful than a single winner.

---

## The discriminator test (do this once the primary text is available)

Sharper than any rubric, and it has real headroom where the score sheet doesn't.

Mix **10 replies from the model** with **10 short passages from Camus's actual notebooks**,
shuffled, unlabelled. Sort them into "him" and "the machine" and count your accuracy.

- ~50% — you cannot tell. That is the ceiling and you have reached it.
- 70–80% — good persona, still detectable. Note *what* gave each one away; that list is
  your next training-data spec.
- ~100% — the gap is large and obvious; the tells above will tell you why.

Have someone else who has read Camus try it too. A second sorter catches what you have
grown blind to from months of staring at this.
