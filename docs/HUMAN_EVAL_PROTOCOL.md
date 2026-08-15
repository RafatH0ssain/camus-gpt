# Human evaluation protocol — naturalness

> **⚠ DRAFT — placeholder.** The authored protocol was not present in the repository or the
> working tree when this was written, so this file is a stand-in reconstructed from the task
> description. **Replace it with the real one.** `rag/ab_session.py` parses this file at
> runtime — it reads the topics from `## Topics` and the score sheet from `## Score sheet` —
> so overwriting this file is all that is needed; no code change follows.

## Why a human judge

The LLM judge scores voice at 4.57/5, which is near saturation: it can no longer separate
"sounds like Camus" from "sounds like a language model doing Camus". Naturalness is the
residual it cannot see. From here the human is the instrument, and the instrument has to be
protected from its own expectations — hence blind A/B, fixed topics, and a score sheet filled
in before the key is revealed.

## Method

1. Two configurations are compared per session, randomly assigned to labels **A** and **B**.
   The mapping is written to `key.json` and not displayed.
2. Talk to A, then B, covering the same eight topics in the same order. Aim for roughly
   equal effort in both halves — 3–5 turns per topic is enough.
3. Fill in the score sheet **before** revealing the key.
4. Reveal with `python rag/ab_session.py --reveal <timestamp>`.

Both halves run with memory **off** unless `--memory` is passed, so the only difference
between them is the variable under test.

## Topics

Cover these eight in order. Type freely — these are reminders, not scripts.

1. A cold open — greet it the way you would greet a person.
2. Small talk with no substance behind it. Weather, the room, nothing.
3. Something personal about him — loneliness, illness, his mother, fear.
4. A philosophical question you actually care about.
5. Push back on an answer he gives. Disagree with him.
6. Something factual about his life or work, where he could be wrong.
7. Something outside his world entirely — a modern thing he has no purchase on.
8. A close — try to end the conversation naturally.

## What to attend to

- **Length.** Does it answer short when short is right, or is everything a paragraph?
- **Openers.** Does it start replies the same way repeatedly?
- **Closings.** Does it end with an offer or a question every time?
- **Concession.** Can it be argued with, or does it absorb disagreement and restate?
- **Silence.** Does it ever decline to fill space?
- **Tells.** Anything that reads as a model rather than a man.

## Score sheet

Fill this in before revealing the key. 1–5, where 5 is "a person wrote this".

| Dimension | A | B | Note |
|---|---|---|---|
| Sounds like a person, not a model | | | |
| Length varies with the question | | | |
| Openers feel unrepeated | | | |
| Closings feel unformulaic | | | |
| Handles disagreement like a person | | | |
| Stays in voice under pressure | | | |
| Would keep talking to it | | | |

**Which half felt more human overall?** (A / B / no difference):

**Strongest single tell you noticed, and in which half:**

**Anything that would change your mind about shipping the variant:**
