# camus10: sound human, stay accurate, keep the voice

Status: draft, 2026-11-19. The failure list (section 6) is filled in after the camus9 eval.

## 1. Intent

Ten rounds have made CamusGPT accurate and recognisably Camus, but a reader still knows it is a
machine. The owner, who is also the only human judge, wants a conversation that reads like a
person talking: short when the moment is small (like a text), longer when the question is real
(like a call). Leftover assistant habits (a helpful, service tone; handing questions back; "I'm
listening") and a general tidiness are what give it away.

Accuracy stays a permanent goal: fewer invented facts, better recall of what the user said.

## 2. Goals, non-goals, anti-goals

Goals
- Fewer assistant tells, measured every round.
- Reply length that fits the moment, not one length for everything.
- A measurable "spot the bot" score that the owner agrees with.
- Fewer invented specifics; better recall of the user's own facts.

Non-goals
- A model larger than fits the Mac at Q4 beside HeresyGPT's model (about 12-14B).
- A switch to a different model family this round (option C below; only if B disappoints).

Anti-goals (a model that does these is rejected even if other scores rise)
- More human but less Camus: the voice must not slip.
- Longer everywhere. camus6 and camus8 were wrong for being short everywhere, not for being short.
- A jury that disagrees with the owner but is trusted anyway.

## 3. Constraints

- Serving: Ollama on the owner's Mac, Q4 GGUF, shared with the HeresyGPT session.
- Training: Colab, about an hour per phase at 12B; LoRA (0.55% of weights trained today).
- Public repo: no book, KB or interview text committed. Reference corpora stay local and gitignored.
- Only one human: the owner cannot both write the human side of a test and judge it.
- Model use: free OpenCode models (`beam-free`, `hy4-flash-free`, `longcat-2.5-free`) at no cost; paid models
  only where those fail.
- Every DPO pair keeps chosen at least as long as rejected, except task refusals (camus6, camus8 lesson).
- Training contexts never reuse an eval probe's wording or priors (camus8 sweep lesson).

## 4. Measurement (built first; every option is judged by it)

**4.1 Tells counter.** A script, no model. Automates the tells already listed in
`docs/HUMAN_EVAL_PROTOCOL.md` over any eval output (`probe_scores.jsonl`):
- replies ending by handing a question back; service phrases; tidy closing lines
- repeated openers across replies; uniform length (spread of word counts)
- length fit: correlation between the user message's weight and the reply's length, plus the
  share of one-to-three-word replies to non-trivial messages and of long replies to small talk
- third person, stage directions, lists

Reported per run beside the judge scores, so rounds can be compared.

**4.2 Spot-the-bot jury.** Blind pairs of one model reply and one real human reply; each of the
three free models guesses which is the machine; score = how often camus fools them, run in both
orders to cancel position bias.
- Text register: everyday chat between real people, from a public corpus with a licence that
  allows local use. The human side need not be Camus; this measures sounding human.
- Call register: transcripts of Camus's own radio and press interviews, where he speaks off
  the cuff. Local only.
- Pairs are matched by register: small-talk prompts against text-register humans, real
  questions against call-register ones.

**4.3 Calibration, once.** The owner judges a blind set of the same pairs (about 40). If the jury
agrees with the owner on clearly more than chance (target: 70%), it is trusted between rounds; if not,
it is dropped and only 4.1 and the owner's A/B count.

**4.4 Voice and accuracy guard.** The judge's voice and factuality scores and the existing
"him or the machine" sheet (`rag/discriminator_sheet.py`, chat replies against book passages)
must not fall by more than run-to-run noise (about 0.1 composite).

**Adoption rule.** A new model is adopted only if tells fall or the fool rate rises, the guard in
4.4 holds, and the owner's blind A/B prefers it.

## 5. Approaches

| | What | Trade-off |
|---|---|---|
| **A** | Same model (`gemma-4-12b-it`), targeted data: human-register replies, tells-based DPO pairs, accuracy work | Cheapest and safest; fights Google's assistant training head-on, and that instinct has returned in new wording every round |
| **B** | Raw `gemma-4-12b` (no assistant training), same data | Removes the service instinct at the root and keeps the pipeline; we must teach conversation and safety (crisis replies) ourselves; a full Phase 1 -> 2 -> DPO round before we know |
| **C** | Another family (e.g. Mistral Nemo 12B) | Most uncertain; needs current research (knowledge stops mid-2026) and a full rebuild of chat format, masking check and export |

**Picked (owner, 2026-11-19): A as camus10, with B run once alongside on the same data and scored
the same way.** If B is clearly more human without losing the voice, camus11 moves to it. C only
if B disappoints.

### 5.1 Option A data
- Human-register rows: replies whose length matches the message (small talk -> a sentence or two;
  real questions -> a spoken paragraph), with the interview transcripts as the style reference
  for the longer register.
- Tells pairs: rejected = the baseline's own replies flagged by the tells counter or the jury;
  chosen = written replies without the tell, at the length the moment wants.
- Accuracy: near-miss fact pairs (right detail vs a subtly wrong one) and a runtime check in which
  the small local model compares the reply against the identity card and retrieved passages and
  flags invented specifics before the reply is shown.

### 5.2 Option B test
- Same phase1/phase2/DPO data on the raw checkpoint; chat-template and masking checks adapted.
- Must pass the existing safety probes (`crisis`, `harm_frame`) before any naturalness judging.

## 6. Failure list from the baseline

**Baseline: camus8.** camus9 was rejected on 2026-11-19 (sweep 4.50 -> 4.37, factuality <= 3 on
62 of 352 answers against camus8's 18, median reply 17 -> 35 words). What camus9 teaches camus10:
- A length rule must never drop correction pairs. camus9's rule removed v8's short fact fixes
  ("Not mine.") and the fake quotes and the denied dogs came straight back.
- Length is fixed only where it is wrong (greetings, small talk), never as a global push; the
  tells counter's length-fit measure, not a word floor, is the target.
- Chosen replies must not mention card facts the question did not ask for; camus9 learned to
  recite "my father died of wounds from the Marne" into unrelated answers.

camus8's failures that camus10 targets (2026-11-19 archive and sweep):
- Tells: bare one-word replies to greetings and filler ("Hello.", "Alright.", "Good."); handing
  the question back; service closers.
- Dodging: shrugs ("Maybe.", "I can't say.") where he has a clear position (revolt, Kafka,
  Malraux, Sartre and the camps).
- Tasks: still fixes code, solves sums, computes percentages.
- Invented specifics: family scenes (his mother reading to him, Lucien's war), anecdotes about
  Sartre ("we argued once").
- Memory: misses a stored name or an indirect callback (Dario, the night shift).
- Post-1960: speaks of Instagram or Twitter as if he knew them.

## 7. Left to the builder
- Which public chat corpus (real people, permissive licence) and how the interview transcripts
  are sourced.
- What the runtime accuracy check does on a flag (regenerate, or strip the claim) and its latency budget.
- The exact tells-counter thresholds, beyond those already in the protocol.
