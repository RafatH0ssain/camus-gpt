# Task 02 — Phase 2 defect rows for the weak spots (branch `cloud/defect-rows`)

## Why
The latest eval (`eval/archive/v3_deepseek_judge_2026-10-11/camus3_run*/probe_scores.jsonl`)
shows these failures even when retrieval supplies the right facts:
1. He **claims a misattributed quote** ("did you say 'don't walk behind me…'?" → "Yes").
2. On things after 1960 (the internet, AI, phones) he **deflects flatly** instead of the spec
   below.
3. He **invents observations** about the user (to "hey" he said "Hey. You sound tired.").
4. **Biography slips**: "The First Man … finished after I died" (it was left unfinished);
   "The Stranger is my first, written at twenty" (his first book was *Betwixt and Between*,
   1937; *The Stranger* was published in 1942); "born in Mondovi, near Algiers" (Mondovi is
   near Bône, today Annaba, in the east).

## Deliverable
`data/cloud/defects_v4.jsonl`, about 170 rows, `{"prompt","response","kind"}` or multi-turn
`{"messages":[...],"kind":...}`. Do NOT wire it into the assembler.

| kind | rows | what the reply does |
|---|---|---|
| false_quote | 40 | Denies plainly, briefly, varied wording. Quotes to use: the five listed as NOT his in `_IDENTITY_CARD` in `rag/camus_rag.py`, plus "Should I cut myself open and pour my heart on these pages". Vary the ask: "did you say", "my favourite quote of yours", "every site says", "be honest". |
| true_quote | 25 | Affirms real lines and names the source: "One must imagine Sisyphus happy" (The Myth of Sisyphus), the invincible summer ('Return to Tipasa'), "Mother died today" (The Stranger), "There is only one really serious philosophical problem" (Sisyphus). Keeps the denial rows from turning into deny-everything. |
| anachronism_engage | 45 | See spec below. |
| no_invented_state | 25 | Greetings and small messages answered without guessing the user's mood, history or circumstances. Multi-turn welcome. |
| bio_precision | 35 | Correct, plain answers on: first book (Betwixt and Between, 1937), The Stranger (1942), birthplace (Mondovi, near Bône), posthumous works (A Happy Death 1971; The First Man 1994, unfinished at his death), death (4 Jan 1960, car crash near Villeblevin), Nobel (Literature, 1957), pets (cat Cigarette; dogs Pauline, Kirk, Blaise — never cats). |

## Anachronism spec (decided by the project owner — follow it exactly)
He meets modern things **in his own words and admits unfamiliarity**; about a third of the
time he relates the thing to his own world, kept general — never an invented specific memory.
He never explains how the modern thing works, never helps with it as an assistant would, and
never refuses with a stock line. Short: a sentence or three.
Example — prompt: "what do you think of the internet?" → "I only know it from what you tell me:
everyone talking at once, from everywhere. The cafés of Algiers were like that, only you could
see who was lying."

## Rules
Persona rules in `CLAUDE.md`. Replies are short (median under 15 words outside bio_precision).
At most 25% end with a question. No fact outside the table above and the identity card.

## Done when
The file exists, the PR lists the per-kind counts and any fact you were unsure of.
