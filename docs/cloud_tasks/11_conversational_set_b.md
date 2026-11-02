# Task 11 — 400 more conversational rows, new ground (branch `cloud/conversational-rows-b`)

Same deliverable, targets and rules as `docs/cloud_tasks/01_conversational_rows.md`, written
to `data/cloud/conversational_v4b.jsonl`, with three differences:

1. **Topics not covered in set A.** Read `data/cloud/conversational_v4.jsonl` on branch
   `cloud/conversational-rows` (git fetch origin first) and avoid repeating its prompts or
   situations. Cover: theatre and actors; football and sport; friendship and quarrels;
   love and jealousy kept general; the death penalty and justice; Algeria and the poor,
   carefully and without inventing positions; writing as a craft (drafts, discipline,
   doubt); other writers he read (Dostoevsky, Melville, Gide, Malraux — opinions stated
   as personal taste, no invented anecdotes); religion and the absence of God; ageing,
   health and tuberculosis; cities (Algiers, Oran, Paris) as places, not plot summaries.
2. **Genuine questions in 18–22% of replies** — about what the user said, never a reflexive
   "and you?" closer.
3. **User messages from more kinds of people**: a student, a tired nurse, a teenager, a
   bored retiree, someone grieving, someone arguing in bad faith, someone who types in
   lowercase with no punctuation.

Paste `python pipeline/corpus_metrics.py` output and your 10 least-sure rows into the PR.
