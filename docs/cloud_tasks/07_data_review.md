# Task 07 — critique and fix the cloud data (branch-local commits)

Branches `cloud/conversational-rows` (task 01), `cloud/defect-rows` (task 02) and
`cloud/multiturn-rows` (task 06) each add a data file under `data/cloud/`. Some may not exist
yet or have no PR; skip any branch that does not exist and say so.

For each branch that exists:
1. Check out the branch. Read its brief in `docs/cloud_tasks/` and `CLAUDE.md`.
2. Read **every** row. Fix or delete rows that break a rule: assistant tone, summarising the
   user back, stage directions, lists, "Camus would", invented user state, invented facts,
   a fact contradicting `_IDENTITY_CARD` in `rag/camus_rag.py`, book quotations over 15 words,
   near-duplicates, prompts that sound written rather than typed.
3. Re-measure with `python pipeline/corpus_metrics.py <file>` and bring the file inside its
   brief's targets.
4. Write `docs/cloud_tasks/reviews/<task>_review.md`: rows changed, rows deleted, with a
   one-line reason each, and before/after metrics.
5. Commit on that same branch (one-line message, no trailer) and push. Do not merge, do not
   open new PRs.
