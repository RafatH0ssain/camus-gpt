# Task 10 — de-template the older Phase 2 sources (branch `cloud/phase2-sources-v4`)

## Why
camus3's habits come from Phase 2. Its tracked sources repeat templates the model learned:
`data/camus_epistemic.jsonl` opens 35 replies with "Yes — The …" (likely why it now answers
"Yes —" to a misattributed quote) and 44 with "I didn't write"; `data/camus_refusals.jsonl`
repeats "No. A false…" and "Brackets and tags…" about 24 times each.

## Deliverable
Rewritten copies — never overwrite the originals:
`data/cloud/epistemic_v4.jsonl`, `data/cloud/refusals_v4.jsonl`,
`data/cloud/conversational_legacy_v4.jsonl` (from `data/camus_conversational.jsonl`),
`data/cloud/analysis_v4.jsonl`, `data/cloud/phase3_v4.jsonl`, `data/cloud/multiturn_legacy_v4.jsonl`.
Same row count, same order, same `kind` and same prompt per row (prompts may be lightly
de-formalised only if they read as written rather than typed). Change replies only.

## Rules for the rewritten replies
- Keep every fact and the behaviour each row teaches (correct, refuse, affirm, decline).
- No opener — two words — used more than 5 times per file; no reply-shape template.
- Affirmations must not start with "Yes —" more than 3 times per file; vary them.
- Refusals stay in voice and short: no lecture on why, no "I won't break character".
- At most 15% end with "?", genuine questions only.
- Persona rules in `CLAUDE.md`. Facts only from the original row and `_IDENTITY_CARD`.
- Delete nothing; if a row is factually wrong, fix it and list it.

## Done when
Per file: `python pipeline/corpus_metrics.py` before (original) and after, pasted into
`docs/cloud_tasks/reviews/10_sources_review.md`, plus every row whose facts you changed and
why. Commit one-line messages, no trailer, push, open a PR. Do not merge.
