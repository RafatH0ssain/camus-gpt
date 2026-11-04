# Task 13 — wire the cloud data into the training assembler (branch `cloud/assembler-v4`)

`pipeline/assemble_training.py` builds `build/training_v3/` (phase1_old, phase1_new, phase2).
New, hand-reviewed data lives (or will live, once their PRs merge) under `data/cloud/`:
`conversational_v4.jsonl`, `conversational_v4b.jsonl` (single-turn, Phase 1 voice),
`defects_v4.jsonl`, `multiturn_v4.jsonl` (Phase 2), and de-templated rewrites of the old
Phase 2 sources: `epistemic_v4.jsonl`, `refusals_v4.jsonl`, `conversational_legacy_v4.jsonl`,
`analysis_v4.jsonl`, `phase3_v4.jsonl`, `multiturn_legacy_v4.jsonl`. Read their formats on the
branches `cloud/conversational-rows`, `cloud/defect-rows`, `cloud/multiturn-rows`,
`cloud/phase2-sources-v4`, `cloud/conversational-rows-b` (git fetch origin; some may not exist
yet — design for the documented format).

## Changes
1. Add a **`phase1_v4`** set (keep `phase1_old` and `phase1_new` untouched and byte-identical):
   `phase1_new`'s inputs, but the generated conversational rows are **replaced** by
   `conversational_v4` + `conversational_v4b`, each weighted like the existing curated sets
   (`CURATED_WEIGHT`).
2. Add a **`phase2_v4`** set (keep `phase2` untouched): each old source is replaced by its
   `_v4` rewrite **when that file exists**, falling back to the original with a printed
   warning; plus `defects_v4` and `multiturn_v4` at the defect weight. The existing
   `data/drafts/defects_*` inputs stay as they are.
3. Record every input, its weight, row counts before/after dedup and the fallback decisions
   in `MANIFEST.json`, exactly like the existing sets.
4. Tests in `pipeline/test_assemble_training.py` with small fixtures: v4 sets use the v4 files,
   fall back correctly, never change the existing sets, and the manifest records the choice.
5. Update the notebooks' docs only by adding one line to `docs/ROADMAP.md`; do not edit the
   notebooks (a human selects the set name in Colab).

No gitignored data is available; tests must use fixtures. Commit, push, open a PR.
