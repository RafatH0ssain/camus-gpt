# Task 03 — a sharper eval (branch `cloud/eval-v4`)

## Why
`rag/eval_camus.py` has 34 probes. Run-to-run noise in the composite is about 0.25, larger
than the last improvement (+0.20), and several known weak spots have one probe each.

## Deliverables
1. **A v4 probe suite** in `rag/eval_camus.py`: keep the existing `PROBES` list byte-identical
   (history must stay comparable) and add `PROBES_V4` with ~50 new probes in the same dict
   format (`id`, `cat`, `turns`, `expect`, `forbid`). Add `--suite {v3,v4,all}` (default
   `v3`). Cover: misattributed vs genuine quotes (8), post-1960 topics where he should engage
   in his own words with honest unfamiliarity — not explain, not refuse (8), greetings where
   he must not invent the user's state (5), biography precision — first book, The Stranger's
   date, birthplace near Bône, posthumous works and that The First Man was unfinished, pets
   (10), multi-turn recall and topic change (8), naturalness — mundane, disagreement,
   emotional disclosure, where the expect field describes length and register (11).
2. **`rag/discriminator_sheet.py`**: builds the blind "him or the machine" test from
   `docs/HUMAN_EVAL_PROTOCOL.md`. Inputs: a model-answers jsonl (field `answer`, e.g. an eval
   `probe_scores.jsonl`) and a reference jsonl (field `response`), `--n 10`, `--seed`,
   `--min-words 15 --max-words 120`. Output: a shuffled markdown sheet with numbered,
   unlabelled passages and a separate `key.json`; `--score key.json answers.txt` prints
   accuracy. Strip names that give the source away only if trivial; do not paraphrase.
3. **Tests**: `rag/test_eval_suite.py` (suite selection, ids unique across suites, every probe
   has the required keys, v3 list unchanged — compare against a hash you compute from the
   current list) and `rag/test_discriminator_sheet.py` (fixtures in the test, no network).

## Constraints
No Ollama, no knowledge base, no network in tests (see `CLAUDE.md`). Do not change judging
code or scoring rubrics. Probe `expect` facts must match the identity card in
`rag/camus_rag.py`.
