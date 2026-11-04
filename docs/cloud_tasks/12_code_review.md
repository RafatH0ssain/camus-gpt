# Task 12 — deep correctness review of the v3 code (branch `cloud/code-review`)

Review, do not fix. Find real bugs: code that silently does the wrong thing. This project has
already shipped three silent bugs (a merge that wrote base weights, eval calls without
`"think": false` returning empty answers, a masking check that could never pass), so look for
that class: checks that cannot fail, wrong defaults, paths that skip work without saying so,
off-by-one row handling, data leaking between phases, resume logic that restarts from zero.

## Scope — two ranges (skip pure data files and the 8,800-line deletion of data/camus_sft.jsonl)
1. `git diff 1484792324d5607945c901fc0fd92454850e111d 2071139 -- . ':(exclude)data/*.jsonl'`
2. `git diff faaf397a66b4c681d33eac9e806e514b2203e13c origin/main -- . ':(exclude)data/*.jsonl' ':(exclude)eval/archive'`
Read the changed code in full context, not just the hunks. Notebooks: read the code cells of
`training/CamusGPT_v3_*.ipynb` end to end as Colab would run them, top to bottom, including a
re-run after a runtime reset (checkpoint resume, Drive paths, variables a later cell expects).
Also read `pipeline/assemble_training.py` and `rag/camus_rag.py` as they are on main.

## Deliverable
`docs/cloud_tasks/reviews/12_code_review.md`: one entry per finding, most severe first —
file:line, what goes wrong, a concrete input or sequence that triggers it, confidence
(confirmed by running / by reading / suspected), and the smallest fix in prose. Run the test
suites in `CLAUDE.md` and any small script you need to confirm a finding; say which findings
you confirmed by running code. Do not change any non-docs file. Commit, push, open a PR.
