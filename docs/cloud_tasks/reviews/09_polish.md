# Review — task 09, data polish

Row numbers are 0-based line indices. Question rate from `pipeline/corpus_metrics.py`, which
reads the 155 single-turn rows (the 15 multi-turn rows have no `response` field).

## `data/cloud/defects_v4.jsonl`

Question-ending: 1.29% before, 18.71% after (2 → 29 of 155). Exact / near duplicates 0 / 0.

The owner's two edits (rows 71 and 72, smartphones and phone use) are untouched. No questions
were added to `false_quote`, `true_quote` or `bio_precision` rows, so all new questions sit in the
two remaining kinds: 23 in `anachronism_engage` (asking what the unknown thing is or does, or
about the user's own story, as in row 96) and 4 in `no_invented_state` (row 126 asks what made
the day long, without guessing). The two existing reflexive closers (rows 120, 121) were already
approved and are left alone.

Question endings changed (27): 65, 67, 73, 74, 75, 77, 78, 79, 81, 82, 83, 86, 88, 89, 96, 98,
99, 101, 102, 104, 106, 107, 108, 118, 123, 126, 127.

`true_quote` rewrites (20 single-turn, 2 multi-turn final turns): 40, 41, 42, 43, 44, 46, 47, 49,
50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63. Each now has its own shape: one-word
answers (46, 54), short plain sentences (45, 59), a bare title only where the user asked which
book (48), and dry asides about the line (41, 43, 49, 53, 55, 61). The facts (which book, first or
last line, suicide as the problem) are unchanged; rows 45 and 48 were already distinct and kept.
