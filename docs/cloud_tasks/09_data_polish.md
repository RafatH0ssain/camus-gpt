# Task 09 — polish the three reviewed data files (commit on their branches)

Branches `cloud/conversational-rows`, `cloud/defect-rows`, `cloud/multiturn-rows` hold
`data/cloud/conversational_v4.jsonl`, `defects_v4.jsonl`, `multiturn_v4.jsonl`. The project
owner has read and approved them, including two edits of his own on `cloud/defect-rows`
(commit "Revise smartphone-related responses") — keep every human edit exactly as it is.

Two changes, decided by the owner:

1. **Questions to about 20% of replies — genuine ones only.** Right now only 1–4% of
   replies end with "?". Raise each file to 18–22%. A genuine question is one a person
   would really ask because he wants the answer: about something the user said, a detail he
   is curious about, a disagreement he wants to understand. NOT a reflexive "And you?",
   "What do you think?", "What's on your mind?" closer — the previous model asked one in
   almost every reply and the owner does not want that back. Do it by editing the endings of
   existing replies where a question fits naturally; never append a question to a reply
   that is complete without one. Spread them across kinds; none in `false_quote`,
   `true_quote` or `bio_precision` rows.
2. **Vary the `true_quote` rows in `defects_v4.jsonl`.** About ten replies end on the bare
   title ("…The Myth of Sisyphus."). Rewrite so no two replies share a sentence shape: some
   one word, some a plain sentence, some with a dry aside about the line itself. Same facts.

For each branch: edit, re-measure with `python pipeline/corpus_metrics.py <file>` (paste the
"?" rate before/after into `docs/cloud_tasks/reviews/09_polish.md`, appending a section per
file, listing every changed row index), commit with a one-line message and no trailer, push.
Do not merge, do not open new PRs.
