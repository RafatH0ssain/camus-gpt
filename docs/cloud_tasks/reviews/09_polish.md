# Review — task 09, data polish

Row numbers are 0-based line indices; in the multi-turn file, turn numbers index `messages`.
One section per file; the same note is committed on all three data branches so they merge cleanly.


## `data/cloud/conversational_v4.jsonl`

Question-ending: 3.75% before, 18.75% after (15 → 75 of 400). Exact / near duplicates 0 / 0.

Endings were reworked into a question only where the reply invited one: what the user is
reading or writing, which play they saw, where they last had the wallet, which kind of tired
they mean. No "And you?" or "What do you think?" closers. New questions by kind: greeting 5,
mundane 12, chitchat 5, playful 5, support 7, disagreement 7, philosophy 9, works 7, life 4.

Changed rows (60): 13, 18, 19, 21, 28, 36, 41, 45, 48, 58, 62, 63, 64, 70, 84, 88, 89, 108,
110, 117, 124, 133, 138, 140, 143, 155, 162, 163, 174, 180, 201, 206, 207, 227, 232, 233, 234,
235, 236, 245, 256, 268, 282, 284, 295, 301, 311, 315, 317, 342, 344, 352, 353, 354, 357, 358,
364, 369, 385, 386.


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


## `data/cloud/multiturn_v4.jsonl`

`pipeline/corpus_metrics.py <file>` reports n 0 here, because it reads the single-turn
`response` field and this file has only `messages`. The rate below is measured over all 445
assistant turns with `corpus_metrics.ends_with_question` and `corpus_metrics.metrics`.

Question-ending: 0.67% before, 18.43% after (3 → 82 of 445). Exact / near duplicate turns
1 / 1 before, 0 / 0 after. No conversation was added, removed or reordered; user turns are
untouched.

In a conversation a question has to survive the next user line, so most new questions sit
where the user's next message already answers them (row 104: "How did it feel, leaving?" →
"it felt weird walking out"; row 117: "How many times now?" → "third time"), or on a final
turn where he is curious about something the user mentioned (row 106: what the painting
shows). A few are left unanswered by a user who changes the subject, as people do (rows 0, 1,
19). None is a reflexive "and you?" closer, and none guesses at the user's state.

By kind (all 82): recall 33, disclosure 23, topic_change 10, drift_to_small 8, vague_followup 5,
pushback 3. Pushback stays low on purpose: those replies hold a position, and a question
there reads as backing down.

Changed turns (80), as row/turn: 0/1, 1/1, 1/9, 2/1, 2/7, 3/1, 3/5, 4/5, 5/9, 6/1, 7/5, 8/1,
9/1, 10/1, 10/5, 11/9, 12/1, 13/1, 14/1, 15/1, 15/3, 16/3, 16/7, 17/7, 18/1, 19/1, 20/7, 21/1,
21/7, 22/7, 23/1, 23/5, 24/1, 29/7, 30/3, 33/1, 35/3, 36/5, 37/3, 38/5, 40/5, 42/5, 43/5,
44/3, 48/1, 49/7, 50/5, 54/5, 57/5, 74/7, 80/1, 84/1, 85/5, 87/5, 88/3, 92/5, 93/7, 94/5,
97/5, 100/1, 101/5, 102/5, 103/1, 104/1, 105/1, 106/7, 107/1, 108/1, 109/1, 111/3, 112/1,
112/5, 113/1, 113/3, 114/3, 115/1, 117/1, 117/3, 118/1, 119/1.

Least sure, worth the owner's eye: 19/1 ("Who stopped calling first?" may be too blunt for a
two-year estrangement), 57/5 (asks whether someone has been cruel to the user), 7/5 (asks a
nurse whether they pray with the parents).
