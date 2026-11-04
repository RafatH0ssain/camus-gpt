# Review — task 09, data polish

Row numbers are 0-based line indices. Question rate from `pipeline/corpus_metrics.py`.

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
