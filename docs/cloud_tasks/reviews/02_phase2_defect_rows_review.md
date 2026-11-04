# Review — task 02, `data/cloud/defects_v4.jsonl`

All 170 rows read (155 single-turn, 15 multi-turn). 24 rows changed, 0 deleted. Row numbers
are 0-based line indices. Kind counts are unchanged and match the brief: false_quote 40,
true_quote 25, anachronism_engage 45, no_invented_state 25, bio_precision 35. Every fact was
checked against the brief's table and `_IDENTITY_CARD`. No factual errors found.

## Rows changed

| row | kind | reason |
|---|---|---|
| 15 | false_quote | "I'd argue with the first half anyway" put an opinion in his mouth that the card does not support. |
| 21 | false_quote | Exact duplicate of row 6 ("No."). |
| 22 | false_quote | "That one is falsely mine" was garbled English. |
| 30 | false_quote | Exact duplicate of row 17 ("No. Not mine."). |
| 33 | false_quote | "They quote wrongly" implied a misquote of a real line, not a false attribution. |
| 58 | true_quote | Exact duplicate of row 43. |
| 67 | anachronism_engage | "A machine that answers, you say" attributed a description the user never gave (invented user state). |
| 71 | anachronism_engage | Same problem: "a little window, you say". |
| 73 | anachronism_engage | Stock line "your time, not mine", repeated in rows 81 and 93. The spec forbids stock refusals. |
| 74 | anachronism_engage | One of 17 replies opening "I don't know …"; opener varied. |
| 75 | anachronism_engage | "Beyond what you say": the user said nothing about Twitter. Opener also varied. |
| 81 | anachronism_engage | Stock line "That's your world, not mine". |
| 82 | anachronism_engage | Opener variety. |
| 84 | anachronism_engage | "You tell me" attributed words to the user; the reply now relates it to his own cinema. |
| 88 | anachronism_engage | Opener variety. |
| 91 | anachronism_engage | Opener variety. |
| 93 | anachronism_engage | Stock line "your century's invention, not mine"; he now admits ignorance in his own terms. |
| 94 | anachronism_engage | Near-duplicate of row 76 ("wouldn't know where to begin/start"). |
| 95 | anachronism_engage | Guessed what a dating app is; opener also varied. |
| 97 | anachronism_engage | Opener variety. |
| 105 | anachronism_engage | "You say" attributed a description to the user. |
| 112 | no_invented_state | Exact duplicate reply ("Hello."). |
| 116 | no_invented_state | Exact duplicate reply ("Hello."). |
| 122 | no_invented_state | Exact duplicate reply ("Hello."). |

The repetition across false_quote and true_quote replies (short denials, "The Myth of
Sisyphus.") is kept on purpose. The brief asks for brief, plain answers, and the variety is in
the asks.

## Metrics

Opener, first-word and duplicate figures come from `pipeline/corpus_metrics.py` and cover
single-turn replies only. The median and question rate cover all assistant turns and were
computed by hand.

| | before | after | target |
|---|---|---|---|
| median words, outside bio_precision | 7 | 7 | under 15 |
| question-ending | 1.6% | 1.6% | 25% or less |
| top-10 two-word opener share | 34.19% | 28.39% | none set |
| most-used opener ("i don't") | 17 | 9 | none set |
| first word "i" share | 23.87% | 18.06% | none set |
| exact / near duplicates | 6 / 9 | 0 / 0 | none set |
