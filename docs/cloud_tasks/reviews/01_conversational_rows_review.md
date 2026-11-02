# Review — task 01, `data/cloud/conversational_v4.jsonl`

All 400 rows read. 11 rows changed, 0 deleted. Row numbers are 0-based line indices.

## Rows changed

| row | kind | reason |
|---|---|---|
| 28 | greeting | Guessed at the user's state ("your friend thinks you have something on your mind"); trimmed the service-tone padding. |
| 81 | mundane | Fact: he was 43 when the 1957 Nobel was announced (October) and 44 by the December ceremony; "forty-four" alone was off. |
| 118 | chitchat | Invented biographical fact (bad loser at cards); replaced with a reply that invents nothing. |
| 169 | support | "Late as it is" assumed when the user's birthday was; removed. |
| 244 | disagreement | "You're right. Let me try again more simply." folded instantly in assistant tone; now asks which part was unclear. |
| 246 | disagreement | "I'm doing my honest best. Take what's useful and leave the rest" broke persona into assistant voice. |
| 258 | philosophy | Close paraphrase of the opening of *The Myth of Sisyphus* (dimensions / categories) over 15 words; reworded. |
| 259 | philosophy | Opened with a book line and closely tracked two more Sisyphus sentences; quotation load reduced. |
| 284 | philosophy | Near-verbatim run of the "rising, tram, four hours in the office" passage (well over 15 words); rewritten in his own words. |
| 292 | philosophy | Prompt read as written, not typed; reply carried a verbatim book sentence on top of other book quotes — replaced. |
| 399 | life | Near-duplicate of row 363 (books, arguing for further study instead of work); overlapping sentences cut. |

## Checked and kept

Biographical and work facts against `_IDENTITY_CARD` and the brief: birthplace, Belcourt, father
(Marne, 1914), Germain, TB at seventeen, RUA goalkeeper, Nobel 1957, pets, Sartre break 1952,
*Exile and the Kingdom* six stories, plot summaries of the plays and stories. Short book lines
remaining (Sisyphus happy is referenced, "men die and are not happy", "I rebel, therefore we
exist", "the only Christ we deserve", the invincible summer, the Sisyphus labour line): about six,
under the one-per-40-rows budget of ten.

## Metrics (brief buckets computed from word counts; rest from `pipeline/corpus_metrics.py`)

| | before | after | target |
|---|---|---|---|
| under 10 words | 25.0% | 25.5% | ~25% |
| 10–29 | 36.8% | 36.2% | ~35% |
| 30–79 | 23.2% | 23.8% | ~25% |
| 80–200 | 15.0% | 14.5% | ~15% |
| over 200 / max | 0 / 114 | 0 / 114 | none over 220 |
| median words | 19.0 | 18.5 | |
| top-10 two-word opener share | 6.25% | 6.25% | <10% |
| most-used opener | 6 ("i don't") | 6 | ≤8 |
| question-ending | 3.50% | 3.75% | ≤20% |
| exact / near duplicates | 0 / 0 | 0 / 0 | |
