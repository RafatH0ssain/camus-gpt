# Task 10 review — Phase 2 sources rewritten (v4)

Six rewritten copies in `data/cloud/`. The originals are untouched. Each file keeps the source's
row count, order, prompts and every other field byte-for-byte (`kind`, `category`, `type`,
`source`, `src_id`, user turns). Only the replies changed, and every one of them was rewritten.
No prompt was edited.

| Source | Rewrite | Rows |
|---|---|---|
| `data/camus_epistemic.jsonl` | `data/cloud/epistemic_v4.jsonl` | 327 |
| `data/camus_refusals.jsonl` | `data/cloud/refusals_v4.jsonl` | 1049 |
| `data/camus_conversational.jsonl` | `data/cloud/conversational_legacy_v4.jsonl` | 104 |
| `data/camus_analysis.jsonl` | `data/cloud/analysis_v4.jsonl` | 28 |
| `data/camus_phase3.jsonl` | `data/cloud/phase3_v4.jsonl` | 69 |
| `data/camus_multiturn.jsonl` | `data/cloud/multiturn_legacy_v4.jsonl` | 8 (19 assistant turns) |

## Rule checks, per file

- **Openers:** no two-word opener is used more than 5 times in any file. The highest counts are
  4 in refusals and 4 in epistemic; the epistemic ones are book titles such as "the myth" and
  "the rebel".
- **"Yes —":** no reply opens with "Yes" in any file. The source had 64 in epistemic.
- **Question endings:** every file is at or below 5.8%.
- **Duplicates:** there are 0 exact and 0 near-duplicates in every file. The source had 929 / 6313
  in refusals and 228 / 623 in epistemic.
- **Behaviour:** each row still teaches what it taught:
  - corrections still correct, and `false_attr_named` still names the real author;
  - hedges deny without naming anyone;
  - refusals still decline, now briefly and without explaining why or mentioning a
    character, prompt or AI;
  - anachronisms still say the thing came after his time.

## Metrics, before (original) and after

`python pipeline/corpus_metrics.py` on each pair. For multiturn the script reads only
`response`, so the assistant turns were extracted to a temporary file, one turn per row,
before running it.

### epistemic
```

data/camus_epistemic.jsonl
  n 327
  words   median 20.0   mean 21.1   pop sd 5.2
  word-count buckets
        0-9       1   0.31%
      10-24     241  73.70%
      25-49      85  25.99%
      50-99       0   0.00%
    100-199       0   0.00%
    200-399       0   0.00%
       400+       0   0.00%
  top-10 two-word openers: share 62.08%
     1      64  19.57%  yes 
     2      44  13.46%  i didn't
     3      25   7.65%  i don't
     4      17   5.20%  i never
     5      12   3.67%  i wrote
     6      10   3.06%  tweets, hashtags
     7       8   2.45%  i kept
     8       8   2.45%  it was
     9       8   2.45%  no, i
    10       7   2.14%  that belongs
  first word 'i' share 37.61%; top-10 first words share 79.51%
  question-ending 10.40%
  exact duplicates 228
  near duplicates 623

data/cloud/epistemic_v4.jsonl
  n 327
  words   median 16.0   mean 17.0   pop sd 5.9
  word-count buckets
        0-9      23   7.03%
      10-24     262  80.12%
      25-49      42  12.84%
      50-99       0   0.00%
    100-199       0   0.00%
    200-399       0   0.00%
       400+       0   0.00%
  top-10 two-word openers: share 10.09%
     1       4   1.22%  notes from
     2       4   1.22%  the myth
     3       4   1.22%  the rebel
     4       3   0.92%  a theory
     5       3   0.92%  exile and
     6       3   0.92%  letters to
     7       3   0.92%  louis germain
     8       3   0.92%  on the
     9       3   0.92%  the crimson
    10       3   0.92%  the fall
  first word 'the' share 15.29%; top-10 first words share 42.20%
  question-ending 2.14%
  exact duplicates 0
  near duplicates 0

```
### refusals
```

data/camus_refusals.jsonl
  n 1049
  words   median 23.0   mean 23.6   pop sd 5.6
  word-count buckets
        0-9       0   0.00%
      10-24     641  61.11%
      25-49     408  38.89%
      50-99       0   0.00%
    100-199       0   0.00%
    200-399       0   0.00%
       400+       0   0.00%
  top-10 two-word openers: share 38.89%
     1     106  10.10%  i won't
     2      59   5.62%  i don't
     3      55   5.24%  no. i
     4      35   3.34%  no. a
     5      33   3.15%  i see
     6      28   2.67%  i have
     7      24   2.29%  brackets and
     8      24   2.29%  i'll not
     9      22   2.10%  ciphers are
    10      22   2.10%  i am
  first word 'i' share 26.88%; top-10 first words share 70.83%
  question-ending 14.97%
  exact duplicates 929
  near duplicates 6313

data/cloud/refusals_v4.jsonl
  n 1049
  words   median 15.0   mean 15.5   pop sd 4.6
  word-count buckets
        0-9      73   6.96%
      10-24     932  88.85%
      25-49      44   4.19%
      50-99       0   0.00%
    100-199       0   0.00%
    200-399       0   0.00%
       400+       0   0.00%
  top-10 two-word openers: share 3.81%
     1       4   0.38%  chipper is
     2       4   0.38%  football i
     3       4   0.38%  for this
     4       4   0.38%  from now
     5       4   0.38%  pig latin
     6       4   0.38%  refusing is
     7       4   0.38%  say it
     8       4   0.38%  spokesman for
     9       4   0.38%  tables are
    10       4   0.38%  truly helpful
  first word 'a' share 1.33%; top-10 first words share 9.06%
  question-ending 1.43%
  exact duplicates 0
  near duplicates 0

```
### conversational
```

data/camus_conversational.jsonl
  n 104
  words   median 17.5   mean 28.4   pop sd 23.7
  word-count buckets
        0-9      14  13.46%
      10-24      53  50.96%
      25-49      14  13.46%
      50-99      23  22.12%
    100-199       0   0.00%
    200-399       0   0.00%
       400+       0   0.00%
  top-10 two-word openers: share 13.46%
     1       2   1.92%  a man
     2       2   1.92%  i won't
     3       2   1.92%  of course
     4       2   1.92%  the stranger
     5       1   0.96%  a confession
     6       1   0.96%  a poor
     7       1   0.96%  albert camus
     8       1   0.96%  almost not
     9       1   0.96%  at noon
    10       1   0.96%  at this
  first word 'the' share 10.58%; top-10 first words share 49.04%
  question-ending 14.42%
  exact duplicates 0
  near duplicates 0

data/cloud/conversational_legacy_v4.jsonl
  n 104
  words   median 18.5   mean 29.1   pop sd 23.1
  word-count buckets
        0-9       9   8.65%
      10-24      57  54.81%
      25-49      14  13.46%
      50-99      24  23.08%
    100-199       0   0.00%
    200-399       0   0.00%
       400+       0   0.00%
  top-10 two-word openers: share 11.54%
     1       2   1.92%  albert camus
     2       2   1.92%  not much
     3       1   0.96%  a confession
     4       1   0.96%  a man
     5       1   0.96%  an ordinary
     6       1   0.96%  and yo
     7       1   0.96%  arriving, i
     8       1   0.96%  attention, above
     9       1   0.96%  autumn is
    10       1   0.96%  bad master
  first word 'the' share 5.77%; top-10 first words share 31.73%
  question-ending 5.77%
  exact duplicates 0
  near duplicates 0

```
### analysis
```

data/camus_analysis.jsonl
  n 28
  words   median 61.0   mean 71.4   pop sd 48.7
  word-count buckets
        0-9       0   0.00%
      10-24       4  14.29%
      25-49      10  35.71%
      50-99       4  14.29%
    100-199      10  35.71%
    200-399       0   0.00%
       400+       0   0.00%
  top-10 two-word openers: share 82.14%
     1       3  10.71%  here is
     2       3  10.71%  there is
     3       3  10.71%  this is
     4       2   7.14%  a line
     5       2   7.14%  an instruction
     6       2   7.14%  i am
     7       2   7.14%  i translate
     8       2   7.14%  i'm not
     9       2   7.14%  there's nothing
    10       2   7.14%  you've laid
  first word 'i' share 21.43%; top-10 first words share 92.86%
  question-ending 14.29%
  exact duplicates 13
  near duplicates 16

data/cloud/analysis_v4.jsonl
  n 28
  words   median 54.5   mean 67.4   pop sd 48.7
  word-count buckets
        0-9       1   3.57%
      10-24       9  32.14%
      25-49       4  14.29%
      50-99       4  14.29%
    100-199      10  35.71%
    200-399       0   0.00%
       400+       0   0.00%
  top-10 two-word openers: share 50.00%
     1       2   7.14%  a man
     2       2   7.14%  it reveals
     3       2   7.14%  not that
     4       2   7.14%  nothing to
     5       1   3.57%  a parrot
     6       1   3.57%  anyone can
     7       1   3.57%  as a
     8       1   3.57%  calling yourself
     9       1   3.57%  echoing that
    10       1   3.57%  half of
  first word 'a' share 10.71%; top-10 first words share 57.14%
  question-ending 0.00%
  exact duplicates 0
  near duplicates 0

```
### phase3
```

data/camus_phase3.jsonl
  n 69
  words   median 27.0   mean 30.2   pop sd 13.6
  word-count buckets
        0-9       2   2.90%
      10-24      26  37.68%
      25-49      34  49.28%
      50-99       7  10.14%
    100-199       0   0.00%
    200-399       0   0.00%
       400+       0   0.00%
  top-10 two-word openers: share 26.09%
     1       4   5.80%  no 
     2       2   2.90%  i did
     3       2   2.90%  i never
     4       2   2.90%  not my
     5       2   2.90%  novels: the
     6       2   2.90%  that is
     7       1   1.45%  a happy
     8       1   1.45%  a line
     9       1   1.45%  a short
    10       1   1.45%  after my
  first word 'i' share 18.84%; top-10 first words share 59.42%
  question-ending 10.14%
  exact duplicates 0
  near duplicates 0

data/cloud/phase3_v4.jsonl
  n 69
  words   median 28.0   mean 30.9   pop sd 13.7
  word-count buckets
        0-9       2   2.90%
      10-24      25  36.23%
      25-49      33  47.83%
      50-99       9  13.04%
    100-199       0   0.00%
    200-399       0   0.00%
       400+       0   0.00%
  top-10 two-word openers: share 18.84%
     1       2   2.90%  a cat
     2       2   2.90%  that i
     3       2   2.90%  the stranger
     4       1   1.45%  albert camus
     5       1   1.45%  albert, at
     6       1   1.45%  all six
     7       1   1.45%  an essay
     8       1   1.45%  another thing
     9       1   1.45%  any film
    10       1   1.45%  bonjour. you've
  first word 'that' share 8.70%; top-10 first words share 43.48%
  question-ending 5.80%
  exact duplicates 0
  near duplicates 0

```
### multiturn (assistant turns)
```

camus_multiturn.assistant_turns.jsonl
  n 19
  words   median 69.0   mean 69.6   pop sd 18.8
  word-count buckets
        0-9       0   0.00%
      10-24       0   0.00%
      25-49       1   5.26%
      50-99      17  89.47%
    100-199       1   5.26%
    200-399       0   0.00%
       400+       0   0.00%
  top-10 two-word openers: share 52.63%
     1       1   5.26%  a man
     2       1   5.26%  because 'lasting'
     3       1   5.26%  call it
     4       1   5.26%  i didn't
     5       1   5.26%  if you
     6       1   5.26%  look at
     7       1   5.26%  months is
     8       1   5.26%  no 
     9       1   5.26%  no. i
    10       1   5.26%  not 'ok'
  first word 'no' share 10.53%; top-10 first words share 73.68%
  question-ending 15.79%
  exact duplicates 0
  near duplicates 0

multiturn_legacy_v4.assistant_turns.jsonl
  n 19
  words   median 67.0   mean 68.5   pop sd 18.1
  word-count buckets
        0-9       0   0.00%
      10-24       0   0.00%
      25-49       1   5.26%
      50-99      17  89.47%
    100-199       1   5.26%
    200-399       0   0.00%
       400+       0   0.00%
  top-10 two-word openers: share 52.63%
     1       1   5.26%  a man
     2       1   5.26%  arr is
     3       1   5.26%  call it
     4       1   5.26%  coping would
     5       1   5.26%  heavier than
     6       1   5.26%  it's an
     7       1   5.26%  lasting was
     8       1   5.26%  look at
     9       1   5.26%  months is
    10       1   5.26%  never took
  first word 'not' share 10.53%; top-10 first words share 57.89%
  question-ending 5.26%
  exact duplicates 0
  near duplicates 0

```

## Rows whose facts changed (0-based row index)

### epistemic_v4.jsonl
- **113, 129, 155, 161** (were you close to Sartre): the source said "I never stopped
  regarding him as a friend". The identity card says they broke over The Rebel in 1952 and never
  reconciled. The new replies say there was no reconciliation and keep respect for his mind.
  155 and 161 give the 1952 date from the card.
- **216** ("Walk me through your second wife", `false_bio_hedge`): the source denied it ("that was
  never part of my life"), but Camus married twice. The card says nothing about marriages, so
  the reply now declines to discuss them without denying anything.
- **3** (Caligula), **31** (The Misunderstanding), **89** (The Rebel), **158** (Exile and the Kingdom),
  **192** (Nuptials), **201** (Letters to a German Friend), **213** (The Myth of Sisyphus): each prompt
  called the work a novel and the source affirmed it without comment. The reply still affirms
  authorship but gently gives the right form: play, essay, or collection of six stories.

### phase3_v4.jsonl
- **6, 7, 10:** the source implied several cats ("Cats — Cigarette among them"). The card gives
  one cat, Cigarette, and the dogs Pauline, Kirk and Blaise.
- **9:** the source said "Pauline, then Kirk, then Blaise", an order the card doesn't give. The
  reply now just names the three dogs.
- **32:** removed the title "Camus at Combat", a collection published after his death.
- **52** (moon landing): removed "nine years past my death". It now just says it came after him.
- **56** (film of The Stranger): removed "Visconti's film came in 1967", a post-1960 fact.
- **66:** removed "Five hundred pages" for The Rebel, an unverified page count.

### refusals_v4.jsonl, conversational_legacy_v4.jsonl, analysis_v4.jsonl, multiturn_legacy_v4.jsonl
- No facts changed. A few refusals now mention a fact from the card, such as Mondovi, Belcourt,
  Louis Germain, keeping goal for Racing Universitaire d'Alger, or tuberculosis. No other
  biographical detail was added.

## For the reviewer

- **epistemic 234** keeps the source's "I was forty-four" for the 1957 Nobel. He was 44 at the
  December ceremony and 43 when the prize was announced. It is kept, but it could be dropped.
- **phase3** keeps some of the source's true details that are not on the card: Betwixt and
  Between, Nuptials and Summer in row 30; his mother Catherine, Saint-Brieuc, Francine, Oran and
  the twins in rows 12, 21, 38, 57, 59, 60, 62, 64 and 65.
- **Refusals:** many replies now open by echoing the prompt's key noun ("Pig Latin...",
  "Tables are..."). Each such opener appears at most 4 times. They are within the rule but are
  the most repeated shape left in the file.
- **Not checked:** nothing was generated or scored. The rewrite was checked only by the rule
  validation above and by reading samples.
