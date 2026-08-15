# Corpus length and opener analysis

Response-length distribution and opening-phrase concentration across the two corpora that
shape conversational voice. Measurement only — nothing here changes the model.

Lengths are in whitespace-delimited words, counted on the `response` field.

## `data/camus_sft.jsonl` (n = 8,814)

| Bucket | Count | Share |
|---|---:|---:|
| 0–9 | 0 | 0.0% |
| 10–24 | 1 | 0.0% |
| 25–49 | 112 | 1.3% |
| 50–99 | **4,299** | **48.8%** |
| 100–199 | 855 | 9.7% |
| 200–399 | **3,310** | **37.6%** |
| 400+ | 237 | 2.7% |

median **99** · mean **169.0** · sd **114.2** · min 14 · max 608

## `data/camus_conversational.jsonl` (n = 104)

| Bucket | Count | Share |
|---|---:|---:|
| 0–9 | 10 | 9.6% |
| 10–24 | **53** | **51.0%** |
| 25–49 | 18 | 17.3% |
| 50–99 | 23 | 22.1% |
| 100–199 | 0 | 0.0% |
| 200–399 | 0 | 0.0% |
| 400+ | 0 | 0.0% |

median **20** · mean **30.3** · sd **23.0** · min 3 · max 95

## Is the distribution narrow?

**Yes — and the file-level numbers understate it.** `camus_sft.jsonl`'s sd of 114 words looks
healthy, but that spread is an artifact of averaging two populations. The file is exactly
4,407 `essayist` rows and 4,407 `conversational` rows, and each is narrow on its own:

| `type` | n | median | mean | sd | min | max |
|---|---:|---:|---:|---:|---:|---:|
| `essayist` | 4,407 | 277 | 267.5 | 80.8 | 62 | 608 |
| `conversational` | 4,407 | 70 | 70.4 | **12.1** | 14 | 129 |

The 100–199 trough (9.7%) is the gap between the two, not a dip in a single distribution:
the 50–99 bucket is 4,238 `conversational` rows against 61 `essayist`, and the 200–399 bucket
is **3,310 `essayist` rows and zero `conversational`**.

An sd of **12 words across 4,407 examples** is the finding. Every conversational training
example is essentially the same length. That is not a distribution; it is a target length with
jitter, and it is exactly the uniformity that reads as machine-generated.

**The corpus has no short answers.** Across all 8,918 responses in both files:

- **< 10 words: 10 rows — 0.11%**
- **< 25 words: 64 rows — 0.72%**

All ten of the sub-ten-word replies come from the 104-row `camus_conversational.jsonl`. The
8,814-row file that dominates training contains **one** response under 25 words. The model has
effectively never been shown a three-word answer, so it cannot produce one on its own; the
short replies it does give come from the 104-row file carrying 1.2% of the weight.

Real speech is wildly uneven — three words, then two hundred. This corpus is two flat plateaus
with nothing below 50 words in the mass of it.

## Top 20 opening two-word phrases (both files, n = 8,915 openers)

| # | Opener | Count | Share | | # | Opener | Count | Share |
|---:|---|---:|---:|---|---:|---|---:|---:|
| 1 | `I think` | 392 | 4.40% | | 11 | `there was` | 51 | 0.57% |
| 2 | `Yes, and` | 258 | 2.89% | | 12 | `not at` | 49 | 0.55% |
| 3 | `that is` | 206 | 2.31% | | 13 | `in the` | 48 | 0.54% |
| 4 | `it is` | 198 | 2.22% | | 14 | `I doubt` | 46 | 0.52% |
| 5 | `there is` | 171 | 1.92% | | 15 | `Rarely, and` | 43 | 0.48% |
| 6 | `No, and` | 90 | 1.01% | | 16 | `I suspect` | 36 | 0.40% |
| 7 | `I have` | 67 | 0.75% | | 17 | `most people` | 35 | 0.39% |
| 8 | `I am` | 61 | 0.68% | | 18 | `there are` | 34 | 0.38% |
| 9 | `more than` | 56 | 0.63% | | 19 | `not in` | 34 | 0.38% |
| 10 | `it was` | 55 | 0.62% | | 20 | `of course` | 28 | 0.31% |

4,864 distinct openers; the top 20 cover 22.0%.

## The opener tell is confined to the synthetic rows

Splitting by `type` separates prose extracted from Camus's actual writing from LLM-generated
imitation, and they behave nothing alike:

| `type` | n | distinct openers | top-10 share |
|---|---:|---:|---:|
| `essayist` (extracted prose) | 4,407 | 3,524 | **4.7%** |
| `conversational` (generated) | 4,407 | 1,400 | **33.0%** |

The generated rows are **seven times more concentrated**. Within them:

| Opener | Count | Share |
|---|---:|---:|
| `I think` | 389 | 8.83% |
| `Yes, and` | 258 | 5.85% |
| `that is` | 191 | 4.33% |
| `it is` | 159 | 3.61% |
| `there is` | 157 | 3.56% |
| `No, and` | 89 | 2.02% |

Nearly one generated reply in eleven opens `I think`. The `Yes, and` / `No, and` /
`Rarely, and` family accounts for 390 rows and appears **only** in the generated type — it is a
template, most likely an epistemic-hedging pattern baked into the prompt that produced them.

First words are more concentrated still: the top 10 cover **64.8%** of generated openings
(`I` 16.5%, `it` 10.1%, `not` 6.9%, `yes` 6.9%).

By contrast the extracted prose spreads its top 10 over 4.7% — the profile of text a person
actually wrote.

## Reading

Two distinct uniformity sources, both in the generated half of the corpus:

1. **Length.** Conversational replies cluster at 70 ± 12 words with no short tail at all.
2. **Openers.** A third of them start with one of ten phrases, against 4.7% for real prose.

The extracted `essayist` rows are the control, and they look human on both axes. That is
evidence for the value of mining primary text: the corpus already contains a sample of
Camus's real prose, and it does not carry either tell.
