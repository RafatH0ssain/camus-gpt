# data/ — training corpus + KB intermediates

This folder mirrors the Google Drive path the Step-3.5 notebook reads from:
`MyDrive/CamusGPT_Training/data/`. Before running
`training/CamusGPT_Step3_5_RefusalSFT.ipynb`, upload these files to that Drive folder.

## Required by the notebook (CHECK 1 pre-flight, cell 3)
The notebook will refuse to start unless all six are present:

| File | What it is | Produced by |
|---|---|---|
| `camus_refusals.jsonl` | adversarial-attack → in-voice refusal pairs (subsampled, ~400) | hand-built (Phase-1/earlier) |
| `camus_conversational.jsonl` | warm/short + substantive/meditation replies (latter upweighted 3×) | hand-built |
| `camus_sft.jsonl` | Phase-1 voice SFT rows (short ones reused to preserve voice, ≤60 words) | Phase-1 corpus |
| `camus_epistemic.jsonl` | anti-sycophancy / anti-confabulation, balanced correction:affirmation | `training/build_epistemic.py` |
| `camus_analysis.jsonl` | analyze-provided-text engage + injection-seam refuse | `training/build_analysis.py` |
| `camus_multiturn.jsonl` | hand-composed multi-turn conversations | `training/build_multiturn.py` |

> **Note on a stale reference:** cell 0's intro prose says it needs
> `camus_dpo_adversarial.jsonl`. That is leftover from before DPO was dropped — the
> notebook title is "…(no DPO)" and the executed CHECK 1 does **not** require it. The
> adversarial refusals come from `camus_refusals.jsonl`. Ignore the DPO filename.

## KB intermediates (also live here, but git-ignored)
`source_chunks.jsonl` (from `kb/ingest_sources.py`) and `kb_extracted.jsonl`
(from `kb/extract_kb.py`) are large, regenerable, and derived from copyrighted PDFs — they
are **not** committed. The deployable KB (`camus_kb_full.jsonl`) and its vectors live on the
Hugging Face dataset repo, not in git.

## What git tracks here
The six training `camus_*.jsonl` files above and this README. Everything else
(`source_chunks.jsonl`, `kb_extracted.jsonl`, any `camus_kb*.jsonl`, `*.npy`) is ignored.
