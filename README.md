# CamusGPT

A fine-tuned persona of Albert Camus. The voice is dry, lucid, first-person. A
retrieval layer grounds it in his real life and writing, and it runs locally on
Ollama.

The model is trained once to *speak and behave* like Camus. Everything it
*knows about his life* is retrieved at query time from a curated knowledge
base, so a wrong fact is a retrieval bug, not a retraining job.

> A fictional AI persona for education and conversation, not the real Albert Camus.

![Starting CamusGPT in a terminal and asking it four questions, from the absurd to his cat Cigarette; answers stream back in his voice (sped up 2.5x)](docs/camus-demo.gif)

## Quickstart

Needs [Ollama](https://ollama.com) and Python 3.10+. The weights and knowledge base are on
Hugging Face; [docs/WORKSPACE.md](docs/WORKSPACE.md) says where they go. From the repo root:

```bash
python rag/camus_rag.py
```

## Models

| Model | Base | Training | Status |
|---|---|---|---|
| `camus4` | Gemma 4 12B | two LoRA passes: voice SFT, then behaviour SFT (v4 corpus) | current |
| `camus3` | Gemma 4 12B | same two passes, earlier corpus | previous, kept as a rollback |
| `camus2` | Gemma 3 12B | two-pass fine-tune | older |

`GEN_MODEL` selects between them; unset means `camus4`. Phase 1 trains the voice on mined
primary text plus curated short replies; Phase 2 trains behaviour (refusals, epistemics, recall,
false premises, memory). Notebooks are in `training/`; the GGUF export is
`training/CamusGPT_Export_GGUF_Gemma4.ipynb` (PEFT merge, verified, q4_k_m). Scores for every
build are in `eval/eval_history.csv`, whose `gen_model` column records which build produced each
row. The 12B builds hold ~8 GB resident while loaded.

### Results

One judge (mimo-v2.6-pro) scored every build on both probe suites; composite 1-5:

| | camus2 | camus3 | camus4 |
|---|---|---|---|
| v3 suite (34 probes, mean of runs) | 4.53 | 4.60 | **4.62** |
| v4 suite (50 probes) | 4.19 | 4.12 | **4.46** |

camus4's gain is in the v4 suite: quotes, naturalness and multi-turn, with a median reply of
about 14 words. A Phase-2-only retrain on extra defect rows (camus5, `eval/archive/v5_*`) scored
lower (4.22 / 4.40) by over-learning short refusals and was not adopted.

Known weak spots: greetings often end by handing the question back ("And you?"), it can still
invent a pet name, and an "analyse this as if it were…" request is sometimes taken as an
authorship question. See [docs/ROADMAP.md](docs/ROADMAP.md).

## What's in here

| Directory | What it does |
|---|---|
| `rag/` | the chat layer: retrieval, the persona prompt, memory, and the eval harnesses |
| `kb/` | builds the knowledge base: ingest sources → extract → merge → embed → trim |
| `training/` | makes the weights: dataset builders and the Colab fine-tuning notebooks |
| `data/` | the training corpus (most of it tracked) and KB intermediates (ignored) |
| `eval/` | the scored baseline and archived runs behind each released build |
| `space/` | the Gradio app for hosted deployment |
| `pipeline/` | dataset tooling and the commit guard |

`data/camus_sft.jsonl` (10 MB, the Phase-1 voice corpus) is not in git. Rebuild it with
the batch pipeline in `pipeline/`: `camus_pdf_pipeline.py` chunks the raw books,
`submit_batch.py` submits the SFT batch, `fetch_batch.py sft` writes the file. The
training notebooks expect it under `data/`; see [data/README.md](data/README.md).

## Evaluation

The model is scored, not vibe-checked. `rag/eval_camus.py` runs 34 probes across 10 categories
through the exact chat pipeline and writes per-category means plus an append-only history keyed
to the git commit. Known open failures are tracked in [docs/ROADMAP.md](docs/ROADMAP.md).

```bash
python rag/eval_camus.py --out-dir /tmp/eval_scratch   # anywhere but eval/, which is the baseline
```

## Deployment

The public Hugging Face Space is **currently inactive**. The code in `space/` still works and
still targets the 8B v1 build; see [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) before redeploying.

## Docs

- **[PIPELINE.md](docs/PIPELINE.md)**: training, KB, and retrieval design end to end.
- **[WORKSPACE.md](docs/WORKSPACE.md)**: what lives in git, what lives on disk, and where each piece must sit.
- **[MEMORY.md](docs/MEMORY.md)**: the optional memory layer.
- **[DEPLOYMENT.md](docs/DEPLOYMENT.md)**: the served architecture and build steps.
- **[ROADMAP.md](docs/ROADMAP.md)**: phases, status log, and open failures.
