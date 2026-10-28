# CamusGPT

A fine-tuned persona of Albert Camus. The voice is dry, lucid, first-person. A
retrieval layer grounds it in his real life and writing, and it runs locally on
Ollama.

The model is trained once to *speak and behave* like Camus. Everything it
*knows about his life* is retrieved at query time from a curated knowledge
base, so a wrong fact is a retrieval bug, not a retraining job.

> **A fictional AI persona for education and conversation. Not the real Albert
> Camus, and not professional advice.** If you are in crisis, call or text
> **988** (US & Canada) or find a local helpline at **findahelpline.com**.

![Starting CamusGPT in a terminal and asking it four questions, from the absurd to his cat Cigarette; answers stream back in his voice (sped up 2.5x)](docs/camus-demo.gif)

## Quickstart

Needs [Ollama](https://ollama.com), Python 3.10+, plus the weights and knowledge
base from the project's Hugging Face repos, which are too large for git. See
[docs/WORKSPACE.md](docs/WORKSPACE.md).

```bash
ollama pull nomic-embed-text
cd rag && ollama create camus -f Modelfile && cd ..   # relative FROM needs this cwd
python kb/build_index.py                              # embed the KB
GEN_MODEL=camus2 python rag/camus_rag.py              # chat
```

Run everything **from the repository root**: the scripts resolve paths against
the working directory, not against themselves.

Useful flags: `--debug` shows what retrieval returned and why; `--memory` enables the optional
persistent memory layer (off by default).

## The two builds

| Model | Base | Status |
|---|---|---|
| `camus2` | 12B, two-pass fine-tune | current |
| `camus` | 8B | previous, kept installed as a rollback |

`GEN_MODEL` selects between them; unset falls back to `camus`. Scores for each are in
`eval/eval_history.csv`, whose `gen_model` column records which build produced each row. The
12B build holds ~8 GB resident while loaded.

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
