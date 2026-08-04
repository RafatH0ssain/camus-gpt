# CamusGPT

A fine-tuned **Llama-3.1-8B** persona of Albert Camus — dry, lucid, first-person — grounded
in his real life and writing by a **RAG** layer. It runs locally via **Ollama**.

Note: This project is no longer hosted on HF Spaces. Send me a message if you have any queries!

> **This is a fictional AI persona for education and conversation — not the real Albert Camus,
> and not professional advice.** If you are in crisis, call or text **988** (US & Canada) or
> find a local helpline at **findahelpline.com**.

## The idea
**Personality lives in the weights; facts live in retrieval.** The model is trained once to
*speak and behave* like Camus; everything it *knows about its own life* comes from a curated +
extracted knowledge base retrieved at query time.

## Repo layout
```
camusgpt/
├── README.md  .gitignore  NOTICE
├── docs/
│   ├── PIPELINE.md             # full training + RAG pipeline write-up
│   └── DEPLOYMENT.md           # how it's served (ZeroGPU) and why
├── training/                   # make the weights
│   ├── README.md               # the Google-Drive layout the notebook expects
│   ├── build_epistemic.py  build_analysis.py  build_multiturn.py   # -> data/camus_*.jsonl
│   ├── CamusGPT_Step3_5_RefusalSFT.ipynb       # Phase-2 refusal-SFT (reads data/ + adapters/)
│   ├── train_camus_8b.py       # Unsloth trainer (3B or 8B via BASE_MODEL)
│   ├── merge_8b.py             # LoRA -> fp16 safetensors
│   └── make_gguf_colab.py      # safetensors -> Q4_K_M GGUF (Colab)
├── kb/                         # build the knowledge base
│   ├── sources_manifest.json  ingest_sources.py  extract_kb.py
│   ├── build_kb.py  merge_kb.py
│   ├── build_index.py          # Ollama nomic index (local)
│   ├── embed_kb_llamacpp.py    # llama.cpp nomic index (Space parity)
│   └── trim_kb.py              # semantic-dedup trim
├── data/                       # training corpus (tracked) + KB intermediates (ignored)
│   └── README.md               # mirrors Drive MyDrive/CamusGPT_Training/data/
├── adapters/                   # camus_sft_lora (Phase-1 voice LoRA; weights ignored)
│   └── README.md               # mirrors Drive MyDrive/CamusGPT_Training/adapters/
├── rag/                        # the chat layer
│   ├── camus_rag.py            # retrieval-augmented chat (boost, 3-tier, task gate)
│   ├── camus_client.py  probe_camus.py  diagnose_retrieval.py
│   ├── Modelfile
└── space/                      # what you push to the HF Space (separate git repo)
    ├── app.py  requirements.txt  README.md
```

## Training inputs (Google Drive layout)
The Phase-2 notebook (`training/CamusGPT_Step3_5_RefusalSFT.ipynb`) runs on Colab and reads a
Drive working folder — stage the repo's `data/` and `adapters/` there:
```
MyDrive/CamusGPT_Training/
├── adapters/camus_sft_lora/        # Phase-1 voice-SFT LoRA (INPUT)
├── data/  camus_sft.jsonl  camus_conversational.jsonl  camus_refusals.jsonl
│          camus_epistemic.jsonl  camus_analysis.jsonl  camus_multiturn.jsonl
└── deploy/  camus.gguf + Modelfile  (OUTPUT)
```

## Quickstart — local (Ollama)
```bash
ollama pull nomic-embed-text
ollama create camus -f rag/Modelfile            # FROM ./camus-8b.Q4_K_M.gguf
python kb/build_kb.py && python kb/merge_kb.py
python kb/build_index.py
python kb/trim_kb.py --dedup 0.88               # optional: trim the noise floor
python rag/camus_rag.py --debug
python rag/probe_camus.py
```

## Quickstart — deploy (ZeroGPU)
See `docs/DEPLOYMENT.md`: fp16 safetensors (`training/merge_8b.py`) → model repo; index via
`kb/embed_kb_llamacpp.py` → dataset repo with the KB; push `space/` to a Gradio ZeroGPU Space;
set `MODEL_REPO` / `KB_REPO`. Run all scripts **from the repo root** so relative paths resolve.

## Docs
- **[docs/PIPELINE.md](docs/PIPELINE.md)** — training, KB, and retrieval design end to end.
- **[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)** — the served architecture and build steps.
