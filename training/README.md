# training/

Makes the weights. The Phase-2 notebook (`CamusGPT_Step3_5_RefusalSFT.ipynb`) runs on Colab
and reads/writes a **Google Drive** working folder — stage the repo's `data/` and `adapters/`
there first:

```
MyDrive/CamusGPT_Training/
├── adapters/camus_sft_lora/     (INPUT  — Phase-1 voice-SFT LoRA)
├── data/                        (INPUT  — the 6 training .jsonl; see ../data/README.md)
└── deploy/                      (OUTPUT — camus.gguf + Modelfile)
```

Flow: load `camus_sft_lora` → merge → attach a fresh guardrail LoRA → train on
refusals + cooperative + epistemic + analysis + multiturn (CHECK 1 file pre-flight,
CHECK 2 response-masking, CHECK 3 behavioral gate) → merge final → refresh tokenizer from
the base model → export GGUF (q4_k_m) + Modelfile to `deploy/`.

`train_camus_8b.py` is the standalone Unsloth port (3B or 8B via `BASE_MODEL`).
`merge_8b.py` / `make_gguf_colab.py` produce the safetensors / GGUF deployment artifacts.
