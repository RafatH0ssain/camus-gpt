# training/

Makes the weights, in Colab. The corpus comes from `pipeline/assemble_training.py`
(see `data/README.md`); the full flow is in `docs/PIPELINE.md`.

- `training/CamusGPT_Phase1_VoiceSFT_Gemma4.ipynb` — voice LoRA on `unsloth/gemma-4-12b-it`.
- `training/CamusGPT_Phase2_SFT_Gemma4.ipynb` — behaviour LoRA on top; the "Phase 2b" cell runs an optional DPO pass (`DPO_SET = None` skips it).
- `training/CamusGPT_Export_GGUF_Gemma4.ipynb` — merges both LoRAs, exports a q4_k_m GGUF + Modelfile for Ollama.

`training/test_v3_notebooks.py` checks the notebooks statically.
