# adapters/ — LoRA adapters

Mirrors the Drive path `CamusGPT_Training/adapters/`.

> **These configs are from the v1 8B adapter** (`unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit`):
> `chat_template.jinja` renders Llama-3 markers, not Gemma. The v2 (Gemma-3-12B) adapter
> configs were never committed here; the notebooks rebuild them from `unsloth/gemma-3-12b-it`.

## `camus_sft_lora/`  (the Phase-1 voice adapter — INPUT to Step 3.5)
The first LoRA SFT pass installs Camus's *voice*. Its output adapter, `camus_sft_lora/`, is
the **input** to `training/CamusGPT_Step3_5_RefusalSFT.ipynb`, whose cell 2 merges it into a
base model and then attaches a *fresh* r=16 guardrail adapter for the refusal-SFT pass:

```python
SFT_ADAPTER  = f"{DRIVE}/adapters/camus_sft_lora"
LOCAL_MERGED = "/content/camus_sft_merged"
```

A LoRA adapter folder typically holds `adapter_config.json`, `adapter_model.safetensors`, and
the tokenizer files. **These weights are binary and are git-ignored** — keep them on Drive (and
optionally mirror to a Hugging Face model repo). Only this README is tracked, so the folder's
role is documented even though the weights aren't in git.

The Step-3.5 run's *own* final output is exported as a GGUF + Modelfile to the Drive `deploy/`
folder (not committed here); see `docs/PIPELINE.md` and `docs/DEPLOYMENT.md`.
