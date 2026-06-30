#!/usr/bin/env python3
"""
merge_8b.py — merge the Phase-1 8B LoRA adapter into fp16 safetensors for GPU serving.

Transformers and ZeroGPU need safetensors (not GGUF), so this merges the adapter onto the
base and saves a full model folder ready to upload to a HF model repo.

Set ADAPTER to your LoRA output dir (adapter_config.json + adapter_model.*) and REPO_ID
to your target HF repo. Requires ~20 GB RAM (Colab High-RAM or a GPU machine).
"""
import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

ADAPTER = "path/to/your/8b-lora-adapter"   # <-- EDIT: your Phase-1 8B LoRA output dir
OUT     = "camus-8b-merged"
REPO_ID = "rafatho/camus-8b-merged"  # <-- EDIT: target HF model repo

print("Loading base + adapter and merging...")
model = AutoPeftModelForCausalLM.from_pretrained(ADAPTER, torch_dtype=torch.float16)
model = model.merge_and_unload()
model.save_pretrained(OUT, safe_serialization=True)

try:
    tok = AutoTokenizer.from_pretrained(ADAPTER)
except Exception:
    base = model.config._name_or_path
    tok = AutoTokenizer.from_pretrained(base)
tok.save_pretrained(OUT)
print(f"Merged model saved to ./{OUT}")