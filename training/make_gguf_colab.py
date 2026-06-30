#!/usr/bin/env python3
"""
make_gguf_colab.py — convert the merged 8B safetensors ALREADY IN this Colab session
into a Q4_K_M GGUF, then upload it to your HF repo so you can pull the ~5GB file locally.

RUN IN COLAB:
    from huggingface_hub import login; login()      # paste a WRITE token (for the upload)
    !python make_gguf_colab.py

Fixes the tokenizer first: the merged folder's tokenizer_config.json names a class
(TokenizersBackend) llama.cpp's converter can't read, so we refresh the tokenizer files
from the base model you trained on (identical tokenizer, converter-friendly).
"""
import os, subprocess, shutil, json
from huggingface_hub import HfApi, hf_hub_download

SRC      = "camus-8b-merged"                      # existing merged-safetensors folder
BASE_TOK = "unsloth/Meta-Llama-3.1-8B-Instruct"   # base you trained on (ungated mirror)
REPO     = "rafatho/camus-gguf"                   # upload target
F16      = "camus-8b-f16.gguf"
GGUF     = "camus-8b.Q4_K_M.gguf"

def run(cmd):
    print("›", cmd); subprocess.run(cmd, shell=True, check=True)

shards = [f for f in os.listdir(SRC) if f.endswith(".safetensors")] if os.path.isdir(SRC) else []
if not shards:
    raise SystemExit(f"No .safetensors found in '{SRC}'. Set SRC to your merged folder path.")
print(f"Using {len(shards)} safetensors shard(s) in {SRC}/")

# --- FIX TOKENIZER: refresh from the base model, then force a converter-friendly class ---
print("Refreshing tokenizer files from the base model...")
for fn in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
    try:
        p = hf_hub_download(BASE_TOK, fn)
        shutil.copy(p, os.path.join(SRC, fn))
        print("  refreshed", fn)
    except Exception as e:
        print("  could not fetch", fn, "-", e)
tc = os.path.join(SRC, "tokenizer_config.json")
if os.path.exists(tc):
    cfg = json.load(open(tc))
    cfg["tokenizer_class"] = "PreTrainedTokenizerFast"   # what llama.cpp's converter expects
    json.dump(cfg, open(tc, "w"))
    print("  set tokenizer_class -> PreTrainedTokenizerFast")
# remove a stale sentencepiece file if present (Llama 3 is BPE, not sentencepiece)
sp = os.path.join(SRC, "tokenizer.model")
if os.path.exists(sp):
    os.remove(sp); print("  removed stray tokenizer.model")

# 1) convert safetensors -> fp16 GGUF
if not os.path.isdir("llama.cpp"):
    run("git clone --depth 1 https://github.com/ggerganov/llama.cpp")
run("pip install -q -r llama.cpp/requirements.txt")
run(f"python llama.cpp/convert_hf_to_gguf.py {SRC} --outfile {F16} --outtype f16")

# 2) build the quantizer, then quantize -> Q4_K_M
run("cmake -S llama.cpp -B llama.cpp/build -DLLAMA_CURL=OFF")
run("cmake --build llama.cpp/build --config Release -j --target llama-quantize")
qbin = next((p for p in ["llama.cpp/build/bin/llama-quantize",
                         "llama.cpp/build/bin/Release/llama-quantize"] if os.path.exists(p)), None)
if not qbin:
    raise SystemExit("llama-quantize not found after build — check the build log above.")
run(f"{qbin} {F16} {GGUF} Q4_K_M")
print(f"Built {GGUF}: {os.path.getsize(GGUF)/1e9:.2f} GB")

# 3) upload to the Hub (reliable way to get it onto your machine)
print("Uploading GGUF to the Hub...")
HfApi().upload_file(path_or_fileobj=GGUF, path_in_repo=GGUF, repo_id=REPO, repo_type="model")
print(f"Uploaded {GGUF} -> {REPO}")

# 4) also offer a direct browser download
try:
    from google.colab import files
    files.download(GGUF)
except Exception as e:
    print("Direct download skipped (use the Hub copy):", e)
