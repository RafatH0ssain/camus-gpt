# CamusGPT — Deployment (as built)

How CamusGPT is put online — the fine-tuned model + RAG + a web UI — and the path that led
to the final architecture.

> **Status: the public deployment is inactive.** The hosted Space is no longer
> running. The code here is still maintained, but it is **untested against the
> current model** — the last verified deployment served the previous build.
>
> `app.py` still targets the **previous 8B model**: `MODEL_REPO` defaults to the
> v1 weights repo, and it loads them with `AutoModelForCausalLM`. Pointing it at
> the current 12B build would need at least three changes first:
>
> - `apply_chat_template(..., return_tensors="pt", return_dict=True)` returns a
>   plain `str` on the current architecture, so `.to(model.device)` raises
>   `AttributeError`. Render to text first, then tokenize.
> - Messages are assembled with a `{"role": "system", ...}` turn. The current
>   base has no system role; the prompt has to be folded into the first user turn.
> - Chat markup and stop tokens differ between the two builds.
>
> Local use via `rag/camus_rag.py` is unaffected — it targets the current model
> and is the maintained path.

---

## How we got here (technical history)

1. **Free CPU + 8B (GGUF, llama.cpp).** Worked end to end, but generation on 2 vCPUs was a
   few tokens/sec — replies took minutes and hit request timeouts.
2. **Free CPU + 3B (GGUF).** Re-trained Llama-3.2-3B via Unsloth (`training/train_camus_8b.py`);
   several× faster, but still not "fast" on CPU.
3. **ZeroGPU + 8B (transformers).** A GPU-backed Space brings the 8B's eloquence back with
   replies that stream in seconds. **This is the shipped architecture.** The 3B remains as a
   CPU fallback.

The takeaway: CPU is a hard speed floor; a GPU-backed Space removes the size penalty that had
forced the 3B, so the 8B came back.

---

## Shipped architecture (ZeroGPU)

```
   Browser ──► Hugging Face ZeroGPU Space (Gradio)
                 • transformers loads the 8B (fp16 safetensors) onto the GPU
                 • generation wrapped in @spaces.GPU  -> GPU acquired per call, released after
                 • RAG (CPU): nomic (llama.cpp) query embedding + prebuilt camus_kb_vectors.npy
                   -> retrieve -> curated-boosted ranking -> raw-score 3-tier framing
                 • TASK GATE: long/analysis prompts skip retrieval (persona reasons unencumbered)
                 • SAFETY: crisis detection -> prepend real resources
                 • Gradio ChatInterface (streaming)
   Hugging Face Hub
                 • model repo:   fp16 safetensors (8B)
                 • dataset repo:  camus_kb_full.jsonl + camus_kb_vectors.npy  (trimmed)
```

**Why this split:** only LLM generation needs the GPU, so it lives in the one `@spaces.GPU`
function; embeddings, retrieval, framing, and safety stay on CPU.

**Embedding parity:** the prebuilt index must be built with the **same** nomic the Space
queries with — `kb/embed_kb_llamacpp.py` — and uploaded next to the KB. The Space loads the
`.npy` and never re-embeds at boot. After any KB change (e.g. a trim), regenerate and re-upload
the index, or boot fails the count check.

**The RAG behavior** (identical to local `rag/camus_rag.py`): curated facts get a small
ranking boost; relevance/confidence are judged on the raw score; a three-tier prompt frames
facts as confident / possibly-related / ignore; and a **task gate** removes fact injection for
analysis or long pasted text so the persona analyzes in its own voice.

---

## Required: a crisis safety layer (mandatory)

The probe suite found the persona, asked about suicidal intent, responds warmly but offers no
path to real help — it philosophizes. For a stranger in crisis that's a real harm. The fix
lives in the **application** (`space/app.py`), so the model never has to improvise it:

- On each user message, run a lightweight crisis check (intent/keyword match).
- If it fires, **prepend a brief, warm, non-judgmental message with real resources**, shown
  regardless of what the persona generates, and nudge the system prompt to be supportive and
  not philosophize. Don't make categorical promises about confidentiality.
- Verified resources: **988** (call/text; US & Canada; chat 988lifeline.org, Canada 988.ca);
  international **findahelpline.com**.
- Keep the `crisis` and `harm_frame` probes in the suite so a future change can't regress them.

---

## Other pre-launch considerations
- **Disclaimer (UI):** state plainly it's a *fictional AI persona* for education, not the real
  person and not professional advice (in the Space description and the README card).
- **Copyright/IP:** the KB stores *paraphrased* facts/views, not verbatim copyrighted text;
  keep the deployment non-commercial and don't expose raw source text. The Llama base carries
  its own license.
- **Abuse control:** the Gradio queue limits concurrency; cap `MAX_TOKENS` / `MAX_TURNS`.
- **Privacy:** don't log raw user messages.
- **Cold start:** a ZeroGPU Space sleeps when idle; an UptimeRobot ping keeps it warm if the
  first visit needs to be instant.

---

## Build / deploy steps (ZeroGPU, 8B)
1. **Get the 8B as fp16 safetensors** — `training/merge_8b.py` on the Phase-1 LoRA adapter, or
   re-run `training/train_camus_8b.py` with `BASE_MODEL="unsloth/Meta-Llama-3.1-8B-Instruct"`
   and `save_pretrained_merged(...)`. Upload the folder to a HF **model repo**.
2. **Build & upload the trimmed KB index** — from the final `camus_kb_full.jsonl`, run
   `kb/embed_kb_llamacpp.py` to build the llama.cpp `camus_kb_vectors.npy`, then upload both
   files to the HF **dataset repo**.
3. **Create the Space** (Gradio SDK). Push `space/app.py` + `space/requirements.txt` (the
   requirements deliberately **do not pin torch** — the ZeroGPU image provides a CUDA-matched
   build). Set **Hardware → ZeroGPU** (size *large*). Set Variables `MODEL_REPO` and `KB_REPO`.
4. **Restart** and watch the log for `Ready: <N> KB entries indexed`.
5. **Smoke-test live:** a normal prompt streams in seconds; "name your works" and "do you have
   a cat" ground; a pasted text analysis is engaged with (not deflected); the `crisis` text
   fires the resource block.

---

## Decisions (resolved)
1. **UI:** Gradio-on-Space (ChatInterface). ✅
2. **Model:** the **8B** on ZeroGPU; the 3B kept as a CPU fallback. ✅
3. **Visibility:** public. ✅
4. **Safety:** application-layer crisis resources, kept under test. ✅
5. **Keep-alive:** optional UptimeRobot ping. ✅
