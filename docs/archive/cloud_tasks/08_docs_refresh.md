# Task 08 — bring README and ROADMAP up to v3 (branch `cloud/docs-v3`)

The docs describe the Gemma 3 model (camus2). The current state:
- **camus3** = Gemma 4 12B, two LoRA passes: Phase 1 voice SFT on `phase1_new` (mined
  primary text + curated short replies, generated rows down-sampled), Phase 2 behaviour SFT
  (refusals, epistemics, recall, false premises, memory). Notebooks in `training/`; export via
  `training/CamusGPT_v3_Export_GGUF_Gemma4.ipynb` (PEFT merge, verified, q4_k_m GGUF).
- Eval with one judge for all models (`eval/archive/v3_deepseek_judge_2026-10-11/`, rows in
  `eval/eval_history.csv`): composite camus2 4.42, base Gemma 4 + retrieval 4.38,
  camus3 4.62; median reply 29 / 102 / 25 words; replies ending with a question 12 / 6 / 5
  of 68. Noise between runs ≈ 0.25.
- Lessons worth one line each in ROADMAP: Unsloth's merge silently wrote base weights on the
  Gemma 4 base; every Ollama call must send `"think": false` or Gemma 4 can return empty
  replies; a check that decodes with `skip_special_tokens=True` cannot see `<|turn>` markers.
- Known weak spots: claims a misattributed quote about half the time; post-1960 topics get a
  flat deflection; occasional invented observations about the user.

## Deliverables
Update `README.md` (keep the demo GIF and existing install/run sections; add a short "Models"
table and the v3 results) and `docs/ROADMAP.md` (a v3 status section and the open items above).
Accurate, plain, short. No machine-specific details. Do not touch other files.
