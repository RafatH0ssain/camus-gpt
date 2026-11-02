# Task 04 — Phase 2 must build its base with a PEFT merge (branch `cloud/phase2-peft-merge`)

## Why
Unsloth's `FastModel.save_pretrained_merged` reported success on this Gemma 4 (unified,
multimodal) base while writing the base weights unchanged — the first export answered like
stock Gemma 4. `training/CamusGPT_v3_Export_GGUF_Gemma4.ipynb` now merges with transformers +
PEFT and proves each merge changed the model. But the Phase 2 notebook
(`training/CamusGPT_v3_Phase2_SFT_Gemma4.ipynb`, the setup cell titled "Rebuild merged
Phase-1 locally") still builds the merged Phase 1 base with `save_pretrained_merged`, so
Phase 2 may have trained on plain Gemma 4.

## Deliverables
1. Rewrite that setup cell: load `unsloth/gemma-4-12b-it` in bf16 with transformers (class
   from `config.architectures[0]`), merge `SFT_ADAPTER` with `PeftModel.from_pretrained(...)
   .merge_and_unload()`, assert >300 LoRA modules attached and a non-zero weight change on
   `model.model.language_model.layers[0].self_attn.q_proj.weight`, save to `LOCAL_MERGED`,
   free GPU memory, then load `LOCAL_MERGED` with `FastModel.from_pretrained(...,
   load_in_4bit=True)` and attach the fresh guardrail LoRA exactly as now. Uninstall
   `torchao` first (Colab's old version makes PEFT refuse adapters). Mirror the export
   notebook's code; do not invent a new approach.
2. Extend `training/test_v3_notebooks.py`: no v3 notebook calls `save_pretrained_merged`;
   the export notebook asserts a weight delta and a behaviour difference from base; the Phase 2
   setup cell uses `merge_and_unload`.
3. Update `docs/ROADMAP.md` with one line explaining the change.

## Constraints
You cannot run Colab. Code cells must parse (`ast.parse` after stripping `!`/`%` lines — add
that as a test too). Keep cell outputs empty. All existing notebook tests must still pass.
