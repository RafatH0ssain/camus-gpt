# Task 12 — correctness review of the v3 code

Scope: the two ranges in the brief, the three `training/CamusGPT_v3_*.ipynb` notebooks read
cell by cell as Colab runs them (including a re-run after a reset), and `pipeline/assemble_training.py`
and `rag/camus_rag.py` as they stand on `main` (b1ab757). No non-docs file was changed.

Notebook references are `file:line` in the `.ipynb` JSON plus the cell number, since the
cells have no line numbers of their own.

## How findings were checked

- **Confirmed by running**: findings 1, 3, 4, 9 (rate only), 13 and 14. The scripts used stubs
  for the gitignored inputs (`data/camus_sft.jsonl` replaced by an empty list); they are
  described in each entry so they can be re-run locally against the real data.
- **By reading**: the rest. No GPU, Ollama, model or Drive in the sandbox, so nothing that
  needs a training run or a live judge could be run.

Test suites (per `CLAUDE.md`):

| suite | Python 3.11 (sandbox default) | Python 3.12 (CI's version) |
|---|---|---|
| `pipeline` | 1 error: `test_temp_sweep` import fails, SyntaxError (finding 13) | 456 tests, OK (skipped=1) on 6 of 7 runs; 1 run hit the flaky test in finding 14 |
| `rag` | 1 error: `test_camus_rag.ThinkFalseTests.test_temp_sweep_ask`, same SyntaxError | 69 tests, OK |
| `training` | 19 tests, OK | 19 tests, OK |

---

## 1. Phase 2: the eval split leaks into train, so early stopping and the "best" checkpoint are picked on training data — HIGH

`training/CamusGPT_v3_Phase2_SFT_Gemma4.ipynb:388` (cell 6), together with
`pipeline/assemble_training.py:532`.

**What goes wrong.** The assembler applies upweighting by *repeating rows* (x2 epistemic,
x3 substantive/meditation/analyze_provided, x2 injection/phase3, x4 multiturn), then writes the
repeated rows into `phase2.jsonl`. The notebook then takes `train_test_split(test_size=0.05)`
over those repeated rows. Every copy of an upweighted row is independent in the split, so most
eval rows have an identical twin in train. `eval_loss` is therefore largely training loss, and it
is what `EarlyStoppingCallback(patience=3)`, `load_best_model_at_end=True` and
`metric_for_best_model="eval_loss"` act on. Overfitting cannot show up on it, early stopping
effectively never fires, and the checkpoint labelled "best" is just the one with the lowest
training loss. Nothing prints a warning.

**Trigger (confirmed by running).** I ran `assemble_training.build_phase2()` on the tracked data
files, with `camus_sft.jsonl` stubbed out as empty, then did a seeded 5% permutation split like
`datasets` does. Result: 1,444 rows, of which 936 are unique; **51 of 73 eval rows (70%) are
verbatim copies of a train row** (seed 3407; seeds 1 and 2 gave 66% and 77%). Adding the 450
weight-1 sft rows dilutes this to about 50%. That is still most of the eval set.

**Fix.** Split before weighting. Simplest: group rows by their messages (the assembler's
`dedup_key`) and split by group, e.g. hash the key and send a group to eval when the hash falls
under 5%, then drop eval-group rows from train. A cleaner option is for the assembler to write
a separate `phase2_eval.jsonl` of unique, unweighted held-out rows and record it in the manifest.

## 2. Phase 1 and Phase 2: resume restarts from a stale checkpoint when the data has changed, and can save an old adapter as new — HIGH

`training/CamusGPT_v3_Phase1_VoiceSFT_Gemma4.ipynb:524` (cell 11; `output_dir` at :391),
`training/CamusGPT_v3_Phase2_SFT_Gemma4.ipynb:570` (cell 10).

**What goes wrong.** Each notebook calls `resume_from_checkpoint=bool(glob(...checkpoint-*))`.
That resumes whenever *any* checkpoint exists in a folder named only by phase (and corpus tag),
never by which data or base the checkpoint came from. Checkpoints stay on Drive after a
successful run (`save_total_limit=2` keeps the last two). The next time you train with a
rebuilt corpus, the run picks them up:

- If the new run has no more steps than the old one, `Trainer` loads the old adapter, sees
  `global_step >= max_steps` and returns after **zero steps**. The next cell writes that **old
  adapter** to `camus3_voice_lora_<corpus>` / `camus3_guardrail_lora`, and the export merges it.
- If the new run has more steps, it continues the old optimizer and LR schedule partway through
  and skips the first N batches of a different dataset.
- In Phase 2 the "base" may also have changed. The merged Phase-1 base is rebuilt from whatever
  Phase-1 adapter is current, so a guardrail checkpoint trained on the previous voice adapter
  resumes on top of the new one.

**Trigger (by reading).** Train Phase 2 to completion. Regenerate `phase2.jsonl` (for example
with the defects drafts added) and its manifest, then run the notebook again. Cell 10 returns
almost at once, cell 11 saves the previous guardrail adapter, and nothing shows that no training
happened except a near-zero `train_runtime`.

**Fix.** Key the checkpoint folder to its inputs, e.g.
`checkpoints/phase2_<sha256[:12] of the corpus>_<PHASE1_CORPUS>`. Phase 1 should use the corpus
sha, which cell 6 already computes. Then resume only from that folder. After the final save,
assert `stats.global_step > 0`, or that `trainer.state.global_step` grew during this call.

## 3. `camus_rag.py`: the mid-session memory flush can never fire — MEDIUM

`rag/camus_rag.py:605` (constant at :417; trim at the lines just above).

**What goes wrong.** With `--memory` on, `history` is trimmed to `HIST_WINDOW` = 32 messages
(16 exchanges) after every turn. The flush condition is
`len(history) // 2 % EXTRACT_EVERY == 0` with `EXTRACT_EVERY = 40`. Because `len(history)//2`
never exceeds 16, it never reaches 40. The comment "mid-session flush so a crash loses nothing"
is never true. The session-end flush also sees only the last 16 exchanges. Older turns survive
only as the rolling summary, which `extract_memories` never reads.

**Trigger (confirmed by running).** I replayed `main()`'s history bookkeeping for 200
exchanges: zero flushes, and at most 16 exchanges held.

**Fix.** Keep a separate exchange counter (`n_exchanges += 1` per turn) and flush on
`n_exchanges % EXTRACT_EVERY == 0`. Extract from the turns being evicted at eviction time, or
keep an untrimmed transcript just for extraction, so evicted turns are not lost to it.

## 4. `camus_rag.py`: the task gate drops retrieval for ordinary biographical questions — MEDIUM

`rag/camus_rag.py:137` (`TASK_CUES`), `:144` (`is_task`).

**What goes wrong.** The cues are matched as substrings anywhere in the message. Three of
them, `"what can you"`, `"what do you make"` and `"the author"`, are also very common ways of
asking about his life. When one matches, `build_turn` skips retrieval completely, and with
`--memory` it skips memory too. The system prompt then carries the "No memory closely matched
this question… do not invent" disclaimer. So the model answers a factual question about his
life with no KB facts, and is told nothing matched.

**Trigger (confirmed by running).** `is_task` returns `True` for "What can you tell me about
Louis Germain?", "What can you tell me about your mother?", "Who is the author of The Plague?"
and "What do you make of your time at Combat?"

**Fix.** Narrow those cues to their task meaning, e.g. "what can you deduce", "what can you
tell about this", "what do you make of this", "the author of this". Or require a quoted or long
text alongside a weak cue. Add the four questions above to `test_camus_rag.py` as must-retrieve
cases.

## 5. Export notebook: re-run skips are decided by file size alone, so a partial or stale GGUF is shipped — MEDIUM

`training/CamusGPT_v3_Export_GGUF_Gemma4.ipynb:74` (`big`), `:92`, `:144`, `:151` (cell 1).

**What goes wrong.** The header says "a re-run skips every stage whose verified output exists".
But "verified" is never recorded: the skip tests are `size > 4 GB` for the q4 file, `> 10 GB` for
the f16 file, and "config.json plus any *.safetensors" for the merged folder. The GGUF magic
check runs only on the pass that wrote the file. Two ways this goes wrong:

- `llama-quantize` dies partway, from a full disk or a runtime hiccup, after writing more than
  4 GB of a roughly 7 GB file. `sh` raises, you re-run the cell, stages 1 and 2 are skipped,
  and stage 3 hashes the truncated file and copies it to Drive as the release.
- In the same runtime you retrain an adapter and re-run the export: every stage is skipped,
  and the previous GGUF is uploaded again with a fresh sha256 line.

A merged folder whose `save_pretrained` stopped after the first shard also passes `hf_ok`.
That case fails loudly later, in the converter.

**Trigger (by reading).** Kill the quantize step after about 5 GB is written, then run the cell
again.

**Fix.** Write each stage's output under a `.partial` name and rename it only after its check
passes, as stage 3 already does for the Drive copy. Re-check the GGUF magic, and ideally a
`gguf` header read, before stage 3 on every run. Write a small stamp file next to the outputs
holding the sha256 of each adapter's `adapter_model.safetensors`, and skip only when the stamp
matches the current adapters.

## 6. Phase 2: the local merged base is reused regardless of which Phase-1 adapter it came from — MEDIUM-LOW

`training/CamusGPT_v3_Phase2_SFT_Gemma4.ipynb:124` (cell 3).

**What goes wrong.** `LOCAL_MERGED = "/content/camus3_sft_merged"` is skipped whenever its
`config.json` exists. The path does not carry `PHASE1_CORPUS`, and the folder has no record of
the adapter it was merged from. Some cases in the same runtime:

- Switching `PHASE1_CORPUS` from `"new"` to `"old"` and re-running cell 3.
- Re-running after a new Phase-1 adapter lands on Drive.
- A "Restart session" after an interrupted save. This one fails loudly instead.

In the first two, Phase 2 silently trains on the previous merge. The merge-delta assert only
runs inside the skipped branch.

**Fix.** Name the folder `camus3_sft_merged_<PHASE1_CORPUS>_<sha12 of adapter_model.safetensors>`,
or write that sha into the folder and compare it before skipping.

## 7. Eval history rows are not comparable: judge errors silently shrink the probe set, and the suite and filters are not recorded — MEDIUM

`rag/eval_camus.py:441` (`append_history`), `:497`.

**What goes wrong.** A probe whose judge call raises gets `judge_error` and is dropped from
`scored`. The report's means and the `eval_history.csv` row are then computed over fewer probes,
and the only trace is `n`. The history row also does not record `--suite`, `--category` or
`--limit`. A v4 run (n=50) and a v3 run (n=34) sit in the same columns, and a v3 run that lost a
probe (n=33) looks like a normal run of the same suite.

**Trigger (by reading; visible in the data).** `eval/eval_history.csv` already has v3 rows
with n=33 next to n=34 rows for the same model and judge (`109d727`, `c0c5893`). Their
composites are compared as if they covered the same probes.

**Fix.** Add `suite`, `category`, `limit` and `judge_errors` columns, and widen the existing
CSV once. Alternatively, retry a failed judge call, and if it still fails, refuse to append a
history row (or mark it partial) rather than averaging over a subset.

## 8. The Ollama judge calls omit `"think": false` — MEDIUM-LOW (suspected)

`rag/eval_camus.py:396-405`, `rag/eval_memory.py:250-258`.

**What goes wrong.** This is the same class as the empty-answer bug already fixed for
generation: `think: false` is sent on every generation call, but not on the `--judge ollama`
path. A thinking-capable judge model (Gemma 4, Qwen 3 and others) can then spend its turn in the
thought channel and return empty or truncated `content`. `parse_judge` raises, the probe gets
`judge_error`, and finding 7 drops it from the means without saying so.

**Trigger (suspected; needs Ollama).** `python rag/eval_camus.py --judge ollama --judge-model gemma4:12b`.

**Fix.** Add `"think": False` to both judge request bodies, and extend `ThinkFalseTests` in
`rag/test_camus_rag.py` to cover them.

## 9. Phase 2 CHECK 2 fails falsely when example 0 is a multi-turn row; its second assert can never fail — LOW

`training/CamusGPT_v3_Phase2_SFT_Gemma4.ipynb:485`, `:490` (cell 8). The same pattern is in the
Phase 1 notebook, cell 9.

**What goes wrong.** `train_on_responses_only` unmasks *every* assistant turn. For a multi-turn
row the trained span starts with the *first* reply, but the check compares it with
`last_assistant(...)`, so `startswith` fails and reports "PARTIAL MIS-MASK" on correct masking.
This is a loud false alarm, not a silent bug. Separately, `norm(expected0) in trained0` follows
from the `startswith` just above it, so the "truncated" assert can never fail. Truncation is
already caught by the `startswith` assert. The real gap is that extra trained text *after* the
reply goes unchecked.

**Trigger (rate confirmed by running).** 32 of 1,444 phase-2 rows (2.2%) have more than one
assistant turn (camus_multiturn x4). Whether row 0 is one depends on the seeded shuffle of the
real corpus. Phase 1's current inputs have no multi-assistant rows, so it is unaffected today.

**Fix.** Compare against the concatenation of all assistant contents, normalised. Replace the
redundant assert with one that the trained text minus the expected text is only turn markers
and whitespace.

## 10. Phase 2 keeps a `messages` column next to `text`, so the trainer might render from it — LOW (suspected)

`training/CamusGPT_v3_Phase2_SFT_Gemma4.ipynb:386` (cell 6).

**What goes wrong.** `Dataset.from_list(convos).map(to_text)` keeps `messages`, `kind` and
`source`. Recent TRL treats a dataset with a `messages` column as conversational and tokenizes
it with `apply_chat_template(messages)`. That path has no `with_system` prepend and no
`enable_thinking=False`, and it ignores `dataset_text_field`. Unsloth's patched trainer probably
uses `text`, and if it did not, CHECK 2's `<|turn>system` assertion would catch it loudly for
rows without a system turn. Phase 1 avoids the question by building the dataset from `text`
alone.

**Fix.** Build the Phase 2 dataset from `{"text", "expected_tail"}` only, as Phase 1 does.

## 11. Assembler: a row present in two Phase 2 groups keeps the first group's weight — LOW (latent)

`pipeline/assemble_training.py:522-526`.

**What goes wrong.** Dedup is global across groups and "first occurrence wins". A row that
appears in, say, `camus_conversational` (x1) and `camus_epistemic` (x2) is kept once at x1.
Its upweighting is lost silently: the manifest shows `rows_after_dedup` per group but no
per-row weight. The tracked inputs have no cross-group duplicates today (the stubbed run
dropped 0), so this is latent. It will matter when the defects drafts are added.

**Fix.** On a cross-group duplicate, keep the higher weight (`max`), or raise an
`AssemblyError` naming both groups.

## 12. Phase 2 pre-flight runs after the expensive merge — LOW

`training/CamusGPT_v3_Phase2_SFT_Gemma4.ipynb:278` (cell 5).

CHECK 1 says it is the "cheap gate so you never start a run that dies halfway". But cell 3 has
already loaded the 16-bit base, merged the adapter and written about 24 GB to `/content` before
CHECK 1 looks for `phase2.jsonl`. A missing corpus costs the whole merge. **Fix:** move the
existence checks, and the manifest sha check, to just after `drive.mount` in cell 3.

## 13. Two environment bugs that break things before any logic runs — LOW (confirmed by running)

- `pipeline/temp_sweep.py:391` nests same-quote f-strings (`f'...{row['per_prompt_n'][k]}...'`
  inside an f-string). That is PEP 701 syntax, so on Python 3.11 or older the module is a
  `SyntaxError`, and `test_temp_sweep` and `test_camus_rag.ThinkFalseTests.test_temp_sweep_ask`
  error on import. CI pins 3.12, so CI is green, but nothing records 3.12 as the minimum.
  **Fix:** use double quotes for the inner subscripts, or state `python >= 3.12` in
  `requirements.txt` or the README.
- `rag/eval_camus.py:37` and `rag/eval_memory.py:37` import `dotenv`, but `python-dotenv` is
  not in `requirements.txt`. A fresh `pip install -r requirements.txt` followed by
  `python rag/eval_camus.py` fails with `ModuleNotFoundError`. The tests stub it, which hides
  this. **Fix:** add `python-dotenv` to `requirements.txt`, or make the import optional.

## 14. Flaky unit test — LOW (confirmed by running)

`pipeline/test_filter_reply_suitability.py:684-698`
(`FailFastTest.test_a_cap_still_stops_at_the_first_batch_with_workers_in_flight`).

This failed once in 7 runs with `'passage 1' != 'passage 0'`. With workers in flight, the
first call to reach the stub is not always passage 0. The code under test is fine; the
assertion assumes an ordering the thread pool does not guarantee. **Fix:** assert that
`seen[0]` is one of the first `max_workers` passages, or that exactly one call was made.

---

## Looked at and found sound

- Phase 1 and Phase 2 merge checks (`wrapped > 300`, nonzero `q_proj` delta) and the export's
  per-adapter delta plus the "differs from base on every probe" behaviour check.
- Manifest sha256, row count and kind histogram checks in both training notebooks. They fail
  loudly, and the row count matches the assembler's post-weight `rows`.
- The export's generated Modelfile matches `rag/Modelfile.camus3` parameter for parameter.
- `think: false` is present on every generation call: `camus_rag.stream_chat`, `util_chat`,
  `eval_camus.ask`, `eval_memory`, `temp_sweep`, `camus_client`.
- The `token_type_ids` probe in both notebooks; `BOOKKEEPING` strips `expected_tail` before
  the real collator sees a batch.
- `discriminator_sheet.py` sampling, key and scoring. Minor: an answer line like `3. him` is
  silently skipped as unanswered, because `"3."` is not a digit string.
