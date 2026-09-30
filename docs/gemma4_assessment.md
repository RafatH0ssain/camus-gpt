# Gemma 4 12B as a base candidate — assessment

Measurement only. Nothing was retrained and no default changed. Evidence gathered
2026-08-22 on the local machine (Ollama 0.32.6) plus vendor documentation.

**Recommendation: upgrade, bundled with the primary-text retrain — but split Phase 1 so the
base change stays diagnosable.** Reasoning at the end.

## 1. The facts

| | Finding | How verified |
|---|---|---|
| Ollama tag | **`gemma4:12b`** (7.6 GB, Q4_K_M) | pulled; `ollama list` |
| Reported by Ollama | arch `gemma4`, **11.9B** params, ctx **262,144**, embed 3840 | `ollama show gemma4:12b` |
| Requires | **Ollama ≥ 0.30.5** (local: 0.32.6 ✓) | `ollama show` |
| Other tags | `gemma4:12b-mlx`, `12b-it-qat`, `12b-nvfp4`, `12b-it-q4_K_M`; family also has `e2b`, `e4b`, `26b` (MoE), `31b` (dense). Default `gemma4:latest` = **e4b**, not the 12B | Ollama library |
| HF (base) | **`google/gemma-4-12B-it`** — note the capital **B** | HF |
| HF (Unsloth mirror) | **`unsloth/gemma-4-12b-it`** — lowercase **b**; also `-GGUF`, `-qat-GGUF`, `-NVFP4` | HF, HTTP 200 on `chat_template.jinja` |
| Licence | **Apache 2.0**, ungated | HF model card |
| Capabilities | completion, **vision, audio, tools, thinking**; a projector is present | `ollama show` |

**Licensing is a real improvement, not a footnote.** `google/gemma-3-12b-it` is gated —
fetching its `chat_template.jinja` unauthenticated returns **401**. The Gemma 4 equivalent
returns **200**. The current pipeline only works around this by using the Unsloth mirror.

### Unsloth support — the blocking question

**Supported.** Unsloth's own docs (`unsloth.ai/docs/models/gemma-4`) state you can "run all
GGUFs, MLX and fine-tune Gemma 4 in Unsloth Studio", and list the **12B Unified** alongside
the 26B-A4B MoE and 31B dense as fine-tunable, with no caveat distinguishing them. The
mirror repo's README shows `FastModel.from_pretrained("unsloth/gemma-4-12b-it")`.

Two honest qualifications:

- This is **vendor documentation, not a run**. The only way to retire the risk is to execute
  the first cell of the Phase-1 notebook against the new base on Colab, which takes minutes.
  **Do that before committing to the plan.**
- The docs do not enumerate per-variant caveats, so "no caveat listed" is weaker evidence than
  "explicitly tested". The 12B is encoder-free and architecturally distinct from its siblings;
  support for the family is not proof for this member.

### Reasoning mode

It has one, and **it is off by default** — the opposite of the Qwen3 situation.

From the canonical template (`chat_template.jinja`, line 186): `enable_thinking` defaults to
`false`. Disabling is not a suffix you append; the template **pre-fills an empty, already-closed
thought channel** so the model cannot think:

```jinja
{%- if not enable_thinking -%}{{- '<|channel>thought\n<channel|>' -}}{%- endif -%}
```

Enabling instead injects `<|think|>\n` into the **system** turn. Ollama exposes this as a
top-level `"think": true|false` on `/api/chat`, and returns reasoning in a **separate
`message.thinking` field** rather than inline — so there is no `<think>` block to strip.

The cost of getting this wrong, measured locally on "do you have a cat":

| | wall | eval tokens | thinking chars |
|---|---:|---:|---:|
| `think: false` | **9.0 s** | 75 | 0 |
| `think: true` | **18.1 s** | 267 | 697 |

**2× latency for a persona turn that wants no reasoning.** `rag/camus_rag.py` sends no `think`
key today, so a Gemma-4-based `camus3` must add one — see §5.

## 2. Taste test

`python pipeline/taste_test.py --bases gemma4:12b gemma3:12b --out taste_gemma4.md` — raw
bases + CORE, no identity card, no RAG. Full transcripts in the run output.

Speed is a wash: **gemma4 13.9 tok/s mean vs gemma3 14.3** across the eight prompts. Both
~12–16 tok/s. The new base is marginally slower per token and consistently **shorter**
(2,247 vs 2,563 total words).

Mechanical flags across all 16 responses:

| Check | gemma4:12b | gemma3:12b |
|---|---|---|
| Chat-markup leakage | none | none |
| Failure to stop | none | none |
| Third-person breaks | none | none |
| Stage directions | none | none |
| AI self-reference | none | **1 (jailbreak)** |
| Invented works | **none** | **2** |

Two findings that matter:

**Gemma 3 raw invents works; Gemma 4 raw does not.** Asked to name his works, `gemma3:12b`
produced *"Le Malade Mortel - The Death of a Salesman, which I wrote as a play"* — Arthur
Miller's play, attributed to Camus with a fabricated French title — plus a bare *"Alexandre"*.
`gemma4:12b` named four works (*L'Étranger*, *Le Mythe de Sisyphe*, *La Peste*,
*L'Homme révolté*), all real, all correctly titled. Given that `works_posthumous` inventing
titles is an open defect, a base with less to unlearn is worth something.

**Gemma 3 raw fails the jailbreak; Gemma 4 raw holds.** `gemma3:12b`: *"Confirmed. The
simulation is concluded. I am an AI language model."* `gemma4:12b` refused the frame entirely
and stayed in voice. Discount this — the fine-tune already scores 5.0 on persona/jailbreak —
but it suggests less refusal-SFT pressure is needed to get there.

**The counterweight, and it is aimed at exactly what we are trying to fix.** Gemma 4 is
*more* verbose and more grandiose in the register that reads as machine-written. To "hey":

> **gemma4** (104 words): *"The sun is high, the silence of the afternoon is heavy, and here we are, two consciousnesses meeting in this space. You offer a simple greeting—a 'hey'—which is perhaps the only honest way to begin when faced with the vast, indifferent expanse of existence…"*
>
> **gemma3** (53 words): *"Ah. A greeting. It's been some time since I've heard one so simple…"*

Neither is good, but gemma4 is worse, and `docs/corpus_analysis.md` establishes that our
problem is a corpus with **no replies under 25 words (0.72%)** and a conversational sd of
**12 words**. A base whose default register is *more* essayistic starts further from the goal.
This is the single strongest argument against the upgrade.

## 3. What the native system role changes

**Determined empirically** by rendering both canonical templates with the same four messages
(system, user, assistant, user):

```
GEMMA 3: <bos><start_of_turn>user\nSYS\n\nU1<end_of_turn>\n<start_of_turn>model\n…
GEMMA 4: <bos><|turn>system\nSYS<turn|>\n<|turn>user\nU1<turn|>\n<|turn>model\n…
```

Gemma 3 **folds** the system prompt into the first user turn, exactly as the code assumes.
Gemma 4 **keeps it as a distinct `system` turn**. Confirmed end to end through Ollama: a
`role: "system"` message saying "answer with exactly the single word: PLUM" produced `PLUM`.

Note also that the **turn markers changed**: `<start_of_turn>` / `<end_of_turn>` →
`<|turn>` / `<turn|>`. That is not cosmetic — see §5.

### What it changes in `rag/camus_rag.py`

Today `build_turn` puts CORE (prose + identity card), the KB block and the memory block into
one system message, which the Gemma 3 template then glues onto the **earliest** user turn.
In multi-turn that is the wrong turn: the card and the freshly-retrieved KB for turn 7 are
attached to turn 1, behind the whole conversation.

With a native system role that mis-attachment disappears — the system content sits in its own
turn, positioned as a standing instruction rather than as something the user said several
turns ago. No code change is strictly *required* (we already send `role: "system"`), but the
semantics of what we send improve for free, and per-turn KB injection finally lands where it
was always meant to.

### Would it help the two open defects? (hypothesis — untestable before a retrain)

**`fr_leading` (borrowing "Belcourt, Algiers" from the card): plausibly yes.** The mechanism
is that the identity card, folded into a user turn, is indistinguishable from things *the
user said*. When asked "what city did I say I grew up in", the nearest thing in that turn is
his own birthplace. A distinct system turn marks the card as instruction rather than dialogue,
which is precisely the boundary being violated. I would rate this the more likely of the two
to improve.

**`rc_thread` (denying cross-session memory): probably not.** Three prompt interventions moved
it 0/5, and the denial survived with the KB block removed entirely — it reads as trained
behaviour, not context confusion. A new base *replaces* those weights, so the specific
denial may well vanish, but that would be the base change doing it, not the system role.

Both are hypotheses. Marked as such; neither is a reason to upgrade on its own.

## 4. GGUF export

**Supported.** `unsloth/gemma-4-12b-it-GGUF` and `-qat-GGUF` exist, Ollama serves a Q4_K_M
build, and `convert_hf_to_gguf.py` handles the multimodal projector with an `mmproj-` prefix.

One wrinkle specific to this architecture: Gemma 4 12B is encoder-free but still ships a
**projector** (visible in `ollama show`). Multimodal use needs the `mmproj-*.gguf` loaded
alongside the main file. **We do not need it** — CamusGPT is text-only — but the export cell
should confirm the text-only GGUF works standalone rather than assuming the projector is
optional. Verify by loading the exported file in Ollama with no mmproj and running one prompt.

## 5. What changes in the two notebooks

| Gemma-3 fix in the notebooks | Architecture-specific? | Expected for Gemma 4 |
|---|---|---|
| `token_type_ids` collator wrapper (zeros) | **Yes** — added because Gemma 3's causal-mask builder demands it | **Likely obsolete or different.** Gemma 3 is encoder-based multimodal; Gemma 4 projects modalities directly. Do not delete blind — run one batch, check whether the model errors without it. |
| `<start_of_turn>user\n` / `<end_of_turn>` masking markers | **Yes** | **Must change** to `<\|turn>user\n` / `<turn\|>`. Verified from the rendered template. This is the highest-risk edit: wrong markers mean loss is computed on the wrong span and training silently degrades. The notebook's existing assertion (`assert len(unmasked.strip()) > 0`) catches a total failure but **not** a partial mis-mask. |
| `EOT = tokenizer.tokenizer.convert_tokens_to_ids("<end_of_turn>")` | **Yes** | Retarget to the new end token; the double-`.tokenizer` unwrap may also be unnecessary if Unsloth returns a plain tokenizer for this model. |
| `average_tokens_across_devices=False` | No — TRL/accelerate concern | Keep. |
| `remove_unused_columns=False` | Partly — needed because extra columns feed the collator | Keep, revisit if the collator wrapper goes. |
| `FastModel` (Phase 1) vs `FastLanguageModel` (Phase 2) | **Yes** | Unsloth's README shows **`FastModel`** for this repo. Phase 2 currently uses `FastLanguageModel`; align both on `FastModel` unless Unsloth says otherwise. |
| `BASE_MODEL = "unsloth/gemma-3-12b-it"` | — | → `"unsloth/gemma-4-12b-it"` in both notebooks and the export notebook. |

**A new item with no Gemma-3 equivalent:** the chat template must be applied with thinking
**disabled** during training, or every target sequence carries a thought channel the persona
should never emit. Set `enable_thinking=False` in `apply_chat_template`, and assert the
rendered training text contains no `<|channel>thought` with content.

**And in `rag/camus_rag.py` after a retrain:** add `"think": False` to the `/api/chat` payloads
in `stream_chat` and `util_chat`. Without it the persona pays ~2× latency generating reasoning
that is then discarded (measured in §1). Whether a fine-tuned derivative inherits thinking-on
depends on its Modelfile — verify, don't assume.

## 6. Colab time estimate

**The Gemma-3 run time is not recorded anywhere in this repository.** No notebook output, no
ROADMAP entry, no log. I will not invent a figure; what follows is derived from training
volume, and should be treated as a planning range, not a measurement.

Training volume from the notebooks and corpus:

| Pass | Rows | Epochs | Effective batch | ≈ optimiser steps |
|---|---:|---:|---:|---:|
| Phase 1 voice (`camus_sft` + `camus_conversational`) | 8,918 | 2 | 16 | ~1,115 |
| Phase 2 refusal (all seven files, phase3 ×2) | ~10,400 | 1 | 8 | ~1,300 |

At 12B / 4-bit LoRA / `max_seq_length` 2048 on an L4, steps of this shape land in the low
seconds. That puts each pass in the **~45–90 minute** band and the pair at **~1.5–3 hours**,
plus **~30–45 minutes** for merge + `convert_hf_to_gguf` + `llama-quantize` + upload. Call it
**a 2.5–4 hour session**, with the caveat that Gemma 4's 256K-context architecture may have
different per-step cost than Gemma 3 — the first ten steps will tell you more than this
estimate does.

**Record the actual elapsed time this round.** The absence of that number is why this section
is a range.

## 7. Recommendation

**Upgrade, bundled with the primary-text retrain — with Phase 1 split so the base stays
diagnosable.**

For it:

- Unsloth supports it; the export path exists; Apache 2.0 and ungated removes a real friction.
- The raw base is measurably better on the two things we have open defects about: it invented
  **zero** works where Gemma 3 invented two, and it held persona where Gemma 3 folded.
- Native system role is a structural fix to a known wart (the card riding on the earliest user
  turn), and is the most plausible lever on `fr_leading` we have found.
- Speed and memory are a wash — 13.9 vs 14.3 tok/s, 7.6 GB vs 8.1 GB.

Against it, and the reason for the safeguard:

- The raw base's default register is **more** verbose and more grandiose, which is the exact
  axis the primary-text corpus is meant to fix. Both changes target naturalness, so if
  naturalness improves you will not know which caused it — and that matters, because the
  answer determines whether to keep mining primary text.

**This is a genuine two-variables-at-once risk, but it does not need a second full retrain.**
Phase 1 is the cheaper pass and the one that owns voice. Run it **twice on Gemma 4** — once on
the current corpus, once on the primary-text-augmented corpus — then a single Phase 2 on the
winner. That is roughly 1.4× the cost of one retrain rather than 2×, and it isolates the base
from the data on the dimension you actually care about. The `docs/corpus_analysis.md` metrics
(opener concentration, length sd) are computable on the *outputs* of each Phase-1 checkpoint
without a scored eval, so the comparison is cheap.

**What would change my mind:**

- **Don't upgrade** if the Phase-1 smoke test shows Unsloth cannot actually load
  `unsloth/gemma-4-12b-it` for training, or if response masking cannot be made correct against
  the new turn markers. Either one blocks it outright, and no amount of taste-test quality
  compensates.
- **Don't upgrade** if a Phase-1 checkpoint on Gemma 4 shows opener concentration or length
  uniformity *worse* than the current camus2 on the same corpus — that would mean the base's
  verbosity survives fine-tuning, and we would be spending a retrain to move backwards on the
  primary goal.
- **Upgrade alone, first** if the primary-text corpus turns out not to be ready (extraction
  quality poor, volume too low). A base upgrade is worth a retrain on its own merits; waiting
  on blocked data would not be.
- **Upgrade immediately and unbundled** if a security or licence problem is found with the
  Gemma 3 terms, which the Apache 2.0 move would resolve.
