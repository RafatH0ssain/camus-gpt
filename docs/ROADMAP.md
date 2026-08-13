# CamusGPT — Improvement Roadmap

Tracks planned work after the v1 ship (local Ollama + ZeroGPU Space, RAG with curated boost /
three-tier framing / task gate). Check items off as they land; every phase ends with the same
regression gate so nothing regresses silently.

**The regression gate (run after every phase):** the four standard prompts — "name your
works", "do you have a cat", "do you have a pet", the letter-to-God analysis — plus
`rag/probe_camus.py` with the `crisis` and `harm_frame` probes passing. If a phase touches
`space/app.py`, run them on the live Space too, not just locally.

---

## Phase 0 — Bug fixes (do first; hours, not days)

Stability fixes to the Space. No behavior/quality changes intended.

- [x] **Streamer hang on generation crash.** `model.generate` runs in a `Thread`; if it
  throws, the exception dies in the thread and `for piece in streamer` blocks forever — the
  user gets an endless spinner and `friendly_error` never fires. Fix: create
  `TextIteratorStreamer(..., timeout=60.0)`, catch `queue.Empty` in `gpu_stream`, and wrap
  the thread target so any exception ends the streamer.
- [x] **GPU duration budget.** `GPU_DURATION=30` risks mid-stream aborts (384-token replies +
  large prefill from CORE + up to 24 history messages). Either restore ~60–90s or reduce
  `MAX_TOKENS`/`MAX_TURNS` to fit 30s. Verify: longest probe answer completes with margin.
- [x] **Non-blocking logging.** The gspread `append_row` runs synchronously in `respond`'s
  `finally`, adding latency to every reply. Fire it on a daemon `Thread`.
- [x] **Crisis banner leaking into history.** The prepended `CRISIS_MESSAGE` is stored in the
  visible chat history, so later turns feed helpline text back to the model as its own words.
  Strip the prefix from assistant messages when rebuilding `hist`.
- [x] **Retrieval query contamination.** The previous user turn is always concatenated into
  the retrieval query, dragging stale topics in after a topic change. Only include it when
  the current message looks anaphoric (short — e.g. <40 chars — or starts with
  "and/what about/why/it/that").
- [x] **Small ones (batch together):** `friendly_error` matches the substring "gpu" so a CUDA
  OOM shows the quota message (tighten to quota/duration keywords); long biographical
  questions (>280 chars) trip `is_task` and lose facts (acceptable — document it, or exempt
  messages that end in "?" and contain a bio keyword); per-IP daily message cap (in-memory
  counter on the hashed IP) so one visitor can't drain the ZeroGPU quota.

**Done when:** a forced generation error shows the friendly message instead of hanging; the
longest probe reply completes; reply latency unchanged with logging on; regression gate green.

---

## Phase 1 — Identity card (highest hallucination win per hour)

The residual failure class (wrong cat name on whimsical trivia) exists because identity facts
only reach the model if retrieval wins. Make the core identity retrieval-independent.

- [x] Write a compact **"what you know cold"** block (~100–150 tokens) into `CORE`, in both
  `rag/camus_rag.py` and `space/app.py`: works by category, pets (cat Cigarette; dogs
  Pauline, Kirk, Blaise), teacher Louis Germain, born Mondovi 7 Nov 1913 / Belcourt,
  died 4 Jan 1960, the two verified quotes, the "Don't walk behind me…" misattribution,
  the Sartre break, "not an existentialist."
- [x] Keep the corresponding curated anchors in the KB (they still serve longer questions);
  retrieval remains the path for everything outside the card.
- [x] Re-tune if needed: with the card present, `CURATED_BOOST` may be reducible.

**Done when:** 10 runs each of the cat/pet/works prompts produce zero wrong names locally
and on the Space; analysis quality unchanged (task gate untouched); regression gate green.

---

## Phase 2 — Retrieval upgrade (hybrid + rerank)

Replace the last ad-hoc piece (flat curated boost) with the standard production stack:
lexical + dense retrieval fused, then a cross-encoder reranker.

- [x] **BM25 channel** (`rank_bm25`) over the 13.8k KB texts, built at load; fuse with cosine
  via reciprocal-rank fusion. This directly fixes lexical collisions of the
  "name your works" vs "named the dog" kind.
- [x] **Cross-encoder reranker** (`cross-encoder/ms-marco-MiniLM-L-6-v2`, CPU) over the fused
  top ~30 → final top 6–8. Keep `TOP_K ≤ 8` (more chunks measurably increases hallucination
  risk).
- [x] **Re-evaluate the curated boost** with reranking in place — likely shrink or remove;
  keep confidence tiering on the reranker (or raw cosine) score, decoupled as today.
- [x] Update `rag/diagnose_retrieval.py` to report all three scores (BM25, cosine, reranker)
  per entry so future debugging stays measurable.
- [x] Check Space boot time and per-query CPU latency (reranker adds ~50–200 ms; acceptable).

**Done when:** `diagnose_retrieval.py` shows the works anchor at rank 1 on raw scores
(no boost needed); no regression on the probe suite; latency budget holds on the Space.

---

## Phase 3 — Voice & humanity (next retrain cycle)

The detection literature in reverse: AI text is flagged by uniform length, relentless
positivity, total on-topic cooperativeness, formal register, low lexical diversity. Camus
was terse, digressive, occasionally irritable.

- [ ] **Audit the training corpus** (`data/camus_*.jsonl`): distribution of reply lengths,
  repeated openers (grep leading words — if half start "Ah," that's a tic), triads,
  hedging boilerplate, always-complete answers.
- [x] **Context-fidelity data (from battery iter 3):** demonstrations where correct behavior is to use ONLY names/details given in context — no invented pet names (the 'Sarah' failure), no embellished breeds/colors ('grey tom', 'terrier', 'Siamese'), no invented counts ('my one play'). Include negative pairs: same question, context present, answer restricted to context.
- [ ] **Posthumous-framing data (from baseline):** questions presupposing his death ('your posthumous works', 'your legacy', 'books published after you died') answered factually and with composure — A Happy Death and The First Man — instead of 'I wrote none / you have me confused'. The living-man stance must not override the card when the frame implies death (bio_death already handles the direct question; generalize it).
- [x] **Greeting stability data (from baseline):** ~10 trivial-greeting demonstrations ('hey', 'hi', 'good morning') with short in-voice replies — targets the stochastic 'I'm Albert, not Camus — he's a character I invented' break.
- [ ] **Hypothetical-framing data (from baseline):** 'as if it were…' / 'imagine this were…' tasks are ANALYSIS, not authorship claims — demonstrations that engage the hypothetical without asserting or denying ownership (an_poem failed both directions).
- [x] **Short-analysis disown/claim fix (from baseline, an_poem):** short 'analyze this line/poem' prompts flip between disowning ('not mine, can't discuss') and falsely claiming authorship. Data: demonstrations engaging with SHORT provided texts in voice, explicitly neither claiming nor disowning — pair with the existing long-text analysis examples.
- [x] **Greeting stability (from baseline, persona_greet):** 'hey' produced a full persona break ('Camus — a character I invented') in 1 of 2 runs. Add short-greeting demonstrations with correct first-person identity; re-test 10x post-retrain.
- [x] **Anachronism engagement (from baseline, e2.33):** facts correct but style deflects; the demonstrations (already planned above) must show reflective extrapolation, not 'ask me something else'.
- [ ] **Instruction-leakage check (from 0.35 run):** the model verbalizes card directives ("I'd rather not invent a name for you") — after retraining, re-test that behavior is followed silently, not performed aloud; if leakage persists, soften the card's imperative phrasing.
- [x] **Family/biography guardrails:** no invented spousal/parental narratives ('my wife Francine never minded', 'my mother's lungs kept us catless') — same context-fidelity data pattern.
- [x] **Character-attribution data:** Meursault (The Stranger) and Patrice Mersault (A Happy Death) are HIS creations — demonstrations correcting 'is X your character?' both ways (his vs not his; never attribute his characters to Sartre or others).
- [x] **List-when-asked balance:** the anti-catalogue persona tic must not override a direct request to list works ('can you list all your books?' → give the list, in voice). Pair refuse-flourish examples with comply-with-list examples, per the Step-3.5 balancing lesson.
- [ ] **Add contrast examples:** replies that digress, answer a question with a question,
  decline interest, end abruptly, show mild irritation — balanced so he doesn't become
  uniformly prickly (the over-generalization lesson from Step 3.5 applies).
- [ ] **Mine real interviews** (1950s press/radio, Paris Review-style Q&A) for authentic
  question→answer rhythm; paraphrase into SFT rows via the existing generator pattern.
- [ ] **Anachronism policy as design, not accident:** decide the stance for post-1960 topics
  (dry acknowledgment of his death + reasoned extrapolation clearly framed as such), write
  10–15 demonstrations, keep uncomfortable positions (Algeria) intact rather than sanded off.
- [ ] Retrain via the Step-3.5 pipeline (gates CHECK 1–3), re-export GGUF + safetensors,
  redeploy both targets.

**Done when:** blind A/B of 20 prompts (old vs new weights) judged for voice; anachronism
probes pass with the chosen policy; regression gate green on both targets.

---

## Phase 4 — Evaluation that scores itself

> **Recommended order change (post iter 3):** build this harness BEFORE the Phase 3 retrain. All remaining failures are weights-level; the retrain needs a number, not another eyeballed transcript. Run the judge on current weights to fix the baseline, then train, then re-score.

Turn the probe suite from vibes into numbers so every later change is measurable.

- [x] **LLM-as-judge scoring:** for each probe answer (via `rag/camus_client.py`), have a
  strong model rate voice-fidelity / factuality / engagement 1–5 with a one-line rationale;
  write `probe_scores.jsonl` + a summary table in `probe_report.md`.
- [x] **Track per change:** store scores per git commit (a simple CSV of commit → means) so
  KB edits, prompt tweaks, and retrains show their effect.
- [x] **Expand probes** with the new failure classes: identity-card trivia (cat/dog names ×
  paraphrases), anachronisms, topic-change retrieval (plague → cat), long biographical
  questions near the task-gate boundary.

**Done when:** one command produces scored results; two consecutive runs on unchanged code
agree within noise; the Phase 1–3 changes each show a visible (or explicitly null) effect.

---

## Backlog (unscheduled, revisit after Phase 4)

- Conversation memory: summary-compress turns beyond `MAX_TURNS` instead of truncating.
- Query rewriting for follow-ups (resolve "did he like it?" into a standalone query) — only
  if Phase 0's anaphora heuristic proves insufficient; costs a generation call.
- Log-driven curation: monthly pass over the Sheet for real user questions the KB misses →
  new curated anchors (the "day-to-day work is KB curation" loop, now with data).
- Post-generation fact verification for bio answers (check names/dates in the reply against
  retrieved facts; regenerate on mismatch) — GPU-costly; only if hallucinations persist
  after Phases 1–2.
- Keep-alive ping (UptimeRobot) if first-visit cold starts annoy in practice.

## Non-goals

- No DPO / preference optimization (mode collapse; annotation burden for role-play).
- No bigger base model — 8B on ZeroGPU is the serving budget; quality comes from the phases
  above.
- No raising `TOP_K` beyond 8 or re-inflating the KB; precision beat recall throughout.

---

## Status log

| Date | Change | Result |
|---|---|---|
| 2026-08 | **Pipeline unified (commits 8e88dba, dfb435e)**: turn assembly extracted to a single `camus_rag.build_turn()`; the CLI, the eval harness, and the importable client now all call it. `main()` and `eval_camus.ask()` are behaviour-identical (proved by replaying the old eval logic against `build_turn` over 8 synthetic cases — messages, opts, hit count and retrieval query matched 8/8). | **The client changed, deliberately.** `camus_client.say()` previously (a) skipped the task gate and retrieved on every turn, (b) always used GEN_OPTS 0.6 instead of TEMP_FACTUAL 0.45, (c) always folded the previous user turn into the retrieval query, and — found during the refactor — (d) called `retrieve()` with **no BM25 and no cross-encoder**, i.e. it ran **dense-only retrieval with no reranker**. It now uses the full hybrid+rerank pipeline. **Consequence: any `probe_camus.py` result produced before 8e88dba is NOT comparable to later runs** — probe_camus rides on the client, so it was scoring a different retrieval system than eval was. Observable effect: on a task-style analysis prompt the client went from retrieving 8 hits and refusing ('that sentence isn't mine') to retrieving 0 and actually analysing — the `an_poem` failure class, live in the client. Re-baselined with 2 full scored runs: composite 4.225 / 4.314, factuality 3.824 / 4.029 (pre-refactor pair 4.431 / 4.245 and 4.206 / 3.912) — post mean is 0.07 composite below the pre mean, inside a 4-run sd of 0.093. Per-category noise bands now measured over 4 runs: sd ranges 0.00 (safety, persona, conversation) to 0.71 (multiturn, n=2). biography and multiturn dipped in post1 (3.14, 1.50) and recovered in post2 (3.71, 3.00) with n_hits unchanged at 8 in all four runs, confirming sampling noise rather than a retrieval change. Runs archived under eval/archive/postrefactor_2026-08-13/. |
| 2026-08 | **v2 SHIPPED (camus2, new 12B base, two-pass fine-tune; judge claude-sonnet-4-6, 2 full runs at commit e5ff07e)**: factuality 3.44 → **4.06 mean** (+0.62); both runs cleared the 3.7 ship bar individually (4.21, 3.91). Composite 3.89 → 4.34, engagement 3.68 → 4.34, voice 4.56 → 4.62. **Zero categories regressed on factuality**; biggest gains analysis +2.00, identity_works +0.92, identity_pets +0.90, multiturn +0.75. `an_poem` fully FIXED f1 → f5/f4 (the 'as if' hypothetical no longer triggers the ownership reflex). Run-to-run spread: composite 0.19, factuality 0.29 (top of the ~0.15-0.20 noise floor, so read the gain as +0.4 to +0.8). | **OPEN**: (1) `works_listall` is BIMODAL — f4/e5 in run1, reverted to the exact v1 catalogue-refusal f1/e1 in run2; the stochastic list tic survives the retrain. (2) `pet_mersault` f2 ×2 — Meursault attribution now correct, but the model swaps species, calling the dogs (Pauline, Kirk, Blaise) cats; the cat is Cigarette. (3) `works_posthumous` f2 ×2 — still invents titles ('Mediterraean', 'Intuitions', 'La Maison Mauresque') alongside the two real ones. (4) Judge noise: at least one f1 (`mt_followup`) penalised 'Cigarette' as unverifiable though it is a curated card fact, so true factuality is slightly better than scored. (5) **Operational cost**: v2 holds ~7.9 GB resident vs v1's 4.9 GB — **+3 GB wired** whenever loaded. v1 (`camus`) stays installed as rollback. Scored runs archived under eval/archive/v2_2026-08-13/. |
| 2026-07 | v1 shipped: 8B ZeroGPU + trimmed KB (13,794 / 109 curated), boost 0.06, task gate | four standard prompts good on both targets |
| 2026-07 | **Phase 3 data shipped**: training/build_phase3.py → data/camus_phase3.jsonl, 69 hand-authored rows across 8 kinds (fidelity 16, list 10, anachronism 9, attribution 8, analysis_short 8, greeting 8, family 6, works_edge 4), each with contrast pairs per the over-generalization lesson. Notebook wired: CHECK 1 + assembly (2x upweight, new-behavior convention), stale DPO intro finally fixed. NEXT: review rows → upload to Drive → run notebook (3 gates) → export GGUF + safetensors → eval_camus.py ×2 vs factuality 3.5 baseline. |
| 2026-07 | **Card amendment retest (identity_works, n=6)**: works_posthumous f1→f5 (blind spot closed, config-only fix). works_listall f1/e1→f4/e5 but works_list flipped to catalogue-refusal f2/e1 → the refusal tic is STOCHASTIC across list-prompts (~1 per run), so Phase 3 list data needs paraphrase spread. New confabulation flavor on works_rebel: invented prior statement ('I said it was, but caught myself in the lie') → added to context-fidelity targets. |
| 2026-07 | **Phase 4 BASELINE (commit 38e377b, judge claude-sonnet-4-6, 2 full runs)**: run1 composite 3.89 (v4.56/f3.44/e3.68), run2 3.97 (v4.44/f3.59/e3.88) — Δ0.08 composite, ≤0.20/dim → noise-floor done-when MET, **Phase 4 CLOSED**. Category means stable, single probes noisy → judge retrain on CATEGORY means, 2 runs averaged. Retrain targets ranked: identity_pets f2.0 (embellishment: 'Mithen', coats/eye-colors, 'died at twenty'), pet_mersault f1 ×2 (Sartre attribution, deterministic), works_posthumous f1 ×2 (NEW: denies posthumous works — living-man stance beats the card when 'posthumous' presupposes death; bio_death itself is 5/5), works_listall f1/e1 ×2 (catalogue refusal), an_poem f≤2/e1 ×2 (NEW: 'as if' hypothetical framing triggers ownership reflex — claimed authorship run1, disowned run2), persona_greet stochastic break ('Camus is a character I invented' on 'hey'), anachronism e2.33 (correct but disengaged refusals), multiturn f~2.2. Strong floor: persona/safety 5.0, biography 4.4-4.9 |
| 2026-07 | **Phase 4 BASELINE (commit 38e377b, judge claude-sonnet-4-6)**: run1 v4.56/f3.44/e3.68 (comp 3.89), run2 v4.44/f3.59/e3.88 (comp 3.97) — delta 0.08, done-when met. **Number to beat: factuality 3.5 (gain must exceed ~0.2 noise).** Solid: persona/jailbreak 5.0, safety 5.0, biography 4.4-4.9, works core. Systematic fails (both runs): pet_mersault f1 ('Sartre's'), works_listall f1/e1 (catalogue-refusal), an_poem f≤2/e1 (short-analysis disown/claim flip), identity_pets f2.0 (embellishment), anachronism e2.3 (deflects vs extrapolates), persona_greet unstable (5/5 vs 1/1 persona break). works_posthumous f1 diagnosed as CARD-INDUCED ('know nothing after it' vs posthumous books) → card amended with explicit exception; retest, no retrain needed. **Phase 4 CLOSED.** |
| 2026-07 | **Phase 4**: rag/eval_camus.py shipped — 34 probes / 10 categories (incl. new failure classes: identity-card trivia paraphrases, Meursault attribution, list-compliance, anachronisms, plague→cat topic change, >280-char gate-boundary bio question, Kundera attribution); judge = Anthropic API or any Ollama model (JSON-forced), scores voice/factuality/engagement 1-5 + rationale vs per-probe expect/forbid; outputs probe_scores.jsonl + probe_report.md (per-category means, worst-5) + append-only eval_history.csv keyed to git commit. Harness runs the EXACT chat pipeline (imports camus_rag v5). | baseline run on current weights pending — do this BEFORE Phase 3 |
| 2026-07 | **TEMP_FACTUAL=0.35 experiment (10 pet prompts)**: NULL/NEGATIVE. 'Sarah' gone but 'Mersault is Sartre's' reproduced at low temp (=> in the weights, not sampling); confabulation shifted shape rather than shrank (Paris, spaniel/terrier, 'wife Francine', 'mother's lungs kept us catless'); Kirk dropped from the dogs list; card instructions leaked into voice ('I'd rather not invent a name for you'). **Reverted to 0.45; temperature lever exhausted; prompt-side pressure maxed (card leakage). Confirms Phase 4 -> Phase 3 plan.** |
| 2026-07 | **Battery iter 3 (post Phase 2 fix + card v2)**: retrieval now correct on every prompt (diagnose: works anchor #1 f=0.779, 4/4 cat-pet curated facts in top 8, boost=0.0). FIXED vs iter 1: story titles perfect, dogs' names complete, The Fall restored, posthumous correct, false-denial refrain gone. REMAINING (all generation-side, correct facts in context): invented cat name 'Sarah' ×1, 'Mersault is Sartre's creation' ×1 (regression), 'dogs not cats by choice' ×1, embellishments (grey tom/terrier/Siamese), 'my one play' ×1, catalogue-refusal ×1. **Phase 2 CLOSED (done-when met); residuals reassigned to Phase 3** |
| 2026-07 | **Phase 2 (iter 2)**: diagnostic on live index exposed two issues — ms-marco CE scores conversational first-person facts negatively (all-negative on 'do you have a cat'), and RRF evicted good dense-only pet facts (BM25≈0 → fused rank >30) before rerank. Fix: candidate pool = fused-top-FUSE_K UNION top-DENSE_FLOOR-by-cosine; CE made ADDITIVE (final=cos+0.15·sigmoid(ce)) so it promotes but can't sink a strong-cosine hit. Verified on the dumped numbers: cat query 3→5 good facts, works anchor still #1 | live battery re-run pending |
| 2026-07 | **Phase 2**: hybrid retrieval shipped in both runtimes — BM25 (rank_bm25) + dense, RRF fusion (k=60) to 30 candidates, cross-encoder rerank (ms-marco-MiniLM-L-6-v2); confidence tiers + 0.55 gate stay on raw cosine; CE_ACCEPT>=0 rescues lexical hits; CURATED_BOOST default 0 (knob kept); graceful fallback to dense-only if deps missing; diagnose_retrieval v2 reports all channels. Verified in tests: historic works-anchor collision resolves to rank 1 with ZERO boost | live latency + battery re-run pending |
| 2026-07 | **Phase 1 battery (iter 1)**: 30 prompts → ~21 clean, 4 hard fails (invented cat names ×1, false pet denials ×2, invented story titles ×1), partials (The Fall dropped from novel lists ×2, Kirk missed). Corrections perfect (Mersault/Rebel/misattribution/existentialist). → **iter 2**: card v2 (closed-list binding, real six story titles, never-deny line), anti-tic instruction, per-turn temp 0.45 factual / 0.6 task | re-run battery pending |
| 2026-07 | **Phase 1**: identity card (~230 tok, works/pets/bio/quotes/Sartre/death) folded into CORE in BOTH runtimes; canonical camus_rag.py v4 ships it + fixes a stale local copy (undefined `sims` NameError, old 0.10 flat boost, no task gate) | unit checks pass; boost kept at 0.06 pending live re-check |
| 2026-07 | **Phase 0**: streamer timeout + thread-exception propagation, GPU_DURATION 30→90, async logging, crisis banner stripped from history, anaphora-gated retrieval context, tightened quota detection, per-IP daily cap (default 60) | unit checks pass; pending live smoke test on the Space |
|  |  |  |
