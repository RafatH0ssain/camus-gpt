# Naturalness research — merged, ranked report

Synthesis of `data/drafts/research_A_humanlike.md` (human-likeness) and
`data/drafts/research_B_camus.md` (sounding like Camus) against this repo's plan: Gemma 12B
LoRA SFT via Unsloth, served by Ollama, hybrid RAG + always-on identity card, blind human A/B
plus an OpenCode-model LLM judge, DPO abandoned after mode collapse. Only sources present in
those two files are cited, with the URLs copied from them. Nothing here was measured against
this repo's model; mappings are inference and marked where the source says so. `[U]` = the raw
file tags the item UNVERIFIED.

---

## 1. Top 10 changes, ranked by expected impact ÷ cost

| # | Change | Evidence (short) | Where it lands | Cost | Risk |
|---|---|---|---|---|---|
| 1 | Rebuild the conversational SFT set around a human length *distribution*, not a target length | DailyDialog ~14.6 tokens/utterance, https://aclanthology.org/I17-1099/ ; repo corpus is sd 12 w with 0.72% under 25 w (`docs/corpus_analysis.md`) | data | M | generators still emit 70±12; needs new rows |
| 2 | Enable `min_p` (~0.05–0.1) and raise `temperature` (~1.0–1.2) | Nguyen et al., https://arxiv.org/abs/2407.01082 ; Ollama ships `min_p` at 0.0 | Ollama options | L | Gemma not among tested families; high temp without `min_p` incoheres |
| 3 | Cap `num_predict` to force length variation and shorter turns | Laban et al., https://arxiv.org/abs/2505.06120 | Ollama options | L | truncation mid-reply |
| 4 | Segment corpus by translator; weight post-1947 texts up, Gilbert down | Kaplansky, https://journals.openedition.org/palimpsestes/1583?lang=en ; Rose, https://www.lrb.co.uk/the-paper/v42/n09/jacqueline-rose/pointing-the-finger ; Marris `[U]`, https://www.theparisreview.org/blog/2022/09/28/deep-emotion-plain-speech-camuss-plague | data | M | provenance metadata incomplete for existing chunks |
| 5 | Put the date *before* the question and date-tag retrieved passages, surfacing only ≤ 1960-01-04 | TCFT, https://arxiv.org/abs/2605.14636 ; REVERIEMEM, https://arxiv.org/abs/2606.25632 ; ExAnte, https://arxiv.org/abs/2505.19533 | retrieval + CORE prompt | L–M | date filter can empty a session's hits |
| 6 | Rebuild CORE on the three-layer persona architecture: concrete mundane identity, explicit instruction to be less polished/less capable | Jones & Bergen, https://arxiv.org/abs/2503.23674 and https://www.pnas.org/doi/10.1073/pnas.2524472123 | CORE prompt | L | win rates not reproduced (`[U]` UC Merced); persona gains no facts (Zheng, https://arxiv.org/abs/2311.10054) |
| 7 | Add stylometry to the eval: expressive-punctuation rate and Burrows' Delta vs his real text, with named entities masked | Études françaises 2020 `[U]` author, https://www.erudit.org/fr/revues/etudfr/2020-v56-n2-etudfr05582/1072479ar ; Burrows, DOI 10.1093/llc/17.3.267 ; Brad et al., https://arxiv.org/abs/2112.05125 | eval | L | Delta formulas behind paywall; topic confound |
| 8 | Add zero-cost mechanical metrics: length CV, question-back rate, opener-repeat rate, not-x-but-y density, MATTR-500 (guard) | EQ-Bench Slop Score, https://eqbench.com/slop-score.html ; Zheng et al., arXiv:2306.05685 | eval | L | slop tool under-reports on short chat turns |
| 9 | Add synthetic anti-sycophancy SFT turns (decline false premises) and an input reframer at serve time | Wei et al., https://arxiv.org/abs/2308.03958 ; Dubois et al., https://arxiv.org/abs/2602.23971 | data + retrieval | L–M | over-correction to uniformly prickly voice |
| 10 | Ground all personal claims in retrieved context and post-filter unattributed biography | Sun et al., https://arxiv.org/abs/2608.04570 | retrieval + eval | M | 35–49% over-inference is the documented baseline |

Ranking note: 1–3 fix the two *measured* defects (length uniformity, opener concentration) at the
lowest cost; 4–7 are Camus-specific and mostly measurement; 9–10 attack the documented
confabulation class last because they need retrieval changes.

---

## 2. Area-by-area findings and next actions

### Data

- **Claim.** A few hundred curated human-style turns move style further than tens of thousands
  generated. **Evidence:** LIMA, https://arxiv.org/abs/2305.11206 ; Zhan et al. (340 examples
  strip RLHF protections), https://arxiv.org/abs/2311.05553. **Action:** budget the LoRA set in
  the hundreds–low thousands, hand-picked; stop scaling generated conversational rows.
- **Claim.** Recursive training on generated data removes distribution tails first, and needs a
  fixed real-data slice each round. **Evidence:** Curse of Recursion, https://arxiv.org/abs/2305.17493 ;
  nature version, https://www.nature.com/articles/s41586-024-07566-y ; MAD, https://arxiv.org/abs/2307.01850 ;
  Seddik et al., https://arxiv.org/abs/2404.05090. **Action:** keep a non-regenerating slice of
  extracted notebook/essay prose in every generation round and never re-generate it.
- **Claim.** Persona-conditioned training actively teaches inventing a user's profile; EmpatheticDialogues is the nearest analogue to a non-sycophantic, push-back voice. **Evidence:** PersonaChat, https://aclanthology.org/P18-1205/ ; EmpatheticDialogues, https://aclanthology.org/P19-1534/. **Action:** keep user-profile conditioning out of SFT loss; sample a small human dialogue slice into the length histogram.
- **Claim.** An additive rubric over named defects yields a thresholdable filter plus per-defect diagnostics; unlikelihood negatives are a cheap inconsistency fix. **Evidence:** FineWeb-Edu, https://arxiv.org/html/2406.17557 ; Don't Say That!, https://aclanthology.org/2020.acl-main.428/. **Action:** score each turn on the six defects separately and add "what he must not say about the user" negatives.

### Training

- **Claim.** Preference optimisation is a diversity tax; DPO collapses lexical diversity to
  74–92% of baseline and degrades writing quality. **Evidence:** Kirk et al.,
  https://arxiv.org/abs/2310.06452 ; Antislop, https://arxiv.org/abs/2510.15061 ; mode-collapse
  attribution, https://openreview.net/pdf?id=3pDMYjpOxk. **Action:** the repo's DPO ban is correct;
  if a preference stage is ever wanted, use FTPO-style token penalties, not DPO.
- **Claim.** Plain CE SFT also narrows output diversity; an entropy regulariser is a candidate fix.
  **Evidence:** Li et al., https://arxiv.org/abs/2408.16673 ; collapse localisation,
  https://arxiv.org/abs/2604.16027. **Action:** measure diversity at the base checkpoint and after
  the LoRA so collapse is attributed to the stage that caused it.
- **Claim.** Fine-tuning on a teacher's outputs inherits the teacher's style, not its facts, and benign SFT degrades alignment. **Evidence:** False Promise of Imitating Proprietary LLMs, https://arxiv.org/abs/2305.15717 ; Qi et al. `[U]`, https://arxiv.org/abs/2310.03693. **Action:** treat generated rows as a style risk, not a facts source; keep training from the base checkpoint.
- **Claim.** Assigning a persona is itself a sycophancy vector in role-play, with a larger effect
  size than synthetic-data fixes. **Evidence:** Too Nice to Tell the Truth,
  https://aclanthology.org/2026.acl-long.1421/. **Action:** test the same prompts under
  high- and low-agreeableness framings and score opinion-validation rate.
- **Claim.** DPO has a documented length bias; preference pairs must be length-matched if used at
  all. **Evidence:** https://arxiv.org/abs/2305.18290 ; SimPO, https://arxiv.org/abs/2405.14734.
  **Action:** not applicable unless a preference stage returns; note in the training notebook.

### Decoding (what Ollama actually supports)

- **Claim.** Only `temperature`, `top_k`, `top_p`, `min_p`, `repeat_penalty`, `repeat_last_n`,
  `num_ctx`, `num_predict`, `num_keep`, `seed`, `stop` are exposed; `typical_p` is deprecated.
  **Evidence:** https://docs.ollama.com/modelfile.md , https://raw.githubusercontent.com/ollama/ollama/main/api/types.go ,
  https://docs.ollama.com/openai.md. **Action:** audit the Modelfile against this list; do not
  design around anything else.
- **Claim.** `min_p` is the best-evidenced available diversity lever and ships disabled.
  **Evidence:** Nguyen et al., https://arxiv.org/abs/2407.01082. **Action:** set `min_p` 0.05–0.1
  and raise temperature, then re-run the opener/length metrics.
- **Claim.** `repeat_penalty` penalises tokens, not sequences, and fights a persona that
  deliberately repeats key terms. **Evidence:** TensorRT-LLM sampling docs,
  https://github.com/nvidia/tensorrt-llm/blob/14863f0c23b6c33f8f6003c665fd0d3461a228bc/docs/source/features/sampling.md.
  **Action:** do not raise `repeat_penalty` for style; use it only for degenerate loops.
- **Claim.** Penalties may be silently ignored on older Ollama runners. **Evidence:** issue
  https://github.com/ollama/ollama/issues/15783 ; 0.30 blog,
  https://ollama.com/blog/improved-performance-and-model-support-with-gguf. **Action:** run
  `ollama --version`, then verify `repeat_penalty: 5.0` actually changes a loop-inducing reply.
- **Claim.** `typical_p`/eta optimises for the model's own mean and is deprecated. **Evidence:**
  https://aclanthology.org/2022.findings-emnlp.249.pdf. **Action:** reject on both grounds.

### Prompting

- **Claim.** A persona prompt is worth large measured human-likeness gains on the same base, and
  the winning persona was boring and specific. **Evidence:** Jones & Bergen,
  https://arxiv.org/abs/2503.23674. **Action:** give Camus mundane embodied texture (a street, a
  body, small annoyances), not literary opinions only.
- **Claim.** Ending replies with a question back is the cue human interrogators used most reliably
  to catch non-humans. **Evidence:** Jones & Bergen, https://arxiv.org/abs/2503.23674.
  **Action:** treat the question-back tic as the first behavioural defect to fix; target
  *unprompted* questioning, not the removal of all questions.
- **Claim.** "Forcing a persona" is a named, human-recognised failure; winning personas were
  explicitly told to be less capable. **Evidence:** Jones & Bergen 2024,
  https://arxiv.org/abs/2405.08007. **Action:** measure per-turn in-voice consistency and allow
  drift; instruct lower capability/polish.
- **Claim.** Personas do not reliably improve factual accuracy. **Evidence:** Zheng et al.,
  https://arxiv.org/abs/2311.10054. **Action:** keep the identity card as a style/eval lever;
  facts stay in retrieval (matches README's split).
- **Claim.** Input-level reframing (declarative→question) beats telling the model not to be
  sycophantic. **Evidence:** https://arxiv.org/abs/2602.23971. **Action:** add a serving-layer
  reframer; run the acceptance test on confidently-asserted user turns.
- **Claim.** A self-critique-and-revise pass is the one transferable piece of Constitutional AI.
  **Evidence:** https://arxiv.org/pdf/2212.08073 and
  https://www.anthropic.com/research/constitutional-ai-harmlessness-from-ai-feedback.
  **Action:** add a style-critic pass checking openers, length uniformity, aphorism density,
  fabricated biography — not refusals.

### Retrieval-as-style (his own passages as exemplars)

- **Claim.** No paper exists on style-aware retrieval; the nearest support is tuning-free
  personalisation and few-shot style prompting. **Evidence:** TICL,
  https://arxiv.org/abs/2502.08972 ; Jemama & Kumar, https://arxiv.org/abs/2509.24930.
  **Action:** retrieve by stylometric similarity to the reply target as an experiment, not a
  design assumption.
- **Claim.** Few-shot style examples give up to 23.5× higher style-matching accuracy than zero-shot
  and prompting strategy matters more than model size. **Evidence:** https://arxiv.org/abs/2509.24930.
  **Action:** prefer style-selected RAG exemplars over a longer persona description.
- **Claim.** CoSER's given-circumstance acting scores against the author's real scene, making
  training and eval tasks identical. **Evidence:** https://arxiv.org/abs/2502.09082. **Action:**
  build a small GCA set from real Camus scenes and reconstruct them.
- **Claim.** "Factual overreach" from shared retrieval/parametric memory is a named failure; tag
  each retrieved passage with who could see it. **Evidence:** REVERIEMEM,
  https://arxiv.org/abs/2606.25632. **Action:** tag every KB passage with a date (and genre) and
  filter at retrieval.
- **Claim.** One model's style can be imported by rewriting generic dialogue through
  author-conditioned prompts. **Evidence:** OpenCharacter,
  https://arxiv.org/abs/2501.15427. **Action:** use response-rewriting for mined notebook
  passages to keep grammar in-distribution.

### Evaluation (including stylometry)

- **Claim.** Score style separately from content or one failure hides the other. **Evidence:**
  CharacterBot, https://arxiv.org/abs/2502.12988 ; neural-text authorship attribution,
  https://aclanthology.org/2020.emnlp-main.673/. **Action:** report a Camus-style score and a
  content/factuality score separately, never a single in-character number.
- **Claim.** Burrows' Delta over the ~150 most common words is the cheapest reference-free
  Camus-distance metric. **Evidence:** DOI 10.1093/llc/17.3.267 ;
  https://academic.oup.com/dsh/article-abstract/17/3/267/929277 ; function words,
  https://aclanthology.org/W14-0908/. **Action:** implement per-chunk Delta against per-translator
  reference sets; use it as an untrained baseline before any classifier.
- **Claim.** Named entities inflate any apparent "Camus score"; mask them. **Evidence:** Brad et
  al., https://arxiv.org/abs/2112.05125. **Action:** strip names/entities before stylometric
  scoring and build style-vs-topic splits.
- **Claim.** Expressive-punctuation rate is a translator-portable, nearly free Camus signature.
  **Evidence:** Études françaises 2020 `[U]`, https://www.erudit.org/fr/revues/etudfr/2020-v56-n2-etudfr05582/1072479ar.
  **Action:** measure question/exclamation/exclamation-dots rate in generated text vs the
  translated corpus.
- **Claim.** LLM judges show position, verbosity and self-preference bias; a fine-tuned model is a
  bad judge of itself. **Evidence:** Zheng et al., arXiv:2306.05685 ; Panickssery et al.,
  https://arxiv.org/abs/2404.13076. **Action:** use a different model family, run A/B *and* B/A,
  truncate to 4000 chars, aggregate with Glicko-2 Elo, and keep one mechanical metric.
- **Claim.** An LLM judge may reward the overused tropes being eliminated ("slop bias"). **Evidence:**
  EQ-Bench, https://eqbench.com/creative_writing.html. **Action:** keep mechanical slop and
  MATTR-500 metrics alongside the judge; MATTR-500 must not fall below baseline.
- **Claim.** Judge metrics replicate poorly against human experts; validate on a small sample.
  **Evidence:** The Oscars of AI Theater, https://arxiv.org/html/2407.11484v4. **Action:** validate
  any automatic in-character metric on a Camus sample before trusting it.
- **Claim.** Decision-based probes and dimension-targeted queries surface character better than
  free chat. **Evidence:** PersonaGym, https://arxiv.org/abs/2407.18416 ; CharacterBench,
  https://arxiv.org/abs/2412.11912. **Action:** write sparse/dense decision probes ("would Camus
  choose X?") as well as open chat.
- **Claim.** A human discriminator test mixing model replies with real notebook passages is
  sharper than a rubric. **Evidence:** the repo's own `docs/HUMAN_EVAL_PROTOCOL.md`; supported by
  Underwood's finding that humans can still tell fine-tuned output from authentic text,
  https://arxiv.org/abs/2505.00030. **Action:** run the 10-vs-10 notebook sorting test once
  primary text is available.
- **Claim.** Human-likeness metrics should be distributional, not per-sample (burstiness is
  unusable per item). **Evidence:** HANSEN, https://aclanthology.org/2023.findings-emnlp.916.
  **Action:** compute length/opener/burstiness statistics over many turns, never as a per-example gate.
- **Claim.** MAUDE and a small style classifier ("Polite Score" pattern) are the verified
  reference-free substitutes; CHARM could not be located. **Evidence:**
  https://aclanthology.org/2020.acl-main.220/ ; https://aclanthology.org/2023.eacl-srw.9/.
  **Action:** train a small Camus/anti-Camus style classifier and report accuracy per epoch.

### Translation register

- **Claim.** The English corpus blends translators who are measurably different speakers; Gilbert
  cut sentence count ~31% and added words. **Evidence:** Kaplansky,
  https://journals.openedition.org/palimpsestes/1583?lang=en ; Rose,
  https://www.lrb.co.uk/the-paper/v42/n09/jacqueline-rose/pointing-the-finger. **Action:** tag
  every chunk with translator/work/genre; measure sentence count per chunk by translator.
- **Claim.** The incipit exists in at least four different English forms across translators.
  **Evidence:** Munos, https://books.openedition.org/pulm/12555?lang=en ; Bloom `[U]`,
  https://www.newyorker.com/books/page-turner/lost-in-translation-what-the-first-line-of-the-stranger-should-be.
  **Action:** decide and document which rendering is canonical for training; never average them.
- **Claim.** Translators reliably inflate Camus's flatness into metaphor; the flatness is
  authorial and was revised toward fewer figures. **Evidence:** Marris `[U]`,
  https://www.theparisreview.org/blog/2022/09/28/deep-emotion-plain-speech-camuss-plague ;
  Flinta, https://eprints.whiterose.ac.uk/id/eprint/236811/1/Flinta_Estranged_From_Himself_Camus_Etranger.pdf.
  **Action:** add a "figurative language inflation" check against the source style target.
- **Claim.** The notebooks were never one book: three volumes, three translators, plus a 1970
  selection. **Evidence:** Nobel bibliography,
  https://www.nobelprize.org/prizes/literature/1957/camus/bibliography. **Action:** treat mined
  notebook passages as a per-translator dialect slice, not a single Camus voice.
- **Claim.** A per-translator model can separate translator identity from author signal.
  **Evidence:** ALMs, https://arxiv.org/abs/2401.12005 ; cross-language attribution,
  https://aclanthology.org/L14-1167/. **Action:** fine-tune one small LM per translator and score
  generations by register.
- **Claim.** Translationese is detectable via part-of-speech perplexity. **Evidence:** Bizzoni et
  al., https://aclanthology.org/2020.iwslt-1.34/. **Action:** use a PoS-perplexity scorer as a
  corpus filter and on RAG hits.
- **Claim.** Different translations are complementary, not a defect to harmonise away. **Evidence:**
  Benjamin `[U]`, general theory,
  https://www.konstfack.se/PageFiles/46686/Walter%20Benjamin%20-%20The%20task%20of%20the%20Translator.pdf.
  **Action:** cure the blend by explicit curation (segment, weight, choose), not by averaging.

### Temporal grounding

- **Claim.** The post-1960 failure is a named, benchmarked, unsolved class; prompting does not fix
  it. **Evidence:** TimeChara, https://arxiv.org/abs/2405.18027 ; set-the-clock backward
  alignment 2.8×, https://arxiv.org/abs/2402.16797 ; prompting caution,
  https://arxiv.org/abs/2505.00030. **Action:** keep the temporal gate outside the model; treat
  anachronism as a measured failure rate.
- **Claim.** Boundary *awareness* and boundary *compliance* must be scored separately; compliance
  is the dominant failure. **Evidence:** CHARM, https://arxiv.org/abs/2609.01352. **Action:** adopt
  the two-axis metric and the "re-ask the base model" leakage-vs-suppression trick.
- **Claim.** Refusal loses to narrating around the anachronism. **Evidence:** RoleBreak,
  https://arxiv.org/abs/2409.16727. **Action:** train in-character scene context so the model
  answers around post-1960 topics rather than hard-refusing.
- **Claim.** Response / refusal / attempt should be first-class supervised labels; a calibrated
  confidence gate mitigates hallucination. **Evidence:** RoleMRC,
  https://arxiv.org/html/2502.11387v1 ; RoleFact, https://arxiv.org/abs/2406.17260. **Action:**
  add these labels to the defect rows and calibrate a logprob threshold on held-out data.
- **Claim.** A trained second-pass critic cuts temporal leakage far more than prompting or SFT, and
  a prefix date beats a suffix. **Evidence:** TCFT, https://arxiv.org/abs/2605.14636. **Action:**
  put `1960-01-04` before the question and add a (question, cutoff, draft) verifier.
- **Claim.** Typesetter-/history-model grounding and leakage as a separate eval axis. **Evidence:**
  TypewriterLM, https://arxiv.org/abs/2606.02991 ; ExAnte, https://arxiv.org/abs/2505.19533.
  **Action:** add a leakage-rate metric to the eval; do not rely on a system prompt saying he died.
- **Claim.** Models cannot reliably bind facts to validity intervals; the best hit only 11% of
  facts. **Evidence:** TimeStress, https://arxiv.org/abs/2502.01220. **Action:** attach validity
  intervals to Camus facts in the KB rather than relying on the model.
- **Claim.** Serving precision changes leakage estimates. **Evidence:** HindsightBench,
  https://arxiv.org/abs/2607.18867. **Action:** re-measure leakage at whatever quantisation Ollama
  actually serves.
- **Claim.** Leakage remains even with explicit cutoffs, and a larger model does not inherently
  fix it. **Evidence:** ExAnte, https://arxiv.org/abs/2505.19533. **Action:** treat the refusal
  path as a retrieval/eval responsibility.

---

## 3. Considered and rejected

- **Samplers Ollama does not expose.** `typical_p`/eta (`https://aclanthology.org/2022.findings-emnlp.249.pdf`),
  XTC (`https://llm-samplers.readthedocs.io/en/stable/_modules/llm_samplers/xtc.html` — `[U]`
  attribution, arXiv:2608.22758 not opened), DRY (llama-cpp-python bindings
  `https://github.com/jamepeng/llama-cpp-python` — `[U]`, arXiv:2608.22761 not opened),
  `mirostat*`, `logit_bias`, `banned_strings`: all absent from `modelfile.md` and `types.go`
  (`https://docs.ollama.com/modelfile.md`, `https://raw.githubusercontent.com/ollama/ollama/main/api/types.go`).
  XTC and DRY are theoretically ideal for repeated openers but cannot be run here; the
  approximation is data plus `min_p`+temperature.
- **The Antislop sampler.** Real and peer-reviewed (`https://arxiv.org/abs/2510.15061`) but
  unavailable through Ollama. Its FTPO-trained Gemma-3-12B weights are claimed `[U]`
  (`sam-paech/gemma-3-12b-it-antislop-lora` — cards not opened); its `calculate_over_represented_words.ipynb`
  is reusable, and output post-filter+regenerate is the only in-house substitute.
- **Commercial AI detectors as an optimisation target.** Detectors score predictability, not
  authorship, and have a 61% false-positive rate on non-native-language essays; any detector's
  advantage trends to zero as imitation improves. Evidence:
  https://arxiv.org/abs/2304.02819 ; https://aclanthology.org/2025.findings-acl.194/ ; the
  detection survey, https://aclanthology.org/2025.cl-1.8/. Use only as a one-way false-positive
  regression ratchet.
- **DPO and preference pairs.** Already banned by the repo (`docs/ROADMAP.md` non-goals) and the
  research agrees: DPO collapses lexical diversity (`https://arxiv.org/abs/2510.15061`) and has a
  length bias (`https://arxiv.org/abs/2305.18290`). No new action.
- **Persona prompts to fix facts or sycophancy.** Personas do not reliably improve factual QA
  (`https://arxiv.org/abs/2311.10054`), and no peer-reviewed measurement of persona-prompt effect
  on sycophancy was located (a stated gap in research A).
- **Scaling the model or the KB.** Research B's PersonaGym notes scale does not buy persona
  fidelity (`https://arxiv.org/abs/2407.18416`); the repo's TOP_K≤8 and 8B-serving non-goals are
  consistent.
- **UltraChat-style length-optimised synthetic chat.** Deliberately optimised length/diversity
  statistics; plausible cause of the uniformity defect (`https://arxiv.org/abs/2305.14233`).
  Exclude.
- **Contrastive decoding / steering (CAA, PPLM, GeDi) as persona steering.** CAA machinery is
  useful as an eval probe (`https://arxiv.org/abs/2312.06681`) but steering itself is too invasive
  for production; PPLM needs vocab/logit access Ollama lacks (`https://arxiv.org/abs/1912.02164`).
  The persona-steering use of these is `[U]`/unfetched in research B.

---

## 4. Conflicts with the current plan

1. **The training set's shape is the defect.** The repo generates uniform ~70±12-word
   conversational rows (`docs/corpus_analysis.md`); research says a tight length prior *is* the
   uniform-length symptom and that opener variety is a data-distribution problem. The repo's most
   recent naturalness session tested only a CORE *(prompt)* variant (`core:lean`), which cannot move
   a data-distribution defect — and the session confirmed it did not.
2. **Effort is going to the prompt while the measured cause is the corpus.** `core:lean` was
   rejected and no retrain followed; research ranks the SFT length/opener distribution as change
   #1.
3. **Measurement is confounded by history truncation, not weights.** The 2026-09 status log
   records that with memory off the model sees the last 4 exchanges and loses everything earlier
   silently. Research on multi-turn degradation (`https://arxiv.org/abs/2505.06120`) and long-term
   memory/abstention (`https://xiaowu0162.github.io/long-mem-eval`) says no naturalness A/B can
   separate truncation from weights-level confabulation until this is fixed. The repo marks this
   OPEN; the research makes it a blocker for the A/B.
4. **The corpus blends translators with no translator tags.** Research says segmenting by
   translator is the precondition for Camus-specificity (`https://journals.openedition.org/palimpsestes/1583?lang=en`);
   the repo's SFT rows carry a `type` field but no translator/genre/work provenance.
5. **The anachronism plan is prompt/data policy; research says the reliable gate is external.**
   `docs/ROADMAP.md` Phase 3 plans demonstrations and a stance; ExAnte and CHARM find prompting
   insufficient and the dominant failure is compliance
   (`https://arxiv.org/abs/2505.19533`, https://arxiv.org/abs/2609.01352). The repo plans no
   date-tagged retrieval filter and no second-pass leakage critic.
6. **The LLM judge rewards the wrong thing for this rewrite.** Repo keeps an LLM judge for phases
   other than naturalness; MT-Bench found verbosity bias and EQ-Bench found "slop bias" — a judge
   can reward the long, aphoristic register being removed
   (`https://eqbench.com/creative_writing.html`, arXiv:2306.05685). The repo's retirement of the
   judge for naturalness is aligned; its use for factuality is not affected.
7. **No stylometry in the eval.** Research ranks Burrows' Delta and expressive-punctuation rate as
   the cheapest Camus-likeness instruments; the repo evaluates with LLM-judge probe scores only.
8. **The lived direction of temporal alignment is under-used.** Set the Clock's backward alignment
   (2.8×) is presented as the strongest published support for fine-tuning a model "back to 1960"
   (`https://arxiv.org/abs/2402.16797`), while the repo treats temporal handling as prompt policy
   rather than a training objective.

---

## 5. Unverified

Everything below is tagged UNVERIFIED, partially verified, or NOT FOUND in the raw files. Never
present these as established.

**Research A — sources, venues, numbers, leads**
- UltraChat full author list `[U]` (https://arxiv.org/abs/2305.14233); Curse of Recursion *Nature* DOI/volume `[U]`, guessed DOI resolves elsewhere (https://arxiv.org/abs/2305.17493).
- NOT OPENED / partial: Qi et al. (https://arxiv.org/abs/2310.03693); Jannai et al. "Human or Not?" (arXiv ID from a citing list only); UC Merced replication (https://escholarship.org/content/qt7k75p125/qt7k75p125.pdf, title/authors unconfirmed).
- OSF data release https://osf.io/jk7bw `[U]` (highest-value unfetched item); Computational Turing Test https://arxiv.org/html/2511.04195v1 `[U]`; TuringBench https://aclanthology.org/2021.findings-emnlp.172.pdf `[U]`; Inverse Turing Bench https://arxiv.org/html/2606.21844v2 `[U]`.
- Liang et al. fetch blocked (429), body via search index (https://arxiv.org/abs/2304.02819); Weber-Wulff et al. SNIPPET only (DOI 10.1007/s40979-023-00146-z); Sadasivan et al. SNIPPET only (arXiv:2303.11156).
- OpenAI classifier withdrawal `[U]`, do not cite; Sharma et al. venue `[U]` (arXiv:2310.13548); mechanistic sycophancy authors `[U]` (https://arxiv.org/abs/2508.02087); "How RLHF Amplifies Sycophancy" SNIPPET (https://arxiv.org/html/2602.01002v1).
- "Constitution or Collapse?" arXiv:2504.04918 lead only; 21-focal-words authors `[U]` (https://arxiv.org/abs/2412.11385); LoCoMo `[U]` (DOI 10.18653/v1/2024.acl-long.747).
- Personalisation leads `[U]`: arXiv:2601.11000, arXiv:2601.13722, ACL 2026 pp. 511–529, arXiv:2603.11394, arXiv:2602.01885.
- Antislop model cards `[U]` (`sam-paech/gemma-3-12b-it-antislop`, `-lora`); ExLlamaV2 `banned_strings`, KoboldCpp v1.76, llama-dry-docs.site not opened; XTC attribution `[U]`; DRY attribution `[U]` (§3 URLs).
- EQ-Bench CW v3 release date `[U]`; Slop Score page undated; Longform page SNIPPET (https://eqbench.com/creative_writing_longform.html).
- Appendix A.3 empirical tests: Ollama 0.30 penalty restoration; effective `repeat_penalty` default; mirostat removal; no figure measured against this repo's model.

**Research B — papers, counts, pages, names**
- Neeko counts `[U]` (https://github.com/weiyifan1023/Neeko); RoleKE-Bench counts `[U]` (https://aclanthology.org/2025.emnlp-main.1689).
- NOT FOUND: "Neeko — modular RPG agent", "PersonaBench", TempHalluBench, TimeTravel/GoogleRefresh, John O'Brien, Peter M. Adams, Anthony M. Conti, David Amand; RPBench-Auto no paper; CharActer (ACL 2022) dropped; contrastive decoding for persona steering and TwinVoice not fetched; TimeBench ambiguous; ChronoQA candidates not opened; LLMLagBench arXiv:2511.12116 and TIDE arXiv:2608.08512 `[U]`.
- Authorship Embeddings `[U]`; LM-PAN `[U]`; IA3 `[U]`; FELIX/GeDi/DExperts/StyleDrop/InstantStyle/LoRA-as-style-vector/PART/Valla not fetched; style-aware retrieval no paper; multi-translator conflation lead Mikros (2018) not opened.
- Snippets only: Études françaises 2020 author name not captured (bot challenge); Coquet body scanned images; Fletcher *Encyclopedia of the Essay* blog reproduction; Bloom/Liebling/Marris pieces; Bloom *Complete Notebooks* policy; Lamb LARB; Gay-Crosier ProQuest not opened; Benjamin "Task of the Translator"; QUADERNA.
- Ministral Camus commit history / lifetime downloads unverified (API 113/month); collision-model lifetime counts superseded by monthly figures.
- Character.AI / Talkie AI / "Camus par deux IA" `[U]`; BnF *Fonds Albert Camus* acquisition `[U]` (https://www.bnf.fr/fr/actualites/entree-du-fonds-albert-camus-dans-les-collections-de-la-bibliotheque-nationale-de-france).
