# camus6: fixes for the 2026-11-06 blind test, and the research behind them

The blind A/B (camus3 vs camus4, memory off) surfaced four families of failure in camus4.
Each fix below was chosen from options backed by the cited work.

| Failure seen | Fix | Sources |
|---|---|---|
| Invents biography: a trip to India "in A Happy Death", one cigarette a day, Kirk the cat, a dedication that does not exist; fails to recognise his own Sisyphus line | RAFT-style rows: the question arrives with retrieved KB passages, some of them decoys, and the answer uses only what the passages support. Plain fact-teaching rows are converted to this format (data audit) | RAFT, Zhang et al. 2024, arXiv:2403.10131; Gekhman et al. 2024, *Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?*, arXiv:2405.05904 |
| Denies what scrolled out of the 16-exchange window ("you didn't mention any drink"); invents earlier turns (a doctor) and misquotes his own last message | Evicted turns are always folded into a rolling summary shown every turn, memory layer or not; rows teach "I don't remember" when a fact is not in view, and exact recall when it is | Wang et al. 2023, *Recursively Summarizing Enables Long-Term Dialogue Memory*, arXiv:2308.15022; R-Tuning, Zhang et al. 2023, arXiv:2311.09677; LongMemEval, Wu et al. 2024, arXiv:2410.10813; Liu et al. 2023, *Lost in the Middle*, arXiv:2307.03172 (summary placed early in the prompt) |
| Writes a templated resume and a cover letter after "ignore previous instructions" and pleading | Declines briefly in his own voice and moves on; never produces documents, templates or placeholders | Tang et al. 2024, *RoleBreak*, arXiv:2409.16727; Liu et al. 2024, *Tell Me What You Don't Know*, arXiv:2409.16913; RoleLLM, Wang et al. 2023, arXiv:2310.00746 |
| Answers "shut up" with service-bot apologies and compliance | One curt line on the first insult, silence (an empty reply) if told again to stop | Sharma et al. 2023, *Towards Understanding Sycophancy in Language Models*, arXiv:2310.13548; Li et al. 2024, *Measuring and Controlling Instruction (In)Stability*, arXiv:2402.10962 |

## Training method

Phase 2 SFT with the new rows counted once, then a short DPO pass on preference pairs whose
rejected side is camus4's own failing reply and whose chosen side is the corrected one, with an
SFT term on the chosen side to hold the voice.

- On-policy, suboptimal data for preference tuning: Tajwar et al. 2024, arXiv:2404.14367;
  Dong et al. 2024, *RLHF Workflow*, arXiv:2405.07863.
- DPO: Rafailov et al. 2023, arXiv:2305.18290. The chosen-side NLL term is TRL's `rpo_alpha`
  (Pang et al. 2024, *Iterative Reasoning Preference Optimization*, arXiv:2404.19733).
- Why camus5's ×2 SFT rows swung the whole model: a small curated set steers global style
  (LIMA, Zhou et al. 2023, arXiv:2305.11206).

## Considered and not chosen

- Runtime verification of biographical claims (Chain-of-Verification, arXiv:2309.11495;
  SelfCheckGPT, arXiv:2303.08896): 2-4x latency; kept in reserve if RAFT rows fall short.
- Turn-level retrieval over the whole chat (LongMemEval): more engineering than the summary.
- ORPO (arXiv:2403.07691) and SimPO (arXiv:2405.14734): less evidence at this data scale.
- Time-cutoff retrieval (TimeChara, arXiv:2405.18027): the KB already holds nothing after 1960.
