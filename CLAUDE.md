# CamusGPT — rules for agents working on this repo

A Camus persona LLM: Gemma 4 12B fine-tuned in two LoRA passes (voice, then behaviour), served by
Ollama with retrieval over a knowledge base. Training runs on Colab; inference and eval run
locally. Read `docs/ROADMAP.md` for status.

## Hard rules

- **Public repo.** Never commit copyrighted book text. A quotation of Camus is allowed only when
  it is short (under 15 words) and rare. Mined book passages live in gitignored files and in
  `build/` — never move them into tracked files.
- **No machine-specific details** in files or commits: no absolute home paths, usernames,
  device names or OS-specific setup notes.
- **Commit messages: one line, imperative, no body, no `Co-Authored-By` trailer.**
- **Work on a branch named `cloud/<task>` and open a pull request. Never merge, never push to
  `main`.** A human reviews every PR locally, where the model and the knowledge base exist.
- Keep diffs small and in the surrounding code's style. No new dependencies unless a brief
  says so.

## What is NOT available in a cloud sandbox

- No GPU, no Ollama, no fine-tuned model: you cannot generate or score model answers.
- Gitignored data is absent: the knowledge base (`camus_kb_full.jsonl`, `camus_kb_vectors.npy`),
  `data/camus_sft.jsonl`, `data/drafts/`, `data/primary_*`, `build/`.
- Tests must therefore pass with no network, no Ollama and no knowledge base. Use fixtures and
  mocks. Install only `numpy requests rank-bm25` for the tests (`sentence-transformers` is not
  needed for any unit test).

Run the tests with:

```
python -m unittest discover -s pipeline -p "test_*.py"
python -m unittest discover -s rag -p "test_*.py"
python -m unittest discover -s training -p "test_v3_notebooks.py"
```

## Persona and data conventions

- Single-turn rows: `{"prompt": ..., "response": ..., "kind": ...}`. Multi-turn rows:
  `{"messages": [{"role": "user"|"assistant", "content": ...}, ...], "kind": ...}`.
- He speaks in the first person as a man of his time (1913–1960). Never "Camus would say",
  never stage directions in parentheses, never lists or headers, never service tone
  ("Great question", "I'd be happy to help"), never summarising the user back.
- Length varies like real speech: many replies are a few words, some are a paragraph.
  Few replies end by handing a question back, and only a genuine one.
- Facts about his life must match `docs/` and the identity card in `rag/camus_rag.py`
  (`_IDENTITY_CARD`). When unsure of a fact, leave it out rather than invent it.
