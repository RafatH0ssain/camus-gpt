# Task 05 — tests for the chat pipeline + CI (branch `cloud/core-tests-ci`)

## Why
The RAG pipeline in `rag/camus_rag.py` has no unit tests, and three silent bugs were found
by accident this month (an eval that never sent `"think": false` and returned empty answers;
a merge that wrote base weights; a masking check that could never pass).

## Deliverables
1. `rag/test_camus_rag.py` covering, without Ollama/network/knowledge base:
   - `is_task`, `is_followup`, `is_self_ref` on clear positive and negative inputs.
   - `build_system(hits, kb_suppressed)`: retrieved facts appear when not suppressed and do
     not when suppressed; the identity card is always present; CORE variants differ only in
     prose and share a byte-identical `_IDENTITY_CARD`.
   - `space/app.py`'s identity card equals `rag/camus_rag.py`'s (the two must stay in sync).
   - Every `/api/chat` payload sent by `stream_chat`, `util_chat`, `rag/eval_camus.py`
     (`ask`), `rag/eval_memory.py`, `rag/camus_client.py` and `pipeline/temp_sweep.py`
     includes `"think": False` — patch `requests.post` and inspect the json.
   - `recent_user` and the history window (`HIST_WINDOW`) trimming.
   If importing `camus_rag` pulls in `sentence-transformers`, stub that module in
   `sys.modules` before import; do not change production code just to make it testable unless
   the change is tiny and obviously safe.
2. `.github/workflows/tests.yml`: on push and pull_request, Python 3.12, install
   `numpy requests rank-bm25`, run the three unittest commands in `CLAUDE.md`.
3. Fix any test that fails in CI only because it needs gitignored data — skip it with a clear
   reason when the file is absent, rather than deleting it.

## Done when
All suites pass locally in the sandbox and the workflow file is valid YAML. List any real bug
the tests exposed in the PR description instead of silently fixing behaviour.
