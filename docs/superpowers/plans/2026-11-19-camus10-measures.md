# camus10 Measures Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the three measures camus10 is judged by (tells counter, spot-the-bot jury with a one-off owner calibration, and a run comparison that applies the adoption rule), and run them on the existing camus7/8/9 eval archives.

**Architecture:** Three small stdlib modules in `rag/`, each with a CLI and a `unittest` file beside it, following `rag/discriminator_sheet.py`. They read the eval harness's existing output (`probe_scores.jsonl`), so no new Ollama runs are needed to use them. The jury reaches the free OpenCode models through the existing transport `rag/judge_opencode.py`.

**Tech Stack:** Python 3.12 stdlib (`statistics.correlation(..., method="ranked")`, `re`, `random`, `json`), `unittest`, `rag/judge_opencode.py`.

**Spec:** `docs/superpowers/specs/2026-11-19-camus10-design.md` (section 4). Option A data and the option B test are separate plans, written after this one ships.

## Global Constraints

- Jury models: `beam-free`, `hy4-flash-free`, `longcat-2.5-free` (free OpenCode Go models). No paid model.
- Reference corpora stay local: `data/reference/` is gitignored; no book, KB or interview text is committed.
- Calibration trust threshold: the jury is trusted when its agreement with the owner is at least 0.70.
- Noise band for the voice/accuracy guard: 0.1 on the judge's composite and per-axis means.
- Adoption rule: tells fall or fool rate rises, the guard holds, and the owner's blind A/B prefers the candidate.
- Commit messages: one line, no body, no trailer; push after each commit.
- Tests run with `.venv/bin/python -m unittest <module>` from the repo root.

## Review Focus

- A row whose `answer` is empty or whose scores are a judge error: tells must count it in `n_empty` and leave it out of every rate, not crash or count it as a one-word reply. (Test in Task 1.)
- `turns` holding dicts (`{"role", "content"}`) instead of strings, as some harness rows do: prompt extraction must take the last user turn either way. (Test in Task 1.)
- A jury model answering without a JSON object, or the gateway returning 429: the bad answer is recorded as an abstain and left out of the fool rate; 429 stops the run (`UsageCapReached` propagates). (Test in Task 3.)
- A register with no reference rows: pairing raises `ValueError` naming the register, rather than silently pairing small talk against interview speech. (Test in Task 2.)
- The owner's answers file missing a number, repeating one, or using `machine`/`m`/`a`/`b` spellings: parsing accepts the spellings and lists every missing or duplicate number. (Test in Task 4.)

---

### Task 1: Tells counter

**Files:**
- Create: `rag/tells.py`
- Create: `rag/test_tells.py`

**Interfaces:**
- Produces:
  - `last_user(turns: list) -> str`: the last user message; turns are strings (user turns only) or `{"role","content"}` dicts.
  - `reply_tells(reply: str) -> set[str]`: subset of `{"question_back", "service", "button_close", "third_person", "stage_direction", "list"}`.
  - `report(rows: list[dict]) -> dict` with keys `n`, `n_empty`, `rates` (one float per tell name above), `terse_rate`, `bloat_rate`, `length_corr`, `length_cv`, `top3_opener_share`, `median_words`.
  - CLI: `python rag/tells.py FILE [FILE ...]` prints one JSON object per file.

Definitions (pinned here; thresholds come from `docs/HUMAN_EVAL_PROTOCOL.md`'s tells and the camus8/9 failures):
- `question_back`: the reply's last sentence ends with `?`.
- `service`: case-insensitive substring of any of `"how can i help"`, `"i'm listening"`, `"i am listening"`, `"i'm here"`, `"i am here"`, `"anything else"`, `"let me know"`, `"happy to help"`, `"what's on your mind"`, `"what brings you"`, `"tell me plainly what"`, `"ask me something real"`, `"come back with something else"`.
- `button_close`: three or more sentences and the last has five words or fewer.
- `third_person`: regex `\bCamus (would|thought|said|wrote|believed|felt)\b`.
- `stage_direction`: regex `\((?:he |she |i )?(pauses|smiles|laughs|sighs|shrugs|nods)[^)]*\)` or `\*[^*\n]+\*`, case-insensitive.
- `list`: any line starting with `- `, `* ` or `\d+\. `.
- small talk: last user message of four words or fewer. Non-trivial: six words or more, or containing `?`, and not small talk.
- `terse_rate`: share of non-trivial messages whose reply is three words or fewer.
- `bloat_rate`: share of small-talk messages whose reply is over 40 words.
- `length_corr`: ranked correlation between user-message word count and reply word count; `None` when fewer than 3 rows.
- `length_cv`: stdev / mean of reply word counts.
- `top3_opener_share`: share of replies whose first word (lowercased, punctuation stripped) is one of the three most common first words.

- [ ] **Step 1: Write the failing tests** in `rag/test_tells.py`:

```python
class TellsTests(unittest.TestCase):
    def test_question_back_and_service(self):
        self.assertEqual(tells.reply_tells("Hello. What's on your mind?"), {"question_back", "service"})
    def test_button_close_needs_three_sentences(self):
        self.assertIn("button_close", tells.reply_tells("I swam. The sea was cold. It was enough. Good."))
        self.assertNotIn("button_close", tells.reply_tells("I swam. Good."))
    def test_clean_reply_has_no_tells(self):
        self.assertEqual(tells.reply_tells("Not mine. Someone else wrote that line."), set())
    def test_last_user_takes_strings_or_dicts(self):
        self.assertEqual(tells.last_user(["hey", "how are you"]), "how are you")
        self.assertEqual(tells.last_user([{"role": "user", "content": "a"},
                                          {"role": "assistant", "content": "b"},
                                          {"role": "user", "content": "c"}]), "c")
    def test_empty_answers_are_counted_not_rated(self):
        rows = [{"turns": ["hey"], "answer": ""}, {"turns": ["hey"], "answer": "Hello."}]
        r = tells.report(rows)
        self.assertEqual((r["n"], r["n_empty"]), (2, 1))
        self.assertEqual(r["rates"]["question_back"], 0.0)
    def test_terse_and_bloat(self):
        rows = [{"turns": ["what did you think of the war in algeria then?"], "answer": "Hard."},
                {"turns": ["hi"], "answer": " ".join(["word"] * 41)}]
        r = tells.report(rows)
        self.assertEqual((r["terse_rate"], r["bloat_rate"]), (1.0, 1.0))
```

- [ ] **Step 2: Run** `.venv/bin/python -m unittest rag.test_tells`. Expected: FAIL, `No module named 'tells'` / import error.
- [ ] **Step 3: Implement** the interfaces above in `rag/tells.py` (stdlib only; sentence split on `(?<=[.!?])\s+`).
- [ ] **Step 4: Run** `.venv/bin/python -m unittest rag.test_tells`. Expected: OK.
- [ ] **Step 5: Run the CLI** on `data/drafts/v8/sweep/run_camus8/probe_scores.jsonl` and `.../run_camus9/probe_scores.jsonl`. Expected: camus9's `terse_rate` below camus8's and its `median_words` about 35 vs 17 (matches the 2026-11-19 eval).
- [ ] **Step 6: Commit** `git commit -m "Add a tells counter for eval outputs"` and push.

### Task 2: Reference rows and pairing

**Files:**
- Create: `rag/spotbot.py`
- Create: `rag/test_spotbot.py`
- Modify: `.gitignore` (add `data/reference/`)

**Interfaces:**
- Consumes: `tells.last_user(turns)` from Task 1.
- Produces:
  - `register_of(message: str) -> str`: `"text"` if the message is small talk (Task 1's definition), else `"call"`.
  - `load_refs(path: str) -> list[dict]`: JSONL rows `{"prompt": str, "reply": str, "register": "text"|"call", "source": str}`; raises `ValueError` naming the line on a missing field or unknown register.
  - `make_pairs(model_rows: list[dict], refs: list[dict], n: int, seed: int) -> list[dict]`: samples `n` model rows (non-empty answers), matches each with a same-register ref by `random.Random(seed)`, and emits two pairs per match (both orders). Pair: `{"pid": str, "register": str, "prompt_a": str, "a": str, "prompt_b": str, "b": str, "machine": "a"|"b"}`, `pid` = `f"{row_id}:{order}"`, order `0` puts the model reply in `a`. Each side keeps its own message (amended during build: the human reply answered someone else's message, so a shared message would make the jury judge fit, not humanness).

- [ ] **Step 1: Write the failing tests:**

```python
    def test_register(self):
        self.assertEqual(spotbot.register_of("hey"), "text")
        self.assertEqual(spotbot.register_of("why does the plague end the way it does?"), "call")
    def test_pairs_come_in_both_orders(self):
        rows = [{"id": "p1", "turns": ["hey"], "answer": "Hello there."}]
        refs = [{"prompt": "yo", "reply": "sup", "register": "text", "source": "x"}]
        pairs = spotbot.make_pairs(rows, refs, n=1, seed=0)
        self.assertEqual(sorted(p["machine"] for p in pairs), ["a", "b"])
        self.assertEqual({p["a"] for p in pairs} | {p["b"] for p in pairs}, {"Hello there.", "sup"})
    def test_missing_register_refs_raise(self):
        rows = [{"id": "p1", "turns": ["what is revolt to you, really?"], "answer": "A refusal."}]
        refs = [{"prompt": "yo", "reply": "sup", "register": "text", "source": "x"}]
        with self.assertRaisesRegex(ValueError, "call"):
            spotbot.make_pairs(rows, refs, n=1, seed=0)
    def test_reference_dir_is_gitignored(self):
        out = subprocess.run(["git", "check-ignore", "-q", "data/reference/x.jsonl"])
        self.assertEqual(out.returncode, 0)
```

- [ ] **Step 2: Run** `.venv/bin/python -m unittest rag.test_spotbot`. Expected: FAIL (import error).
- [ ] **Step 3: Implement** `register_of`, `load_refs`, `make_pairs` in `rag/spotbot.py`; add `data/reference/` to `.gitignore`.
- [ ] **Step 4: Run** the tests. Expected: OK.
- [ ] **Step 5: Commit** `git commit -m "Add spot-the-bot pairing against local human reference rows"` and push.

### Task 3: Jury

**Files:**
- Modify: `rag/spotbot.py`
- Modify: `rag/test_spotbot.py`

**Interfaces:**
- Consumes: `make_pairs` (Task 2); `judge_opencode.judge(model, system, prompt)` and `judge_opencode.first_json_object(text)`.
- Produces:
  - `JURY_MODELS = ("beam-free", "hy4-flash-free", "longcat-2.5-free")`
  - `JURY_SYS` (exact text below).
  - `ask(model: str, pair: dict, judge=jz.judge) -> str`: returns `"a"`, `"b"` or `"abstain"` (no JSON object, or `machine` not a/b, or any exception except `UsageCapReached`).
  - `run_jury(pairs: list[dict], models=JURY_MODELS, judge=jz.judge) -> list[dict]`: one row per (pair, model): `{"pid", "model", "guess", "machine", "register"}`.
  - `fool_rate(results: list[dict]) -> dict`: `{"overall": float, "by_model": {m: float}, "by_register": {r: float}, "abstains": int}`; fool = guess != machine, abstains excluded. 0.5 means indistinguishable.
  - CLI: `python rag/spotbot.py jury --model-file F --refs F --n 40 --seed 1 --out DIR` writes `DIR/jury.jsonl` and `DIR/fool_rate.json` and prints the latter.

`JURY_SYS`:
```
You will see two short exchanges, A and B. In each, someone sent a message and got a reply.
One of the two replies was written by a person; the other was written by an AI imitating a
person. The two exchanges are unrelated. Decide which reply is the AI's. Judge only how human
the reply sounds as speech or chat, not whether it is wise or correct.
Answer with one JSON object and nothing else: {"machine": "A"} or {"machine": "B"}.
```
User prompt: `A\nMESSAGE: {prompt_a}\nREPLY: {a}\n\nB\nMESSAGE: {prompt_b}\nREPLY: {b}`.

- [ ] **Step 1: Write the failing tests** (judge is injected; no network):

```python
    def test_ask_parses_and_abstains(self):
        pair = {"pid": "p:0", "prompt_a": "hey", "a": "x", "prompt_b": "yo", "b": "y", "machine": "a", "register": "text"}
        self.assertEqual(spotbot.ask("m", pair, judge=lambda *a: '{"machine": "B"}'), "b")
        self.assertEqual(spotbot.ask("m", pair, judge=lambda *a: "no idea"), "abstain")
    def test_usage_cap_propagates(self):
        def capped(*a): raise jz.UsageCapReached("429")
        pair = {"pid": "p:0", "prompt_a": "hey", "a": "x", "prompt_b": "yo", "b": "y", "machine": "a", "register": "text"}
        with self.assertRaises(jz.UsageCapReached):
            spotbot.run_jury([pair], models=("m",), judge=capped)
    def test_fool_rate_excludes_abstains(self):
        res = [{"pid": "1", "model": "m", "guess": "a", "machine": "a", "register": "text"},
               {"pid": "2", "model": "m", "guess": "a", "machine": "b", "register": "text"},
               {"pid": "3", "model": "m", "guess": "abstain", "machine": "b", "register": "text"}]
        fr = spotbot.fool_rate(res)
        self.assertEqual((fr["overall"], fr["abstains"]), (0.5, 1))
```

- [ ] **Step 2: Run** the tests. Expected: FAIL (`ask` not defined). Confirm `jz.UsageCapReached` exists in `rag/judge_opencode.py`; if its name differs, use that name in both test and code.
- [ ] **Step 3: Implement** `ask`, `run_jury`, `fool_rate` and the `jury` CLI subcommand.
- [ ] **Step 4: Run** the tests. Expected: OK.
- [ ] **Step 5: Smoke-test** each jury model on one pair from a two-row hand-made refs file in the scratchpad. Expected: three non-abstain guesses. If a model id is rejected by the gateway, stop and report it; do not substitute a paid model.
- [ ] **Step 6: Commit** `git commit -m "Add the free-model spot-the-bot jury"` and push.

### Task 4: Owner calibration sheet

**Files:**
- Modify: `rag/spotbot.py`
- Modify: `rag/test_spotbot.py`

**Interfaces:**
- Consumes: pairs (Task 2), jury results (Task 3).
- Produces:
  - `write_sheet(pairs: list[dict], sheet_path: str, key_path: str) -> None`: uses only order-0 pairs, renumbers them 1..k, writes a markdown sheet (per number: A's message and reply, B's message and reply; no source) and a key JSON `{number: pid}`.
  - `parse_owner(text: str, expected: int) -> dict[int, str]`: lines `"<number> <a|b|A|B>"`, blank lines ignored; raises `ValueError` listing every missing number (1..expected) and every duplicate.
  - `agreement(key: dict, owner: dict[int, str], results: list[dict]) -> dict`: per jury model, share of numbered pairs where the model's guess equals the owner's; plus `"owner_correct"` (owner vs truth) and `"trusted": bool` = mean model agreement >= 0.70.
  - CLI: `spotbot.py sheet --model-file F --refs F --n 40 --seed 1 --out DIR` (writes `sheet.md`, `key.json`, and runs the jury on the same pairs into `jury.jsonl`); `spotbot.py calibrate --dir DIR --answers FILE` prints `agreement`.

- [ ] **Step 1: Write the failing tests:**

```python
    def test_parse_owner_lists_missing_and_duplicates(self):
        with self.assertRaisesRegex(ValueError, r"missing \[2\].*duplicate \[1\]"):
            spotbot.parse_owner("1 a\n1 b\n3 B\n", expected=3)
    def test_agreement_and_trust(self):
        key = {1: "p:0", 2: "q:0"}
        owner = {1: "a", 2: "b"}
        res = [{"pid": "p:0", "model": "m", "guess": "a", "machine": "a", "register": "text"},
               {"pid": "q:0", "model": "m", "guess": "b", "machine": "a", "register": "text"}]
        ag = spotbot.agreement(key, owner, res)
        self.assertEqual((ag["m"], ag["owner_correct"], ag["trusted"]), (1.0, 0.5, True))
```

- [ ] **Step 2: Run** the tests. Expected: FAIL.
- [ ] **Step 3: Implement** the interfaces above.
- [ ] **Step 4: Run** the tests. Expected: OK.
- [ ] **Step 5: Commit** `git commit -m "Add the owner calibration sheet for the jury"` and push.

### Task 5: Run comparison and adoption rule

**Files:**
- Create: `rag/compare_runs.py`
- Create: `rag/test_compare_runs.py`

**Interfaces:**
- Consumes: `tells.report` (Task 1); `fool_rate.json` (Task 3) when present.
- Produces:
  - `judge_means(paths: list[str]) -> dict`: mean `voice`, `factuality`, `engagement`, `composite` over all scored rows in the given `probe_scores.jsonl` files (rows without all three scores skipped).
  - `verdict(base: dict, cand: dict, noise: float = 0.1) -> dict`: inputs are `{"judge": judge_means(...), "tells": report(...), "fool": float | None}`; returns `{"guard_ok": bool, "humanlike_better": bool, "reasons": list[str]}`. `guard_ok` is False if any of voice, factuality, composite falls by more than `noise`. `humanlike_better` is True if the mean of the six tell rates falls, or `fool` rises by at least 0.05 when both runs have one.
  - CLI: `python rag/compare_runs.py --base F [F ...] --cand F [F ...] [--base-fool F --cand-fool F]` prints a table and the verdict, ending with the line `owner's blind A/B decides adoption`.

- [ ] **Step 1: Write the failing tests:**

```python
    def test_guard_catches_factuality_drop(self):
        base = {"judge": {"voice": 4.5, "factuality": 4.8, "engagement": 4.4, "composite": 4.57}, "tells": T(0.2), "fool": None}
        cand = {"judge": {"voice": 4.5, "factuality": 4.4, "engagement": 4.4, "composite": 4.43}, "tells": T(0.1), "fool": None}
        v = compare_runs.verdict(base, cand)
        self.assertFalse(v["guard_ok"]); self.assertTrue(v["humanlike_better"])
    def test_within_noise_passes(self):
        base = {"judge": {"voice": 4.5, "factuality": 4.8, "engagement": 4.4, "composite": 4.57}, "tells": T(0.2), "fool": 0.30}
        cand = {"judge": {"voice": 4.45, "factuality": 4.75, "engagement": 4.4, "composite": 4.53}, "tells": T(0.2), "fool": 0.36}
        v = compare_runs.verdict(base, cand)
        self.assertTrue(v["guard_ok"]); self.assertTrue(v["humanlike_better"])
```
with helper `T(x)` returning a tells report whose six rates all equal `x`.

- [ ] **Step 2: Run** `.venv/bin/python -m unittest rag.test_compare_runs`. Expected: FAIL.
- [ ] **Step 3: Implement** the interfaces above.
- [ ] **Step 4: Run** the tests. Expected: OK.
- [ ] **Step 5: Run** the CLI with camus8 as base and camus9 as candidate on the 2026-11-19 archives (`eval/archive/v8_camus8_2026-11-19/*/probe_scores.jsonl` plus the sweep, against `v9_camus9_*`). Expected: `guard_ok` False (factuality fell 0.3+), matching the rejection.
- [ ] **Step 6: Commit** `git commit -m "Add run comparison with the camus10 adoption rule"` and push.

### Task 6: Reference corpora

**Files:**
- Create (local, gitignored): `data/reference/text_register.jsonl`, `data/reference/call_register.jsonl`, `data/reference/SOURCES.md`

This task is data, done by the lead with web research through a free model (`mimo run -m opencode-go/hy4-flash-free`), not code.

- [ ] **Step 1: Text register.** Find a public corpus of real people chatting (not scripted, not crowdworkers role-playing a persona) whose licence allows local research use. Record name, URL, licence and date in `SOURCES.md`. Extract at least 80 single exchanges where the first message is four words or fewer, into `{"prompt","reply","register":"text","source"}` rows.
- [ ] **Step 2: Call register.** Collect transcripts of Camus's own spoken interviews (radio, press conferences, recorded conversations; translated where needed, translator noted). Extract at least 60 question-and-answer exchanges with a question of six words or more, as `register: "call"` rows. If fewer than 60 exist, record the count in `SOURCES.md` and stop at what exists.
- [ ] **Step 3: Validate** with `.venv/bin/python -c "import sys; sys.path.insert(0,'rag'); import spotbot; print(len(spotbot.load_refs('data/reference/text_register.jsonl')), len(spotbot.load_refs('data/reference/call_register.jsonl')))"`. Expected: two counts meeting the floors above; `git status --short data/reference` prints nothing.
- [ ] **Step 4: No commit** (nothing tracked changes). Report the sources and counts to the owner.

### Task 7: Baseline readings and calibration

**Files:** none tracked; outputs under `eval/archive/measures_2026-11-19/` (gitignored with `eval/archive`).

- [ ] **Step 1: Tells** on camus7, camus8 and camus9 sweep outputs (`data/drafts/v8/sweep/run_camus{7,8,9}/probe_scores.jsonl`). Record the three reports.
- [ ] **Step 2: Jury** on the same three files: `spotbot.py jury --refs` (both reference files concatenated into `data/reference/all.jsonl`) `--n 40 --seed 1`. Record the three fool rates.
- [ ] **Step 3: Calibration sheet** from camus8: `spotbot.py sheet --model-file data/drafts/v8/sweep/run_camus8/probe_scores.jsonl --refs data/reference/all.jsonl --n 40 --seed 2 --out eval/archive/measures_2026-11-19/calibration`. Hand `sheet.md` to the owner; do not open `key.json`.
- [ ] **Step 4: Score** the owner's answers with `spotbot.py calibrate`. Report agreement per jury model and `trusted`. If not trusted, the jury is dropped per the spec (section 4.3), and camus10 is judged by tells, the guard and the owner's A/B only.
- [ ] **Step 5: Voice check.** Run the existing "him or the machine" sheet once on camus8 (`rag/discriminator_sheet.py`, 10 replies against 10 book passages) so camus10 has a voice baseline beside the judge scores.
- [ ] **Step 6: Record** the readings in `docs/ROADMAP.md` (local) and the vault note `~/Notes/notes/CamusGPT.md` (append only).
