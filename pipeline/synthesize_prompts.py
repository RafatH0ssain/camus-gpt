#!/usr/bin/env python3
"""
synthesize_prompts.py — turn mined primary passages into SFT rows by synthesising the
user side of the conversation.

Reads data/primary_candidates.jsonl (rows {source, chunk_id, passage, words}) and
writes data/primary_rows.jsonl:

    {"prompt": <synthesised message>, "response": <passage verbatim>,
     "kind": "primary", "source": ..., "src_id": <chunk_id>}

The passages are his own writing, but they were published as monologues: nobody
ever wrote to him about them. A row whose prompt is a fabricated question is a
row that teaches the model to answer a question nobody asked, so the prompt is
synthesised by asking a model, in batches, for the message the passage would
plausibly be a reply to.

Selection (the counts print under --dry-run):
  * dropped: reported speech (opens on a quotation mark), plan notes (opens on a
    number then ")" or "."), "cf." cross-references, page references;
  * notebook sources ("Notebooks ..."): 3-80 words, up to 1500 in total across
    the three volumes, sampled uniformly at random (seed 13) — shortest-first
    would quietly hand back a corpus of two-word fragments;
  * every other source: 20-120 words, at most 150 per source, random sample
    (seed 13). Every pool is sorted by passage text and holds one row per
    passage, so the seed picks the same passages from the same passages whatever
    order the candidate file is in and whatever extra fields its rows carry: a
    resumed run and a full run agree on what the file should contain.
  * the caps are counted against the output file, not against the candidates:
    a run that already spent part of a source's budget gets only the remainder,
    so no number of reruns can put more than 150 rows of a source in the file.

Synthesis: the instruction is one system message and a batch of 20 passages goes
out per call, through rag/judge_opencode.py (key loading, headers, transport and
the 429 abort all come from there). The model is the opencode-go provider's
space-bunny-free — jz.DEFAULT_MODEL — because the Zen gateway spells it without
the "opencode-go/" provider prefix and answers HTTP 400 "Model is unavailable" to
the prefixed name.

Each finished batch is appended and flushed before the next call, so a run that
dies halfway leaves a usable file, and a rerun skips the passages already written
rather than paying for them twice. The resume key is the passage text itself,
lowercased with its whitespace collapsed. A chunk id is the wrong key for that: it
names a chunk of many passages and is rewritten whenever the candidate file is
re-mined, so a resumed run re-sent passages the file already held. The file this
was measured on held 3394 rows and 2473 distinct passages, every repeated passage
byte-identical and filed under the same chunk id, which is the signature of a
second run working from a stale snapshot rather than of a different passage.

Failure handling, in the order it matters:
  * HTTP 429 or a usage-limit error: the run stops at once, prints how far it got
    and exits 0. The gateway caps a window, not a run, and a retried batch would
    only earn the same 429 — so nothing is retried, and the partial file stands.
  * any other error: the batch is retried once, then skipped and its ids logged,
    so one bad batch cannot cost the other hundred.

    python pipeline/synthesize_prompts.py --dry-run
    python pipeline/synthesize_prompts.py --limit 40
    python pipeline/synthesize_prompts.py
"""
import argparse
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "rag"))

import judge_opencode as jz  # noqa: E402  (key loading, headers, transport, 429 abort)

MODEL = jz.DEFAULT_MODEL   # the same model the opencode-go provider calls space-bunny-free
BATCH = 20
SEED = 13

NOTEBOOK_PREFIX = "Notebooks"
NOTEBOOK_MIN_WORDS, NOTEBOOK_MAX_WORDS = 3, 80
NOTEBOOK_TOTAL = 1500
OTHER_MIN_WORDS, OTHER_MAX_WORDS = 20, 120
OTHER_PER_SOURCE = 150
NOTEBOOK_BUDGET = "\0notebooks"   # the three notebook volumes share one budget

MAX_PROMPT_WORDS = 25
LEAK_NGRAM = 5
MAX_QUESTIONS_PER_BATCH = 5

DROP_RULES = (
    ("reported speech (opens on a quotation mark)",
     re.compile("^[\u201c\u201d\u201e\u201f\"\u00ab\u00bb]")),
    ("plan notes (opens on a number then ) or .)",
     re.compile(r"^\d+[).]")),
    ("cross-references (contains cf.)",
     re.compile(r"\bcf\.", re.I)),
    ("page references (p. 12)",
     re.compile(r"\bpp?\.\s*\d")),
)

FORBIDDEN = re.compile(r"\b(camus|books?|notebooks?)\b", re.I)
WORD = re.compile(r"[^\W\d_]+", re.UNICODE)

SYSTEM = (
    "You write the user side of chat transcripts for a model whose only text is a "
    "writer's notebooks and essays. For each passage you are given, write ONE message "
    "that a real person might send in a conversation, chosen so that the passage would "
    "be a natural reply to it.\n"
    "Rules:\n"
    "- exactly one message per passage, carrying the id you were given;\n"
    "- at most 25 words each;\n"
    "- lowercase and fragments are fine; do not tidy them into finished sentences;\n"
    "- the person has not read the passage and does not know it exists, so never "
    "quote or paraphrase its distinctive phrasing, and never hint that it comes "
    "from a text;\n"
    "- never mention books, notebooks, or \"Camus\";\n"
    "- no two messages in the batch may be the same.\n"
    "\n"
    "The one rule you keep breaking: at most 5 of the 20 messages may end in a "
    "question mark. A batch of questions is the worst possible answer, because a "
    "person asking a question expects a reply that addresses it, and these "
    "passages usually do not. Fifteen of the twenty must be statements: remarks, "
    "complaints, disagreements, instructions, orders, asides, one-word answers, "
    "half-finished thoughts, or a mood the person is in. A statement works just as "
    "well as a question for a passage to be a reply to — \"nobody in this town "
    "bothers with that any more\" invites exactly the kind of answer the passage "
    "gives.\n"
    "Reply with ONLY a JSON array of objects, one per passage and in the same order: "
    "[{\"id\": <the id given>, \"prompt\": <the message>}]. No prose, no code fences."
)

CAP_PHRASE = re.compile(r"(usage|rate)\s*[-_ ]?limit", re.I)
CAP_BODY = re.compile(
    r"(too many requests|quota|insufficient (quota|balance|credits)|"
    r"exceeded your current|monthly limit|429)", re.I)


def passage_words(passage):
    return len(str(passage).split())


def row_words(row):
    words = row.get("words")
    if isinstance(words, int) and words > 0:
        return words
    return passage_words(row.get("passage", ""))


def drop_rule(passage):
    """Label of the first rule the passage breaks, else None. Order matters: the
    counts are reported per rule and a passage is counted once."""
    text = str(passage or "")
    for label, pattern in DROP_RULES:
        if pattern.search(text):
            return label
    return None


def is_notebook(source):
    return str(source or "").startswith(NOTEBOOK_PREFIX)


def normalise(text):
    """Passage text as the resume key reads it: lowercase, whitespace collapsed.

    The passage is the row, not the chunk it came from. A chunk id names a chunk
    of many passages and is rewritten whenever the candidate file is re-mined, so
    two files holding the same passage can disagree about it — the text cannot.
    """
    return " ".join(str(text or "").split()).lower()


def budget_key(source):
    """What a source spends against: its own name, or one shared notebook budget,
    because the 1500 is a cap on the notebooks together and not per volume."""
    return NOTEBOOK_BUDGET if is_notebook(source) else str(source or "")


def cap_for(source):
    return NOTEBOOK_TOTAL if is_notebook(source) else OTHER_PER_SOURCE


class Written:
    """What the output file already holds: the passages in it, and how many rows
    each source has there. The caps are counted against this, so a run that is
    picking up where the last one stopped spends only what is left."""

    def __init__(self, rows=()):
        self.texts = set()
        self.notebooks = set()
        self.per_source = {}
        for row in rows:
            text = normalise(row.get("response"))
            if not text:
                continue
            self.texts.add(text)
            if is_notebook(row.get("source")):
                self.notebooks.add(text)
            else:
                self.per_source.setdefault(str(row.get("source", "")), set()).add(text)

    def spent(self, budget):
        """Rows the file holds against one budget, counted once per passage so a
        file that still holds an old duplicate cannot starve a source."""
        if budget == NOTEBOOK_BUDGET:
            return len(self.notebooks)
        return len(self.per_source.get(budget, ()))


def pool(rows):
    """A sampling pool in an order of its own: sorted by passage text, one row per
    passage. Sorting is what makes the sample depend on the passages rather than on
    the order the candidate file happens to be in, and the dedupe is why a passage
    filed under two chunk ids is offered to the model once, not twice."""
    unique = {}
    for row in rows:
        text = row_key(row)
        if text:
            unique.setdefault(text, row)
    return [unique[text] for text in sorted(unique)]


def plan_todo(selected, written):
    """The rows to synthesise now, in selection order, and the budget each source
    spends on them.

    Three things are dropped: a passage the output file already holds (the resume),
    the same passage twice in the selection (one row per passage, whatever the
    candidate file did), and anything past a cap — where a cap counts the rows the
    file already has, so the plan is a statement about the finished corpus rather
    than about one run of it."""
    used = {}
    todo, seen = [], set()
    for row in selected:
        text = row_key(row)
        budget = budget_key(row.get("source"))
        if not text or text in seen or text in written.texts:
            continue
        if used.get(budget, 0) + written.spent(budget) >= cap_for(row.get("source")):
            continue
        used[budget] = used.get(budget, 0) + 1
        seen.add(text)
        todo.append(row)
    return todo, used


def select_rows(rows, seed=SEED, written=None):
    """The passages to synthesise prompts for, and the counts behind that choice.

    Sampling is one random.Random(seed) used in a fixed order — notebooks first,
    then the other sources by name — over pools that are sorted by passage text and
    hold one row per passage. So the seed and the passages decide the sample, and
    not the order or the extra fields of the candidate file: a rerun picks the same
    passages as the run before it.

    `written` is the output file's state; the caps are counted against it, so a
    resumed run takes only the rows the file has room for.
    """
    written = written or Written()
    dropped = Counter()
    kept = []
    for row in rows:
        label = drop_rule(row.get("passage"))
        if label is None:
            kept.append(row)
        else:
            dropped[label] += 1

    notebook_pool = pool([row for row in kept
                          if is_notebook(row.get("source"))
                          and NOTEBOOK_MIN_WORDS <= row_words(row) <= NOTEBOOK_MAX_WORDS])
    per_source = {}
    for row in kept:
        if not is_notebook(row.get("source")) \
                and OTHER_MIN_WORDS <= row_words(row) <= OTHER_MAX_WORDS:
            per_source.setdefault(str(row.get("source", "")), []).append(row)
    per_source = {source: pool(entries)
                  for source, entries in sorted(per_source.items())}

    rng = random.Random(seed)
    sampled = rng.sample(notebook_pool, min(NOTEBOOK_TOTAL, len(notebook_pool)))
    for source in sorted(per_source):
        candidates = per_source[source]
        sampled.extend(rng.sample(candidates, min(OTHER_PER_SOURCE, len(candidates))))

    selected, used = plan_todo(sampled, written)
    stats = {
        "input": len(rows),
        "dropped": dropped,
        "notebook_pool": len(notebook_pool),
        "notebook_kept": used.get(NOTEBOOK_BUDGET, 0),
        "already": len(written.texts),
        "other_sources": {source: {"eligible": len(candidates),
                                   "already": written.spent(source),
                                   "kept": used.get(source, 0)}
                          for source, candidates in per_source.items()},
    }
    return selected, stats


def print_selection(stats, selected, out_path=None):
    print(f"input passages {stats['input']}")
    print("removed (first rule that matches, so the counts add up):")
    for label, _ in DROP_RULES:
        print(f"  {stats['dropped'][label]:>6}  {label}")
    print(f"  {sum(stats['dropped'].values()):>6}  total removed")
    print(f"  {stats['input'] - sum(stats['dropped'].values()):>6}  left after the drop rules")
    print(f"\nnotebook passages {NOTEBOOK_MIN_WORDS}-{NOTEBOOK_MAX_WORDS} words: "
          f"{stats['notebook_pool']} eligible, {stats['notebook_kept']} to go out "
          f"(cap {NOTEBOOK_TOTAL}, uniform sample, seed {SEED})")
    print(f"other sources {OTHER_MIN_WORDS}-{OTHER_MAX_WORDS} words, "
          f"max {OTHER_PER_SOURCE} each:")
    for source, counts in stats["other_sources"].items():
        print(f"  {counts['kept']:>6} / {counts['eligible']:>6}  {source}"
              f"  ({counts['already']} already written)")
    print(f"\n{stats['already']} distinct passages already in "
          f"{out_path or 'the output file'}; the caps above are counted with them")
    print(f"selected {len(selected)} passages")


def batch_payload(batch):
    """The user message for one batch: the passages, each with the id the reply
    must carry. Batch-local ids, so the model has a short number to echo back."""
    return json.dumps([{"id": index, "passage": str(row.get("passage", ""))}
                       for index, row in enumerate(batch)], ensure_ascii=False)


def call_model(system, user, key=None):
    """One chat/completions call through the shared Zen transport.

    jz.judge is the whole transport: key, headers, timeout, token budget, the
    one retry it makes of an empty or truncated reply, and the 429 abort. A reply
    that arrives as a JSON array still carries JSON objects, which is what
    jz.judge checks a reply for, so nothing here has to re-validate it.
    """
    return jz.judge(MODEL, system, user, key=key)


def _json_candidates(text):
    stripped = jz._strip_fences(str(text or ""))
    candidates = [stripped]
    start, end = stripped.find("["), stripped.rfind("]")
    if 0 <= start < end:
        candidates.append(stripped[start:end + 1])
    return candidates


def parse_reply(text):
    """The model's answer as a list of {"id", "prompt"} dicts.

    A fenced array, an array wrapped in prose, and a single object that holds one
    all parse; a reply with none of them raises ValueError, which the caller
    treats as a failed batch like any other transport error.
    """
    for candidate in _json_candidates(text):
        try:
            value = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        if isinstance(value, dict):
            for key in ("prompts", "batch", "results", "messages"):
                inner = value.get(key)
                if isinstance(inner, list):
                    return [item for item in inner if isinstance(item, dict)]
            if "prompt" in value:
                return [value]
    raise ValueError(f"no JSON array of prompts in reply: {str(text)[:120]}")


def align(batch, items):
    """Pair each passage with its prompt.

    The reply is matched on id. When the ids do not cover the batch — the model
    drops one, or renumbers from 1 — a reply of exactly the batch's length is
    taken in order instead, which is the one case where position is as good a
    key as id. Anything else leaves the unmatched passages without a prompt, and
    they are counted as rejected rather than paired with somebody else's reply.
    """
    by_id = {}
    for item in items:
        if "prompt" in item:
            try:
                by_id[int(item["id"])] = item["prompt"]
            except (KeyError, TypeError, ValueError):
                continue
    if set(by_id) == set(range(len(batch))):
        return [by_id[index] for index in range(len(batch))]
    if len(items) == len(batch):
        return [item.get("prompt") for item in items]
    return [by_id.get(index) for index in range(len(batch))]


def _words(text):
    return [word.lower() for word in WORD.findall(str(text or ""))]


def leaks(prompt, passage):
    """True when the prompt repeats the passage verbatim for LEAK_NGRAM words in a
    row — the one constraint a caller cannot check by reading the prompt alone."""
    grams = {tuple(_words(passage)[i:i + LEAK_NGRAM])
             for i in range(max(0, len(_words(passage)) - LEAK_NGRAM + 1))}
    prompt_words = _words(prompt)
    return any(tuple(prompt_words[i:i + LEAK_NGRAM]) in grams
               for i in range(max(0, len(prompt_words) - LEAK_NGRAM + 1)))


def prompt_fault(prompt, passage, seen):
    """Why a reply's prompt is unusable, else None. Each fault maps to the
    instruction it breaks, so the counts say which rule the model keeps missing.

    `seen` holds every prompt already written, so an exact repeat anywhere in the
    run — not only inside one batch — is a fault: two identical prompts make two
    rows that differ only in the answer."""
    if not isinstance(prompt, str) or not prompt.strip():
        return "empty prompt"
    text = " ".join(prompt.split())
    if len(text.split()) > MAX_PROMPT_WORDS:
        return f"over {MAX_PROMPT_WORDS} words"
    if FORBIDDEN.search(text):
        return "mentions a book, notebook or Camus"
    if leaks(text, passage):
        return "quotes the passage"
    if text.lower() in seen:
        return "repeats a prompt already written"
    return None


def is_usage_limit_error(error):
    """A cap that arrives as an ordinary HTTP error rather than a 429: the gateway
    words it several ways and all of them mean the window is spent."""
    text = str(error)
    return bool(CAP_PHRASE.search(text) or CAP_BODY.search(text))


def load_rows(path):
    """JSONL rows; a line that will not parse is left out."""
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def written_state(path):
    """The output file's state, read once at the start of a run: the passages in it
    and how many rows each source has there. A rerun resumes from these instead of
    paying for the same passages twice."""
    return Written(load_rows(path) if os.path.exists(path) else [])


def row_key(row):
    """What identifies a passage across files and across re-mines: its text."""
    return normalise(row.get("passage"))


def output_row(prompt, row):
    return {"prompt": prompt, "response": str(row.get("passage", "")),
            "kind": "primary", "source": row.get("source"),
            "src_id": row.get("chunk_id")}


def batches(items, size):
    return [items[i:i + size] for i in range(0, len(items), size)]


def synthesise(batch, reply, seen):
    """The rows for one parsed reply, plus the faults counted by reason.

    The question cap is enforced here as well as asked for in the instruction.
    The model reached for a question on four rows in five when only told to vary
    the form, and a corpus where most prompts are questions teaches the model
    that a question always gets answered — which is exactly the behaviour that
    makes it confident and wrong. So a batch that overruns the cap keeps its
    first MAX_QUESTIONS_PER_BATCH questions and has the rest written as
    statements: the surplus rows are dropped and counted, which is the honest
    outcome, rather than silently rewriting the model's own words.
    """
    pairs = align(batch, reply)
    rows, faults = [], Counter()
    asked = 0
    for row, prompt in zip(batch, pairs):
        passage = str(row.get("passage", ""))
        fault = prompt_fault(prompt, passage, seen)
        if fault is None and prompt.rstrip().endswith("?"):
            if asked < MAX_QUESTIONS_PER_BATCH:
                asked += 1
            else:
                fault = "over the batch question quota"
        if fault is not None:
            faults[fault] += 1
            continue
        seen.add(" ".join(prompt.split()).lower())
        rows.append(output_row(prompt, row))
    return rows, faults


def skip_ids(batch):
    """The chunk ids of a skipped batch, for the log line that names what was lost."""
    return [row.get("chunk_id") for row in batch]


def fetch(batch, number, key, call, stats):
    """One batch's reply, or (None, None) when the batch was skipped and logged.

    The two failure kinds are kept apart here, because they call for opposite
    answers. A cap (429, or a usage-limit body on another status) returns a stop
    reason and is never retried: the window is spent, so a second call earns the
    same answer. Every other error is retried once — jz.judge already retries a
    reply it cannot use once, so this is the retry on top of that — and then the
    batch is skipped with its ids logged, so one bad batch cannot end the run.
    """
    for attempt in (1, 2):
        stats["calls"] += 1
        try:
            return call(SYSTEM, batch_payload(batch), key=key), None
        except jz.UsageCapReached as exc:
            return None, str(exc)
        except Exception as exc:  # noqa: BLE001 - retried once, then skipped
            if is_usage_limit_error(exc):
                return None, f"{type(exc).__name__}: {exc}"
            if attempt == 1:
                continue
            stats["skipped"].append(skip_ids(batch))
            print(f"  batch {number}: skipped after one retry — {exc}", file=sys.stderr)
    return None, None


def run(selected, out_path, batch_size=BATCH, key=None, call=call_model):
    """Synthesise every selected passage, appending each finished batch at once.

    The passages the output file already holds are dropped before the first call,
    and the caps are counted against the file, so a run that is killed and started
    again neither pays twice nor pushes a source past its cap.

    Returns (stats, stop_reason). stop_reason is None unless a usage cap ended
    the run, in which case the file on disk is everything written so far.
    """
    todo, _ = plan_todo(selected, written_state(out_path))
    stats = {"selected": len(selected), "todo": len(todo), "calls": 0,
             "batches": 0, "written": 0, "rejected": 0, "skipped": [],
             "faults": Counter()}
    stop_reason = None
    seen = set()

    with open(out_path, "a", encoding="utf-8") as out:
        for number, batch in enumerate(batches(todo, batch_size), 1):
            reply, stop_reason = fetch(batch, number, key, call, stats)
            if stop_reason is not None:
                break
            if reply is None:
                continue
            stats["batches"] += 1
            try:
                items = parse_reply(reply)
            except ValueError as exc:
                stats["skipped"].append(skip_ids(batch))
                print(f"  batch {number}: skipped, reply would not parse — {exc}",
                      file=sys.stderr)
                continue
            rows, faults = synthesise(batch, items, seen)
            for row in rows:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            stats["written"] += len(rows)
            stats["rejected"] += sum(faults.values())
            stats["faults"].update(faults)
            print(f"  batch {number}: {len(batch)} passages, {len(rows)} rows, "
                  f"{stats['calls']} calls, {stats['written']} written")
    return stats, stop_reason


def print_summary(stats, stop_reason, out_path):
    print(f"\nselected {stats['selected']}  to synthesise {stats['todo']}")
    print(f"batches {stats['batches']}  calls {stats['calls']}  "
          f"rows written {stats['written']}  replies rejected {stats['rejected']}")
    for reason, count in sorted(stats["faults"].items()):
        print(f"  {count:>6}  {reason}")
    if stats["skipped"]:
        print(f"  skipped batches {len(stats['skipped'])}: ids {stats['skipped']}")
    print(f"rows -> {out_path}  ({written_keys_count(out_path)} rows in the file)")
    if stop_reason is not None:
        print(f"stopped on a usage cap, not a failure: {stop_reason}")
        print(f"  {stats['todo'] - stats['written'] - stats['rejected']} passages still "
              f"to do; re-run the same command to continue where this stopped.")


def written_keys_count(path):
    """How many rows the output file holds now, written and resumed alike."""
    if not os.path.exists(path):
        return 0
    return len(load_rows(path))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--in", dest="src", default="data/primary_candidates.jsonl")
    ap.add_argument("--out", default="data/primary_rows.jsonl")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the selection counts and stop")
    ap.add_argument("--limit", type=int, default=None,
                    help=f"synthesise prompts for at most N passages")
    ap.add_argument("--batch", type=int, default=BATCH,
                    help=f"passages per call (default {BATCH})")
    args = ap.parse_args()

    selected, stats = select_rows(load_rows(args.src), written=written_state(args.out))
    print_selection(stats, selected, args.out)
    if args.dry_run:
        print(f"\ndry run: {len(selected)} passages would go out in "
              f"{-(-len(selected) // args.batch)} calls of {args.batch}")
        return 0

    if args.limit is not None:
        selected = selected[:args.limit]
    key = jz.load_key()
    run_stats, stop_reason = run(selected, args.out, args.batch, key=key)
    print_summary(run_stats, stop_reason, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())