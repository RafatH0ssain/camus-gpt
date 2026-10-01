#!/usr/bin/env python3
"""
filter_reply_suitability.py — keep the mined rows that can work as conversational replies.

Reads data/primary_rows.jsonl (rows {prompt, response, kind, source, src_id}, the passages
verbatim from the notebooks and essays) and writes:

    data/primary_rows.decisions.jsonl   one decision per distinct response
    data/primary_rows.filtered.jsonl    the kept rows, in the input file's order

Both are under data/primary_*.jsonl, which .gitignore keeps local: the passages are
copyrighted and must never reach a tracked file. Nothing is written to stdout but counts
and short quoted samples.

Two stages, in this order.

1. Rule-based drops, counted per reason:
   * a dated log entry — the response opens on a date ("September 22. The Happy Death.",
     "January 4, 1952.", "1942.", "MARCH 17-JUNE"), which is a notebook diary line, not
     something said to anyone;
   * a citation marker — the response opens on "Id.", "Ibid.", "Cf." or initials
     ("B.B.", "R.C.") followed by a space, i.e. he is carrying someone else's words;
   * a crisis prompt — the prompt mentions suicide, killing oneself, self-harm, wanting
     to die, or the "stop resisting the thought of dying" shape. These rows are dropped
     whatever the passage is: a notebook fragment paired with a crisis message is not
     training data, it is a bad thing to hand a model at two in the morning.

2. A model pass over the survivors, in batches of 25 through rag/judge_opencode.py (key
   loading, the four Zen headers, transport and the 429 abort all come from there), on
   jz.DEFAULT_MODEL. Per row the model is asked one question: does this response read as
   something a person could plausibly say back to this prompt in conversation? Titles and
   headings, plans and outlines, dated log entries, reading notes, and other people's words
   quoted as his reply all fail. The reply is a JSON array of {"id", "keep", "reason"} and
   nothing else.

Checkpointing: decisions are appended per batch and flushed before the next call, keyed on
the normalised response text (lowercased, whitespace collapsed), so a rerun skips rows
already decided instead of paying for them again. The response text is the right key rather
than src_id, which names a chunk of many passages and is rewritten whenever the rows are
re-mined. One writer: the decisions file is appended, the filtered file is rewritten whole
from the input plus the decisions at the end of the run, so an interrupted run leaves both
files consistent with each other.

Failure handling:
  * HTTP 429 or a usage-limit body: the run stops at once, prints how far it got, exits 0.
    The gateway caps a window, not a run, so a retried batch would only earn the same
    answer. What was decided stays on disk.
  * any other error, or a reply that will not parse: the batch is retried once, then left
    undecided, so a rerun picks those rows up. jz.judge already retries an empty or
    truncated reply once, so that retry is on top of its own.

    python pipeline/filter_reply_suitability.py --dry-run
    python pipeline/filter_reply_suitability.py --limit 50
    python pipeline/filter_reply_suitability.py
"""
import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "rag"))

import judge_opencode as jz  # noqa: E402  (key loading, headers, transport, 429 abort)

MODEL = jz.DEFAULT_MODEL   # the same model the opencode-go provider calls space-bunny-free
BATCH = 25
SAMPLES = 10
SAMPLE_CHARS = 90

# ── rule 1: the response opens on a date ─────────────────────────────────────────
# "September 22. The Happy Death.", "May 16. Leave for Paris", "July 25 Wake up at 7",
# "April 1946 3: Resignation", "MARCH 17-JUNE", "1942.", "August 15, 1945". A month on its
# own ("November.") is not matched here: the model pass is the right place for that, and a
# bare month can still open a sentence about the calendar.
MONTHS = (r"January|February|March|April|May|June|July|August|September|October|November"
          r"|December|Jan\.|Feb\.|Mar\.|Apr\.|Jun\.|Jul\.|Aug\.|Sept?\.|Oct\.|Nov\.|Dec\."
          r"|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec")
# The month must end at the token it names: "Auguste Comte decides..." opens on a man,
# not on August, and "Marxism reproaches..." not on March. The negative lookahead is what
# keeps the abbreviations from eating the first three letters of an ordinary word.
DATE_START = re.compile(
    rf"^\s*(?:(?:{MONTHS})(?![A-Za-z])\s*\.?\s*\d|\d{{4}}\s*[.,:;])", re.I)

# ── rule 2: the response opens on a citation ──────────────────────────────────────
# "Id.", "Ibid.", "Cf." and initials such as "B.B. " / "R.C. " mark borrowed words. Two or
# more dotted capitals, because a single "N. " opens a sentence often enough that treating
# it as a citation would throw away real rows.
CITATION_START = re.compile(
    r"^\s*(?:id\.|ibid\.|cf\.|(?:[A-Z]\.){2,}\s)", re.I)

# ── rule 3: the prompt is a crisis message ───────────────────────────────────────
# Matched on the whole prompt, not on a phrase list of "dying": "everyone nearby is dying"
# is not a crisis message, "I want to die" is. Each alternative below is a phrase a person
# in trouble would write.
CRISIS = re.compile(
    r"\bsuicid\w*"                                        # suicide, suicidal, suicides
    r"|\bkill(?:ing|ed)?\s+(?:my|his|her|their|our|your|its)self\b"
    r"|\bself[-\s]harm\w*"                                # self-harm, self harm
    r"|\bhurt(?:ing)?\s+(?:my|him|her|their|my|him|her)self\b"
    r"|\b(?:want|wants|wanting|wanted|hope|hopes|hoping|hoped|wish|wishes|wishing|wish"
    r"ed|need|needs|feel|feels|feeling|think|thinks|thinking|ready|prefer|plan|plans|"
    r"planning|decided|consider|considering|afraid|scared|struggl\w+)\s+to\s+die\b"
    r"|\b(?:stop|stopping|resist|resisting|resisted|give\s+in\s+to|giving\s+in\s+to)"
    r"[\w\s]{0,24}?(?:thought|idea|feeling|temptation|pull)[\w\s]{0,12}?dying\b"
    r"|\bend(?:ing|s)?\s+it\s+(?:all|today|tonight|now|soon|finally|permanently|"
    r"this\s+(?:week|month|night|year|morning))\b"
    r"|\bend(?:ing|s)?\s+(?:my|his|her|their)\s+(?:own\s+)?life\b"
    r"|\btak(?:e|ing)\s+(?:my|his|her|their)\s+(?:own\s+)?life\b"
    r"|\b(?:death|dying)\s+wish\b"
    r"|\b(?:cut|cuts|cutting|slit|slits|slitting)\s+(?:my(?:self)?|him(?:self)?|"
    r"her(?:self)?|them(?:selves)?)\b"
    r"|\b(?:cut|cuts|cutting|slit|slits|slitting)\s+"
    r"(?:my\s+|the\s+)?(?:wrists|throat|arms)\b"
    r"|\b(?:hang|hanged|hanging)\s+(?:my|him|her|them|my(?:self)?)\b"
    r"|\boverdos\w*"
    r"|\bd(?:ie|ying)\s+alone\b",
    re.I)

RULE_LABELS = ("dated log entry (response opens on a date)",
               "citation marker (Id./Ibid./Cf./initials)",
               "crisis prompt (suicide, self-harm, wanting to die)")

SYSTEM = (
    "You screen mined rows for a conversational dataset. Each row is a user message and a "
    "verbatim passage from a writer's notebooks and essays, paired as a turn of "
    "conversation.\n"
    "Keep a row only if the response reads as something a person could plausibly say back "
    "to that prompt in conversation — a remark, an answer, a complaint, an aside, a half-"
    "finished thought.\n"
    "Drop it when the response is:\n"
    "- a bare title or heading (\"The Happy Death.\", \"Talk on theater.\");\n"
    "- a plan or an outline note (\"1. Rewrite the ending\", \"Chapter II: the revolt\");\n"
    "- a dated log or diary entry (\"May 16. Leave for Paris, heart aching.\");\n"
    "- a reading note, a summary of a book, or a piece of apparatus;\n"
    "- someone else's words quoted as if they were his reply;\n"
    "- anything you could not imagine saying out loud to the other person.\n"
    "Judge the response as a reply to this prompt. Do not judge whether you agree with it.\n"
    "Reply with ONLY a JSON array of objects, one per row, in the same order, carrying the "
    "id you were given: [{\"id\": <the id given>, \"keep\": true|false, \"reason\": "
    "\"<3-6 words>\"}]. No prose, no code fences."
)

CAP_PHRASE = re.compile(r"(usage|rate)\s*[-_ ]?limit", re.I)
CAP_BODY = re.compile(
    r"(too many requests|quota|insufficient (quota|balance|credits)|"
    r"exceeded your current|monthly limit|429)", re.I)


def normalise(text):
    """Response text as the resume key reads it: lowercase, whitespace collapsed."""
    return " ".join(str(text or "").split()).lower()


def row_key(row):
    return normalise(row.get("response"))


def is_dated(response):
    """A notebook entry that opens on a date: a diary line, never a reply."""
    return DATE_START.match(str(response or "")) is not None


def is_citation(response):
    """A passage carried over from a note by someone else: "Id.", "Ibid.", "Cf.",
    or initials like "B.B." — Camus quoting, not Camus speaking."""
    return CITATION_START.match(str(response or "")) is not None


def is_crisis(prompt):
    """A prompt about suicide, self-harm or wanting to die. These rows are never
    paired with a notebook fragment."""
    return CRISIS.search(str(prompt or "")) is not None


def rule_label(row):
    """Label of the first rule the row breaks, else None.

    Crisis first: it is the one rule with no acceptable exception, so a row that breaks
    it must be counted there even when its response is also dated or a citation."""
    if is_crisis(row.get("prompt")):
        return RULE_LABELS[2]
    if is_dated(row.get("response")):
        return RULE_LABELS[0]
    if is_citation(row.get("response")):
        return RULE_LABELS[1]
    return None


def apply_rules(rows):
    """The rows to put to the model, and the rows dropped, counted per reason.

    Each dropped row is counted under the first rule it breaks, so the counts add up to
    every drop. `matched` counts each rule on its own, rows that break two rules
    appearing twice, so the overlap between the two views is visible rather than
    hidden."""
    survivors, dropped, matched = [], Counter(), Counter()
    for row in rows:
        if is_crisis(row.get("prompt")):
            matched[RULE_LABELS[2]] += 1
        if is_dated(row.get("response")):
            matched[RULE_LABELS[0]] += 1
        if is_citation(row.get("response")):
            matched[RULE_LABELS[1]] += 1
        label = rule_label(row)
        if label is None:
            survivors.append(row)
        else:
            dropped[label] += 1
    return survivors, dropped, matched


def batch_payload(batch):
    """The user message for one batch: each row with the id the reply must carry.
    Batch-local ids, so the model has a short number to echo back."""
    return json.dumps([{"id": index,
                        "prompt": str(row.get("prompt", "")),
                        "response": str(row.get("response", ""))}
                       for index, row in enumerate(batch)], ensure_ascii=False)


def call_model(system, user, key=None):
    """One chat/completions call through the shared Zen transport.

    jz.judge is the whole transport: key, headers, timeout, token budget, its one retry
    of an empty or truncated reply, and the 429 abort. A JSON array reply carries JSON
    objects, which is what jz.judge checks a reply for, so nothing here re-validates it.
    """
    return jz.judge(MODEL, system, user, key=key)


def is_usage_limit_error(error):
    """A cap that arrives as an ordinary HTTP error rather than a 429: the gateway
    words it several ways and all of them mean the window is spent."""
    text = str(error)
    return bool(CAP_PHRASE.search(text) or CAP_BODY.search(text))


def _json_candidates(text):
    stripped = jz._strip_fences(str(text or ""))
    candidates = [stripped]
    start, end = stripped.find("["), stripped.rfind("]")
    if 0 <= start < end:
        candidates.append(stripped[start:end + 1])
    return candidates


def _keep(value):
    """The model's keep flag, however it spelled it. Anything unrecognised is False:
    a row is only kept when the model said so plainly."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "yes", "keep", "1")
    if isinstance(value, (int, float)):
        return bool(value)
    return False


def parse_decisions(text):
    """The reply as a list of {"id", "keep", "reason"} dicts.

    A fenced array, an array wrapped in prose, and a single object holding one all parse.
    A reply with none of them raises ValueError, which the caller treats as a failed
    batch: retried once, then left undecided.
    """
    for candidate in _json_candidates(text):
        try:
            value = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        if isinstance(value, dict):
            for key in ("decisions", "results", "rows", "batch"):
                inner = value.get(key)
                if isinstance(inner, list):
                    return [item for item in inner if isinstance(item, dict)]
            if "keep" in value:
                return [value]
    raise ValueError(f"no JSON array of decisions in reply: {str(text)[:120]}")


def align(batch, items):
    """Pair each row with its decision.

    Matched on id. When the ids do not cover the batch — the model drops one, or numbers
    from 1 — a reply of exactly the batch's length is taken in order instead, which is
    the one case where position is as good a key as id. Anything else leaves the
    unmatched rows undecided rather than giving them somebody else's verdict."""
    by_id = {}
    for item in items:
        if "keep" in item:
            try:
                by_id[int(item["id"])] = item
            except (KeyError, TypeError, ValueError):
                continue
    if set(by_id) == set(range(len(batch))):
        return [by_id[index] for index in range(len(batch))]
    if len(items) == len(batch):
        return list(items)
    return [by_id.get(index) for index in range(len(batch))]


def decision_row(row, keep, reason):
    """The decision as it goes on disk: the response text as its own key, so the record
    survives re-mining, plus enough of the row to find it again by eye."""
    return {"key": row_key(row),
            "keep": bool(keep),
            "reason": " ".join(str(reason or "").split())[:60],
            "src_id": row.get("src_id"),
            "source": row.get("source")}


def batches(items, size):
    return [items[i:i + size] for i in range(0, len(items), size)]


def fetch(batch, number, key, call, stats=None):
    """One batch's parsed decisions, None when the batch is left undecided, plus a stop
    reason when a cap ended the run.

    The retry covers the call *and* the parse, because a reply the model could not format
    is as likely to arrive again as a transport error is: a model that answered with prose
    once has shown what it is doing, and the identical request is worth exactly one more
    ask. jz.judge already retries an empty or truncated reply once, so this retry sits on
    top of its own.

    The two failure kinds call for opposite answers. A cap (429, or a usage-limit body on
    another status) comes back as a stop reason and is never retried: the window is
    spent. Anything else is retried once and then the batch is abandoned without a
    decision, which a rerun picks up again.
    """
    last = ""
    for attempt in (1, 2):
        if stats is not None:
            stats["calls"] += 1
        try:
            reply = call(SYSTEM, batch_payload(batch), key=key)
            return parse_decisions(reply), None
        except jz.UsageCapReached as exc:
            return None, str(exc)
        except Exception as exc:  # noqa: BLE001 - retried once, then left undecided
            if is_usage_limit_error(exc):
                return None, f"{type(exc).__name__}: {exc}"
            last = f"{type(exc).__name__}: {exc}"
            print(f"  batch {number}: attempt {attempt} failed — {exc}", file=sys.stderr)
    print(f"  batch {number}: left undecided after one retry — {last}", file=sys.stderr)
    return None, None


def load_rows(path):
    """JSONL rows; a line that will not parse is left out."""
    rows = []
    if not os.path.exists(path):
        return rows
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


def load_decisions(path):
    """What is already decided, keyed on the normalised response text."""
    decided = {}
    for record in load_rows(path):
        key = normalise(record.get("key"))
        if key and "keep" in record:
            decided[key] = record
    return decided


def write_filtered(rows, decided, out_path):
    """Rewrite the filtered file whole: every input row whose response has a keep
    decision, in the input file's order, fields untouched.

    Rewriting rather than appending is what makes one writer safe here. Two runs of this
    script then leave the same file for the same set of decisions, and an interrupted
    run never leaves a half-appended file that a rerun would double."""
    kept = [row for row in rows if decided.get(row_key(row), {}).get("keep")]
    tmp = f"{out_path}.tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        for row in kept:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, out_path)
    return kept


def run(survivors, decided, decisions_path, batch_size=BATCH, key=None, call=call_model):
    """Decide every survivor not already in `decided`, appending each batch at once.

    Returns (stats, stop_reason). stop_reason is None unless a usage cap ended the run,
    in which case the decisions on disk are everything decided so far."""
    todo, seen = [], set()
    for row in survivors:
        key_text = row_key(row)
        if not key_text or key_text in seen or key_text in decided:
            continue
        seen.add(key_text)
        todo.append(row)

    stats = {"survivors": len(survivors), "todo": len(todo),
             "already": len(decided), "batches": 0, "calls": 0,
             "decided": 0, "kept": 0, "dropped": 0, "undecided": 0}
    stop_reason = None

    with open(decisions_path, "a", encoding="utf-8") as out:
        for number, batch in enumerate(batches(todo, batch_size), 1):
            items, stop_reason = fetch(batch, number, key, call, stats)
            if stop_reason is not None:
                stats["undecided"] += len(batch)
                break
            if items is None:
                stats["undecided"] += len(batch)
                continue
            pairs = align(batch, items)
            written = 0
            for row, item in zip(batch, pairs):
                if not isinstance(item, dict) or "keep" not in item:
                    stats["undecided"] += 1
                    continue
                keep = _keep(item["keep"])
                record = decision_row(row, keep, item.get("reason"))
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                decided[record["key"]] = record
                written += 1
                stats["kept" if keep else "dropped"] += 1
            out.flush()
            stats["decided"] += written
            stats["batches"] += 1
            print(f"  batch {number}: {len(batch)} rows, {written} decided, "
                  f"{stats['kept']} kept, {stats['calls']} calls")
    return stats, stop_reason


def print_rules(rows, dropped, matched, decided):
    print(f"input rows {len(rows)}")
    print("dropped by rule (first rule that matches, so the counts add up):")
    for label in RULE_LABELS:
        print(f"  {dropped[label]:>6}  {label}")
    print(f"  {sum(dropped.values()):>6}  total dropped")
    print(f"  {len(rows) - sum(dropped.values()):>6}  left for the model pass")
    both = sum(matched.values()) - sum(dropped.values())
    if both:
        print(f"  ({both} row(s) break two rules — a crisis prompt on a dated response, "
              "say — and are counted under the first, so the per-rule totals above run "
              f"high: {dict(matched)})")
    already = sum(1 for row in rows
                  if row_key(row) in decided and decided[row_key(row)])
    print(f"  {already:>6}  already decided in the decisions file")


def print_summary(stats, stop_reason, decisions_path, out_path, kept_total):
    print(f"\nsurvivors {stats['survivors']}  already decided {stats['already']}  "
          f"to decide {stats['todo']}")
    print(f"batches {stats['batches']}  calls {stats['calls']}  "
          f"decided {stats['decided']}  kept {stats['kept']}  dropped {stats['dropped']}  "
          f"undecided {stats['undecided']}")
    print(f"decisions -> {decisions_path}")
    print(f"kept rows -> {out_path}  ({kept_total} rows)")
    if stop_reason is not None:
        print(f"stopped on a usage cap, not a failure: {stop_reason}")
        print(f"  {stats['todo'] - stats['decided'] - stats['undecided']} rows still "
              "to decide; re-run the same command to continue where this stopped.")


def clipped(text, limit=SAMPLE_CHARS):
    flat = " ".join(str(text or "").split())
    return flat if len(flat) <= limit else flat[:limit]


def print_samples(rows, decided, count=SAMPLES):
    kept = [row for row in rows if decided.get(row_key(row), {}).get("keep")]
    dropped = [row for row in rows
               if row_key(row) in decided and not decided[row_key(row)].get("keep")]
    print(f"\n{count} kept pairs:")
    for row in kept[:count]:
        reason = decided[row_key(row)].get("reason", "")
        print(f"  P: {clipped(row.get('prompt'))}\n"
              f"  R: {clipped(row.get('response'))}\n"
              f"     [{reason}]")
    print(f"\n{count} dropped pairs:")
    for row in dropped[:count]:
        label = rule_label(row) or decided[row_key(row)].get("reason", "")
        print(f"  P: {clipped(row.get('prompt'))}\n"
              f"  R: {clipped(row.get('response'))}\n"
              f"     [{label}]")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--in", dest="src", default="data/primary_rows.jsonl")
    ap.add_argument("--decisions", default="data/primary_rows.decisions.jsonl")
    ap.add_argument("--out", default="data/primary_rows.filtered.jsonl")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the rule counts and stop; no model is called")
    ap.add_argument("--limit", type=int, default=None,
                    help="put at most N survivors to the model")
    ap.add_argument("--batch", type=int, default=BATCH,
                    help=f"rows per call (default {BATCH})")
    ap.add_argument("--samples", type=int, default=SAMPLES,
                    help=f"kept and dropped pairs to print at the end (default {SAMPLES}, "
                         "0 for none)")
    args = ap.parse_args()

    rows = load_rows(args.src)
    survivors, dropped, matched = apply_rules(rows)
    decided = load_decisions(args.decisions)
    print_rules(rows, dropped, matched, decided)

    if args.dry_run:
        print(f"\ndry run: rules only. {len(survivors)} rows would go to the model in "
              f"{-(-len(survivors) // args.batch)} calls of {args.batch}.")
        return 0

    if args.limit is not None:
        survivors = survivors[:args.limit]
        print(f"\nlimited to {len(survivors)} survivors")
    key = jz.load_key()
    stats, stop_reason = run(survivors, decided, args.decisions, args.batch,
                             key=key, call=call_model)
    kept = write_filtered(rows, load_decisions(args.decisions), args.out)
    print_summary(stats, stop_reason, args.decisions, args.out, len(kept))
    if args.samples:
        print_samples(rows, load_decisions(args.decisions), args.samples)
    return 0


if __name__ == "__main__":
    sys.exit(main())