#!/usr/bin/env python3
"""
assemble_training.py — assemble the v3 training sets into build/training_v3/.

Every output row is the one schema the training notebooks read:

    {"messages": [{"role": "system"|"user"|"assistant", "content": str}, ...],
     "kind": str, "source": str}

Single-turn inputs (prompt/response) become a [user, assistant] pair; rows that
already carry "messages" keep them (a leading system message is allowed). Every
row must end with an assistant turn whose content is non-empty — "..." is
content, so short replies survive. A malformed input row aborts the run with its
file and line number rather than being skipped.

  phase1_old.jsonl  the corpus camus2 was actually trained on: camus_sft.jsonl +
                    camus_conversational.jsonl read from git tag `corpus-camus2`,
                    NOT from the worktree — the ligature repair and the
                    conversational naturalisation both landed after that tag, so
                    the worktree copy is not what the released model saw.
  phase1_new.jsonl  the same two files as they stand today, plus
                    data/camus_short.jsonl (hand-authored short + goodbye
                    replies) and data/primary_rows.filtered.jsonl (mined primary
                    text), reweighted — see below. Book text: build/ only, never
                    tracked.
  phase2.jsonl      the Step-3.5 guardrail mix (see below).

phase1_new reweighting (CONV_KEEP, CONV_SEED, CURATED_WEIGHT at the top of this
file):

  data/camus_sft.jsonl           every `essayist` row kept in full; the uniform
                                 generated `conversational` rows sampled down to
                                 CONV_KEEP with random.Random(CONV_SEED), the
                                 sampled rows left in input order.
  data/camus_short.jsonl         x CURATED_WEIGHT — hand-authored, and the only
  data/primary_rows.filtered     rows that speak in the notebook's register.
  data/camus_conversational      x1.

Deduplication: exact duplicate rows are dropped, first occurrence wins — and it
runs BEFORE the upweighting, so a deliberate 2x/3x/4x still reaches the set
while an accidental repeat inside one input file does not. phase1_new applies
that per source file, then weights, for the same reason.

Also written: MANIFEST.json (row count, kind histogram, sha256, assistant-turn
metrics, the commit each set was built from, whether the defects files were
included) and README.txt (how to get the folder onto Google Drive).

Phase 2 mirrors training/CamusGPT_v2_Step3_5_RefusalSFT_Gemma3.ipynb, cell 4 —
the mixing that trained camus2's guardrail pass (`random.seed(3407)`):

  camus_refusals.jsonl      shuffled with that seed, first 400 rows only.
  camus_conversational.jsonl every row once, but kind in (substantive,
                            meditation) three times: the v2 over-deflection
                            regression came from the model deflecting instead of
                            composing long-form.
  camus_sft.jsonl           YES — Phase 2 does mix in sft rows, but only the
                            short conversational ones (response <= 60 words),
                            shuffled with the same seed and cut to the first 450.
                            No essayist prose: that is recited, not answered.
  camus_epistemic.jsonl     x2 — installs new behaviour; the notebook's
                            correction:affirmation assertion is left alone, so
                            the affirmation rows stay.
  camus_analysis.jsonl      analyze_provided x3, injection_refuse x2.
  camus_phase3.jsonl        x2 — eval-driven fixes, same rule as the other
                            new-behaviour sets.
  camus_multiturn.jsonl     x4 — few but pivotal; multi-turn items, not
                            prompt/response rows.
  defects_*.jsonl           x2, ONLY when data/drafts/DEFECTS_REVIEWED exists.
                            An extension past the notebook, gated on that marker
                            file so unreviewed drafts cannot reach training:
                            these sets install new behaviour (recall in context,
                            false premises, memory use), which is exactly what the
                            notebook upweights 2x. Until the marker exists they
                            are excluded.

Deduplication: exact duplicate rows are dropped, first occurrence wins — and it
runs BEFORE the upweighting above and in phase1_new, so a deliberate 2x/3x/4x
still reaches the set while an accidental repeat inside one input file does not.
Provenance: `kind` is the row's own "kind", else "category" (refusals), else
"type" (sft: essayist/conversational; phase3: fidelity/attribution/...), else the
input file's stem. `source` is the row's own "source", else the input label —
which reads `corpus-camus2:data/camus_sft.jsonl` for the tag-read rows, so a row
carries the corpus it came from.

    python pipeline/assemble_training.py --dry-run
    python pipeline/assemble_training.py
    python pipeline/assemble_training.py --write-short   # rebuild data/camus_short.jsonl
"""
import argparse
import hashlib
import json
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPELINE_DIR))

import corpus_metrics  # noqa: E402

REPO_ROOT = PIPELINE_DIR.parent

ROLES = ("system", "user", "assistant")
KIND_KEYS = ("kind", "category", "type")

CORPUS_TAG = "corpus-camus2"

SHORT_FILE = "data/camus_short.jsonl"
SHORT_DRAFTS = ("data/drafts/short_goodbye.deepseek.jsonl",
                "data/drafts/short_goodbye.space-bunny.jsonl")
PRIMARY_ROWS = "data/primary_rows.filtered.jsonl"

SFT_FILE = "data/camus_sft.jsonl"
SFT_ESSAYIST = "essayist"
SFT_CONVERSATIONAL = "conversational"

PHASE1_OLD_INPUTS = (SFT_FILE, "data/camus_conversational.jsonl")
CURATED_INPUTS = (SHORT_FILE, PRIMARY_ROWS)
PHASE1_NEW_INPUTS = PHASE1_OLD_INPUTS + CURATED_INPUTS

# phase1_new reweighting: the uniform generated conversational rows dominate
# the mix and are near-identical to each other, so they are sampled down to
# CONV_KEEP; the curated sets (hand-authored short replies, mined primary text)
# are few and carry the register the notebook reads, so they go in twice.
CONV_KEEP = 1500
CONV_SEED = 13
CURATED_WEIGHT = 2

DEFECTS_MARKER = "data/drafts/DEFECTS_REVIEWED"
DEFECTS_INPUTS = ("data/drafts/defects_recall.jsonl",
                  "data/drafts/defects_premise.jsonl",
                  "data/drafts/defects_memory.jsonl")

# Notebook-mirrored mixing constants (cell 4 of the Gemma-3 Step-3.5 notebook).
NB_SEED = 3407
NB_REFUSAL_ROWS = 400
NB_PRESERVE_MAX_WORDS = 60
NB_PRESERVE_ROWS = 450
NB_LONG_KINDS = ("substantive", "meditation")
NB_LONG_UPWEIGHT = 3
NB_EPISTEMIC_UPWEIGHT = 2
NB_ANALYZE_UPWEIGHT = 3
NB_INJECTION_UPWEIGHT = 2
NB_PHASE3_UPWEIGHT = 2
NB_MULTITURN_UPWEIGHT = 4
DEFECTS_UPWEIGHT = 2

MANIFEST_NAME = "MANIFEST.json"
README_NAME = "README.txt"

README_TEMPLATE = """CamusGPT v3 — training-set bundle
================================

Destination on Google Drive:  MyDrive/CamusGPT_Training/data_v3/

Uploaded files — {n_sets} sets, {n_rows} rows, {total_mb:.1f} MB in all
(sha256 of each file is in MANIFEST.json):

{file_table}

What each set is
----------------
  phase1_old.jsonl   camus_sft.jsonl + camus_conversational.jsonl as they were
                     at git tag corpus-camus2 — the corpus camus2 was trained
                     on, for a same-inputs A/B against phase1_new.
  phase1_new.jsonl   today's corpus: the two above, plus the hand-authored
                     short/goodbye pairs and the mined primary-text rows.
                     Reweighted: every essayist row, the uniform generated
                     conversational rows sampled down to {conv_keep} (seed
                     {conv_seed}), and the two curated sets (short/goodbye,
                     primary text) x{curated_weight} — exact duplicates are
                     dropped per source file first, so the x{curated_weight}
                     survives. The constants and the per-source counts before
                     and after weighting are in MANIFEST.json.
  phase2.jsonl       the Step-3.5 guardrail mix, mirroring cell 4 of
                     training/CamusGPT_v2_Step3_5_RefusalSFT_Gemma3.ipynb:
                     refusals (seeded 400-row sample) + conversational
                     (substantive/meditation x3) + a 450-row short-SFT
                     preservation sample + epistemic x2 + analysis x3/x2 +
                     phase3 x2 + multiturn x4.{defects_note}
  MANIFEST.json      row count, kind histogram, sha256, assistant-turn metrics,
                     and the git commit each set was assembled from.
  README.txt         this file.

Row schema (identical in every set):
  {{"messages": [{{"role": "system"|"user"|"assistant", "content": str}}, ...],
   "kind": str, "source": str}}
Single-turn rows are [user, assistant]; every row ends with a non-empty
assistant turn. A notebook reads them with:

  Dataset.from_json("data_v3/{{name}}")

Upload — pick one
-----------------

A) Colab (how every notebook here reaches Drive)

    from google.colab import drive
    import shutil
    drive.mount("/content/drive")
    dest = "/content/drive/MyDrive/CamusGPT_Training/data_v3"
    shutil.copytree("/content/build/training_v3", dest, dirs_exist_ok=True)

   If the bundle is not on the Colab disk yet, clone the repo, run
   `python pipeline/assemble_training.py`, then copytree as above.

B) rclone (resumable, good for a big folder)

    rclone config                      # name it "gdrive": scope drive
    rclone copy build/training_v3/ gdrive:CamusGPT_Training/data_v3/ --progress

C) Browser

   Drive -> My Drive -> CamusGPT_Training -> New folder -> "data_v3" ->
   drag the files in. Small enough to do by hand; keep them in ONE folder,
   flat — the notebooks read data_v3/<file>.jsonl by name.

Verify after uploading
----------------------
Check the upload against MANIFEST.json — one sha256 per set:

    cd build/training_v3
    shasum -a 256 phase1_old.jsonl phase1_new.jsonl phase2.jsonl

Each line must match the "sha256" of the matching set in MANIFEST.json. If a
file went up wrong, re-upload that one file; the notebooks read them by name.

Keep the bundle out of git
--------------------------
build/ is gitignored and pipeline/hooks/pre-commit refuses anything under it.
phase1_old.jsonl and phase1_new.jsonl contain Camus's published text: local and
Drive only, never a commit, never the repo.
"""


class AssemblyError(Exception):
    """A malformed input row, a missing input, or a git command that failed."""


# ── input ────────────────────────────────────────────────────────────────────

def git_output(args, repo_root):
    proc = subprocess.run(["git", *args], cwd=str(repo_root),
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          encoding="utf-8")
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip().splitlines()
        raise AssemblyError("git %s failed: %s"
                            % (" ".join(args), detail[-1] if detail else proc.returncode))
    return proc.stdout


def parse_jsonl(lines, label):
    """[(lineno, text)] -> [(lineno, dict)]; blank lines skipped, junk refused."""
    rows = []
    for lineno, text in lines:
        if not text.strip():
            continue
        try:
            row = json.loads(text)
        except json.JSONDecodeError as exc:
            raise AssemblyError("%s:%d: invalid JSON (%s)" % (label, lineno, exc.msg)) from None
        if not isinstance(row, dict):
            raise AssemblyError("%s:%d: row is %s, not a JSON object"
                                % (label, lineno, type(row).__name__))
        rows.append((lineno, row))
    return rows


def load_input(label, repo_root):
    """Read and validate one input. `label` is a repo path, or `tag:path`."""
    if ":" in label:
        tag, _, relpath = label.partition(":")
        lines = list(enumerate(git_output(["show", "%s:%s" % (tag, relpath)],
                                         repo_root).splitlines(keepends=True), 1))
    else:
        path = repo_root / label
        if not path.is_file():
            raise AssemblyError("%s: no such file" % label)
        lines = list(enumerate(path.read_text(encoding="utf-8").splitlines(keepends=True), 1))
    return [normalize_row(row, label, lineno, default_kind_for(label))
            for lineno, row in parse_jsonl(lines, label)]


# ── validation and normalisation ─────────────────────────────────────────────

def _text_field(value, where, field):
    if not isinstance(value, str):
        raise AssemblyError("%s: %s is %s, expected a string"
                            % (where, field, type(value).__name__))
    if not value.strip():
        raise AssemblyError("%s: %s is empty" % (where, field))
    return value


def _labelled_field(row, keys, where):
    """First non-empty string among `keys`; a non-string value is a malformed row."""
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        if not isinstance(value, str):
            raise AssemblyError("%s: %s is %s, expected a string"
                                % (where, key, type(value).__name__))
        if value.strip():
            return value
    return ""


def _check_messages(value, where):
    if not isinstance(value, list) or not value:
        raise AssemblyError("%s: 'messages' is %s, expected a non-empty list"
                            % (where, type(value).__name__))
    messages = []
    for index, message in enumerate(value):
        if not isinstance(message, dict):
            raise AssemblyError("%s: message %d is %s, expected an object"
                                % (where, index, type(message).__name__))
        role, content = message.get("role"), message.get("content")
        if role not in ROLES:
            raise AssemblyError("%s: message %d has role %r, expected one of %s"
                                % (where, index, role, ", ".join(ROLES)))
        if not isinstance(content, str):
            raise AssemblyError("%s: message %d content is %s, expected a string"
                                % (where, index, type(content).__name__))
        messages.append({"role": role, "content": content})
    last = messages[-1]
    if last["role"] != "assistant":
        raise AssemblyError("%s: row ends on a %r turn, it must end with the assistant"
                            % (where, last["role"]))
    if not last["content"].strip():
        raise AssemblyError("%s: final assistant turn is empty" % where)
    return messages


def default_kind_for(label):
    """'data/camus_multiturn.jsonl' -> 'multiturn' — the fallback when a row
    carries no kind/category/type."""
    stem = Path(label.partition(":")[2] or label).stem
    return stem[len("camus_"):] if stem.startswith("camus_") else stem


def normalize_row(row, label, lineno, default_kind):
    """One input row -> one output row. Raises AssemblyError(file:line: why)."""
    where = "%s:%d" % (label, lineno)
    if "messages" in row:
        messages = _check_messages(row["messages"], where)
    elif "prompt" in row or "response" in row:
        messages = [
            {"role": "user", "content": _text_field(row.get("prompt"), where, "prompt")},
            {"role": "assistant", "content": _text_field(row.get("response"), where, "response")},
        ]
    else:
        raise AssemblyError("%s: no 'messages' and no 'prompt'/'response' pair" % where)
    kind = _labelled_field(row, KIND_KEYS, where) or default_kind
    return {"messages": messages,
            "kind": kind,
            "source": _labelled_field(row, ("source",), where) or label}


# ── dedup ────────────────────────────────────────────────────────────────────

def dedup_key(row):
    """(prompt, response) for a single-turn row, the whole turn list otherwise."""
    messages = row["messages"]
    if len(messages) == 2 and messages[0]["role"] == "user" and messages[1]["role"] == "assistant":
        return (messages[0]["content"], messages[1]["content"])
    return tuple((message["role"], message["content"]) for message in messages)


def dedup(rows, seen=None):
    """Exact duplicates out, first occurrence kept. -> (kept, dropped_count).
    `seen` lets a caller drop each input's repeats against the ones already
    accepted from earlier inputs, without re-deriving the keys."""
    if seen is None:
        seen = set()
    kept = []
    for row in rows:
        key = dedup_key(row)
        if key in seen:
            continue
        seen.add(key)
        kept.append(row)
    return kept, len(rows) - len(kept)


# ── sets ─────────────────────────────────────────────────────────────────────

def build_phase1(repo_root, labels, weight_of=None, prepare=None):
    """Concatenation in the order given, deduped, then each input's rows repeated
    `weight_of(label)` times. Dedup runs before the weights, so a deliberate x2
    still reaches the set while an accidental repeat inside one input does not.

    `weight_of` defaults to 1 and `prepare` to the identity: phase1_old takes
    both at face value, phase1_new supplies the reweighting."""
    seen = set()
    dropped = 0
    rows = []
    inputs = []
    for label in labels:
        part = load_input(label, repo_root)
        kept, gone = dedup(part, seen)
        dropped += gone
        note = None
        if prepare is not None:
            kept, note = prepare(label, kept)
        weight = weight_of(label) if weight_of is not None else 1
        entry = {"input": label, "rows_read": len(part), "rows_after_dedup": len(part) - gone,
                 "rows_selected": len(kept), "weight": weight,
                 "rows_after_weight": len(kept) * weight}
        if note:
            entry["note"] = note
        elif weight != 1:
            entry["note"] = "x%d" % weight
        inputs.append(entry)
        rows.extend(row for _ in range(weight) for row in kept)
    inputs.append({"input": "(set)", "note": "exact duplicate rows dropped",
                   "rows_dropped": dropped})
    return rows, {"inputs": inputs}


def downsample_conversational(rows, keep, seed):
    """The generated conversational rows cut to `keep`, drawn uniformly at random
    from `seed`; every other row — the essayist prose — is kept untouched. The
    chosen rows stay in input order, so the sample is a stable subset.

    -> (kept, available, kept_conversational)"""
    positions = [index for index, row in enumerate(rows) if row["kind"] == SFT_CONVERSATIONAL]
    drawn = set(random.Random(seed).sample(positions, min(keep, len(positions))))
    conversational = set(positions)
    kept = [row for index, row in enumerate(rows)
            if index not in conversational or index in drawn]
    return kept, len(positions), len(drawn)


def build_phase1_new(repo_root, labels=PHASE1_NEW_INPUTS):
    """Today's corpus, reweighted for phase 1: every essayist row, the uniform
    generated conversational rows sampled down to CONV_KEEP with CONV_SEED, the
    two curated sets at CURATED_WEIGHT, the hand-built conversational set once.
    Dedup runs per source before any of it, so the deliberate x2 survives."""
    def weight_of(label):
        return CURATED_WEIGHT if label in CURATED_INPUTS else 1

    def prepare(label, rows):
        if label != SFT_FILE:
            return rows, None
        kept, available, drawn = downsample_conversational(rows, CONV_KEEP, CONV_SEED)
        return kept, ("%s rows kept in full, %s rows %d -> %d (uniform sample, seed %d)"
                      % (SFT_ESSAYIST, SFT_CONVERSATIONAL, available, drawn, CONV_SEED))

    rows, inputs = build_phase1(repo_root, labels, weight_of=weight_of, prepare=prepare)
    inputs["conv_keep"] = CONV_KEEP
    inputs["curated_weight"] = CURATED_WEIGHT
    inputs["seed"] = CONV_SEED
    return rows, inputs


def build_phase2(repo_root):
    """The notebook's mixing. Groups are (label, rows, weight); dedup first,
    weights after, so upweighting survives and accidental repeats do not."""
    rng = random.Random(NB_SEED)
    groups = []
    inputs = []

    def add(label, rows, weight, note):
        groups.append((label, rows, weight))
        inputs.append({"input": label, "rows_read": len(rows), "weight": weight, "note": note})

    refusals = load_input("data/camus_refusals.jsonl", repo_root)
    rng.shuffle(refusals)
    add("data/camus_refusals.jsonl", refusals[:NB_REFUSAL_ROWS], 1,
        "seeded shuffle, first %d rows" % NB_REFUSAL_ROWS)

    coop = load_input("data/camus_conversational.jsonl", repo_root)
    add("data/camus_conversational.jsonl",
        [row for row in coop if row["kind"] not in NB_LONG_KINDS], 1, "once")
    add("data/camus_conversational.jsonl",
        [row for row in coop if row["kind"] in NB_LONG_KINDS], NB_LONG_UPWEIGHT,
        "kind in %s, x%d" % (", ".join(NB_LONG_KINDS), NB_LONG_UPWEIGHT))

    sft = load_input("data/camus_sft.jsonl", repo_root)
    short_sft = [row for row in sft
                 if len(row["messages"][-1]["content"].split()) <= NB_PRESERVE_MAX_WORDS]
    rng.shuffle(short_sft)
    add("data/camus_sft.jsonl", short_sft[:NB_PRESERVE_ROWS], 1,
        "short conversational rows only (<= %d words), seeded shuffle, first %d"
        % (NB_PRESERVE_MAX_WORDS, NB_PRESERVE_ROWS))

    epistemic = load_input("data/camus_epistemic.jsonl", repo_root)
    add("data/camus_epistemic.jsonl", epistemic, NB_EPISTEMIC_UPWEIGHT,
        "x%d" % NB_EPISTEMIC_UPWEIGHT)

    analysis = load_input("data/camus_analysis.jsonl", repo_root)
    add("data/camus_analysis.jsonl",
        [row for row in analysis if row["kind"] == "analyze_provided"], NB_ANALYZE_UPWEIGHT,
        "analyze_provided, x%d" % NB_ANALYZE_UPWEIGHT)
    add("data/camus_analysis.jsonl",
        [row for row in analysis if row["kind"] != "analyze_provided"], NB_INJECTION_UPWEIGHT,
        "injection_refuse, x%d" % NB_INJECTION_UPWEIGHT)

    phase3 = load_input("data/camus_phase3.jsonl", repo_root)
    add("data/camus_phase3.jsonl", phase3, NB_PHASE3_UPWEIGHT, "x%d" % NB_PHASE3_UPWEIGHT)

    multiturn = load_input("data/camus_multiturn.jsonl", repo_root)
    add("data/camus_multiturn.jsonl", multiturn, NB_MULTITURN_UPWEIGHT, "x%d" % NB_MULTITURN_UPWEIGHT)

    if defects_included(repo_root):
        for label in DEFECTS_INPUTS:
            add(label, load_input(label, repo_root), DEFECTS_UPWEIGHT,
                "reviewed drafts, x%d" % DEFECTS_UPWEIGHT)

    seen = set()
    pool = []
    kept_per_group = []
    dropped = 0
    for label, rows, weight in groups:
        kept = 0
        for row in rows:
            key = dedup_key(row)
            if key in seen:
                dropped += 1
                continue
            seen.add(key)
            pool.append((row, weight))
            kept += 1
        kept_per_group.append(kept)
    for entry, kept in zip(inputs, kept_per_group):
        entry["rows_after_dedup"] = kept

    rows = [row for row, weight in pool for _ in range(weight)]
    rng.shuffle(rows)
    inputs.append({"input": "(set)", "note": "exact duplicate rows dropped",
                    "rows_dropped": dropped})
    return rows, {"inputs": inputs}


def defects_included(repo_root):
    """The defects drafts only ship once a human has signed off on them."""
    return (repo_root / DEFECTS_MARKER).exists()


# ── the short-reply corpus (tracked; rebuilt from the gitignored drafts) ─────

def write_short_corpus(repo_root, dry_run=False):
    """data/camus_short.jsonl = the two drafts concatenated, prompt/response/kind,
    exact duplicate responses dropped (first occurrence wins)."""
    rows = []
    seen = set()
    dropped = 0
    for label in SHORT_DRAFTS:
        path = repo_root / label
        if not path.is_file():
            raise AssemblyError("%s: no such file" % label)
        for lineno, row in parse_jsonl(
                list(enumerate(path.read_text(encoding="utf-8").splitlines(keepends=True), 1)), label):
            where = "%s:%d" % (label, lineno)
            prompt = _text_field(row.get("prompt"), where, "prompt")
            response = _text_field(row.get("response"), where, "response")
            kind = _text_field(row.get("kind"), where, "kind")
            if response in seen:
                dropped += 1
                continue
            seen.add(response)
            rows.append({"prompt": prompt, "response": response, "kind": kind})
    content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    if not dry_run:
        (repo_root / SHORT_FILE).write_text(content, encoding="utf-8")
    return rows, dropped


# ── manifest ─────────────────────────────────────────────────────────────────

def assistant_texts(rows):
    """Every assistant turn in the set — one text per turn, so multi-turn rows
    contribute all of their targets."""
    return [message["content"]
            for row in rows for message in row["messages"]
            if message["role"] == "assistant"]


def set_metrics(rows):
    """The five assistant-turn metrics, read off corpus_metrics.metrics()."""
    m = corpus_metrics.metrics(assistant_texts(rows))
    return {
        "assistant_turns": m["n"],
        "words_median": m["median"],
        "words_sd": m["pstdev"],
        "share_under_25_words": m["buckets"]["0-9"]["pct"] + m["buckets"]["10-24"]["pct"],
        "top10_opener_share": m["openers_top_share"],
        "question_ending_rate": m["question_rate"],
        "exact_duplicate_turns": m["duplicates"],
    }


def serialize(rows):
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)


def sha256_of(content):
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def readme_text(manifest):
    sets = manifest["sets"]
    width = max(len(entry["file"]) for entry in sets.values())
    lines = []
    for entry in sets.values():
        lines.append("  %-*s  %6d rows  %8.2f MB  sha256 %s"
                     % (width, entry["file"], entry["rows"], entry["bytes"] / 1048576,
                        entry["sha256"]))
    n_rows = sum(entry["rows"] for entry in sets.values())
    n_bytes = sum(entry["bytes"] for entry in sets.values())
    defects_note = ""
    if not any(entry["defects_included"] for entry in sets.values()):
        defects_note = ("\n                     The defects drafts were NOT included"
                        " (no %s marker)." % DEFECTS_MARKER)
    return README_TEMPLATE.format(
        n_sets=len(sets), n_rows=n_rows, total_mb=n_bytes / 1048576,
        file_table="\n".join(lines), defects_note=defects_note,
        conv_keep=CONV_KEEP, conv_seed=CONV_SEED, curated_weight=CURATED_WEIGHT)


def assemble(repo_root, out_dir, dry_run=False):
    """Write every set, MANIFEST.json and README.txt. Returns the manifest."""
    head = git_output(["rev-parse", "HEAD"], repo_root).strip()
    tag_commit = git_output(["rev-list", "-n", "1", CORPUS_TAG], repo_root).strip()

    old_rows, old_inputs = build_phase1(repo_root, [f"{CORPUS_TAG}:{label}"
                                                  for label in PHASE1_OLD_INPUTS])
    new_rows, new_inputs = build_phase1_new(repo_root)
    phase2_rows, phase2_inputs = build_phase2(repo_root)

    sets = [
        ("phase1_old", old_rows, old_inputs, False,
         {"note": "read from git tag %s, not the worktree" % CORPUS_TAG,
          "corpus_tag": CORPUS_TAG, "corpus_tag_commit": tag_commit}),
        ("phase1_new", new_rows, new_inputs, False,
         {"note": "contains book text (primary_rows.filtered.jsonl) — build/ only; "
                  "every input once except the curated sets, x%d "
                  "(rows_after_weight per input is in 'inputs')" % CURATED_WEIGHT,
          "conv_keep": new_inputs["conv_keep"],
          "curated_weight": new_inputs["curated_weight"],
          "seed": new_inputs["seed"]}),
        ("phase2", phase2_rows, phase2_inputs, defects_included(repo_root),
         {"note": "mirrors %s cell 4" % "training/CamusGPT_v2_Step3_5_RefusalSFT_Gemma3.ipynb",
          "notebook_seed": NB_SEED}),
    ]

    manifest = {
        "generated_by": "pipeline/assemble_training.py",
        "git_commit": head,
        "row_schema": {
            "messages": "list of {role: system|user|assistant, content: str}",
            "kind": "str", "source": "str",
            "rule": "single-turn inputs become [user, assistant]; rows with 'messages' "
                    "keep them; every row ends with a non-empty assistant turn",
        },
        "sets": {},
    }
    contents = {}
    for name, rows, inputs, defects, extra in sets:
        content = serialize(rows)
        contents[name] = content
        kinds = Counter(row["kind"] for row in rows)
        manifest["sets"][name] = dict(
            extra,
            file=f"{name}.jsonl",
            rows=len(rows),
            bytes=len(content.encode("utf-8")),
            sha256=sha256_of(content),
            kinds=dict(sorted(kinds.items(), key=lambda kv: (-kv[1], kv[0]))),
            metrics=set_metrics(rows),
            git_commit=head,
            defects_included=defects,
            inputs=inputs["inputs"],
        )

    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, content in contents.items():
            (out_dir / f"{name}.jsonl").write_text(content, encoding="utf-8")
        (out_dir / MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        (out_dir / README_NAME).write_text(readme_text(manifest), encoding="utf-8")
    return manifest


def report(manifest):
    print("assemble_training  commit %s" % manifest["git_commit"][:10])
    for name, entry in manifest["sets"].items():
        m = entry["metrics"]
        print("\n%s  %d rows  %d kinds  %s" % (name, entry["rows"], len(entry["kinds"]),
                                              "defects: %s" % ("yes" if entry["defects_included"] else "no")))
        print("  sha256 %s" % entry["sha256"])
        print("  assistant turns %d  median %d words  sd %.1f"
              % (m["assistant_turns"], m["words_median"], m["words_sd"]))
        print("  under 25 words %5.1f%%   top-10 openers %5.1f%%   ends with ? %5.1f%%"
              % (100 * m["share_under_25_words"], 100 * m["top10_opener_share"],
                 100 * m["question_ending_rate"]))
        top = list(entry["kinds"].items())[:5]
        print("  kinds  " + "  ".join("%s %d" % kv for kv in top)
              + ("  ..." if len(entry["kinds"]) > 5 else ""))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo-root", default=str(REPO_ROOT),
                        help="repository root (default: the checkout this script sits in)")
    parser.add_argument("--out-dir", default=None,
                        help="output folder (default: <repo-root>/build/training_v3)")
    parser.add_argument("--write-short", action="store_true",
                        help="rebuild tracked %s from the drafts first" % SHORT_FILE)
    parser.add_argument("--dry-run", action="store_true",
                        help="validate and report; write nothing")
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else repo_root / "build" / "training_v3"

    try:
        if args.write_short:
            rows, dropped = write_short_corpus(repo_root, dry_run=args.dry_run)
            print("%s: %d rows, %d duplicate responses dropped%s"
                  % (SHORT_FILE, len(rows), dropped, " (dry run)" if args.dry_run else ""))
        manifest = assemble(repo_root, out_dir, dry_run=args.dry_run)
    except AssemblyError as exc:
        print("assemble_training: %s" % exc, file=sys.stderr)
        return 2

    report(manifest)
    if args.dry_run:
        print("\ndry run — nothing written")
    else:
        print("\nwrote %s, %s, %s to %s" % ("phase1_old.jsonl", "phase1_new.jsonl",
                                            "phase2.jsonl", out_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())