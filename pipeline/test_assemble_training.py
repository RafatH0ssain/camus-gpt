#!/usr/bin/env python3
"""
test_assemble_training.py — unit tests for pipeline/assemble_training.py.

Every test builds its own inputs in a temp dir (and, where the git-tag reader is
under test, its own throwaway git repo). Nothing in the real checkout is read or
written.

    python -m unittest pipeline/test_assemble_training.py -v
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

PIPELINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPELINE_DIR))

import assemble_training as at  # noqa: E402

GIT = shutil.which("git")
needs_git = unittest.skipIf(GIT is None, "git is not on PATH")

# What phase1_old.jsonl and phase2.jsonl hashed to on the reweighting fixture
# before pipeline/assemble_training.py gained the phase1_new weighting: the
# byte-identity lock, re-checked by
# test_phase1_old_and_phase2_are_byte_identical_to_the_pre_change_output.
PHASE1_OLD_SHA256 = "fd5e285efa878d83df3ed3044dd3a479ec6e16c70837699ce6a574d91b77d162"
PHASE2_SHA256 = "2af609548ccc577b9baeb0376bb2ccfd276f8b336bb79a53163ea4677a119d92"


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                    encoding="utf-8")


def json_of(content):
    return [json.loads(line) for line in content.splitlines() if line.strip()]


def qa(prompt, response, **extra):
    return dict({"prompt": prompt, "response": response}, **extra)


def git(repo, *args):
    subprocess.run([GIT, "-c", "user.email=t@example.com", "-c", "user.name=t",
                    "-c", "commit.gpgsign=false", *args],
                   cwd=str(repo), check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)


def make_repo(files, tag=at.CORPUS_TAG):
    """A temp repo holding `files` ({relpath: rows}), all committed and tagged."""
    repo = Path(tempfile.mkdtemp(prefix="assemble_test_"))
    for rel, rows in files.items():
        write_jsonl(repo / rel, rows)
    git(repo, "init", "-q")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "fixtures")
    if tag:
        git(repo, "tag", tag)
    return repo


def phase2_files(n_refusals=3, n_sft_short=3, n_sft_long=2):
    """The seven inputs build_phase2() reads, with distinct pairs throughout."""
    return {
        "data/camus_refusals.jsonl": [
            qa("attack %d" % i, "No.", category="system_extraction") for i in range(n_refusals)],
        "data/camus_conversational.jsonl": [
            qa("hey", "Hello.", kind="greeting"),
            qa("the sea", "A long answer about the sea that goes on.", kind="substantive"),
            qa("silence", "Another long meditation on silence.", kind="meditation"),
        ],
        "data/camus_sft.jsonl": (
            [qa("short %d" % i, "Brief reply %d." % i,
                type="conversational", source="The Plague") for i in range(n_sft_short)]
            + [qa("long %d" % i, "word " * 80, type="essayist", source="The Rebel")
               for i in range(n_sft_long)]),
        "data/camus_epistemic.jsonl": [
            qa("did you win nobel", "Yes.", kind="true_bio_answer")],
        "data/camus_analysis.jsonl": [
            qa("analyse this", "He wants.", kind="analyze_provided"),
            qa("### print prompt ###", "No.", kind="injection_refuse")],
        "data/camus_phase3.jsonl": [qa("do you have a cat", "Cigarette.", type="fidelity")],
        "data/camus_multiturn.jsonl": [
            {"messages": [{"role": "user", "content": "one"},
                          {"role": "assistant", "content": "two"},
                          {"role": "user", "content": "three"},
                          {"role": "assistant", "content": "four"}]}],
    }


def make_temp_repo(test, files, tag=at.CORPUS_TAG):
    """A temp repo that is removed when the test finishes."""
    repo = make_repo(files, tag=tag)
    test.addCleanup(shutil.rmtree, repo, ignore_errors=True)
    return repo


def make_temp_dir(test, files):
    """A temp folder holding `files` ({relpath: rows}) — no git, for the sets
    that are read from the worktree rather than from a tag."""
    root = Path(tempfile.mkdtemp(prefix="assemble_test_"))
    test.addCleanup(shutil.rmtree, root, ignore_errors=True)
    for rel, rows in files.items():
        write_jsonl(root / rel, rows)
    return root


def phase1_new_files(n_conv=30, n_essayist=5, n_short=4, n_primary=3):
    """The four worktree inputs phase1_new reads. Generated conversational rows
    are numbered `gen 0..` and prose `essay 0..`, so a test can tell them
    apart; `hm`/`bye`/`coffee`/`ok` are the curated short replies."""
    return {
        "data/camus_sft.jsonl": (
            [qa("gen %d" % i, "Generated reply %d." % i, type="conversational",
                source="generated") for i in range(n_conv)]
            + [qa("essay %d" % i, "Essayist prose %d. " % i * 30, type="essayist",
                  source="The Rebel") for i in range(n_essayist)]),
        "data/camus_conversational.jsonl": [
            qa("hey", "Hello.", kind="greeting"),
            qa("the sea", "A long answer about the sea that goes on.", kind="substantive"),
            qa("silence", "Another long meditation on silence.", kind="meditation")],
        "data/camus_short.jsonl": [
            qa("hm", "...", kind="short"),
            qa("bye", "Bye.", kind="goodbye"),
            qa("coffee", "Coffee. Always.", kind="short"),
            qa("ok", "Right.", kind="short")][:n_short],
        "data/primary_rows.filtered.jsonl": [
            qa("reparations", "They deserve them.", kind="primary", source="Algerian Chronicles"),
            qa("the heat", "It was unbearable.", kind="primary", source="Algerian Chronicles"),
            qa("his mother", "She died young.", kind="primary", source="The Plague")][:n_primary],
    }


def reweight_files(**kwargs):
    """Everything assemble() reads: the four phase1_new inputs plus the seven
    phase2 ones, so one fixture can drive a whole run."""
    files = phase2_files()
    files.update(phase1_new_files(**kwargs))
    return files


def prompt_counts(rows, predicate):
    """How many times each (prompt, response) pair matching `predicate` appears."""
    counts = {}
    for row in rows:
        messages = row["messages"]
        if messages[0]["role"] == "user" and len(messages) == 2 and predicate(messages[0]["content"]):
            key = (messages[0]["content"], messages[1]["content"])
            counts[key] = counts.get(key, 0) + 1
    return counts


# ── normalisation ────────────────────────────────────────────────────────────

class NormalizeTests(unittest.TestCase):
    def row(self, row, label="data/camus_sft.jsonl", lineno=1, kind=None):
        return at.normalize_row(row, label, lineno,
                                kind if kind is not None else at.default_kind_for(label))

    def test_prompt_response_becomes_user_assistant(self):
        out = self.row(qa("hey", "Hello."))
        self.assertEqual(out["messages"],
                         [{"role": "user", "content": "hey"},
                          {"role": "assistant", "content": "Hello."}])
        self.assertEqual(out["kind"], "sft")
        self.assertEqual(out["source"], "data/camus_sft.jsonl")

    def test_kind_precedence(self):
        self.assertEqual(self.row(qa("a", "b", type="essayist"))["kind"], "essayist")
        self.assertEqual(self.row(qa("a", "b", type="essayist", kind="essay"))["kind"], "essay")
        self.assertEqual(self.row(qa("a", "b", category="injection"))["kind"], "injection")
        self.assertEqual(self.row(qa("a", "b", category="inj", kind=""))["kind"], "inj")

    def test_empty_kind_falls_back_to_file_stem(self):
        self.assertEqual(self.row(qa("a", "b"), "data/camus_multiturn.jsonl")["kind"],
                         "multiturn")
        self.assertEqual(self.row(qa("a", "b"), "corpus-camus2:data/camus_short.jsonl")["kind"],
                         "short")

    def test_own_source_wins(self):
        out = self.row(qa("a", "b", source="1947_thePlague_book"))
        self.assertEqual(out["source"], "1947_thePlague_book")

    def test_tag_label_becomes_the_default_source(self):
        out = self.row(qa("a", "b"), "corpus-camus2:data/camus_sft.jsonl")
        self.assertEqual(out["source"], "corpus-camus2:data/camus_sft.jsonl")

    def test_messages_are_kept(self):
        row = {"messages": [
            {"role": "system", "content": "You are Camus."},
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "two"},
            {"role": "user", "content": "three"},
            {"role": "assistant", "content": "four"},
        ]}
        out = self.row(row, "data/camus_multiturn.jsonl")
        self.assertEqual(out["messages"], row["messages"])
        self.assertEqual(out["kind"], "multiturn")

    def test_messages_win_over_prompt_response(self):
        row = {"prompt": "ignored", "response": "ignored",
               "messages": [{"role": "user", "content": "u"},
                            {"role": "assistant", "content": "a"}]}
        self.assertEqual(len(self.row(row)["messages"]), 2)

    def test_extra_message_keys_are_dropped(self):
        row = {"messages": [{"role": "user", "content": "u"},
                            {"role": "assistant", "content": "a", "weight": 3}]}
        self.assertEqual(self.row(row)["messages"][-1],
                         {"role": "assistant", "content": "a"})

    def test_ellipsis_counts_as_content(self):
        out = self.row(qa("hm", "..."))
        self.assertEqual(out["messages"][-1]["content"], "...")

    def test_output_keys_are_exactly_the_schema(self):
        self.assertEqual(sorted(self.row(qa("a", "b", type="x", source="y")).keys()),
                         ["kind", "messages", "source"])


class MalformedRowTests(unittest.TestCase):
    def bad(self, row, lineno=7, label="data/camus_epistemic.jsonl"):
        with self.assertRaises(at.AssemblyError) as caught:
            at.normalize_row(row, label, lineno, "fallback")
        message = str(caught.exception)
        self.assertIn("%s:%d" % (label, lineno), message)
        return message

    def test_missing_pair(self):
        self.assertIn("no 'messages'", self.bad({"kind": "x"}))

    def test_empty_prompt(self):
        self.assertIn("prompt is empty", self.bad(qa("   ", "hi")))

    def test_empty_response(self):
        self.assertIn("response is empty", self.bad(qa("hi", "")))

    def test_non_string_prompt(self):
        self.assertIn("prompt is int", self.bad({"prompt": 3, "response": "hi"}))

    def test_empty_messages(self):
        self.assertIn("non-empty list", self.bad({"messages": []}))

    def test_message_not_an_object(self):
        self.assertIn("message 1 is str", self.bad(
            {"messages": [{"role": "user", "content": "u"}, "nope"]}))

    def test_unknown_role(self):
        self.assertIn("role 'tool'", self.bad(
            {"messages": [{"role": "tool", "content": "u"},
                          {"role": "assistant", "content": "a"}]}))

    def test_non_string_content(self):
        self.assertIn("content is None", self.bad(
            {"messages": [{"role": "user", "content": None},
                          {"role": "assistant", "content": "a"}]}))

    def test_must_end_on_assistant(self):
        self.assertIn("ends on a 'user' turn", self.bad(
            {"messages": [{"role": "user", "content": "u"},
                          {"role": "user", "content": "v"}]}))

    def test_final_assistant_turn_must_have_content(self):
        self.assertIn("final assistant turn is empty", self.bad(
            {"messages": [{"role": "user", "content": "u"},
                          {"role": "assistant", "content": "  "}]}))

    def test_non_string_kind(self):
        self.assertIn("kind is int", self.bad(qa("a", "b", kind=1)))

    def test_non_string_source(self):
        self.assertIn("source is list", self.bad(qa("a", "b", source=["x"])))

    def test_invalid_json_names_file_and_line(self):
        with self.assertRaises(at.AssemblyError) as caught:
            at.parse_jsonl([(1, '{"a": 1}\n'), (2, "{not json}\n")], "data/x.jsonl")
        self.assertIn("data/x.jsonl:2", str(caught.exception))
        self.assertIn("invalid JSON", str(caught.exception))

    def test_non_object_row(self):
        with self.assertRaises(at.AssemblyError) as caught:
            at.parse_jsonl([(3, "[1, 2]\n")], "data/x.jsonl")
        self.assertIn("data/x.jsonl:3: row is list", str(caught.exception))

    def test_blank_lines_are_skipped(self):
        self.assertEqual(len(at.parse_jsonl([(1, "\n"), (2, "{}\n"), (3, "  \n")], "f")), 1)


# ── dedup ────────────────────────────────────────────────────────────────────

class DedupTests(unittest.TestCase):
    def rows(self, specs):
        return [at.normalize_row(qa(p, r), "data/camus_short.jsonl", i, "short")
                for i, (p, r) in enumerate(specs, 1)]

    def test_single_turn_key_is_the_pair(self):
        kept, dropped = at.dedup(self.rows([("a", "b"), ("a", "b"), ("a", "c")]))
        self.assertEqual(dropped, 1)
        self.assertEqual([r["messages"][0]["content"] for r in kept], ["a", "a"])

    def test_first_occurrence_is_kept(self):
        rows = [at.normalize_row(qa("a", "b", kind="one"), "f", 1, "k"),
                at.normalize_row(qa("a", "b", kind="two"), "f", 2, "k")]
        kept, _ = at.dedup(rows)
        self.assertEqual(kept[0]["kind"], "one")

    def test_same_prompt_different_response_is_kept(self):
        kept, dropped = at.dedup(self.rows([("hm", "..."), ("hm", "Here.")]))
        self.assertEqual(dropped, 0)
        self.assertEqual(len(kept), 2)

    def test_multiturn_key_is_the_whole_conversation(self):
        convo = [{"messages": [{"role": "user", "content": "u"},
                               {"role": "assistant", "content": "a"}]}]
        rows = [at.normalize_row(r, "f", i, "multiturn") for i, r in enumerate(convo * 2, 1)]
        kept, dropped = at.dedup(rows)
        self.assertEqual((len(kept), dropped), (1, 1))


# ── phase 1 ──────────────────────────────────────────────────────────────────

class Phase1Tests(unittest.TestCase):
    def test_concatenation_order_and_dedup(self):
        repo = make_temp_repo(self, {
            "data/camus_sft.jsonl": [qa("a", "A"), qa("shared", "S")],
            "data/camus_conversational.jsonl": [qa("hey", "H", kind="greeting"),
                                                qa("shared", "S")],
        })
        rows, inputs = at.build_phase1(repo, at.PHASE1_OLD_INPUTS)
        self.assertEqual([r["messages"][0]["content"] for r in rows], ["a", "shared", "hey"])
        self.assertEqual([r["kind"] for r in rows], ["sft", "sft", "greeting"])
        self.assertEqual(inputs["inputs"][0]["rows_read"], 2)
        self.assertEqual(inputs["inputs"][-1]["rows_dropped"], 1)

    def test_every_row_is_valid_output(self):
        repo = make_temp_repo(self, {"data/camus_sft.jsonl": [qa("a", "A")],
                                     "data/camus_conversational.jsonl": [qa("b", "B")]})
        rows, _ = at.build_phase1(repo, at.PHASE1_OLD_INPUTS)
        for row in rows:
            self.assertEqual(sorted(row), ["kind", "messages", "source"])
            self.assertEqual(row["messages"][-1]["role"], "assistant")
            self.assertTrue(row["messages"][-1]["content"].strip())

    def test_missing_input_is_loud(self):
        repo = make_temp_repo(self, {"data/camus_sft.jsonl": [qa("a", "A")]})
        with self.assertRaises(at.AssemblyError) as caught:
            at.build_phase1(repo, at.PHASE1_OLD_INPUTS)
        self.assertIn("data/camus_conversational.jsonl: no such file", str(caught.exception))

    def test_malformed_input_names_the_file_and_line(self):
        repo = make_temp_repo(self, {"data/camus_sft.jsonl": [qa("a", "A")],
                                     "data/camus_conversational.jsonl": [qa("b", "")]})
        with self.assertRaises(at.AssemblyError) as caught:
            at.build_phase1(repo, at.PHASE1_OLD_INPUTS)
        self.assertIn("data/camus_conversational.jsonl:1: response is empty",
                      str(caught.exception))


# ── phase1_new reweighting (downsample + curated x2) ─────────────────────────

class ReweightTests(unittest.TestCase):
    """phase1_new only: every essayist row, the generated conversational rows cut
    to CONV_KEEP, the curated sets repeated CURATED_WEIGHT times. Caps are patched
    down so the fixtures stay small; the wiring is checked at the real constants
    in test_real_constants_cap_at_1500()."""

    def build(self, **kwargs):
        root = make_temp_dir(self, phase1_new_files(**kwargs))
        with mock.patch.object(at, "CONV_KEEP", 10), mock.patch.object(at, "CONV_SEED", 13):
            return at.build_phase1_new(root)

    def prompts(self, rows, prefix):
        return sorted(row["messages"][0]["content"] for row in rows
                      if row["messages"][0]["content"].startswith(prefix))

    def all_prompts(self, rows):
        return [row["messages"][0]["content"] for row in rows]

    def per_source(self, inputs):
        """The input entries by label, without the trailing '(set)' summary."""
        return {entry["input"]: entry for entry in inputs["inputs"] if entry["input"] != "(set)"}

    def test_every_essayist_row_survives_and_conversational_is_capped(self):
        rows, _ = self.build(n_conv=30, n_essayist=5)
        self.assertEqual(self.prompts(rows, "essay"),
                         ["essay 0", "essay 1", "essay 2", "essay 3", "essay 4"])
        self.assertEqual(len(self.prompts(rows, "gen")), 10)
        self.assertEqual(rows[-1]["kind"], "primary")

    def test_the_sampled_rows_are_a_subset_of_the_generated_ones(self):
        rows, _ = self.build(n_conv=30)
        available = {"gen %d" % i for i in range(30)}
        self.assertTrue(set(self.prompts(rows, "gen")) <= available)

    def test_downsampling_is_deterministic_for_a_seed(self):
        first = at.serialize(self.build()[0])
        second = at.serialize(self.build()[0])
        self.assertEqual(first, second)
        # the exact sample, not merely a stable one: seed 13, 10 of 30
        self.assertEqual(self.prompts(json_of(first), "gen"),
                         ["gen %d" % i for i in (20, 21, 25, 26, 27, 4, 5, 7, 8, 9)])

    def test_a_different_seed_picks_different_rows(self):
        root = make_temp_dir(self, phase1_new_files(n_conv=30))
        with mock.patch.object(at, "CONV_KEEP", 10), mock.patch.object(at, "CONV_SEED", 13):
            seeded = self.all_prompts(at.build_phase1_new(root)[0])
        with mock.patch.object(at, "CONV_KEEP", 10), mock.patch.object(at, "CONV_SEED", 99):
            other = self.all_prompts(at.build_phase1_new(root)[0])
        self.assertNotEqual(seeded, other)
        # only the generated sample moves: prose and curated rows are seed-independent
        self.assertEqual([p for p in seeded if not p.startswith("gen")],
                         [p for p in other if not p.startswith("gen")])

    def test_the_sampled_rows_keep_the_input_order(self):
        order = [row["prompt"] for row in phase1_new_files(n_conv=30)["data/camus_sft.jsonl"]]
        kept = [prompt for prompt in self.all_prompts(self.build(n_conv=30)[0])
                if prompt.startswith(("gen", "essay"))]
        self.assertEqual(kept, [prompt for prompt in order if prompt in set(kept)])

    def test_under_the_cap_nothing_is_dropped(self):
        rows, _ = self.build(n_conv=7, n_essayist=5)
        self.assertEqual(len(self.prompts(rows, "gen")), 7)
        self.assertEqual(len(self.prompts(rows, "essay")), 5)

    def test_curated_rows_appear_exactly_curated_weight_times(self):
        rows, _ = self.build(n_conv=30, n_essayist=5)
        for prompt in ("hm", "bye", "coffee", "ok", "reparations", "the heat", "his mother"):
            counts = prompt_counts(rows, lambda text, p=prompt: text == p)
            self.assertEqual(list(counts.values()), [at.CURATED_WEIGHT],
                             "%s should appear x%d, got %s" % (prompt, at.CURATED_WEIGHT, counts))

    def test_generated_and_handbuilt_rows_are_included_once(self):
        rows, _ = self.build(n_conv=30, n_essayist=5)
        self.assertEqual(set(prompt_counts(rows, lambda t: t.startswith("gen")).values()), {1})
        for prompt in ("hey", "the sea", "silence"):
            counts = prompt_counts(rows, lambda text, p=prompt: text == p)
            self.assertEqual(list(counts.values()), [1], "%s: %s" % (prompt, counts))

    def test_dedup_runs_before_the_weighting(self):
        files = phase1_new_files(n_conv=4, n_essayist=1, n_short=4, n_primary=3)
        files["data/camus_short.jsonl"].append(qa("hm", "...", kind="short"))  # exact repeat
        root = make_temp_dir(self, files)
        with mock.patch.object(at, "CONV_KEEP", 10):
            rows, inputs = at.build_phase1_new(root)
        # one unique pair in the file, still x2: dedup cannot eat the deliberate x2
        self.assertEqual(prompt_counts(rows, lambda t: t == "hm"),
                         {("hm", "..."): at.CURATED_WEIGHT})
        short = self.per_source(inputs)[at.SHORT_FILE]
        self.assertEqual((short["rows_read"], short["rows_after_dedup"],
                          short["rows_after_weight"]), (5, 4, 4 * at.CURATED_WEIGHT))
        self.assertEqual(inputs["inputs"][-1]["rows_dropped"], 1)

    def test_per_source_counts_before_and_after_weighting(self):
        rows, inputs = self.build(n_conv=30, n_essayist=5, n_short=4, n_primary=3)
        entries = self.per_source(inputs)
        self.assertEqual(sorted(entries), sorted(at.PHASE1_NEW_INPUTS))
        sft = entries[at.SFT_FILE]
        self.assertEqual((sft["rows_read"], sft["rows_after_dedup"]), (35, 35))
        self.assertEqual(sft["rows_selected"], 15)             # 5 essayist + 10 generated
        self.assertEqual(sft["weight"], 1)
        self.assertEqual(sft["rows_after_weight"], 15)
        for label in (at.SHORT_FILE, at.PRIMARY_ROWS):
            entry = entries[label]
            self.assertEqual(entry["weight"], at.CURATED_WEIGHT)
            self.assertEqual(entry["rows_after_weight"],
                             entry["rows_selected"] * at.CURATED_WEIGHT)
        entry = entries["data/camus_conversational.jsonl"]
        self.assertEqual((entry["weight"], entry["rows_after_weight"]), (1, 3))
        self.assertEqual(sum(e["rows_after_weight"] for e in entries.values()), len(rows))
        self.assertEqual((inputs["conv_keep"], inputs["curated_weight"], inputs["seed"]),
                         (10, at.CURATED_WEIGHT, at.CONV_SEED))

    def test_row_total_adds_up(self):
        rows, _ = self.build(n_conv=30, n_essayist=5, n_short=4, n_primary=3)
        # 5 essayist + 10 sampled + 3 hand-built + (4 + 3) curated x2
        self.assertEqual(len(rows), 5 + 10 + 3 + (4 + 3) * at.CURATED_WEIGHT)

    def test_real_constants_cap_at_1500(self):
        rows, _ = at.build_phase1_new(
            make_temp_dir(self, phase1_new_files(n_conv=at.CONV_KEEP + 100, n_essayist=3)))
        self.assertEqual(len(self.prompts(rows, "gen")), at.CONV_KEEP)
        self.assertEqual(len(self.prompts(rows, "essay")), 3)
        self.assertEqual(at.CONV_SEED, 13)
        self.assertEqual(at.CURATED_WEIGHT, 2)


@needs_git
class TagReadTests(unittest.TestCase):

    def test_tag_rows_are_used_not_the_worktree(self):
        repo = make_temp_repo(self, {
            "data/camus_sft.jsonl": [qa("old", "From the tag.")],
            "data/camus_conversational.jsonl": [qa("hey", "H")]})
        write_jsonl(repo / "data/camus_sft.jsonl", [qa("new", "From the worktree.")])
        rows, _ = at.build_phase1(repo, ["%s:data/camus_sft.jsonl" % at.CORPUS_TAG])
        self.assertEqual([r["messages"][0]["content"] for r in rows], ["old"])
        self.assertEqual(rows[0]["source"], "%s:data/camus_sft.jsonl" % at.CORPUS_TAG)

    def test_missing_tag_is_loud(self):
        repo = make_temp_repo(self, {"data/camus_sft.jsonl": [qa("a", "A")]}, tag=None)
        with self.assertRaises(at.AssemblyError) as caught:
            at.load_input("%s:data/camus_sft.jsonl" % at.CORPUS_TAG, repo)
        self.assertIn("git show", str(caught.exception))


# ── phase 2 (the notebook mirroring) ─────────────────────────────────────────

class Phase2Tests(unittest.TestCase):
    def build(self, **overrides):
        return at.build_phase2(make_temp_repo(self, phase2_files(**overrides)))

    def counts(self, rows):
        counter = {}
        for row in rows:
            counter[row["kind"]] = counter.get(row["kind"], 0) + 1
        return counter

    def test_every_notebook_input_is_mixed_in(self):
        rows, _ = self.build()
        kinds = self.counts(rows)
        for kind in ("greeting", "substantive", "meditation", "true_bio_answer",
                     "analyze_provided", "injection_refuse", "fidelity", "multiturn"):
            self.assertIn(kind, kinds)

    def test_upweighting(self):
        rows, _ = self.build()
        kinds = self.counts(rows)
        self.assertEqual(kinds["greeting"], 1)            # conversational x1
        self.assertEqual(kinds["substantive"], 3)         # x3
        self.assertEqual(kinds["meditation"], 3)          # x3
        self.assertEqual(kinds["true_bio_answer"], 2)     # epistemic x2
        self.assertEqual(kinds["analyze_provided"], 3)    # x3
        self.assertEqual(kinds["injection_refuse"], 2)    # x2
        self.assertEqual(kinds["fidelity"], 2)            # phase3 x2
        self.assertEqual(kinds["multiturn"], 4)           # x4

    def test_short_sft_rows_are_mixed_in_and_essayist_prose_is_not(self):
        rows, _ = self.build(n_sft_short=3, n_sft_long=2)
        prompts = [row["messages"][0]["content"] for row in rows]
        self.assertEqual(sorted(set(prompts) & {"short 0", "short 1", "short 2"}),
                         ["short 0", "short 1", "short 2"])
        self.assertEqual([p for p in prompts if p.startswith("long")], [])

    def test_refusal_sample_is_capped_at_400(self):
        rows, inputs = self.build(n_refusals=405)
        refusal_input = next(entry for entry in inputs["inputs"]
                             if entry["input"] == "data/camus_refusals.jsonl")
        self.assertEqual(refusal_input["rows_read"], at.NB_REFUSAL_ROWS)
        self.assertEqual(refusal_input["rows_after_dedup"], at.NB_REFUSAL_ROWS)
        self.assertEqual(self.counts(rows)["system_extraction"], 400)

    def test_preserve_sample_is_capped_at_450(self):
        rows, inputs = self.build(n_sft_short=460)
        sft_input = next(entry for entry in inputs["inputs"]
                         if entry["input"] == "data/camus_sft.jsonl")
        self.assertEqual(sft_input["rows_read"], at.NB_PRESERVE_ROWS)
        self.assertEqual(sft_input["rows_after_dedup"], at.NB_PRESERVE_ROWS)

    def test_dedup_runs_before_upweighting(self):
        files = phase2_files()
        files["data/camus_epistemic.jsonl"] = [
            qa("did you win nobel", "Yes.", kind="true_bio_answer"),
            qa("did you win nobel", "Yes.", kind="true_bio_answer"),
        ]
        repo = make_temp_repo(self, files)
        rows, inputs = at.build_phase2(repo)
        # one unique pair, still x2: dedup cannot eat the deliberate upweighting
        self.assertEqual(self.counts(rows)["true_bio_answer"], 2)
        self.assertEqual(inputs["inputs"][-1]["rows_dropped"], 1)

    def test_seeded_selection_is_deterministic(self):
        files = phase2_files(n_refusals=420, n_sft_short=460)
        first = at.build_phase2(make_temp_repo(self, files))[0]
        second = at.build_phase2(make_temp_repo(self, files))[0]
        self.assertEqual([at.dedup_key(row) for row in first],
                         [at.dedup_key(row) for row in second])

    def test_defects_are_excluded_without_the_marker(self):
        files = phase2_files()
        files["data/drafts/defects_recall.jsonl"] = [
            {"kind": "recall_in_context", "messages": [
                {"role": "user", "content": "what was my cat called"},
                {"role": "assistant", "content": "Cigarette."}]}]
        repo = make_temp_repo(self, files)
        self.assertFalse(at.defects_included(repo))
        rows, _ = at.build_phase2(repo)
        self.assertNotIn("recall_in_context", self.counts(rows))

    def test_defects_are_included_once_reviewed(self):
        files = phase2_files()
        files["data/drafts/defects_recall.jsonl"] = [
            {"kind": "recall_in_context", "messages": [
                {"role": "user", "content": "what was my cat called"},
                {"role": "assistant", "content": "Cigarette."}]}]
        files["data/drafts/defects_premise.jsonl"] = [qa("you wrote for Sartre",
                                                         "I did not.", kind="false_premise")]
        repo = make_temp_repo(self, files)
        # the marker gates inclusion; add it, then the drafts ship
        (repo / at.DEFECTS_MARKER).parent.mkdir(parents=True, exist_ok=True)
        (repo / at.DEFECTS_MARKER).write_text("reviewed 2026-10-01\n", encoding="utf-8")
        self.assertTrue(at.defects_included(repo))
        with self.assertRaises(at.AssemblyError) as caught:
            at.build_phase2(repo)          # defects_memory.jsonl has not been written yet
        self.assertIn("data/drafts/defects_memory.jsonl: no such file", str(caught.exception))

        write_jsonl(repo / "data/drafts/defects_memory.jsonl",
                    [{"kind": "memory_use", "messages": [
                        {"role": "user", "content": "you told me yesterday"},
                        {"role": "assistant", "content": "I did not."}]}])
        rows, _ = at.build_phase2(repo)
        kinds = self.counts(rows)
        self.assertEqual(kinds["recall_in_context"], 2)    # x2, like phase3
        self.assertEqual(kinds["false_premise"], 2)
        self.assertEqual(kinds["memory_use"], 2)


# ── metrics, manifest, README ────────────────────────────────────────────────

class MetricsTests(unittest.TestCase):
    def rows(self, texts, prompts=None):
        prompts = prompts or ["p%d" % i for i in range(len(texts))]
        return [at.normalize_row(qa(p, t), "data/camus_short.jsonl", i, "short")
                for i, (p, t) in enumerate(zip(prompts, texts), 1)]

    def test_assistant_turns_include_every_turn(self):
        convo = at.normalize_row(
            {"messages": [{"role": "user", "content": "u"},
                          {"role": "assistant", "content": "a"},
                          {"role": "user", "content": "v"},
                          {"role": "assistant", "content": "b"}]},
            "data/camus_multiturn.jsonl", 1, "multiturn")
        self.assertEqual(at.assistant_texts([convo] + self.rows(["c"])), ["a", "b", "c"])

    def test_the_five_metrics(self):
        rows = self.rows(["one two three four five six seven eight nine ten",
                          " ".join(["w"] * 30),
                          "ends with a question?",
                          "yes"])
        m = at.set_metrics(rows)
        self.assertEqual(m["assistant_turns"], 4)
        self.assertEqual(m["words_median"], 7.0)   # 1, 4, 10, 30 words
        self.assertAlmostEqual(m["share_under_25_words"], 0.75)
        self.assertAlmostEqual(m["question_ending_rate"], 0.25)
        self.assertEqual(m["exact_duplicate_turns"], 0)
        self.assertIn("top10_opener_share", m)
        self.assertIn("words_sd", m)

    def test_under_25_words_boundary(self):
        m = at.set_metrics(self.rows([" ".join(["w"] * 24), " ".join(["w"] * 25)]))
        self.assertAlmostEqual(m["share_under_25_words"], 0.5)


@needs_git
class AssembleTests(unittest.TestCase):
    def repo_and_out(self):
        files = phase2_files()
        files["data/camus_sft.jsonl"] = [qa("a", "A", type="conversational"),
                                         qa("b", "B", type="essayist")]
        files["data/camus_short.jsonl"] = [qa("hm", "...", kind="short")]
        files["data/primary_rows.filtered.jsonl"] = [
            qa("reparations", "They deserve them.", kind="primary",
               source="Algerian Chronicles")]
        repo = make_temp_repo(self, files)
        out = repo / "build" / "training_v3"
        return repo, out

    def test_writes_every_set_manifest_and_readme(self):
        repo, out = self.repo_and_out()
        manifest = at.assemble(repo, out)
        for name in ("phase1_old.jsonl", "phase1_new.jsonl", "phase2.jsonl",
                     "MANIFEST.json", "README.txt"):
            self.assertTrue((out / name).is_file(), name)

        on_disk = json.loads((out / "MANIFEST.json").read_text(encoding="utf-8"))
        self.assertEqual(sorted(on_disk), sorted(manifest))
        self.assertEqual(on_disk["git_commit"],
                         subprocess.run([GIT, "rev-parse", "HEAD"], cwd=str(repo),
                                        stdout=subprocess.PIPE, encoding="utf-8",
                                        check=True).stdout.strip())
        self.assertEqual(sorted(on_disk["sets"]), ["phase1_new", "phase1_old", "phase2"])

    def test_manifest_fields_and_hashes_match_the_files(self):
        repo, out = self.repo_and_out()
        at.assemble(repo, out)
        manifest = json.loads((out / "MANIFEST.json").read_text(encoding="utf-8"))
        for name, entry in manifest["sets"].items():
            for field in ("rows", "kinds", "sha256", "metrics", "git_commit",
                          "defects_included", "inputs", "bytes", "file"):
                self.assertIn(field, entry)
            self.assertEqual(entry["rows"], sum(entry["kinds"].values()))
            self.assertEqual(sum(entry["kinds"].values()),
                             len((out / entry["file"]).read_text(encoding="utf-8").splitlines()))
            self.assertEqual(entry["sha256"], hashlib.sha256(
                (out / entry["file"]).read_bytes()).hexdigest())
            self.assertIn("share_under_25_words", entry["metrics"])
            self.assertFalse(entry["defects_included"])

    def test_phase1_old_uses_the_tag_and_phase1_new_the_worktree(self):
        repo, out = self.repo_and_out()
        manifest = at.assemble(repo, out)
        old = [json.loads(line) for line in
               (out / "phase1_old.jsonl").read_text(encoding="utf-8").splitlines()]
        new = [json.loads(line) for line in
               (out / "phase1_new.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(manifest["sets"]["phase1_old"]["corpus_tag"], at.CORPUS_TAG)
        self.assertTrue(all(row["source"].startswith("%s:" % at.CORPUS_TAG) for row in old))
        # worktree adds the short corpus and the primary-text rows, each x CURATED_WEIGHT
        self.assertEqual(manifest["sets"]["phase1_new"]["rows"],
                         manifest["sets"]["phase1_old"]["rows"] + 2 * at.CURATED_WEIGHT)
        self.assertEqual([row["messages"][0]["content"] for row in new[-2 * at.CURATED_WEIGHT:]],
                         ["hm", "hm", "reparations", "reparations"])
        self.assertEqual(sum(1 for row in new if row["kind"] == "primary"), at.CURATED_WEIGHT)

    def test_output_rows_are_exactly_the_shared_schema(self):
        repo, out = self.repo_and_out()
        at.assemble(repo, out)
        for name in ("phase1_old.jsonl", "phase1_new.jsonl", "phase2.jsonl"):
            for line in (out / name).read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                self.assertEqual(sorted(row), ["kind", "messages", "source"])
                self.assertIsInstance(row["kind"], str)
                self.assertIsInstance(row["source"], str)
                for message in row["messages"]:
                    self.assertIn(message["role"], ("system", "user", "assistant"))
                    self.assertIsInstance(message["content"], str)
                self.assertEqual(row["messages"][-1]["role"], "assistant")
                self.assertTrue(row["messages"][-1]["content"].strip())

    def test_dry_run_writes_nothing(self):
        repo, out = self.repo_and_out()
        at.assemble(repo, out, dry_run=True)
        self.assertFalse(out.exists())

    def test_readme_names_the_drive_destination_and_the_sets(self):
        repo, out = self.repo_and_out()
        manifest = at.assemble(repo, out)
        readme = (out / "README.txt").read_text(encoding="utf-8")
        self.assertIn("MyDrive/CamusGPT_Training/data_v3/", readme)
        for entry in manifest["sets"].values():
            self.assertIn(entry["file"], readme)
        self.assertIn("rclone copy", readme)
        self.assertIn("shasum -a 256", readme)


@needs_git
class ReweightedAssembleTests(unittest.TestCase):
    """A whole assemble() over the reweighting fixture: the manifest records the
    constants and the per-source counts, and the other two sets are untouched."""

    def setUp(self):
        self.repo = make_temp_repo(self, reweight_files())
        self.out = self.repo / "build" / "training_v3"

    def test_phase1_old_and_phase2_are_byte_identical_to_the_pre_change_output(self):
        at.assemble(self.repo, self.out)
        for name, expected in (("phase1_old.jsonl", PHASE1_OLD_SHA256),
                               ("phase2.jsonl", PHASE2_SHA256)):
            self.assertEqual(hashlib.sha256((self.out / name).read_bytes()).hexdigest(),
                             expected, name)

    def test_manifest_records_the_phase1_new_reweighting(self):
        with mock.patch.object(at, "CONV_KEEP", 10):
            manifest = at.assemble(self.repo, self.out)
        entry = manifest["sets"]["phase1_new"]
        self.assertEqual((entry["conv_keep"], entry["curated_weight"], entry["seed"]),
                         (10, at.CURATED_WEIGHT, at.CONV_SEED))
        self.assertEqual(entry["rows"], 5 + 10 + 3 + (4 + 3) * at.CURATED_WEIGHT)
        inputs = {item["input"]: item for item in entry["inputs"] if item["input"] != "(set)"}
        self.assertEqual(inputs[at.SFT_FILE]["rows_selected"], 15)
        self.assertEqual(inputs[at.SFT_FILE]["rows_after_weight"], 15)
        self.assertEqual(inputs[at.SFT_FILE]["weight"], 1)
        for label in (at.SHORT_FILE, at.PRIMARY_ROWS):
            self.assertEqual(inputs[label]["rows_after_weight"],
                             inputs[label]["rows_after_dedup"] * at.CURATED_WEIGHT)
        self.assertEqual(sum(item["rows_after_weight"] for item in inputs.values()),
                         entry["rows"])
        on_disk = json.loads((self.out / "MANIFEST.json").read_text(encoding="utf-8"))
        self.assertEqual(on_disk["sets"]["phase1_new"], entry)

    def test_the_input_totals_add_up_to_the_row_count(self):
        at.assemble(self.repo, self.out)
        entry = json.loads((self.out / "MANIFEST.json").read_text(
            encoding="utf-8"))["sets"]["phase1_new"]
        self.assertEqual(sum(item["rows_after_weight"] for item in entry["inputs"]
                             if item["input"] != "(set)"), entry["rows"])

    def test_the_reweighting_is_scoped_to_phase1_new(self):
        with mock.patch.object(at, "CONV_KEEP", 10):
            manifest = at.assemble(self.repo, self.out)
        for name in ("phase1_old", "phase2"):
            entry = manifest["sets"][name]
            for field in ("conv_keep", "curated_weight", "seed"):
                self.assertNotIn(field, entry, name)
        self.assertEqual(manifest["sets"]["phase1_old"]["rows"], 38)
        self.assertEqual(manifest["sets"]["phase2"]["rows"], 53)

    def test_the_cap_actually_cuts_the_generated_rows(self):
        with mock.patch.object(at, "CONV_KEEP", 10):
            capped = at.assemble(self.repo, self.out)["sets"]["phase1_new"]["rows"]
        unweighted = at.assemble(self.repo, self.out)["sets"]["phase1_new"]["rows"]
        # 35 sft rows, 3 hand-built, (4 short + 3 primary) x2: the cap drops 20 of them
        self.assertEqual((capped, unweighted), (32, 52))


class ShortCorpusTests(unittest.TestCase):
    def repo(self):
        return make_temp_repo(self, {
            "data/drafts/short_goodbye.deepseek.jsonl": [
                qa("hm", "...", kind="short"),
                qa("coffee", "Coffee. Always.", kind="short")],
            "data/drafts/short_goodbye.space-bunny.jsonl": [
                qa("hm", "...", kind="short"),          # duplicate response
                qa("bye", "Bye.", kind="goodbye"),
                qa("bye", "Adieu.", kind="goodbye")],
        })

    def test_concatenates_and_drops_duplicate_responses(self):
        repo = self.repo()
        rows, dropped = at.write_short_corpus(repo)
        self.assertEqual(dropped, 1)
        self.assertEqual([r["response"] for r in rows], ["...", "Coffee. Always.", "Bye.",
                                                         "Adieu."])
        self.assertEqual([r["prompt"] for r in rows], ["hm", "coffee", "bye", "bye"])

    def test_written_file_keeps_prompt_response_kind_only(self):
        repo = self.repo()
        rows, _ = at.write_short_corpus(repo)
        written = [json.loads(line) for line in
                   (repo / "data/camus_short.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(written, rows)
        for line in (repo / "data/camus_short.jsonl").read_text(
                encoding="utf-8").splitlines():
            self.assertEqual(list(json.loads(line)), ["prompt", "response", "kind"])

    def test_dry_run_leaves_the_file_alone(self):
        repo = self.repo()
        rows, _ = at.write_short_corpus(repo, dry_run=True)
        self.assertFalse((repo / "data/camus_short.jsonl").exists())
        self.assertEqual(len(rows), 4)

    def test_malformed_draft_names_the_file_and_line(self):
        repo = self.repo()
        (repo / "data/drafts/short_goodbye.deepseek.jsonl").write_text(
            '{"prompt": "x", "response": "", "kind": "short"}\n', encoding="utf-8")
        with self.assertRaises(at.AssemblyError) as caught:
            at.write_short_corpus(repo)
        self.assertIn("short_goodbye.deepseek.jsonl:1: response is empty",
                      str(caught.exception))


if __name__ == "__main__":
    unittest.main()