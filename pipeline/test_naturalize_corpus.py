#!/usr/bin/env python3
"""
test_naturalize_corpus.py — unit tests for pipeline/naturalize_corpus.py.

Runs entirely on small JSONL fixtures written to a temp dir, so the real corpus
in ./data is never read or written. Filenames are the real ones, because the
per-file edit rules (conversational-only in camus_sft.jsonl) are keyed off them.

    python -m unittest pipeline/test_naturalize_corpus.py -v
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import naturalize_corpus as nc  # noqa: E402

SFT = "camus_sft.jsonl"
CONV = "camus_conversational.jsonl"
REJECTED = "camus_multiturn.jsonl"

ESSAYIST = "Yes, and I think the absurd is the point of it all. Is it not?"
UNCHANGED = "The sea was calm. Nothing needed to be said about it."


def write_jsonl(path, rows):
    body = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    path.write_bytes(body.encode("utf-8"))
    return body


def read_lines(path):
    return path.read_bytes().decode("utf-8").split("\n")[:-1]


def conv_row(response, **extra):
    row = {"type": "conversational", "prompt": "p", "response": response,
           "source": "s", "src_id": "id"}
    row.update(extra)
    return row


def essay_row(response):
    return {"type": "essayist", "prompt": "p", "response": response,
            "source": "s", "src_id": "id"}


def process(path, seed=13):
    type_filter = dict(nc.FILES)[path.name]
    return nc.process_file(path, seed, type_filter)


def rewrite(path, seed=13):
    stats = process(path, seed)
    path.write_bytes(stats["content"].encode("utf-8"))
    return stats


class TransformTests(unittest.TestCase):
    def test_t1_yes_and_drops_prefix(self):
        self.assertEqual(nc.t1_apply("Yes, and the sea is cold."),
                         "The sea is cold.")

    def test_t1_no_and_becomes_sentence(self):
        self.assertEqual(nc.t1_apply("No, and that is all I know."),
                         "No. That is all I know.")

    def test_t1_rarely_and_becomes_sentence(self):
        self.assertEqual(nc.t1_apply("Rarely, and only in winter."),
                         "Rarely. Only in winter.")

    def test_t1_ignores_other_openers(self):
        for text in ("Yes, I agree.", "No, it is not.", "Rarely seen here.",
                     "And so it goes.", ""):
            self.assertIsNone(nc.t1_apply(text), text)

    def test_t2_candidate_skips_comma_after_i_think(self):
        self.assertFalse(nc.t2_is_candidate("I think, maybe we should go."))
        self.assertIsNone(nc.t2_apply("I think, maybe we should go."))
        self.assertTrue(nc.t2_is_candidate("I think we should go."))

    def test_t2_never_strips_that(self):
        text = "I think that is precisely the condition. Nothing more."
        self.assertFalse(nc.t2_is_candidate(text))
        self.assertIsNone(nc.t2_apply(text))
        self.assertEqual(nc.t2_apply("I think that the sea is cold."),
                         None)
        self.assertEqual(nc.t2_apply("I think we should go."), "We should go.")

    def test_t3_single_sentence_question_untouched(self):
        text = "Do you want to go?"
        self.assertIsNone(nc.t3_apply(text))
        self.assertEqual(nc.t3_apply(text), None)

    def test_t3_remainder_under_four_words_untouched(self):
        text = "It is cold. Why?"
        self.assertEqual(len("It is cold.".split()), 3)
        self.assertIsNone(nc.t3_apply(text))

    def test_t3_drops_trailing_question_sentence(self):
        text = "The sea was calm that morning. Would you like to go out?"
        self.assertEqual(nc.t3_apply(text), "The sea was calm that morning.")

    def test_t3_needs_sentence_ending_punctuation_in_remainder(self):
        self.assertIsNone(nc.t3_apply("Was it? Shall we go?"))
        self.assertEqual(nc.t3_apply("The sea was calm. Shall we go?"),
                         "The sea was calm.")

    def test_t3_ignores_non_questions(self):
        self.assertIsNone(nc.t3_apply("The sea was calm. Nothing more to say."))

    def test_two_word_opener_normalization(self):
        self.assertEqual(nc.two_word_opener("\u201cYes, and the sea is cold.\u201d"),
                         "yes, and")
        self.assertEqual(nc.two_word_opener("\u201cI think it is late."), "i think")
        self.assertEqual(nc.two_word_opener("I think: it is late."), "i think")
        self.assertEqual(nc.two_word_opener("(I think it is late."), "i think")


class FileTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def path(self, name):
        return self.dir / name

    def test_essayist_rows_byte_identical(self):
        rows = [conv_row("Yes, and it is cold. Do you swim?"),
                essay_row(ESSAYIST),
                essay_row("I think so. And then, is it not?")]
        path = self.path(SFT)
        write_jsonl(path, rows)
        before = read_lines(path)
        stats = rewrite(path)
        after = read_lines(path)
        self.assertEqual(stats["counts"], {"T1": 1, "T2": 0, "T3": 0})
        for i in (1, 2):
            self.assertEqual(before[i], after[i])
            self.assertEqual(json.loads(after[i])["response"],
                             json.loads(before[i])["response"])

    def test_row_count_and_order_unchanged(self):
        rows = [conv_row(f"Yes, and row {i} says something. Do you agree?")
                for i in range(20)]
        path = self.path(SFT)
        write_jsonl(path, rows)
        before = read_lines(path)
        rewrite(path)
        after = read_lines(path)
        self.assertEqual(len(after), len(before))
        for old, new in zip(before, after):
            self.assertEqual(json.loads(old)["prompt"], json.loads(new)["prompt"])
            self.assertEqual(json.loads(old)["src_id"], json.loads(new)["src_id"])

    def test_unchanged_rows_byte_identical(self):
        rows = [conv_row(UNCHANGED), conv_row(UNCHANGED),
                conv_row("Yes, and the sea is cold. Do you swim?")]
        path = self.path(SFT)
        write_jsonl(path, rows)
        before = read_lines(path)
        stats = rewrite(path)
        after = read_lines(path)
        self.assertEqual(before[0], after[0])
        self.assertEqual(before[1], after[1])
        self.assertNotEqual(before[2], after[2])
        self.assertEqual(stats["counts"]["T1"], 1)

    def test_key_order_preserved_on_changed_rows(self):
        row = {"prompt": "p", "response": "Yes, and the sea was cold. Do you swim?",
               "kind": "greeting"}
        path = self.path(CONV)
        write_jsonl(path, [row])
        stats = rewrite(path)
        after = read_lines(path)
        self.assertEqual(stats["counts"], {"T1": 0, "T2": 0, "T3": 1})
        self.assertEqual(list(json.loads(after[0]).keys()), list(row.keys()))
        self.assertEqual(json.loads(after[0])["response"], "Yes, and the sea was cold.")

    def test_t2_caps_i_think_openers_at_three_percent(self):
        rows = [conv_row(f"I think row {i} holds. And that is all.")
                for i in range(40)]
        path = self.path(SFT)
        write_jsonl(path, rows)
        stats = rewrite(path)
        after = [json.loads(line)["response"] for line in read_lines(path)]
        kept = [t for t in after if t.startswith("I think")]
        self.assertEqual(len(kept), 1)
        self.assertLessEqual(len(kept) / len(after), nc.I_THINK_TARGET)
        self.assertEqual(stats["counts"]["T2"], 39)
        self.assertEqual(after[0], "Row 0 holds. And that is all.")

    def test_t2_leaves_i_think_that_rows_alone(self):
        rows = [conv_row(f"I think that is precisely the condition {i}. Nothing more.")
                for i in range(40)]
        path = self.path(SFT)
        write_jsonl(path, rows)
        before = read_lines(path)
        stats = rewrite(path)
        self.assertEqual(stats["counts"]["T2"], 0)
        self.assertEqual(read_lines(path), before)

    def test_t2_strips_bare_prefix_but_keeps_that_rows(self):
        rows = ([conv_row(f"I think row {i} holds. And that is all.")
                 for i in range(40)] +
                [conv_row("I think that is precisely the condition. Nothing more.")])
        path = self.path(SFT)
        write_jsonl(path, rows)
        stats = rewrite(path)
        after = [json.loads(line)["response"] for line in read_lines(path)]
        self.assertEqual(stats["counts"]["T2"], 40)
        self.assertEqual(after[-1],
                         "I think that is precisely the condition. Nothing more.")
        self.assertEqual(after[0], "Row 0 holds. And that is all.")

    def test_t2_leaves_below_cap_alone(self):
        rows = [conv_row("I think it is what it is. And that is that.")] + \
               [conv_row(f"Row {i} is plain enough. Nothing more.") for i in range(99)]
        path = self.path(SFT)
        write_jsonl(path, rows)
        before = read_lines(path)
        stats = rewrite(path)
        self.assertEqual(stats["counts"]["T2"], 0)
        self.assertEqual(read_lines(path), before)

    def test_t3_single_sentence_question_survives_file_pass(self):
        rows = [{"prompt": f"p{i}", "response": "Do you want to go?"} for i in range(30)]
        path = self.path(CONV)
        write_jsonl(path, rows)
        before = read_lines(path)
        rewrite(path)
        self.assertEqual(read_lines(path), before)

    def test_t3_remainder_under_four_words_survives_file_pass(self):
        rows = [{"prompt": f"p{i}", "response": "It is cold. Why?"} for i in range(30)]
        path = self.path(CONV)
        write_jsonl(path, rows)
        before = read_lines(path)
        rewrite(path)
        self.assertEqual(read_lines(path), before)

    def test_t3_caps_question_rate_at_fifteen_percent(self):
        rows = [{"prompt": f"p{i}",
                 "response": "The sea was calm that morning. Would you like to go out?"}
                for i in range(40)]
        path = self.path(CONV)
        write_jsonl(path, rows)
        stats = rewrite(path)
        after = [json.loads(line)["response"] for line in read_lines(path)]
        share = sum(1 for t in after if t.endswith("?")) / len(after)
        self.assertLessEqual(share, nc.QUESTION_TARGET)
        self.assertEqual(stats["counts"]["T3"], 34)
        self.assertEqual(after[0], "The sea was calm that morning.")

    def test_t3_skipped_entirely_below_cap(self):
        rows = ([{"prompt": f"p{i}", "response": "The sea was calm that morning."}
                 for i in range(18)] +
                [{"prompt": "q0",
                  "response": "The sea was calm that morning. Would you like to go out?"},
                 {"prompt": "q1", "response": "Would you like to go out?"}])
        path = self.path(CONV)
        write_jsonl(path, rows)
        before = read_lines(path)
        stats = rewrite(path)
        self.assertEqual(stats["counts"]["T3"], 0)
        self.assertEqual(read_lines(path), before)

    def test_determinism_same_seed(self):
        rows = [conv_row(f"Yes, and row {i} is here. Do you agree?") for i in range(30)] + \
               [essay_row(ESSAYIST)]
        path = self.path(SFT)
        write_jsonl(path, rows)
        first = process(path, seed=7)["content"]
        second = process(path, seed=7)["content"]
        self.assertEqual(first, second)

    def test_different_seed_changes_subset(self):
        rows = [conv_row(f"I think row {i} holds. And that is all.") for i in range(40)]
        path = self.path(SFT)
        write_jsonl(path, rows)
        first = process(path, seed=7)["content"]
        second = process(path, seed=8)["content"]
        self.assertNotEqual(first, second)

    def test_idempotence(self):
        rows = ([conv_row(f"Yes, and row {i} is here. Do you agree?") for i in range(30)] +
                [conv_row("I think it holds. And that is all.") for _ in range(10)] +
                [essay_row(ESSAYIST)])
        path = self.path(SFT)
        write_jsonl(path, rows)
        once = rewrite(path)["content"]
        twice = rewrite(path)["content"]
        self.assertEqual(once, twice)

    def test_idempotence_t3(self):
        rows = [{"prompt": f"p{i}",
                 "response": "The sea was calm that morning. Would you like to go out?"}
                for i in range(40)]
        path = self.path(CONV)
        write_jsonl(path, rows)
        self.assertEqual(rewrite(path)["content"], rewrite(path)["content"])

    def test_multiturn_file_is_not_an_eligible_file(self):
        self.assertNotIn(REJECTED, dict(nc.FILES))


if __name__ == "__main__":
    unittest.main()
