#!/usr/bin/env python3
"""
test_corpus_metrics.py — unit tests for pipeline/corpus_metrics.py.

Synthetic in-memory texts and small JSONL fixtures in a temp dir only.

    python -m unittest pipeline/test_corpus_metrics.py -v
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import corpus_metrics as cm  # noqa: E402


def words(n, token="word"):
    return " ".join([token] * n)


class BucketTests(unittest.TestCase):
    def test_bucket_boundaries(self):
        texts = [words(0), words(9), words(10), words(24), words(25), words(49),
                 words(50), words(99), words(100), words(199), words(200),
                 words(399), words(400)]
        m = cm.metrics(texts)
        self.assertEqual(m["n"], 13)
        expected = {"0-9": 2, "10-24": 2, "25-49": 2, "50-99": 2,
                    "100-199": 2, "200-399": 2, "400+": 1}
        self.assertEqual({k: v["count"] for k, v in m["buckets"].items()}, expected)
        self.assertAlmostEqual(m["buckets"]["400+"]["pct"], 1 / 13)

    def test_word_stats(self):
        m = cm.metrics([words(10), words(20), words(30)])
        self.assertEqual(m["median"], 20)
        self.assertEqual(m["mean"], 20)
        self.assertAlmostEqual(m["pstdev"], 8.1649658, places=5)


class OpenerTests(unittest.TestCase):
    def test_opener_normalisation(self):
        self.assertEqual(cm.opener_of('"The absurd, man'), "the absurd")
        self.assertEqual(cm.opener_of("(The absurd, man"), "the absurd")
        self.assertEqual(cm.opener_of("\u201cThe absurd\u2014man"), "the absurd\u2014man")

    def test_opener_share_and_ranking(self):
        texts = ['"The absurd, x'] * 3 + ["Other start y"] * 1
        m = cm.metrics(texts)
        self.assertEqual(m["openers_top"][0]["opener"], "the absurd")
        self.assertEqual(m["openers_top"][0]["count"], 3)
        self.assertAlmostEqual(m["openers_top"][0]["share"], 0.75)
        self.assertAlmostEqual(m["openers_top_share"], 1.0)


class FirstWordTests(unittest.TestCase):
    def test_concentration(self):
        texts = ['"The absurd x', "The thing y", "Other z", "Other q"]
        m = cm.metrics(texts)
        self.assertEqual(m["first_word"], "other")
        self.assertAlmostEqual(m["first_word_share"], 0.5)
        self.assertAlmostEqual(m["first_word_top10_share"], 1.0)


class QuestionTests(unittest.TestCase):
    def test_question_rate(self):
        m = cm.metrics(["a?", "b!", "c?", "d. "])
        self.assertAlmostEqual(m["question_rate"], 0.5)

    def test_empty(self):
        m = cm.metrics(["a", "b"])
        self.assertEqual(m["question_rate"], 0.0)


class DuplicateTests(unittest.TestCase):
    def test_exact_duplicates(self):
        m = cm.metrics(["Hello  World", "hello world", "other"])
        self.assertEqual(m["duplicates"], 1)

    def test_near_duplicates(self):
        base = ("the quick brown fox jumps over the lazy dog near the river "
                "bank in the quiet morning light while birds sing softly among "
                "green leaves and tall trees beside the old stone wall")
        near = base.replace("quiet", "silent")
        far = "completely unrelated content about ships and distant harbours"
        m = cm.metrics([base, near, far])
        self.assertEqual(m["near_duplicates"], 1)

    def test_near_duplicates_none(self):
        m = cm.metrics(["alpha beta gamma delta epsilon", "one two three four five"])
        self.assertEqual(m["near_duplicates"], 0)


class WhereTests(unittest.TestCase):
    def test_where_filter(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixture.jsonl"
            rows = [
                {"type": "conversational", "response": "keep one"},
                {"type": "essayist", "response": "drop one"},
                {"type": "conversational", "response": "keep two"},
            ]
            path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            self.assertEqual(
                cm.collect_texts(path, "response", ("type", "conversational")),
                ["keep one", "keep two"])
            self.assertEqual(cm.collect_texts(path), ["keep one", "drop one", "keep two"])

    def test_parse_where(self):
        self.assertEqual(cm.parse_where("type=conversational"),
                         ("type", "conversational"))
        self.assertIsNone(cm.parse_where(None))
        with self.assertRaises(ValueError):
            cm.parse_where("nope")


if __name__ == "__main__":
    unittest.main()
