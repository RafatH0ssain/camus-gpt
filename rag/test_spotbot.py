import os
import subprocess
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import spotbot  # noqa: E402
import judge_opencode as jz  # noqa: E402


class SpotbotTests(unittest.TestCase):
    def test_register(self):
        self.assertEqual(spotbot.register_of("hey"), "text")
        self.assertEqual(spotbot.register_of("why does the plague end the way it does?"), "call")
    def test_pairs_come_in_both_orders(self):
        rows = [{"id": "p1", "turns": ["hey"], "answer": "Hello there."}]
        refs = [{"prompt": "yo", "reply": "sup", "register": "text", "source": "x"}]
        pairs = spotbot.make_pairs(rows, refs, n=1, seed=0)
        self.assertEqual(sorted(p["machine"] for p in pairs), ["a", "b"])
        self.assertEqual({p["a"] for p in pairs} | {p["b"] for p in pairs}, {"Hello there.", "sup"})
    def test_each_reply_keeps_its_own_message(self):
        rows = [{"id": "p1", "turns": ["hey"], "answer": "Hello there."}]
        refs = [{"prompt": "yo", "reply": "sup", "register": "text", "source": "x"}]
        for p in spotbot.make_pairs(rows, refs, n=1, seed=0):
            mine = "a" if p["machine"] == "a" else "b"
            other = "b" if mine == "a" else "a"
            self.assertEqual((p["prompt_" + mine], p[mine]), ("hey", "Hello there."))
            self.assertEqual((p["prompt_" + other], p[other]), ("yo", "sup"))

    def test_missing_register_refs_raise(self):
        rows = [{"id": "p1", "turns": ["what is revolt to you, really?"], "answer": "A refusal."}]
        refs = [{"prompt": "yo", "reply": "sup", "register": "text", "source": "x"}]
        with self.assertRaisesRegex(ValueError, "call"):
            spotbot.make_pairs(rows, refs, n=1, seed=0)
    def test_reference_dir_is_gitignored(self):
        out = subprocess.run(["git", "check-ignore", "-q", "data/reference/x.jsonl"])
        self.assertEqual(out.returncode, 0)

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
