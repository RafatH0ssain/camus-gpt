import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tells  # noqa: E402


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


class OwnerTellsTests(unittest.TestCase):
    def test_banned_phrases_are_service_tells(self):
        for r in ("You are speaking to me across time, so speak plainly. What is it?",
                  "I've learned not to guess about these things.",
                  "Which is fairly rare, these days."):
            self.assertIn("service", tells.reply_tells(r), r)

    def test_staccato_is_a_run_of_fragments(self):
        # the two camus8 replies the owner flagged for structure
        self.assertIn("staccato", tells.reply_tells(
            "Not ingrate. Honest. The medal is on the wall; the price is in the days. Both are true."))
        self.assertIn("staccato", tells.reply_tells(
            "Then keep it for six months. Find another first. Everyone else has a cushion; you "
            "don't get to spend your cushion."))
        # a real Camus interview answer the owner suspected; its sentences are not fragments
        self.assertNotIn("staccato", tells.reply_tells(
            "Yes, I have a liking for energy and conquests. But I soon tire of what I have obtained. "
            "This is my great weakness. I also have a liking for obscurity, for self-effacement. "
            "But the passion for life urges me forward again. In short, I never solve the dilemma."))
