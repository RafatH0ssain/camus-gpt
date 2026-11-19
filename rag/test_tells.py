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
