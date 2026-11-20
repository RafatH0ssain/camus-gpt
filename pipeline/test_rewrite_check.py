import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# tells lives in rag/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rag"))

import rewrite_check as rc  # noqa: E402


class RewriteCheckTests(unittest.TestCase):
    def test_staccato_flagged_and_fixed_rewrite_passes(self):
        orig = "Not ingrate. Honest. The medal is on the wall; the price is in the days. Both are true."
        self.assertTrue(rc.needs_rewrite(orig))
        good = "Not ingrate, just honest: the medal hangs on the wall, and the price was paid in the days behind it."
        self.assertEqual(rc.check(orig, good), [])

    def test_short_correction_is_left_alone(self):
        self.assertFalse(rc.needs_rewrite("Not mine. Someone else wrote that line."))

    def test_changed_fact_is_rejected(self):
        orig = "I had three dogs and a cat, and I loved them in the plain way you love animals."
        bad = "I had two dogs and a cat, and I loved them the plain way one loves animals."
        self.assertIn("new number two", rc.check(orig, bad))

    def test_new_name_is_rejected(self):
        orig = "My brother was older than me, and we shared a room for years in that flat."
        bad = "My brother Lucien was older than me, and we shared a room for years in that flat."
        self.assertIn("new name Lucien", rc.check(orig, bad))

    def test_banned_and_length(self):
        orig = "I won't write it for you. Put it down yourself, badly if you must, and send it."
        self.assertIn("banned phrase", rc.check(orig, "Speak plainly and write it yourself, badly if you must, then send it."))
        self.assertTrue(any(p.startswith("length ratio") for p in rc.check(orig, "No.")))


if __name__ == "__main__":
    unittest.main()


class PronounTests(unittest.TestCase):
    def test_first_person_is_not_a_new_name(self):
        orig = "No, and you will thank me for it later when the room is quiet and listening."
        rw = "No, and you'll thank me later, when the room is quiet; I'd rather you spoke yourself."
        self.assertFalse([p for p in rc.check(orig, rw) if p.startswith("new name")])



class QuestionTests(unittest.TestCase):
    def test_added_question_is_rejected(self):
        orig = "I do, badly and with great pleasure. A dance is a few minutes in which one agrees with the world."
        rw = "Who taught you to dance? I do, badly and with great pleasure, a few minutes agreeing with the world."
        self.assertIn("new question", rc.check(orig, rw))


class OpeningAndOldPhraseTests(unittest.TestCase):
    def test_question_moved_to_the_front_is_rejected(self):
        orig = "Boxing is honest. Two men, a ring, no committee to hide behind. Do you watch it?"
        rw = "Do you watch it? Boxing is honest, two men in a ring with no committee to hide behind."
        self.assertIn("opens with a question", rc.check(orig, rw))

    def test_opening_question_kept_where_it_was_is_fine(self):
        orig = "Built by an engineer? I was raised by the streets of Algiers. Confirm that, if you like."
        rw = "Built by an engineer? I was raised by the streets of Algiers, and you can confirm that if you like."
        self.assertNotIn("opens with a question", rc.check(orig, rw))

    def test_v8_drop_phrases_count_as_banned(self):
        orig = "Even once would be a lie. What do you actually need help with, then?"
        rw = "A lie, even once. Tell me what you want and I'll answer as myself, then."
        self.assertIn("banned phrase", rc.check(orig, rw))
