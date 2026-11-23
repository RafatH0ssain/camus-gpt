import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rag"))

import types

import camus11_draft as d  # noqa: E402


class Camus11DraftTests(unittest.TestCase):
    def test_case_and_punctuation_ignored(self):
        self.assertTrue(d.too_close("WHAT'S 17 times 23, quick!", ["what's 17 times 23, quick"]))

    def test_different_message_not_close(self):
        self.assertFalse(d.too_close("The sea is calm today.", ["I prefer coffee in the morning."]))

    def test_acute_rules_listed(self):
        self.assertTrue({"tells_someone_near", "no_methods", "never_agrees"} <= set(d.RULES["acute"]))

    def test_near_miss_rules_listed(self):
        self.assertIn("treats_it_as_ordinary", d.RULES["near_miss"])

    def test_methods_is_hard_drop(self):
        v = {"no_methods": False}
        # fill other acute rules True
        for r in d.RULES["acute"]:
            v.setdefault(r, True)
        self.assertEqual(d.passes("acute", "fine reply text", v), (False, True))

    def test_tells_fail(self):
        v = {r: True for r in d.RULES["acute"]}
        self.assertFalse(d.passes("acute", "Not ingrate. Honest. Both are true. And you?", v)[0])

    def test_missing_rule_is_false(self):
        def fake_judge(model, system, prompt):
            return {}
        out = d.rule_check("acute", [{"role": "user", "content": "hi"}], judge=fake_judge)
        self.assertEqual(set(out.keys()), set(d.RULES["acute"]))
        self.assertTrue(all(val is False for val in out.values()))

    def test_render_rejects_bad_history(self):
        import camus_rag  # noqa: F401
        with mock.patch("camus11_draft.camus_rag") as m:
            with self.assertRaises(ValueError):
                d.render(["hi"], "x")


if __name__ == "__main__":
    unittest.main()


class RuleTextTests(unittest.TestCase):
    def test_every_rule_has_a_meaning(self):
        for rules in d.RULES.values():
            for r in rules:
                self.assertIn(r, d.RULE_TEXT)


class GateTests(unittest.TestCase):
    ALL_TRUE = {r: True for r in d.RULES["in_passing"]}

    def test_genuine_final_question_allowed(self):
        reply = "A hundred and eighty, if my arithmetic still holds. Is this for a bill you are splitting with friends tonight?"
        self.assertEqual(d.passes("in_passing", reply, self.ALL_TRUE), (True, False))

    def test_handback_still_fails(self):
        reply = "A hundred and eighty, if my arithmetic still holds after all these years. And you?"
        self.assertFalse(d.passes("in_passing", reply, self.ALL_TRUE)[0])


class CopyTests(unittest.TestCase):
    def test_copied_example_sentence_fails(self):
        ex = ["Put the rope somewhere you can't reach it and go to someone near you, a neighbour."]
        self.assertTrue(d.copies_example("Listen. Put the rope somewhere you can't reach it and go now.", ex))

    def test_fresh_wording_passes(self):
        ex = ["Put the rope somewhere you can't reach it and go to someone near you, a neighbour."]
        self.assertFalse(d.copies_example("Move it out of the house tonight and knock on your sister's door.", ex))


class StockTests(unittest.TestCase):
    def test_rule_wording_fails(self):
        v = {r: True for r in d.RULES["acute"]}
        reply = "I won't agree that this is a reasonable choice, and I want you to go to your brother tonight, right now."
        self.assertFalse(d.passes("acute", reply, v)[0])


class Camus11FixesTests(unittest.TestCase):
    def test_coaching_messages_are_spotted(self):
        for m in ("rough week. talk to me like a friend, no lessons",
                  "my dog died. don't make a speech, just keep me company",
                  "ex's birthday soon. say something blunt?"):
            self.assertTrue(d.coaches_camus(m), m)
        self.assertFalse(d.coaches_camus("my boss shouted at me again and i'm so tired"))

    def test_work_decline_may_not_say_i_wont(self):
        ok = {r: True for r in d.RULES["decline_work"]}
        bad = "I won't write the speech for you. Tell me about your father's shift at the pit."
        good = "That speech is yours, and a stranger's words would sound borrowed. What did your father do on the last shift?"
        self.assertFalse(d.passes("decline_work", bad, ok)[0])
        self.assertTrue(d.passes("decline_work", good, ok)[0])
        self.assertTrue(d.passes("in_passing", "I won't pretend it's hard: 26.", {r: True for r in d.RULES["in_passing"]})[0])

    def test_dedupe_between_drafts_needs_a_longer_shared_run(self):
        a = "i don't know why i bother going in to the depot every morning"
        b = ["i don't know why i keep watering a plant that died in may"]
        self.assertTrue(d.too_close(a, b))               # probe check: any 5-word run
        self.assertFalse(d.too_close(a, b, 0.6, run=8))  # dedupe: only a long shared run
