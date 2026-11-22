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
        reply = "A hundred and eighty, if my arithmetic still holds. Are you doing your accounts at this hour or settling an argument?"
        self.assertEqual(d.passes("in_passing", reply, self.ALL_TRUE), (True, False))

    def test_handback_still_fails(self):
        reply = "A hundred and eighty, if my arithmetic still holds after all these years. And you?"
        self.assertFalse(d.passes("in_passing", reply, self.ALL_TRUE)[0])
