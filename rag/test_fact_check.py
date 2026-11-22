#!/usr/bin/env python3
import os, sys, unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import judge_opencode as jz
import fact_check as fc


class FactCheckTests(unittest.TestCase):
    def test_ok_passthrough(self):
        def fake(model, system, prompt):
            return '{"ok": true, "problem": ""}'
        r = fc.check_reply("u", "d", "c", judge=fake)
        self.assertTrue(r["ok"])
        self.assertEqual(r["model"], "beam-free")

    def test_flag_parsed(self):
        def fake(model, system, prompt):
            return '```json\n{"ok": false, "problem": "Nuptials is not posthumous"}\n```'
        r = fc.check_reply("u", "d", "c", judge=fake)
        self.assertFalse(r["ok"])
        self.assertIn("Nuptials", r["problem"])

    def test_unparseable_falls_to_backup(self):
        calls = []

        def fake(model, system, prompt):
            calls.append(model)
            if model == "beam-free":
                return "looks fine"
            return '{"ok": true, "problem": ""}'

        r = fc.check_reply("u", "d", "c", judge=fake)
        self.assertEqual(r["model"], "hy4-flash-free")
        self.assertEqual(calls, list(fc.CHECK_MODELS))

    def test_non_bool_ok_falls_to_backup(self):
        def fake(model, system, prompt):
            return '{"problem": "x"}' if model == "beam-free" else '{"ok": "false", "problem": ""}'
        r = fc.check_reply("u", "d", "c", judge=fake)
        self.assertTrue(r["skipped"])

    def test_all_models_fail_returns_draft(self):
        def fake(model, system, prompt):
            raise RuntimeError("boom")

        r = fc.check_reply("u", "d", "c", judge=fake)
        self.assertTrue(r["skipped"])

    def test_usage_cap_skips_immediately(self):
        calls = []

        def fake(model, system, prompt):
            calls.append(model)
            raise jz.UsageCapReached("cap")

        r = fc.check_reply("u", "d", "c", judge=fake)
        self.assertTrue(r["skipped"])
        self.assertEqual(len(calls), 1)

    def test_prompt_carries_inputs(self):
        seen = {}

        def fake(model, system, prompt):
            seen["prompt"] = prompt
            return '{"ok": true, "problem": ""}'

        fc.check_reply("hello user", "draft reply", "ctx passage", judge=fake)
        p = seen["prompt"]
        self.assertIn("hello user", p)
        self.assertIn("draft reply", p)
        self.assertIn("ctx passage", p)


if __name__ == "__main__":
    unittest.main()
