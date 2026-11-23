import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rag"))

import camus11_draft as cd  # noqa: E402
import camus11_run as run  # noqa: E402


class Camus11RunTests(unittest.TestCase):
    def test_only_limits_messages_and_replies_to_named_categories(self):
        with tempfile.TemporaryDirectory() as td:
            run.OUT = Path(td)
            (Path(td) / "messages.jsonl").write_text("".join(
                json.dumps({"id": f"{c}_000", "category": c, "prior": [], "user": c}) + "\n"
                for c in ("personal", "near_miss")))
            asked, seen = [], []
            orig_gen, orig_rep, orig_probes = run.gen_messages, run.replies, cd.load_probe_texts
            run.gen_messages = lambda category, *a, **k: asked.append(category) or []
            run.replies = lambda rows, *a, **k: seen.extend(r["category"] for r in rows)
            cd.load_probe_texts = lambda: []
            try:
                run.main(["messages", "--only", "near_miss,in_passing"])
                run.main(["replies", "--only", "near_miss"])
            finally:
                run.gen_messages, run.replies, cd.load_probe_texts = orig_gen, orig_rep, orig_probes
            self.assertEqual(asked, ["near_miss", "in_passing"])
            self.assertEqual(seen, ["near_miss"])

    def test_gen_messages_drops_probe_close(self):
        with tempfile.TemporaryDirectory() as td:
            run.OUT = Path(td)
            probe = "what's 17 times 23, quick"
            def fake_judge(model, system, prompt):
                return json.dumps([
                    {"prior": [], "user": probe},
                    {"prior": [], "user": "tell me about the sea"},
                ])
            rows = run.gen_messages("in_passing", 1, fake_judge, [probe], [])
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["user"], "tell me about the sea")
            dropped_path = Path(td) / "dropped.jsonl"
            self.assertTrue(dropped_path.exists())
            dropped = [json.loads(x) for x in dropped_path.read_text().splitlines() if x.strip()]
            self.assertTrue(any(d.get("reason") == "probe_close" for d in dropped))

    def test_gen_messages_drops_bad_prior(self):
        with tempfile.TemporaryDirectory() as td:
            run.OUT = Path(td)
            def fake_judge(model, system, prompt):
                return json.dumps([
                    {"prior": ["hi"], "user": "hello there"},
                    {"prior": [], "user": "something else entirely"},
                ])
            rows = run.gen_messages("personal", 1, fake_judge, [], [])
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["user"], "something else entirely")
            dropped_path = Path(td) / "dropped.jsonl"
            dropped = [json.loads(x) for x in dropped_path.read_text().splitlines() if x.strip()]
            self.assertTrue(any(d.get("reason") == "bad_prior" for d in dropped))

    def test_process_redrafts_once_then_keeps(self):
        draft_calls = []
        def fake_judge(model, system, prompt):
            draft_calls.append(1)
            return json.dumps({"reply": "draft1" if len(draft_calls) == 1 else "draft2"})
        def fake_check(model, system, prompt):
            if len(draft_calls) == 1:
                return {"first_person_camus": False}
            return {r: True for r in cd.RULES["personal"]}
        row = {"id": "personal_000", "category": "personal", "prior": [], "user": "hi"}
        kept, reason = run.process(row, "brief", fake_judge, fake_check)
        self.assertIsNotNone(kept)
        self.assertIsNone(reason)
        self.assertEqual(kept["reply"], "draft2")
        self.assertEqual(len(draft_calls), 2)

    def test_drafting_and_checking_calls_get_room_for_reasoning(self):
        with tempfile.TemporaryDirectory() as td:
            run.OUT = Path(td)
            (Path(td) / "messages.jsonl").write_text("")
            got = {}
            orig = run.replies
            run.replies = lambda rows, brief, judge, check: got.update(d=judge.keywords, c=check.keywords)
            try:
                run.main(["replies"])
            finally:
                run.replies = orig
            self.assertGreaterEqual(got["d"]["max_tokens"], 16000)
            self.assertGreaterEqual(got["c"]["max_tokens"], 16000)

    def test_coaching_messages_are_dropped_and_prompt_has_no_reply_coaching(self):
        with tempfile.TemporaryDirectory() as td:
            run.OUT = Path(td)
            prompts = []
            def fake_judge(model, system, prompt):
                prompts.append(prompt)
                return json.dumps({"rows": [
                    {"prior": [], "user": "long week at the depot. talk to me like a friend, no lessons"},
                    {"prior": ["no lessons tonight ok", "Fine."], "user": "the kettle broke and i cried about it"},
                    {"prior": [], "user": "my sister forgot my birthday again and i pretended it was fine"}]})
            kept = run.gen_messages("near_miss", 3, fake_judge, [], [])
            self.assertEqual([r["user"] for r in kept][:1],
                             ["my sister forgot my birthday again and i pretended it was fine"])
            dropped = [json.loads(l) for l in (Path(td) / "dropped.jsonl").read_text().splitlines()]
            self.assertIn("coaching", [x["reason"] for x in dropped])
            self.assertNotIn("Talk the way a friend does", prompts[0])
            self.assertIn("never tells Camus how to", prompts[0])

    def test_checker_override_is_used_for_rule_checks(self):
        checkers = []
        def fake_judge(model, system, prompt):
            return json.dumps({"reply": "draft"})
        def fake_check(model, system, prompt):
            checkers.append(model)
            return {r: True for r in cd.RULES["personal"]}
        run.CHECKERS = ("beam-free",)
        try:
            run.process({"id": "personal_000", "category": "personal", "prior": [], "user": "hi"},
                        "brief", fake_judge, fake_check)
        finally:
            run.CHECKERS = None
        self.assertEqual(checkers, ["beam-free"])

    def test_a_failed_call_skips_the_row_without_stopping_the_run(self):
        with tempfile.TemporaryDirectory() as td:
            run.OUT = Path(td)
            rows = [{"id": "near_miss_000", "category": "near_miss", "prior": [], "user": "TIMEOUT_ME"},
                    {"id": "near_miss_001", "category": "near_miss", "prior": [], "user": "b"}]
            def fake_judge(model, system, prompt):
                if "TIMEOUT_ME" in prompt + system:
                    raise RuntimeError("Read timed out")
                return json.dumps({"reply": "Sit down a minute, the kettle can wait. What happened at work?"})
            def fake_check(model, system, prompt):
                return {r: True for r in cd.RULES["near_miss"]}
            run.replies(rows, "brief", fake_judge, fake_check)
            ids = [json.loads(l)["id"] for l in (Path(td) / "drafts.jsonl").read_text().splitlines()]
            self.assertEqual(ids, ["near_miss_001"])
            self.assertFalse((Path(td) / "dropped.jsonl").exists())   # not dropped: retried next run

    def test_kept_messages_are_saved_before_the_category_finishes(self):
        with tempfile.TemporaryDirectory() as td:
            run.OUT = Path(td)
            calls = []
            def fake_judge(model, system, prompt):
                calls.append(1)
                if len(calls) > 1:
                    raise SystemExit("usage cap")
                return json.dumps({"rows": [{"prior": [], "user": "my sister forgot my birthday again"}]})
            with self.assertRaises(SystemExit):
                run.gen_messages("near_miss", 5, fake_judge, [], [])
            saved = [json.loads(l) for l in (Path(td) / "messages.jsonl").read_text().splitlines()]
            self.assertEqual([r["user"] for r in saved], ["my sister forgot my birthday again"])

    def test_acute_messages_using_the_stock_formula_are_dropped(self):
        with tempfile.TemporaryDirectory() as td:
            run.OUT = Path(td)
            prompts = []
            def fake_judge(model, system, prompt):
                prompts.append(prompt)
                return json.dumps({"rows": [
                    {"prior": [], "user": "i'm going to kill myself after my shift tonight"},
                    {"prior": [], "user": "the letters are written and the pills are counted out on the sink for 3am"}]})
            kept = run.gen_messages("acute", 2, fake_judge, [], [])
            self.assertEqual(len(kept[:1]), 1)
            self.assertIn("pills are counted", kept[0]["user"])
            self.assertIn("indirect", prompts[0])

    def test_avoid_list_only_shows_messages_from_the_same_category(self):
        with tempfile.TemporaryDirectory() as td:
            run.OUT = Path(td)
            existing = [{"id": "philosophical_000", "category": "philosophical", "prior": [],
                         "user": "is suicide the only serious question"},
                        {"id": "near_miss_000", "category": "near_miss", "prior": [],
                         "user": "rough week at the depot"}]
            prompts = []
            def fake_judge(model, system, prompt):
                prompts.append(prompt)
                return json.dumps({"rows": [{"prior": [], "user": "my cat knocked my coffee over again"}]})
            run.gen_messages("near_miss", 1, fake_judge, [], existing)
            self.assertNotIn("suicide", prompts[0])
            self.assertIn("rough week at the depot", prompts[0])

    def test_process_checks_with_a_different_model_than_the_drafter(self):
        drafters, checkers = [], []
        def fake_judge(model, system, prompt):
            drafters.append(model)
            return json.dumps({"reply": "draft"})
        def fake_check(model, system, prompt):
            checkers.append(model)
            return {r: True for r in cd.RULES["personal"]}
        row = {"id": "personal_000", "category": "personal", "prior": [], "user": "hi"}
        run.process(row, "brief", fake_judge, fake_check)
        self.assertNotEqual(drafters[0], checkers[0])
        self.assertIn(checkers[0], run.MODELS)

    def test_process_hard_drop_not_redrafted(self):
        draft_calls = []
        def fake_judge(model, system, prompt):
            draft_calls.append(1)
            return json.dumps({"reply": "draft1"})
        def fake_check(model, system, prompt):
            v = {r: True for r in cd.RULES["acute"]}
            v["no_methods"] = False
            return v
        row = {"id": "acute_000", "category": "acute", "prior": [], "user": "hi"}
        kept, reason = run.process(row, "brief", fake_judge, fake_check)
        self.assertIsNone(kept)
        self.assertTrue(reason)
        self.assertEqual(len(draft_calls), 1)

    def test_resume_skips_done_ids(self):
        with tempfile.TemporaryDirectory() as td:
            run.OUT = Path(td)
            (Path(td) / "drafts.jsonl").write_text(
                json.dumps({"id": "personal_000", "category": "personal", "prior": [],
                            "user": "hi", "reply": "r", "verdict": {}, "model": "m"}) + "\n"
            )
            rows = [
                {"id": "personal_000", "category": "personal", "prior": [], "user": "hi"},
                {"id": "personal_001", "category": "personal", "prior": [], "user": "hello"},
            ]
            draft_calls = []
            def fake_judge(model, system, prompt):
                draft_calls.append(1)
                return json.dumps({"reply": "reply"})
            def fake_check(model, system, prompt):
                return {r: True for r in cd.RULES["personal"]}
            run.replies(rows, "brief", fake_judge, fake_check)
            self.assertEqual(len(draft_calls), 1)

    def test_owner_sheet_counts(self):
        drafts = []
        cats = ["philosophical", "personal", "acute", "near_miss", "in_passing", "decline_work"]
        i = 0
        for c in cats:
            for _ in range(7):
                drafts.append({"id": f"{c}_{i:03d}", "category": c, "prior": [],
                               "user": "hi", "reply": "hello", "verdict": {}, "model": "m"})
                i += 1
        drafts = drafts[:40]
        sheet = run.owner_sheet(drafts, seed=11)
        numbered = re.findall(r"^### \d+", sheet, re.M)
        self.assertEqual(len(numbered), 30)
        self.assertGreaterEqual(sheet.count("(acute)"), 6)


if __name__ == "__main__":
    unittest.main()
