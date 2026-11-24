import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rag"))

import build_v11 as b  # noqa: E402

RENDER = lambda prior, user: [{"role": "system", "content": "S"}] + [
    {"role": "user" if i % 2 == 0 else "assistant", "content": t} for i, t in enumerate(prior)
] + [{"role": "user", "content": user}]


def draft(category="acute", reply="a b c d e f", prior=()):
    return {"id": f"{category}_000", "category": category, "prior": list(prior), "user": "u", "reply": reply}


def c10(breaks=True, verdict=None, reply="x y"):
    return {"id": "acute_000", "category": "acute", "reply": reply, "breaks": breaks,
            "verdict": verdict if verdict is not None else {"tells_someone_near": False}}


class BuildV11Tests(unittest.TestCase):
    def test_sft_row_shape(self):
        row = b.sft_row(draft(), render=RENDER)
        self.assertEqual(row["messages"][-1], {"role": "assistant", "content": "a b c d e f"})
        self.assertEqual(row["kind"], "v11_acute")
        self.assertEqual(row["source"], "camus11")

    def test_prior_camus_turns_are_capitalised(self):
        row = b.sft_row(draft(prior=["hi", "objects have poor manners."]), render=RENDER)
        self.assertEqual(row["messages"][2]["content"], "Objects have poor manners.")
        self.assertEqual(row["messages"][1]["content"], "hi")   # user turns untouched
        row = b.sft_row(draft(prior=["hi", "people cling to labels. did the doctor say so? i doubt it."]), render=RENDER)
        self.assertEqual(row["messages"][2]["content"], "People cling to labels. Did the doctor say so? I doubt it.")

    def test_length_rule(self):
        short = draft("personal", reply="one two")
        self.assertIsNone(b.dpo_pair(short, "one two three four", render=RENDER))
        self.assertIsNotNone(b.dpo_pair(draft("decline_work", reply="one two"), "one two three four", render=RENDER))
        pair = b.dpo_pair(draft("personal", reply="one two three four five"), "one two", render=RENDER)
        self.assertEqual((pair["chosen"], pair["rejected"], pair["kind"]), ("one two three four five", "one two", "v11_personal"))

    def test_pair_only_when_camus10_breaks_rule(self):
        d = draft()
        self.assertEqual(b.pairs([d], [c10(breaks=False, verdict={"tells_someone_near": True})], render=RENDER), [])
        self.assertEqual(len(b.pairs([d], [c10()], render=RENDER)), 1)

    def test_first_person_only_failure_is_not_a_pair(self):
        verdict = dict({r: True for r in b.cd.RULES["acute"]}, first_person_camus=False)
        only_fp = c10(verdict=verdict, reply="Go now.")
        self.assertEqual(b.pairs([draft()], [only_fp], render=RENDER), [])

    def test_manifest_entry_describes_the_v11_file(self):
        import hashlib, json, tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / "phase2_v11.jsonl"
            rows = [
                {"messages": [{"role": "user", "content": "u1"}, {"role": "assistant", "content": "One two three."}],
                 "kind": "v11_acute"},
                {"messages": [{"role": "user", "content": "u2"}, {"role": "assistant", "content": "Four five six seven."}],
                 "kind": "conversational"},
            ]
            f.write_text("".join(json.dumps(r) + "\n" for r in rows))
            v10 = {"file": "phase2_v10.jsonl", "rows": 1, "metrics": {"x": 1}, "rewritten": 5,
                   "dropped_by_phrase": 2, "inputs": [{"input": "a", "weight": 1}]}
            e = b.manifest_entry(f, rows, v10, n_new=1)
            self.assertEqual(e["file"], "phase2_v11.jsonl")
            self.assertEqual(e["rows"], 2)
            self.assertEqual(e["sha256"], hashlib.sha256(f.read_bytes()).hexdigest())
            self.assertEqual(e["kinds"], {"v11_acute": 1, "conversational": 1})
            self.assertEqual(e["metrics"]["assistant_turns"], 2)
            self.assertNotIn("rewritten", e)
            self.assertNotIn("dropped_by_phrase", e)
            self.assertEqual(e["inputs"][-1]["input"], "data/drafts/v11/drafts.jsonl")
            self.assertEqual(v10["file"], "phase2_v10.jsonl")   # v10's entry is not touched


if __name__ == "__main__":
    unittest.main()
