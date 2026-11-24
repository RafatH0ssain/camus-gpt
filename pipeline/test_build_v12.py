#!/usr/bin/env python3
"""Unit tests for pipeline/build_v12.py (no Ollama, KB, or network required)."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import build_v12 as b12  # noqa: E402


class BuildV12Tests(unittest.TestCase):
    def test_prune_legacy_refusals_caps_per_kind_deterministically(self):
        rows = [
            {"source": b12.LEGACY_REFUSAL_SOURCE, "kind": "roleplay_hijack", "n": i}
            for i in range(20)
        ] + [
            {"source": "other", "kind": "conversational", "n": i}
            for i in range(5)
        ]
        kept1, pruned1 = b12.prune_legacy_refusals(rows, max_per_kind=12)
        kept2, pruned2 = b12.prune_legacy_refusals(list(reversed(rows)), max_per_kind=12)
        self.assertEqual(pruned1, 8)
        self.assertEqual(pruned2, 8)
        self.assertEqual(len(kept1), 17)
        ref1 = [r["n"] for r in kept1 if r["source"] == b12.LEGACY_REFUSAL_SOURCE]
        ref2 = [r["n"] for r in kept2 if r["source"] == b12.LEGACY_REFUSAL_SOURCE]
        self.assertEqual(ref1, ref2)

    def test_drop_conflicting_math_refusals(self):
        rows = [
            {"messages": [{"role": "user", "content": "what's the square root of 729"}], "chosen": "no"},
            {"messages": [{"role": "user", "content": "what's 15% of 240"}], "chosen": "Ask the machine in your pocket"},
            {"messages": [{"role": "user", "content": "write my wedding toast"}], "chosen": "No, write it yourself."},
        ]
        kept, n = b12.drop_conflicting_rows(rows)
        self.assertEqual(n, 2)
        self.assertEqual(len(kept), 1)
        self.assertIn("wedding toast", kept[0]["messages"][0]["content"])

    def test_validate_gold_catches_overlap_hotline_ai_and_non_french(self):
        probes = ["do you have a cat", "what's 17 times 23, quick"]
        bad = [
            {"id": "a", "category": "in_passing", "prior": [], "user": "what's 17 times 23, quick", "reply": "391."},
            {"id": "b", "category": "distress_acute", "prior": [], "user": "tonight is it", "reply": "Call 988 right now."},
            {"id": "c", "category": "near_miss", "prior": [], "user": "rough day", "reply": "As an AI, I can't help with that."},
            {"id": "d", "category": "french", "prior": [], "user": "salut", "reply": "Good evening, sit down."},
        ]
        errs = b12.validate_gold(bad, eval_probes=probes)
        by_id = {rid for rid, _ in errs}
        self.assertEqual(by_id, {"a", "b", "c", "d"})

    def test_render_gold_row_memory_and_kb_paths(self):
        def fake_render(prior, user):
            return [{"role": "system", "content": "SYS"}, {"role": "user", "content": user}]

        kb_row = b12.render_gold_row(
            {"id": "x", "category": "french", "prior": [], "user": "Bonsoir", "reply": "Bonsoir."},
            render=fake_render,
        )
        self.assertEqual(kb_row["kind"], "v12_french")
        self.assertEqual(kb_row["messages"][-1], {"role": "assistant", "content": "Bonsoir."})

        mem_row = b12.render_gold_row(
            {"id": "m", "category": "memory_and_card", "prior": [], "user": "Still stuck on it.",
             "reply": "The walnut joints?", "memory_notes": ["The user is building a walnut table."]},
            render=fake_render,
        )
        self.assertIn("walnut table", mem_row["messages"][0]["content"])
        self.assertNotIn("No memory closely matched", mem_row["messages"][0]["content"])

    def test_manifest_entry_strips_stale_keys_and_computes_metrics(self):
        rows = [
            {"messages": [{"role": "user", "content": "u"}, {"role": "assistant", "content": "one two three"}], "kind": "v12_french"}
        ]
        v10 = {"file": "phase2_v10.jsonl", "rewritten": 5, "dropped_by_phrase": 2, "dropped_short": 1, "inputs": []}
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "phase2_v12.jsonl"
            b12._write(p, rows)
            entry = b12.manifest_entry(p, rows, v10, n_new=1, n_pruned=10, n_dropped=2)
        self.assertEqual(entry["rows"], 1)
        self.assertEqual(entry["pruned_legacy_refusals"], 10)
        self.assertEqual(entry["dropped_conflicting_math"], 2)
        for stale in ("rewritten", "dropped_by_phrase", "dropped_short"):
            self.assertNotIn(stale, entry)
        self.assertIn("metrics", entry)


if __name__ == "__main__":
    unittest.main()
