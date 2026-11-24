#!/usr/bin/env python3
"""Unit tests for rag/memory.py (no Ollama or network required)."""
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import memory as mem  # noqa: E402


def unit(v):
    a = np.asarray(v, dtype=np.float32)
    return a / (np.linalg.norm(a) + 1e-9)


def make_embedder(table, calls=None):
    """Deterministic unit-vector embedder keyed by substring match."""
    def embed_fn(text, prefix=""):
        if calls is not None:
            calls.append((prefix, text))
        low = text.lower()
        for key, vec in table.items():
            if key in low:
                return unit(vec)
        return unit([0.0, 0.0, 0.0, 1.0])
    return embed_fn


class MemoryStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.mem_p = os.path.join(self.tmp.name, "memory.jsonl")
        self.vec_p = os.path.join(self.tmp.name, "memory_vectors.npy")
        self.pen_p = os.path.join(self.tmp.name, "memory_pending.jsonl")

    def store(self, table, calls=None):
        return mem.MemoryStore(make_embedder(table, calls), self.mem_p, self.vec_p, self.pen_p)

    def test_add_bump_merge_and_save_reload(self):
        # cos([1,0,0,0], [0.87,0.493,0,0]) ~= 0.87 (between DEDUP_MERGE=0.85 and DEDUP_DROP=0.90)
        s = self.store({
            "longer": [0.87, 0.493, 0.0, 0.0],
            "tea": [1.0, 0.0, 0.0, 0.0],
            "lisbon": [0.0, 1.0, 0.0, 0.0],
        })
        self.assertEqual(s.add("   "), "empty")
        self.assertEqual(s.add("The user drinks tea.", kind="bogus_kind"), "added")
        self.assertEqual(s.rows[0]["kind"], "fact")

        # >= DEDUP_DROP bumps hits without adding a duplicate row
        self.assertEqual(s.add("The user drinks tea."), "bumped")
        self.assertEqual(len(s.rows), 1)
        self.assertEqual(s.rows[0]["hits"], 2)

        # DEDUP_MERGE..DEDUP_DROP merges fuller text and cue in place
        self.assertEqual(s.add("The user drinks green tea, longer version.", cue="my tea"), "merged")
        self.assertEqual(len(s.rows), 1)
        self.assertIn("longer version", s.rows[0]["text"])
        self.assertEqual(s.rows[0]["cue"], "my tea")
        self.assertEqual(s.rows[0]["hits"], 3)

        self.assertEqual(s.add("The user lives in Lisbon.", kind="fact"), "added")
        s.save()

        reloaded = self.store({"tea": [1.0, 0.0, 0.0, 0.0], "lisbon": [0.0, 1.0, 0.0, 0.0]})
        self.assertEqual(len(reloaded.rows), 2)
        self.assertEqual(reloaded.vecs.shape, (2, 4))

    def test_retrieve_filters_by_threshold_and_bumps_hits(self):
        s = self.store({"tea": [1.0, 0.0, 0.0, 0.0], "lisbon": [0.0, 1.0, 0.0, 0.0]})
        s.add("The user drinks tea.")
        s.add("The user lives in Lisbon.")
        hits = s.retrieve("tea", top_k=3, threshold=0.52)
        self.assertEqual(len(hits), 1)
        self.assertIn("tea", hits[0]["text"])
        self.assertEqual(s.rows[0]["hits"], 2)
        self.assertEqual(s.rows[1]["hits"], 1)

    def test_eviction_drops_fewest_hits_then_oldest(self):
        table = {f"m{i}": [float(i == j) for j in range(4)] for i in range(4)}
        s = self.store(table)
        with mock.patch.object(mem, "MAX_MEMORIES", 2):
            s.add("The user likes m0.")
            s.rows[0]["hits"] = 5
            s.add("The user likes m1.")
            s.rows[1]["last_seen"] = "2020-01-01T00:00:00+00:00"
            s.add("The user likes m2.")
        texts = [r["text"] for r in s.rows]
        self.assertEqual(len(texts), 2)
        self.assertIn("The user likes m0.", texts)
        self.assertNotIn("The user likes m1.", texts)

    def test_stage_probation_promotion_and_single_embed_on_live_dedup(self):
        calls = []
        s = self.store({"chess": [1.0, 0.0, 0.0, 0.0], "piano": [0.0, 1.0, 0.0, 0.0]}, calls=calls)
        res1 = s.stage([{"text": "The user plays chess.", "kind": "unknown", "cue": "my hobby"}], session="s1")
        self.assertEqual(res1["staged"], 1)
        self.assertEqual(res1["promoted"], 0)
        self.assertEqual(len(s.rows), 0)

        res2 = s.stage([{"text": "The user plays chess.", "kind": "fact", "cue": "my hobby"}], session="s2")
        self.assertEqual(res2["promoted"], 1)
        self.assertEqual(len(s.rows), 1)

        # Staging an item already in live store should embed only once (no redundant add() embed)
        calls.clear()
        res3 = s.stage([{"text": "The user plays chess.", "kind": "fact", "cue": "my hobby"}], session="s3")
        self.assertEqual(res3["bumped"], 1)
        self.assertEqual(len(calls), 1)

    def test_forget_purges_live_and_pending(self):
        s = self.store({"chess": [1.0, 0.0, 0.0, 0.0], "piano": [0.0, 1.0, 0.0, 0.0]})
        s.add("The user plays chess.")
        s.save()
        s.stage([{"text": "The user plays piano.", "kind": "fact"}])

        removed = s.forget("piano")
        self.assertEqual(removed, 1)
        # Re-staging piano must stage fresh (sightings=1), NOT promote from old probation
        res = s.stage([{"text": "The user plays piano.", "kind": "fact"}])
        self.assertEqual(res["staged"], 1)
        self.assertEqual(res["promoted"], 0)

        self.assertEqual(s.forget("chess"), 1)
        s.save()
        self.assertEqual(s.rows, [])
        self.assertFalse(os.path.exists(self.vec_p))

    def test_load_rebuilds_mismatched_vectors_and_reindex_updates(self):
        s = self.store({"chess": [1.0, 0.0, 0.0, 0.0], "piano": [0.0, 1.0, 0.0, 0.0]})
        s.add("The user plays chess.")
        s.add("The user plays piano.", cue="my instrument")
        s.save()
        np.save(self.vec_p, np.zeros((1, 4), dtype=np.float32))  # corrupt length mismatch

        reloaded = self.store({"chess": [1.0, 0.0, 0.0, 0.0], "piano": [0.0, 1.0, 0.0, 0.0]})
        self.assertEqual(reloaded.vecs.shape, (2, 4))
        self.assertEqual(reloaded.reindex(), 2)


class ParsingAndFormattingTests(unittest.TestCase):
    def test_parse_extraction_handles_fences_and_trailing_brackets(self):
        raw = (
            '```json\n'
            '[{"text": "The user is named Ada.", "kind": "fact", "cue": "my name"},\n'
            ' {"text": "short", "kind": "fact"},\n'
            ' {"text": "The user dislikes lists.", "kind": "weird"}]\n'
            '```\n'
            'Note: extracted from [turn 1] above.'
        )
        got = mem.parse_extraction(raw)
        self.assertEqual(len(got), 2)
        self.assertEqual(got[0]["text"], "The user is named Ada.")
        self.assertEqual(got[1]["kind"], "fact")  # invalid kind normalized

    def test_parse_extraction_caps_at_five_and_handles_empty(self):
        items = [{"text": f"The user likes item number {i}.", "kind": "fact"} for i in range(8)]
        self.assertEqual(len(mem.parse_extraction(json.dumps(items))), 5)
        self.assertEqual(mem.parse_extraction("nothing durable here"), [])

    def test_extract_memories_skips_short_conversations(self):
        called = []
        out = mem.extract_memories([{"role": "user", "content": "hi"}], lambda m: called.append(m) or "[]")
        self.assertEqual(out, [])
        self.assertEqual(called, [])

    def test_build_memory_block_and_summary_block(self):
        self.assertEqual(mem.build_memory_block("", []), "")
        blk = mem.build_memory_block("The user lives in Lisbon.", [
            {"text": "The user builds tools.", "kind": "fact"},
            {"text": "The user likes brevity.", "kind": "preference"},
            {"text": "Debugging retrieval.", "kind": "thread"},
        ])
        self.assertIn("Lisbon", blk)
        self.assertIn("Recalled from earlier conversations:", blk)
        self.assertIn("How they like to be spoken with:", blk)
        self.assertIn("Left unfinished last time", blk)
        self.assertEqual(mem.build_summary_block(""), "")
        self.assertIn("Earlier in this same conversation", mem.build_summary_block("Talked about tea."))

    def test_load_profile_missing_and_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "profile.md")
            self.assertEqual(mem.load_profile(p), "")
            with open(p, "w", encoding="utf-8") as fh:
                fh.write("  The user is Ada.\n")
            self.assertEqual(mem.load_profile(p), "The user is Ada.")


if __name__ == "__main__":
    unittest.main()
