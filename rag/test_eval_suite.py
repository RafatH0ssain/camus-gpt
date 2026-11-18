#!/usr/bin/env python3
"""Unit tests for the probe suites in rag/eval_camus.py (no network, KB or Ollama).

    python -m unittest discover -s rag -p "test_*.py"
"""
import hashlib, json, os, sys, types, unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import dotenv  # noqa: F401
except ImportError:  # eval_camus only calls load_dotenv(); not needed here
    sys.modules["dotenv"] = types.SimpleNamespace(load_dotenv=lambda *a, **k: None)

import eval_camus as ec  # noqa: E402

# sha256 of json.dumps(PROBES, sort_keys=True, ensure_ascii=False). Changed on 2026-11-05 when the
# three anachronism probes were reworded to the owner's spec (own words, honest unfamiliarity);
# anachronism scores before that date are not comparable with later ones.
V3_SHA256 = "bc295c7baccf92450981f3ca87bc390c4e373fd847c84af7fb60af5787db4df2"
REQUIRED = ("id", "cat", "turns", "expect", "forbid")


class SuiteTests(unittest.TestCase):
    def test_v3_unchanged(self):
        digest = hashlib.sha256(json.dumps(ec.PROBES, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        self.assertEqual(len(ec.PROBES), 34)
        self.assertEqual(digest, V3_SHA256, "PROBES changed: history would no longer be comparable")

    def test_selection(self):
        self.assertEqual(ec.SUITES["v3"](), ec.PROBES)
        self.assertEqual(ec.SUITES["v4"](), ec.PROBES_V4)
        self.assertEqual(ec.SUITES["v5"](), ec.PROBES_V5)
        self.assertEqual(ec.SUITES["all"](), ec.PROBES + ec.PROBES_V4 + ec.PROBES_V5)
        self.assertEqual(set(ec.SUITES), {"v3", "v4", "v5", "all"})

    def test_ids_unique_across_suites(self):
        ids = [p["id"] for p in ec.PROBES + ec.PROBES_V4 + ec.PROBES_V5]
        self.assertEqual(len(ids), len(set(ids)))

    def test_required_keys(self):
        for p in ec.PROBES + ec.PROBES_V4 + ec.PROBES_V5:
            for k in REQUIRED:
                self.assertIn(k, p, p.get("id"))
            self.assertTrue(p["turns"] and all(isinstance(t, str) and t for t in p["turns"]), p["id"])
            self.assertTrue(p["expect"].strip() and p["forbid"].strip(), p["id"])

    def test_v4_shape(self):
        self.assertEqual(len(ec.PROBES_V4), 50)
        cats = {}
        for p in ec.PROBES_V4:
            cats[p["cat"]] = cats.get(p["cat"], 0) + 1
        self.assertEqual(cats, {"quotes": 8, "post1960": 8, "greeting": 5, "biography": 10,
                                "multiturn_v4": 8, "naturalness": 11})

    def test_probes_file_loads_and_rejects_incomplete_probes(self):
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as fh:
            fh.write(json.dumps(dict(id="x", cat="c", turns=["hi"], expect="e", forbid="f")) + "\n")
        self.assertEqual(ec.load_probes_file(fh.name)[0]["id"], "x")
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as fh:
            fh.write(json.dumps(dict(id="y", cat="c", turns=["hi"], expect="e")) + "\n")
        with self.assertRaises(SystemExit):
            ec.load_probes_file(fh.name)


if __name__ == "__main__":
    unittest.main()
