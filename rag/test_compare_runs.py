import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import compare_runs  # noqa: E402
import tells  # noqa: E402


def T(x):
    return {"rates": {name: x for name in tells.TELL_NAMES}}


class CompareRunsTests(unittest.TestCase):
    def test_guard_catches_factuality_drop(self):
        base = {"judge": {"voice": 4.5, "factuality": 4.8, "engagement": 4.4, "composite": 4.57}, "tells": T(0.2), "fool": None}
        cand = {"judge": {"voice": 4.5, "factuality": 4.4, "engagement": 4.4, "composite": 4.43}, "tells": T(0.1), "fool": None}
        v = compare_runs.verdict(base, cand)
        self.assertFalse(v["guard_ok"]); self.assertTrue(v["humanlike_better"])

    def test_within_noise_passes(self):
        base = {"judge": {"voice": 4.5, "factuality": 4.8, "engagement": 4.4, "composite": 4.57}, "tells": T(0.2), "fool": 0.30}
        cand = {"judge": {"voice": 4.45, "factuality": 4.75, "engagement": 4.4, "composite": 4.53}, "tells": T(0.2), "fool": 0.36}
        v = compare_runs.verdict(base, cand)
        self.assertTrue(v["guard_ok"]); self.assertTrue(v["humanlike_better"])

    def test_memory_guard_catches_regression(self):
        base = {"judge": {"voice": 4.5, "factuality": 4.7, "engagement": 4.4, "composite": 4.53},
                "memory": {"voice": 3.8, "memory": 4.5, "naturalness": 4.5, "composite": 4.27},
                "tells": T(0.03), "fool": None}
        cand = {"judge": {"voice": 4.5, "factuality": 4.7, "engagement": 4.4, "composite": 4.53},
                "memory": {"voice": 3.5, "memory": 4.1, "naturalness": 3.9, "composite": 3.83},
                "tells": T(0.02), "fool": None}
        v = compare_runs.verdict(base, cand)
        self.assertFalse(v["guard_ok"])
        self.assertTrue(any("memory" in r for r in v["reasons"]))

    def test_memory_means_from_jsonl(self):
        import json, tempfile
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "memory_scores.jsonl")
            with open(p, "w", encoding="utf-8") as fh:
                fh.write(json.dumps({"id": "a", "scores": {"voice": 4, "memory": 5, "naturalness": 3}}) + "\n")
                fh.write(json.dumps({"id": "b", "scores": {"voice": 2, "memory": 3, "naturalness": 5}}) + "\n")
            m = compare_runs.memory_means([p])
            self.assertAlmostEqual(m["voice"], 3.0)
            self.assertAlmostEqual(m["memory"], 4.0)
            self.assertAlmostEqual(m["naturalness"], 4.0)
            self.assertAlmostEqual(m["composite"], 11.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
