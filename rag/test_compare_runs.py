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


if __name__ == "__main__":
    unittest.main()
