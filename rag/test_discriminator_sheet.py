#!/usr/bin/env python3
"""Unit tests for rag/discriminator_sheet.py. Fixtures are built here; no network, no files
outside a temp dir.

    python -m unittest discover -s rag -p "test_*.py"
"""
import json, os, sys, tempfile, unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import discriminator_sheet as ds  # noqa: E402


def words(tag, n):
    return " ".join(f"{tag}{i}" for i in range(n))


MODEL = [words("m", 20 + i) for i in range(12)] + ["too short", words("long", 200)]
REF = [words("r", 25 + i) for i in range(12)] + ["tiny"]


def write_jsonl(path, field, texts):
    with open(path, "w", encoding="utf-8") as f:
        for t in texts:
            f.write(json.dumps({field: t}) + "\n")


class BuildTests(unittest.TestCase):
    def test_counts_filter_and_verbatim(self):
        passages, key = ds.build(MODEL, REF, 10, 1, 15, 120)
        self.assertEqual(len(passages), 20)
        self.assertEqual(key.count("him"), 10)
        self.assertEqual(key.count("machine"), 10)
        for p, k in zip(passages, key):
            self.assertTrue(15 <= ds.nwords(p) <= 120)
            self.assertIn(p, REF if k == "him" else MODEL)

    def test_seed_deterministic_and_shuffled(self):
        a = ds.build(MODEL, REF, 10, 7, 15, 120)
        self.assertEqual(a, ds.build(MODEL, REF, 10, 7, 15, 120))
        self.assertNotEqual(a, ds.build(MODEL, REF, 10, 8, 15, 120))
        self.assertNotEqual(a[1], ["machine"] * 10 + ["him"] * 10)

    def test_not_enough_eligible(self):
        with self.assertRaises(SystemExit):
            ds.build(MODEL, REF, 13, 1, 15, 120)

    def test_speaker_label_stripped_only(self):
        self.assertEqual(ds.clean("Camus: Hello there."), "Hello there.")
        self.assertEqual(ds.clean("  Albert Camus : Hello."), "Hello.")
        self.assertEqual(ds.clean("The sea, Camus: said."), "The sea, Camus: said.")

    def test_render_has_no_labels(self):
        passages, key = ds.build(MODEL, REF, 3, 2, 15, 120)
        sheet = ds.render(passages)
        for i in range(1, 7):
            self.assertIn(f"## {i}", sheet)
        self.assertNotIn("machine", sheet.split("## 1")[1])


class ScoreTests(unittest.TestCase):
    def test_parse_and_score(self):
        key = ["him", "machine", "him", "machine"]
        ans = ds.parse_answers("1 him\n2: m\n3 machine\n4 0\nnoise\n")
        self.assertEqual(ans, {1: "him", 2: "machine", 3: "machine", 4: "machine"})
        correct, answered, total, per = ds.score(key, ans)
        self.assertEqual((correct, answered, total), (3, 4, 4))
        self.assertEqual(per["him"], [1, 2])

    def test_out_of_range_ignored(self):
        self.assertEqual(ds.score(["him"], {5: "him"})[:3], (0, 0, 1))


class CliTests(unittest.TestCase):
    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            m, r = os.path.join(d, "m.jsonl"), os.path.join(d, "r.jsonl")
            write_jsonl(m, "answer", MODEL)
            write_jsonl(r, "response", REF)
            out, keyf = os.path.join(d, "s.md"), os.path.join(d, "key.json")
            ds.main(["--model", m, "--reference", r, "--n", "5", "--seed", "3",
                     "--out", out, "--key", keyf])
            key = json.load(open(keyf))["key"]
            self.assertEqual(len(key), 10)
            ans = os.path.join(d, "a.txt")
            open(ans, "w").write("\n".join(f"{i} {k}" for i, k in enumerate(key, 1)))
            ds.main(["--score", keyf, ans])  # all correct: must not raise


if __name__ == "__main__":
    unittest.main()
