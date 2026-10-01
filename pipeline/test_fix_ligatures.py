#!/usr/bin/env python3
"""
test_fix_ligatures.py — unit tests for pipeline/fix_ligatures.py.

The join tests run against the real dictionary (/usr/share/dict/words), since the
whole safety argument is "is this joined string a word?". The file tests write
small JSONL fixtures to a temp dir, so the real corpora are never touched, and
they go through main() so the CLI, the byte-preservation rule and the reporting
are all covered.

    python -m unittest pipeline/test_fix_ligatures.py -v
"""
import io
import json
import contextlib
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fix_ligatures as fl  # noqa: E402

WORDS = fl.load_words()

# every one of these has a joined form that is a dictionary word
POSITIVES = (
    ("fi lth", "filth", "A"),
    ("suffi ciently", "sufficiently", "A"),
    ("eff ect", "effect", "A"),
    ("di ff erent", "different", "B"),
    ("fl ight", "flight", "A"),
    ("in fi nite", "infinite", "B"),
    ("snugly fi fty", "snugly fifty", "A"),
)

# none of these joins into a word, or every piece already is one
NEGATIVES = (
    "off the",
    "stuff that",
    "staff meeting",
    "cliff edge",
    "the staff meeting was off the cuff of stuff",
    "a staff man",          # "staffman" is a word, but both pieces are words too
    "fi Lth",               # a capital after the split is not a split
    "off5 fi",
    "suffi zz",
)


def run(*argv):
    """main() with its output captured; returns (exit code, stdout)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = fl.main(list(argv))
    return code, buf.getvalue()


def write_bytes(path, body):
    path.write_bytes(body if isinstance(body, bytes) else body.encode("utf-8"))


def write_jsonl(path, rows):
    body = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    write_bytes(path, body)
    return body


def read_lines(path):
    return path.read_bytes().decode("utf-8").split("\n")[:-1]


class DictionaryTests(unittest.TestCase):
    def test_load_words_lowercases_and_strips(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "words"
            path.write_text("Filth\nSuffi\n\nCIENTLY\n", encoding="utf-8")
            self.assertEqual(fl.load_words(path), {"filth", "suffi", "ciently"})

    def test_load_words_missing_dictionary_is_fatal(self):
        with self.assertRaises(SystemExit):
            fl.load_words("/usr/share/dict/no-such-word-list")

    def test_real_dictionary_has_the_words_the_tests_rely_on(self):
        for word in ("filth", "sufficiently", "effect", "different", "flight",
                     "staffman"):
            self.assertIn(word, WORDS)

    def test_bare_strips_surrounding_punctuation_and_case(self):
        self.assertEqual(fl.bare('"Filth,'), "filth")
        self.assertEqual(fl.bare("suffi"), "suffi")

    def test_ligature_shapes(self):
        self.assertEqual([t for t in ("suffi", "cliff", "off", "effl", "fable")
                          if fl.ends_with_ligature(t)], ["suffi", "cliff", "off", "effl"])
        self.assertEqual([t for t in ("fi", "ffl", "ffi", "suffi", "if")
                          if fl.is_standalone_ligature(t)], ["fi", "ffl", "ffi"])


class JoinTests(unittest.TestCase):
    def test_joins(self):
        for before, after, rule in POSITIVES:
            with self.subTest(before=before):
                fixed, joins = fl.fix_text(before, WORDS)
                self.assertEqual(fixed, after)
                self.assertEqual([j["rule"] for j in joins], [rule])
                self.assertEqual(joins[0]["word"], after.split()[-1])
                self.assertIn(" ", joins[0]["before"])

    def test_leaves_real_words_alone(self):
        for before in NEGATIVES:
            with self.subTest(before=before):
                fixed, joins = fl.fix_text(before, WORDS)
                self.assertEqual(fixed, before)
                self.assertEqual(joins, [])

    def test_keeps_punctuation_of_the_surviving_piece(self):
        self.assertEqual(fl.fix_text("The fi lth, and it.", WORDS)[0],
                         "The filth, and it.")

    def test_several_joins_in_one_text(self):
        text = "the fi lth and the fl ight and di ff erent and suffi ciently"
        fixed, joins = fl.fix_text(text, WORDS)
        self.assertEqual(fixed, "the filth and the flight and different and sufficiently")
        self.assertEqual(len(joins), 4)

    def test_join_is_idempotent(self):
        for before, after, _ in POSITIVES:
            with self.subTest(before=before):
                again, joins = fl.fix_text(after, WORDS)
                self.assertEqual(again, after)
                self.assertEqual(joins, [])

    def test_only_whitespace_between_the_pieces_is_removed(self):
        fixed, _ = fl.fix_text("signifi cant.\n\nThen  more", WORDS)
        self.assertEqual(fixed, "significant.\n\nThen  more")

    def test_sample_before_and_after_are_readable(self):
        _, joins = fl.fix_text("and the fi lth of the town", WORDS)
        self.assertEqual(joins[0]["before"], "and the fi lth of the town")
        self.assertEqual(joins[0]["after"], "and the filth of the town")

    def test_a_standalone_ligature_needs_both_neighbours(self):
        for text in ("ff", "ff x", "x ff"):
            with self.subTest(text=text):
                self.assertEqual(fl.fix_text(text, WORDS)[0], text)

    def test_a_custom_dictionary_decides(self):
        tiny = {"ight", "flight"}   # "fl" itself is deliberately not a word here
        self.assertEqual(fl.fix_text("fl ight", tiny)[0], "flight")
        self.assertEqual(fl.fix_text("filth", tiny)[0], "filth")


class FileTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.rows = [
            {"prompt": "p", "response": "Nothing needed saying.", "id": 0},
            {"prompt": "p", "response": "the fi lth of the town", "id": 1},
            {"prompt": "p", "response": None, "id": 2},
            {"prompt": "p", "id": 3},
            {"prompt": "p", "response": "off the cuff, staff meeting", "id": 4},
            {"prompt": "p", "response": "suffi ciently diffi cult", "id": 5},
        ]
        self.path = self.dir / "camus_sft.jsonl"
        self.body = write_jsonl(self.path, self.rows)
        self.addCleanup(self.tmp.cleanup)

    def rewrite(self, *extra):
        return run(str(self.path), "--field", "response", *extra)

    def test_dry_run_leaves_the_file_untouched(self):
        code, out = self.rewrite("--dry-run")
        self.assertEqual(code, 0)
        self.assertEqual(self.path.read_bytes().decode("utf-8"), self.body)
        self.assertIn("rows changed 2", out)
        self.assertIn("joins 3", out)
        self.assertIn("dry-run", out)

    def test_rewrites_only_the_rows_that_change(self):
        before = read_lines(self.path)
        self.rewrite()
        after = read_lines(self.path)
        self.assertEqual(len(before), len(after))
        for i, (old, new) in enumerate(zip(before, after)):
            if i in (1, 5):
                self.assertNotEqual(old, new, f"row {i} should have changed")
            else:
                self.assertEqual(old, new, f"row {i} must keep its original bytes")
        self.assertEqual(json.loads(after[1])["response"], "the filth of the town")
        self.assertEqual(json.loads(after[5])["response"], "sufficiently difficult")

    def test_row_order_keys_and_count_survive(self):
        self.rewrite()
        rows = [json.loads(line) for line in read_lines(self.path)]
        self.assertEqual([r["id"] for r in rows], [0, 1, 2, 3, 4, 5])
        self.assertEqual([list(r.keys()) for r in rows],
                         [list(r.keys()) for r in self.rows])

    def test_running_twice_changes_nothing_the_second_time(self):
        self.rewrite()
        once = self.path.read_bytes()
        _, out = self.rewrite()
        self.assertEqual(self.path.read_bytes(), once)
        self.assertIn("rows changed 0", out)
        self.assertIn("joins 0", out)
        self.assertIn("no joins", out)

    def test_file_without_joins_is_byte_identical(self):
        clean = self.dir / "camus_clean.jsonl"
        write_jsonl(clean, [{"response": "off the cuff and a staff meeting"}])
        before = clean.read_bytes()
        code, out = run(str(clean), "--field", "response")
        self.assertEqual(code, 0)
        self.assertEqual(clean.read_bytes(), before)
        self.assertIn("rows changed 0", out)

    def test_missing_trailing_newline_is_preserved(self):
        path = self.dir / "camus_nonl.jsonl"
        write_bytes(path, json.dumps({"response": "fi lth"}) + "\n"
                    + json.dumps({"response": "fine"}))
        self.assertEqual(path.read_bytes()[-1:], b"}")
        run(str(path), "--field", "response")
        self.assertEqual(path.read_bytes()[-1:], b"}")

    def test_invalid_json_line_is_left_alone(self):
        path = self.dir / "camus_bad.jsonl"
        write_bytes(path, "{not json}\n" + json.dumps({"response": "fi lth"}) + "\n")
        code, _ = run(str(path), "--field", "response")
        self.assertEqual(code, 0)
        self.assertEqual(read_lines(path)[0], "{not json}")
        self.assertEqual(json.loads(read_lines(path)[1])["response"], "filth")

    def test_samples_show_before_and_after(self):
        _, out = self.rewrite()
        self.assertIn("'the fi lth of the town'", out)
        self.assertIn("'the filth of the town'", out)
        self.assertIn("sample joins (3 of 3)", out)

    def test_every_field_is_repaired(self):
        path = self.dir / "camus_kb.jsonl"
        write_jsonl(path, [{"text": "signifi cant", "quote": None, "id": 7},
                           {"text": "clean text", "quote": "the fi lth", "id": 8}])
        _, out = run(str(path), "--field", "text", "--field", "quote")
        self.assertIn("rows changed 2", out)
        self.assertIn("joins 2", out)
        rows = [json.loads(line) for line in read_lines(path)]
        self.assertEqual(rows[0]["text"], "significant")
        self.assertIsNone(rows[0]["quote"])
        self.assertEqual(rows[1]["text"], "clean text")
        self.assertEqual(rows[1]["quote"], "the filth")

    def test_messages_field_repairs_assistant_turns_only(self):
        path = self.dir / "camus_multiturn.jsonl"
        write_jsonl(path, [{"messages": [
            {"role": "user", "content": "why fi lth?"},
            {"role": "assistant", "content": "because di ff erent"},
        ]}])
        _, out = run(str(path), "--field", "messages")
        self.assertIn("rows changed 1", out)
        messages = json.loads(read_lines(path)[0])["messages"]
        self.assertEqual(messages[0]["content"], "why fi lth?")
        self.assertEqual(messages[1]["content"], "because different")

    def test_a_field_is_required(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                run(str(self.path))


if __name__ == "__main__":
    unittest.main()