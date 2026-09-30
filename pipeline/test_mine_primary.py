#!/usr/bin/env python3
"""
test_mine_primary.py — unit tests for pipeline/mine_primary.py.

All fixtures are synthetic row dicts; nothing under data/ is read or written.

    python -m unittest pipeline/test_mine_primary.py -v
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mine_primary as mp  # noqa: E402


def row(chunk_id, text, source="Source A", stream="voice", **extra):
    base = {"id": chunk_id, "source": source, "stream": stream,
            "date": "1950", "text": text}
    base.update(extra)
    return base


class FilterTests(unittest.TestCase):
    def test_stream_filter(self):
        rows = [row("v1", "A voice chunk with words here.", stream="voice"),
                row("f1", "A fact chunk with words here.", stream="fact")]
        candidates, per_source = mp.mine_rows(rows)
        self.assertEqual([c["chunk_id"] for c in candidates], ["v1"])
        self.assertEqual(per_source["Source A"], 1)

    def test_source_allowlist(self):
        rows = [row("a1", "Alpha source passage here.", source="Alpha"),
                row("b1", "Beta source passage here.", source="Beta")]
        candidates, _ = mp.mine_rows(rows, sources=["Alpha"])
        self.assertEqual([c["chunk_id"] for c in candidates], ["a1"])

    def test_min_words(self):
        rows = [row("c1", "Too short"), row("c2", "This one has enough words.")]
        candidates, _ = mp.mine_rows(rows, min_words=4)
        self.assertEqual([c["chunk_id"] for c in candidates], ["c2"])

    def test_junk_filter(self):
        rows = [row("c1", "This is a clean passage.\n\n\u00a9\u00ae\u2122\u00a7\u00b6\u2206\u2211")]
        candidates, _ = mp.mine_rows(rows)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["passage"], "This is a clean passage.")

    def test_dedup(self):
        rows = [row("c1", "The same passage of text."),
                row("c2", "the  same   passage of text.")]
        candidates, _ = mp.mine_rows(rows)
        self.assertEqual(len(candidates), 1)


class HeaderTests(unittest.TestCase):
    def test_running_header_removed(self):
        rows = [row(f"c{i}", f"CHAPTER ONE\n\nThis is body text number {i} here.")
                for i in range(7)]
        headers = mp.running_headers(rows)
        self.assertIn("CHAPTER ONE", headers["Source A"])
        candidates, per_source = mp.mine_rows(rows)
        self.assertEqual(per_source["Source A"], 7)
        self.assertTrue(all("CHAPTER ONE" not in c["passage"] for c in candidates))

    def test_header_needs_more_than_five_chunks(self):
        rows = [row(f"c{i}", f"CHAPTER ONE\n\nThis is body text number {i} here.")
                for i in range(5)]
        headers = mp.running_headers(rows)
        self.assertNotIn("Source A", headers)

    def test_page_number_and_roman_removed(self):
        self.assertTrue(mp.is_page_or_roman("12"))
        self.assertTrue(mp.is_page_or_roman("XIV"))
        self.assertFalse(mp.is_page_or_roman("civil"))
        self.assertFalse(mp.is_page_or_roman("hello"))
        rows = [row("c1", "42\n\nA real passage of prose follows here.")]
        candidates, _ = mp.mine_rows(rows)
        self.assertEqual(candidates[0]["passage"], "A real passage of prose follows here.")


class SplitTests(unittest.TestCase):
    def test_blank_line_split(self):
        text = "Para one line one.\n\nPara two line one.\ncontinue two."
        self.assertEqual(
            mp.split_passages(text, 150),
            ["Para one line one.", "Para two line one. continue two."])

    def test_sentence_split_when_no_blank_lines(self):
        text = "One two three. Four five six. Seven eight nine."
        self.assertEqual(
            mp.split_passages(text, 4),
            ["One two three.", "Four five six.", "Seven eight nine."])

    def test_sentence_split_never_breaks_a_sentence(self):
        self.assertEqual(
            mp.split_passages("One two three four.", 2),
            ["One two three four."])

    def test_max_words_packs_sentences(self):
        text = "One two. Three four. Five six. Seven eight."
        self.assertEqual(
            mp.split_passages(text, 5),
            ["One two. Three four.", "Five six. Seven eight."])


class EntryTests(unittest.TestCase):
    def test_entry_kept_whole_under_max_words(self):
        text = "APRIL\n\nFirst sentence of the entry. Second one closes it."
        rows = [row("n1", text, entry=True)]
        candidates, _ = mp.mine_rows(rows)
        self.assertEqual(candidates[0]["passage"],
                         "APRIL First sentence of the entry. Second one closes it.")
        self.assertTrue(candidates[0]["entry"])

    def test_entry_never_splits_on_blank_lines(self):
        rows = [row("n1", "One paragraph of an entry.\n\nA second paragraph of it.",
                    entry=True)]
        candidates, _ = mp.mine_rows(rows)
        self.assertEqual(len(candidates), 1)

    def test_entry_over_max_words_splits_on_sentences(self):
        text = ("One two three. Four five six. Seven eight nine. Ten eleven twelve. "
                "Thirteen fourteen.")
        rows = [row("n1", text, entry=True)]
        candidates, _ = mp.mine_rows(rows, min_words=2, max_words=6)
        self.assertEqual([c["passage"] for c in candidates],
                         ["One two three. Four five six.",
                          "Seven eight nine. Ten eleven twelve.", "Thirteen fourteen."])
        self.assertTrue(all(c["entry"] for c in candidates))

    def test_entry_split_never_breaks_a_sentence(self):
        rows = [row("n1", "One two three four five. Six seven.", entry=True)]
        candidates, _ = mp.mine_rows(rows, min_words=2, max_words=3)
        self.assertEqual([c["passage"] for c in candidates],
                         ["One two three four five.", "Six seven."])

    def test_entry_under_min_words_dropped(self):
        rows = [row("n1", "Yes.", entry=True), row("n2", "A whole entry of prose here.")]
        candidates, _ = mp.mine_rows(rows, min_words=4)
        self.assertEqual([c["chunk_id"] for c in candidates], ["n2"])

    def test_plain_rows_are_untouched(self):
        rows = [row("c1", "One paragraph of a chunk.\n\nA second paragraph of it.")]
        candidates, _ = mp.mine_rows(rows)
        self.assertEqual(len(candidates), 2)
        self.assertNotIn("entry", candidates[0])

    def test_entry_page_number_still_stripped(self):
        rows = [row("n1", "42\n\nA real notebook entry of prose follows here.",
                    entry=True)]
        candidates, _ = mp.mine_rows(rows)
        self.assertEqual(candidates[0]["passage"],
                         "A real notebook entry of prose follows here.")

    def test_split_entry_passage_collapses_whitespace(self):
        self.assertEqual(
            mp.split_entry_passage("  one\ntwo  \nthree\n\nfour  ", 150),
            ["one two three four"])
        self.assertEqual(mp.split_entry_passage("   ", 150), [""])


if __name__ == "__main__":
    unittest.main()
