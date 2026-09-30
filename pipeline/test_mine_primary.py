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
        candidates, per_source, _ = mp.mine_rows(rows)
        self.assertEqual([c["chunk_id"] for c in candidates], ["v1"])
        self.assertEqual(per_source["Source A"], 1)

    def test_source_allowlist(self):
        rows = [row("a1", "Alpha source passage here.", source="Alpha"),
                row("b1", "Beta source passage here.", source="Beta")]
        candidates, _, _ = mp.mine_rows(rows, sources=["Alpha"])
        self.assertEqual([c["chunk_id"] for c in candidates], ["a1"])

    def test_min_words(self):
        rows = [row("c1", "Too short"), row("c2", "This one has enough words.")]
        candidates, _, _ = mp.mine_rows(rows, min_words=4)
        self.assertEqual([c["chunk_id"] for c in candidates], ["c2"])

    def test_junk_filter(self):
        rows = [row("c1", "This is a clean passage.\n\n\u00a9\u00ae\u2122\u00a7\u00b6\u2206\u2211")]
        candidates, _, _ = mp.mine_rows(rows)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["passage"], "This is a clean passage.")

    def test_dedup(self):
        rows = [row("c1", "The same passage of text."),
                row("c2", "the  same   passage of text.")]
        candidates, _, _ = mp.mine_rows(rows)
        self.assertEqual(len(candidates), 1)


class JunkFilterTests(unittest.TestCase):
    """The three filters that drop passages which are not Camus talking."""

    def test_editorial_note_names_camus(self):
        for passage in ("The editor notes that Camus wrote this in April.",
                        "Camus kept to his diary every evening.",
                        "Camus's own reply was brief and cold."):
            self.assertTrue(mp.is_editorial_note(passage), passage)
        self.assertFalse(mp.is_editorial_note("The secretary said it was raining."))
        self.assertFalse(mp.is_editorial_note("A Camusian reading of the plague."))
        # The name is matched as capitalised: a lowercase "camus" is a word in
        # running text, not the editors' habit of naming him.
        self.assertFalse(mp.is_editorial_note("In french, camus means nothing."))

    def test_editorial_note_passage_removed(self):
        rows = [row("c1", "A notebook entry of plain prose here."),
                row("c2", "Camus had destroyed the manuscript in the fire.")]
        candidates, _, filtered = mp.mine_rows(rows)
        self.assertEqual([c["chunk_id"] for c in candidates], ["c1"])
        self.assertEqual(filtered["editorial notes (names Camus)"], 1)

    def test_editorial_note_filters_notebook_entries_too(self):
        rows = [row("n1", "APRIL\n\nCamus had not yet read the manuscript.",
                    entry=True)]
        candidates, _, filtered = mp.mine_rows(rows)
        self.assertEqual(candidates, [])
        self.assertEqual(filtered["editorial notes (names Camus)"], 1)

    def test_front_back_matter_phrases(self):
        phrases = ("Typography", "binding design", "ISBN 978-0-19-xxxx",
                   "Copyright 1958", "All Rights Reserved", "Translated by",
                   "translated from the French", "Library of Congress",
                   "Printed in England", "First Edition", "Published by",
                   "Edited by Justin O'Brien", "Introduction by an old friend")
        for phrase in phrases:
            self.assertTrue(mp.is_front_back_matter(f"Colophon: {phrase} here."), phrase)
        self.assertFalse(mp.is_front_back_matter("The rebel is a man who says no."))
        self.assertFalse(mp.is_front_back_matter("He published nothing that year."))

    def test_front_back_matter_passage_removed(self):
        rows = [row("c1", "First Edition. Published by the author in Algiers."),
                row("c2", "A real passage of the letter follows here.")]
        candidates, _, filtered = mp.mine_rows(rows)
        self.assertEqual([c["chunk_id"] for c in candidates], ["c2"])
        self.assertEqual(filtered["front/back matter"], 1)

    def test_apparatus_line_openers(self):
        for passage in ("See also the note on measure.",
                        "Cf. the letter to Sartre of that year.",
                        "Note: the French reads otherwise.",
                        "[Translator's note: the idiom is literal here.",
                        "[Editor's note: the date is disputed.",
                        "12. And the second clause of the entry follows."):
            self.assertTrue(mp.is_apparatus_line(passage), passage)
        for passage in ("One must see the whole of it before judging.",
                        "On 12. And then the second clause follows here.",
                        "A note: of that kind, briefly stated.",
                        "No. 4 was the only one that answered."):
            self.assertFalse(mp.is_apparatus_line(passage), passage)

    def test_apparatus_line_passage_removed(self):
        rows = [row("c1", "See the first volume for the full discussion."),
                row("c2", "A real passage of prose follows here.")]
        candidates, _, filtered = mp.mine_rows(rows)
        self.assertEqual([c["chunk_id"] for c in candidates], ["c2"])
        self.assertEqual(filtered["footnote/apparatus lines"], 1)

    def test_each_filter_counted_separately(self):
        rows = [row("c1", "APRIL\n\nThe editor reports that Camus was terse today.",
                    entry=True),
                row("c2", "First Edition. Published by the author, printed in Paris."),
                row("c3", "Cf. the previous entry, which says the opposite."),
                row("c4", "The typhus reached the fourth house by evening.")]
        candidates, _, filtered = mp.mine_rows(rows)
        self.assertEqual([c["chunk_id"] for c in candidates], ["c4"])
        self.assertEqual(dict(filtered), {
            "editorial notes (names Camus)": 1,
            "front/back matter": 1,
            "footnote/apparatus lines": 1,
        })

    def test_a_passage_is_counted_under_the_first_matching_filter(self):
        # editorial, then front/back matter, then apparatus.
        rows = [row("c1", "Note: Camus, ISBN 0-19-xxxx, colophon and all."),
                row("c2", "See the copyright page for the full details.")]
        candidates, _, filtered = mp.mine_rows(rows)
        self.assertEqual(candidates, [])
        self.assertEqual(dict(filtered),
                         {"editorial notes (names Camus)": 1,
                          "front/back matter": 1})


class HeaderTests(unittest.TestCase):
    def test_running_header_removed(self):
        rows = [row(f"c{i}", f"CHAPTER ONE\n\nThis is body text number {i} here.")
                for i in range(7)]
        headers = mp.running_headers(rows)
        self.assertIn("CHAPTER ONE", headers["Source A"])
        candidates, per_source, _ = mp.mine_rows(rows)
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
        candidates, _, _ = mp.mine_rows(rows)
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
        candidates, _, _ = mp.mine_rows(rows)
        self.assertEqual(candidates[0]["passage"],
                         "APRIL First sentence of the entry. Second one closes it.")
        self.assertTrue(candidates[0]["entry"])

    def test_entry_never_splits_on_blank_lines(self):
        rows = [row("n1", "One paragraph of an entry.\n\nA second paragraph of it.",
                    entry=True)]
        candidates, _, _ = mp.mine_rows(rows)
        self.assertEqual(len(candidates), 1)

    def test_entry_over_max_words_splits_on_sentences(self):
        text = ("One two three. Four five six. Seven eight nine. Ten eleven twelve. "
                "Thirteen fourteen.")
        rows = [row("n1", text, entry=True)]
        candidates, _, _ = mp.mine_rows(rows, min_words=2, max_words=6)
        self.assertEqual([c["passage"] for c in candidates],
                         ["One two three. Four five six.",
                          "Seven eight nine. Ten eleven twelve.", "Thirteen fourteen."])
        self.assertTrue(all(c["entry"] for c in candidates))

    def test_entry_split_never_breaks_a_sentence(self):
        rows = [row("n1", "One two three four five. Six seven.", entry=True)]
        candidates, _, _ = mp.mine_rows(rows, min_words=2, max_words=3)
        self.assertEqual([c["passage"] for c in candidates],
                         ["One two three four five.", "Six seven."])

    def test_entry_under_min_words_dropped(self):
        rows = [row("n1", "Yes.", entry=True), row("n2", "A whole entry of prose here.")]
        candidates, _, _ = mp.mine_rows(rows, min_words=4)
        self.assertEqual([c["chunk_id"] for c in candidates], ["n2"])

    def test_plain_rows_are_untouched(self):
        rows = [row("c1", "One paragraph of a chunk.\n\nA second paragraph of it.")]
        candidates, _, _ = mp.mine_rows(rows)
        self.assertEqual(len(candidates), 2)
        self.assertNotIn("entry", candidates[0])

    def test_entry_page_number_still_stripped(self):
        rows = [row("n1", "42\n\nA real notebook entry of prose follows here.",
                    entry=True)]
        candidates, _, _ = mp.mine_rows(rows)
        self.assertEqual(candidates[0]["passage"],
                         "A real notebook entry of prose follows here.")

    def test_split_entry_passage_collapses_whitespace(self):
        self.assertEqual(
            mp.split_entry_passage("  one\ntwo  \nthree\n\nfour  ", 150),
            ["one two three four"])
        self.assertEqual(mp.split_entry_passage("   ", 150), [""])


if __name__ == "__main__":
    unittest.main()
