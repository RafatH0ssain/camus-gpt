#!/usr/bin/env python3
"""
test_ingest_sources.py — unit tests for the notebook entry splitter in
kb/ingest_sources.py.

All fixtures are synthetic strings; no PDF is opened and nothing under data/ is
read or written.

    python -m unittest kb/test_ingest_sources.py -v
"""
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ingest_sources as ing  # noqa: E402

PB = ing.PAGE_BREAK
MANIFEST = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "sources_manifest.json")


def lines(*rows):
    return "\n".join(rows)


class DateHeadingTests(unittest.TestCase):
    def test_month_only(self):
        for heading in ("APRIL", "March", "  JULY  ", "décembre"):
            self.assertTrue(ing.is_date_heading(heading), heading)

    def test_month_and_day_or_year(self):
        for heading in ("DECEMBER 15", "April 1948", "MAY 1935", "15 April 1942",
                        "January 1942"):
            self.assertTrue(ing.is_date_heading(heading), heading)

    def test_span_of_dates(self):
        self.assertTrue(ing.is_date_heading("September 1937 - April 1939"))
        self.assertTrue(ing.is_date_heading("January 1942 – September 1945"))

    def test_prose_is_not_a_heading(self):
        for line in ("In March of 1937 he wrote to his teacher",
                     "“Yes.”", "the summer of 1938 was hot", "Chapter One",
                     "15 was the number of the bus", ""):
            self.assertFalse(ing.is_date_heading(line), line)

    def test_heading_is_its_own_boundary_but_heads_the_next_entry(self):
        text = lines("An entry that ends here.", "APRIL",
                     "The first hot days of the year.", "Stifling.")
        self.assertEqual(ing.split_entries(text),
                         ["An entry that ends here.",
                          lines("APRIL", "The first hot days of the year.", "Stifling.")])

    def test_entry_starting_with_a_heading_keeps_it(self):
        text = lines("MARCH", "The wind is cold again.", "Snow on the hills.")
        self.assertEqual(ing.split_entries(text), [text])

    def test_two_headings_in_a_row_stay_together(self):
        text = lines("September 1937", "April 1939", "A note on both of them.")
        self.assertEqual(ing.split_entries(text), [text])


class SeparatorTests(unittest.TestCase):
    def test_separator_lines(self):
        for line in ("*", "***", "—", "——", "===", "~~~", "-"):
            self.assertTrue(ing.is_separator(line), line)

    def test_prose_is_not_a_separator(self):
        for line in ("- and then", "He wrote a *", "—", "a-b-c"):
            if line == "—":
                continue
            self.assertFalse(ing.is_separator(line), line)

    def test_separator_ends_both_sides(self):
        text = lines("First entry line one.", "***", "Second entry line one.")
        self.assertEqual(ing.split_entries(text),
                         ["First entry line one.", "Second entry line one."])

    def test_separator_alone_makes_no_entry(self):
        self.assertEqual(ing.split_entries(lines("***", "*")), [])


class BlankLineTests(unittest.TestCase):
    def test_blank_line_splits_and_hard_wraps_do_not(self):
        text = lines("Entry one wraps", "across two lines.", "",
                     "Entry two is a", "single paragraph.")
        self.assertEqual(ing.split_entries(text),
                         [lines("Entry one wraps", "across two lines."),
                          lines("Entry two is a", "single paragraph.")])

    def test_runs_of_blank_lines_collapse(self):
        text = lines("First entry.", "", "", "", "Second entry.")
        self.assertEqual(ing.split_entries(text), ["First entry.", "Second entry."])

    def test_no_blank_lines_means_one_entry(self):
        text = lines("Line one of a single entry.", "Line two of it.",
                     "Line three, still the same entry.")
        self.assertEqual(ing.split_entries(text), [text])


class PageBreakTests(unittest.TestCase):
    def test_page_break_at_a_paragraph_end_splits(self):
        text = PB.join([
            lines("The first entry ends on a complete", "thought at the foot of the page."),
            lines("The second entry opens with a capital", "letter on the next page."),
        ])
        self.assertEqual(len(ing.split_entries(text)), 2)
        self.assertTrue(ing.split_entries(text)[1].startswith("The second entry"))

    def test_page_break_mid_paragraph_does_not_split(self):
        text = PB.join([
            lines("The first entry runs off the foot of the page without"),
            lines("finishing the thought it started."),
        ])
        self.assertEqual(len(ing.split_entries(text)), 1)

    def test_page_break_mid_word_does_not_split(self):
        text = PB.join([lines("A line that breaks after a soft hy-"),
                        lines("phen at the top of the next page.")])
        self.assertEqual(len(ing.split_entries(text)), 1)

    def test_page_break_before_a_lowercase_continuation_does_not_split(self):
        text = PB.join([lines("The first entry ends here on a page."),
                        lines("and continues in lower case across the break.")])
        self.assertEqual(len(ing.split_entries(text)), 1)

    def test_page_break_into_an_underline_heading_splits(self):
        text = PB.join([lines("The first entry ends here on a page."),
                        lines("APRIL", "The first hot days of the year.")])
        entries = ing.split_entries(text)
        self.assertEqual(entries[0], "The first entry ends here on a page.")
        self.assertEqual(entries[1], lines("APRIL", "The first hot days of the year."))

    def test_blank_line_at_the_top_of_a_page_still_splits(self):
        text = PB.join([lines("The first entry ends here on a page."),
                        lines("", "The second entry opens the next page.")])
        self.assertEqual(len(ing.split_entries(text)), 2)


class IndentTests(unittest.TestCase):
    """The indents stand in for the edition's own line offsets."""

    def page(self, offsets):
        return [(float(off), text) for off, text in offsets]

    def test_entry_indent_marks_a_blank_line(self):
        lines_ = self.page([(55, "body line of the first entry"),
                            (55, "still the first entry"),
                            (69, "the start of the second entry"),
                            (55, "more of the second entry")])
        self.assertEqual(ing.entry_start_indices(lines_), {2})

    def test_indent_becomes_a_blank_line_for_the_splitter(self):
        text = "\n".join([
            ing.entry_page_text(self.page([
                (55, "body line of the first entry"),
                (69, "the start of the second entry")])),
            ing.entry_page_text(self.page([
                (55, "more of the second entry"),
                (69, "a third entry begins here")])),
        ])
        self.assertEqual(ing.split_entries(text), [
            "body line of the first entry",
            lines("the start of the second entry", "more of the second entry"),
            "a third entry begins here",
        ])

    def test_running_head_and_page_number_are_not_entry_starts(self):
        lines_ = self.page([(245, "NOTEBOOK I"), (55, "body line of one entry"),
                            (292, "22"), (69, "an entry that starts here")])
        self.assertEqual(ing.entry_start_indices(lines_), {3})

    def test_body_margin_is_the_common_offset(self):
        lines_ = self.page([(300, "header"), (40, "body one"), (41, "body two"),
                            (42, "body three"), (41, "body four"), (40, "body five"),
                            (300, "footer")])
        self.assertEqual(ing.body_margin(lines_), 40)

    def test_blank_lines_are_never_entry_starts(self):
        lines_ = self.page([(55, "a body line of one entry"), (55, "   "),
                            (69, "the real start of another entry")])
        self.assertEqual(ing.entry_start_indices(lines_), {2})


class OnlyTests(unittest.TestCase):
    """--only names a manifest file; a name that is not in the manifest is fatal,
    so a typo cannot silently drop a source from the run."""

    def call_main(self, *only):
        """Run main() with --only *only, returning (SystemExit or None, stderr)."""
        argv = sys.argv
        err = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "chunks.jsonl")
            sys.argv = ["ingest_sources.py", "--manifest", MANIFEST,
                        "--out", out, "--only", *only]
            raised = None
            try:
                with redirect_stdout(io.StringIO()), redirect_stderr(err):
                    try:
                        ing.main()
                    except SystemExit as exc:
                        raised = exc
            finally:
                sys.argv = argv
            self.assertEqual(os.listdir(tmp), [] if raised is not None
                             else ["chunks.jsonl"])
        return raised, err.getvalue()

    def test_unknown_names_exit_listing_every_one(self):
        raised, err = self.call_main("no_such_book.pdf", "also_missing.pdf")
        self.assertIsNotNone(raised)
        self.assertNotEqual(raised.code, 0)
        for name in ("no_such_book.pdf", "also_missing.pdf"):
            self.assertIn(name, str(raised))
            self.assertIn(name, err)

    @unittest.skipIf(ing.fitz is None, "PyMuPDF not installed")
    def test_known_name_does_not_raise(self):
        with open(MANIFEST, encoding="utf-8") as handle:
            known = {s["file"] for s in json.load(handle)["sources"]}
        raised, _ = self.call_main(sorted(known)[0])
        self.assertIsNone(raised)


if __name__ == "__main__":
    unittest.main()
