#!/usr/bin/env python3
"""
test_verify_quotes.py — unit tests for pipeline/verify_quotes.py.

Every fixture is synthetic: made-up sentences in made-up books, none of it
Camus. Nothing under data/ is read or written, and the end-to-end case works
inside a temporary directory.

    python -m unittest pipeline/test_verify_quotes.py -v
"""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import verify_quotes as vq  # noqa: E402


def chunk(chunk_id, source, text):
    return {"id": chunk_id, "source": source, "stream": "voice",
            "date": "1942", "text": text}


def corpus(*chunks):
    return vq.Corpus(list(chunks))


def quote(text, web_status="sourced", claimed_source="", site="test",
          url="https://example.invalid/q"):
    return {"text": text, "claimed_source": claimed_source, "site": site,
            "url": url, "web_status": web_status}


LONG = ("the house was quiet and the street was quiet and nothing moved for a "
        "long while, so he sat down by the window and counted the hours until "
        "morning, and the counting took him past the point of sleep, and past "
        "the point of wanting to sleep, until the light came up grey over the "
        "roofs and the day began its ordinary business of noise and errands. ")


class NormaliseTests(unittest.TestCase):
    def test_lowercase_and_whitespace(self):
        self.assertEqual(vq.normalise("  He   Said\n\nThat  "), "he said that")

    def test_curly_quotes_and_apostrophes_unified(self):
        self.assertEqual(vq.normalise("\u201cL\u2019invitation\u201d \u2018hier\u2019"),
                         "l invitation hier")

    def test_dashes_unified_then_stripped(self):
        self.assertEqual(vq.normalise("a\u2013b \u2014 c \u2212d"), "a b c d")

    def test_punctuation_stripped(self):
        self.assertEqual(vq.normalise("Oui!; \u00abOui\u00bb: (mais) 1,000?"),
                         "oui oui mais 1 000")

    def test_tokenise_agrees_with_normalise(self):
        for text in ("L'\u00e9t\u00e9 \u2014 l\u2019HIVER\u2026", "a.b;c/d!",
                     "", "   ", "3, 14.15 \u2013 1,000"):
            self.assertEqual(vq.tokenise(text), vq.normalise(text).split())

    def test_window_sizes_within_slack(self):
        self.assertEqual(vq.window_sizes(20), list(range(16, 25)))
        self.assertEqual(vq.window_sizes(5), [4, 5, 6])
        self.assertEqual(vq.window_sizes(1), [1])
        self.assertEqual(vq.window_sizes(10), list(range(8, 13)))

    def test_excerpt_keeps_original_punctuation_and_word_cap(self):
        text = "First, he said: \u201cNo \u2014 never!\u201d Then he went out, quietly."
        self.assertEqual(vq.excerpt(text, 0, max_words=6),
                         'First, he said: \u201cNo \u2014 never!\u201d Then')
        self.assertEqual(vq.excerpt(text, 6), "he went out, quietly")
        self.assertEqual(vq.excerpt(text, 99), "")

    def test_excerpt_capped_at_forty_words(self):
        excerpt = vq.excerpt(LONG, 3, max_words=40)
        self.assertEqual(len(excerpt.split()), 40)


class BestMatchTests(unittest.TestCase):
    def test_exact_match_after_normalisation(self):
        books = corpus(chunk("a1", "Book A",
                             "\u201cI know the answer to everything \u2014 and you,\u201d "
                             "he said, and left."))
        match = vq.best_match(books, "i know the answer to everything and you")
        self.assertEqual(match.score, 1.0)
        self.assertEqual(match.source, "Book A")
        self.assertTrue(match.excerpt.startswith(
            "I know the answer to everything \u2014 and you"), match.excerpt)

    def test_exact_match_keeps_line_breaks(self):
        books = corpus(chunk("a1", "Notebook",
                             "January.\nHe wrote: the world is\nnot to be made "
                             "better.\nFebruary."))
        self.assertEqual(vq.best_match(books, "The world is not to be made better.").score,
                         1.0)

    def test_exact_match_reported_once_from_the_shortest_chunk(self):
        books = corpus(chunk("long", "Book B", LONG + "the world is not to be "
                                                 "made better by men like us. "),
                       chunk("short", "Book A", "The world is not to be made better."))
        match = vq.best_match(books, "the world is not to be made better")
        self.assertEqual((match.score, match.source), (1.0, "Book A"))

    def test_rarest_token_decides_the_source(self):
        books = corpus(
            chunk("a1", "Book A",
                  "and the of and the of and the of and the of and the of"),
            chunk("b1", "Book B",
                  "he found a peculiar chronometer that had stopped in the hall"))
        match = vq.best_match(books, "a peculiar chronometer that had stopped")
        self.assertEqual((match.score, match.source), (1.0, "Book B"))

    def test_passage_found_mid_chunk_in_a_long_window(self):
        books = corpus(chunk("a1", "Book A", LONG + "She had never seen such a "
                                                "yellow light on the water. " + LONG))
        match = vq.best_match(books, "she had never seen such a yellow light on "
                                     "the water")
        self.assertEqual(match.score, 1.0)
        self.assertEqual(match.source, "Book A")

    def test_different_translation_lands_in_the_middle_band(self):
        books = corpus(chunk("a1", "Book A",
                             "In the depth of winter I at last discovered within "
                             "myself an invincible summer."))
        match = vq.best_match(books,
                              "In the depth of winter, I finally learned that "
                              "within me there laid an invincible summer.",
                              )
        self.assertEqual(match.source, "Book A")
        self.assertTrue(vq.LIKELY_AT <= match.score < vq.VERIFIED_AT,
                        f"score {match.score} outside the translation band")
        self.assertEqual(vq.match_status(round(match.score, 2)),
                         "likely_translation")

    def test_near_miss_below_the_band_is_not_found(self):
        books = corpus(chunk("a1", "Book A",
                             "Nothing in the ledger resembles the quotation at all."))
        match = vq.best_match(books, "In the depth of winter I finally learned "
                                     "that within me there laid an invincible "
                                     "summer")
        self.assertLess(match.score, vq.LIKELY_AT)
        self.assertEqual(vq.match_status(match.score), "not_found")

    def test_absent_quote_scores_zero_without_a_source(self):
        match = vq.best_match(corpus(chunk("a1", "Book A", "A quiet morning.")),
                              "quincailles metaphysique hydrophobe")
        self.assertEqual(match, vq.Match(0.0, "", ""))

    def test_empty_quote_scores_zero(self):
        books = corpus(chunk("a1", "Book A", "A quiet morning."))
        self.assertEqual(vq.best_match(books, ""), vq.Match(0.0, "", ""))
        self.assertEqual(vq.best_match(books, "   -- \u2026 "), vq.Match(0.0, "", ""))

    def test_weak_signal_still_reports_its_source(self):
        books = corpus(chunk("a1", "Book A",
                             "he thought of the winter and the summer and the "
                             "invincible summer he had heard about"))
        match = vq.best_match(books, "the invincible summer was a lie told by a "
                                     "clerk in another town entirely")
        self.assertGreater(match.score, 0.0)
        self.assertLess(match.score, vq.LIKELY_AT)
        self.assertEqual(match.source, "Book A")
        self.assertTrue(match.excerpt)

    def test_both_corpus_files_are_searched(self):
        quotes = [quote("The world is not to be made better."),
                  quote("A summer that cannot be defeated.")]
        books = corpus(chunk("s1", "Book S", "A summer that cannot be defeated."),
                       chunk("p1", "Book P", "The world is not to be made better."))
        results = vq.verify_all(books, quotes)
        self.assertEqual([r["matched_source"] for r in results], ["Book P", "Book S"])
        self.assertEqual({r["final"] for r in results}, {"verified"})

    def test_duplicate_text_across_the_two_files_is_harmless(self):
        books = corpus(chunk("s1", "Book S", "The world is not to be made better."),
                       chunk("p1", "Book P", "The world is not to be made better."))
        self.assertEqual(vq.best_match(books, "The world is not to be made better.").score,
                         1.0)

    def test_window_bound_never_hides_a_real_match(self):
        pad = " ".join(f"word{index}" for index in range(400))
        books = corpus(chunk("a1", "Book A",
                             f"{pad} the medallion of the chancellor of the city "
                             f"{pad}"))
        match = vq.best_match(books, "the medallion of the chancellor of the city")
        self.assertEqual(match.score, 1.0)
        self.assertTrue(match.excerpt.startswith(
            "the medallion of the chancellor of the city"), match.excerpt)
        self.assertLessEqual(len(match.excerpt.split()), vq.EXCERPT_MAX_WORDS)


class StatusTests(unittest.TestCase):
    def test_match_status_bands(self):
        self.assertEqual(vq.match_status(1.0), "verified")
        self.assertEqual(vq.match_status(vq.VERIFIED_AT), "verified")
        self.assertEqual(vq.match_status(0.8499), "likely_translation")
        self.assertEqual(vq.match_status(vq.LIKELY_AT), "likely_translation")
        self.assertEqual(vq.match_status(0.5999), "not_found")
        self.assertEqual(vq.match_status(0.0), "not_found")

    def test_final_for_every_web_status(self):
        for status, expected in (("sourced", "sourced_not_in_corpus"),
                                 ("misattributed", "misattributed"),
                                 ("disputed", "disputed"),
                                 ("popular_unverified", "unverified"),
                                 ("something_new", "unverified")):
            self.assertEqual(vq.decide(status, "not_found"), expected)

    def test_found_quotes_ignore_the_web_status(self):
        for status in vq.WEB_STATUSES:
            self.assertEqual(vq.decide(status, "verified"), "verified")
            self.assertEqual(vq.decide(status, "likely_translation"),
                             "likely_translation")


class VerifyRowTests(unittest.TestCase):
    def setUp(self):
        self.books = corpus(
            chunk("a1", "Book A", "The world is not to be made better."),
            chunk("a2", "Book B", "Only the sea is calm tonight."))

    def test_row_keeps_input_fields_and_adds_six(self):
        row = quote("The world is not to be made better.", claimed_source="Book A")
        out = vq.verify_row(self.books, row)
        self.assertEqual(sorted(out), sorted(list(row) + ["match_status", "score",
                                                           "matched_source",
                                                           "matched_excerpt", "final"]))
        self.assertEqual(list(out)[:len(row)], list(row))
        for field in ("match_status", "score", "matched_source",
                      "matched_excerpt", "final"):
            self.assertIn(field, out)
        self.assertEqual(out["match_status"], "verified")
        self.assertEqual(out["score"], 1.0)
        self.assertEqual(out["matched_source"], "Book A")
        self.assertEqual(out["final"], "verified")
        self.assertEqual(len(row), 5)

    def test_missing_text_is_handled(self):
        out = vq.verify_row(self.books, {"web_status": "sourced"})
        self.assertEqual(out["match_status"], "not_found")
        self.assertEqual(out["final"], "sourced_not_in_corpus")
        self.assertEqual(out["matched_source"], "")
        self.assertEqual(out["matched_excerpt"], "")

    def test_status_is_read_off_the_rounded_score(self):
        books = corpus(chunk("a1", "Book A", "Nothing relevant here at all."))
        with mock.patch.object(vq, "best_match",
                               return_value=vq.Match(0.847, "Book A", "passage")):
            out = vq.verify_row(books, quote("anything"))
        self.assertEqual(out["score"], 0.85)
        self.assertEqual(out["match_status"], "verified")
        self.assertEqual(out["final"], "verified")

    def test_score_is_rounded_to_two_decimals(self):
        books = corpus(chunk("a1", "Book A",
                             "he walked the long road toward the silent house"))
        with mock.patch.object(vq, "best_match",
                               return_value=vq.Match(0.87654321, "Book A", "x")):
            out = vq.verify_row(books, quote("anything"))
        self.assertEqual(out["score"], 0.88)


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.results = [
            {"text": "A verified line from the book.", "claimed_source": "Book A",
             "site": "s", "url": "u", "web_status": "sourced",
             "match_status": "verified", "score": 1.0,
             "matched_source": "Book A", "matched_excerpt": "A verified line",
             "final": "verified"},
            {"text": "A line the corpus verifies against the web's claim.",
             "claimed_source": "Book B", "site": "s", "url": "u",
             "web_status": "misattributed", "match_status": "verified",
             "score": 0.91, "matched_source": "Book A",
             "matched_excerpt": "A line the corpus", "final": "verified"},
            {"text": "A translation of a line from the book.", "claimed_source": "",
             "site": "s", "url": "u", "web_status": "popular_unverified",
             "match_status": "likely_translation", "score": 0.72,
             "matched_source": "Book A", "matched_excerpt": "A translation",
             "final": "likely_translation"},
            {"text": "From a work we do not hold in the corpus.", "claimed_source": "",
             "site": "s", "url": "u", "web_status": "sourced",
             "match_status": "not_found", "score": 0.12, "matched_source": "",
             "matched_excerpt": "", "final": "sourced_not_in_corpus"},
            {"text": "Denied to him but on the internet.", "claimed_source": "",
             "site": "s", "url": "u", "web_status": "misattributed",
             "match_status": "not_found", "score": 0.31, "matched_source": "",
             "matched_excerpt": "", "final": "misattributed"},
            {"text": "Sourced to the wrong work.", "claimed_source": "",
             "site": "s", "url": "u", "web_status": "disputed",
             "match_status": "not_found", "score": 0.2, "matched_source": "",
             "matched_excerpt": "", "final": "disputed"},
            {"text": "Everyone knows this one and nobody can place it.",
             "claimed_source": "", "site": "s", "url": "u",
             "web_status": "popular_unverified", "match_status": "not_found",
             "score": 0.0, "matched_source": "", "matched_excerpt": "",
             "final": "unverified"},
        ]

    def test_count_table_totals(self):
        web, finals, table = vq.count_table(self.results)
        self.assertEqual(web, ["sourced", "disputed", "misattributed",
                               "popular_unverified"])
        self.assertEqual(finals, ["verified", "likely_translation",
                                  "sourced_not_in_corpus", "misattributed",
                                  "disputed", "unverified"])
        self.assertEqual(table[("sourced", "verified")], 1)
        self.assertEqual(table[("sourced", "sourced_not_in_corpus")], 1)
        self.assertEqual(table[("misattributed", "verified")], 1)
        self.assertEqual(table[("misattributed", "misattributed")], 1)
        self.assertEqual(table[("popular_unverified", "unverified")], 1)
        self.assertEqual(sum(table.values()), len(self.results))

    def test_count_table_grid(self):
        text = vq.format_table(self.results)
        lines = text.splitlines()
        self.assertEqual(len(lines), 6)
        self.assertTrue(lines[0].startswith("web_status"))
        self.assertTrue(lines[1].startswith("sourced"))
        self.assertTrue(lines[-1].startswith("total"))
        self.assertEqual(lines[-1].split()[-1], str(len(self.results)))
        self.assertEqual(sum(int(value) for value in lines[-1].split()[1:-1]),
                         len(self.results))
        for line in lines[1:-1]:
            self.assertEqual(sum(int(value) for value in line.split()[1:-1]),
                             int(line.split()[-1]))

    def test_report_flags_misattributed_and_unverified_only(self):
        text = vq.report(self.results)
        self.assertIn("misattributed / unverified  (2 of 7)", text)
        self.assertIn("Denied to him but on the internet.", text)
        self.assertIn("Everyone knows this one and nobody can place it.", text)
        for absent in ("A verified line from the book.",
                       "From a work we do not hold",
                       "Sourced to the wrong work.",
                       "A translation of a line from the book."):
            self.assertNotIn(absent, text)

    def test_report_flags_a_row_the_corpus_contradicts_the_web_on(self):
        text = vq.report(self.results)
        self.assertIn("web says misattributed, corpus verifies the words  (1)",
                      text)
        self.assertIn("A line the corpus verifies against the web's claim.", text)

    def test_report_says_zero_contradictions_when_there_are_none(self):
        rows = [row for row in self.results
                if not (row["web_status"] == "misattributed"
                        and row["match_status"] == "verified")]
        self.assertIn("web says misattributed, corpus verifies the words  (0)",
                      vq.report(rows))

    def test_preview_is_one_line_and_capped(self):
        long_quote = "\n".join(["word"] * 100)
        one_line = vq.preview(long_quote)
        self.assertNotIn("\n", one_line)
        self.assertEqual(len(one_line), vq.PREVIEW_CHARS)

    def test_flagged_rows_are_one_line_each(self):
        text = vq.report(self.results)
        section = text.split("misattributed / unverified")[1]
        body = section.split("\n\n")[0]
        self.assertEqual(len([line for line in body.splitlines()
                              if line.startswith("  [")]), 2)


class EndToEndTests(unittest.TestCase):
    def write_jsonl(self, path, rows):
        with open(path, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def test_main_writes_output_and_prints_the_three_sections(self):
        with tempfile.TemporaryDirectory() as tmp:
            quotes = [
                quote("The world is not to be made better.", "sourced", "Book A"),
                quote("A summer that cannot be defeated.", "misattributed", "Book Z"),
                quote("Nothing like this anywhere in the books.", "sourced", ""),
                quote("Also nothing like this one.", "popular_unverified", ""),
            ]
            self.write_jsonl(os.path.join(tmp, "quotes.jsonl"), quotes)
            self.write_jsonl(os.path.join(tmp, "chunks1.jsonl"),
                             [chunk("a1", "Book A",
                                    "\u201cThe world is not to be made better,\u201d "
                                    "he said.")])
            self.write_jsonl(os.path.join(tmp, "chunks2.jsonl"),
                             [chunk("a2", "Book B",
                                    "A summer that cannot be defeated, said the "
                                    "notary.")])
            out_path = os.path.join(tmp, "out.jsonl")
            argv = ["verify_quotes.py",
                    "--quotes", os.path.join(tmp, "quotes.jsonl"),
                    "--corpus", os.path.join(tmp, "chunks1.jsonl"),
                    os.path.join(tmp, "chunks2.jsonl"),
                    "--out", out_path]
            buffer = io.StringIO()
            with mock.patch.object(sys, "argv", argv):
                with contextlib.redirect_stdout(buffer):
                    vq.main()
            printed = buffer.getvalue()
            self.assertIn("final by web_status", printed)
            self.assertIn("misattributed / unverified  (1 of 4)", printed)
            self.assertIn("web says misattributed, corpus verifies the words  (1)",
                          printed)
            self.assertIn("A summer that cannot be defeated.", printed)
            with open(out_path, encoding="utf-8") as handle:
                written = [json.loads(line) for line in handle if line.strip()]
            self.assertEqual(len(written), 4)
            self.assertEqual([row["final"] for row in written],
                             ["verified", "verified", "sourced_not_in_corpus",
                              "unverified"])
            self.assertEqual(written[1]["matched_source"], "Book B")
            for row in written:
                self.assertEqual(sorted(row), sorted(
                    list(quote("x")) + ["match_status", "score", "matched_source",
                                        "matched_excerpt", "final"]))
            json.dumps(written)

    def test_load_rows_skips_malformed_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "broken.jsonl")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(chunk("a1", "Book A", "Fine.")) + "\n")
                handle.write("{not json\n")
                handle.write("\n")
                handle.write("[1, 2]\n")
                handle.write(json.dumps(chunk("a2", "Book A", "Also fine.")) + "\n")
            rows, skipped = vq.load_rows(path)
            self.assertEqual([row["id"] for row in rows], ["a1", "a2"])
            self.assertEqual([number for _, number in skipped], [2, 4])

    def test_load_corpus_keeps_row_order_across_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = os.path.join(tmp, "one.jsonl")
            second = os.path.join(tmp, "two.jsonl")
            self.write_jsonl(first, [chunk("a1", "Book A", "One.")])
            self.write_jsonl(second, [chunk("b1", "Book B", "Two.")])
            chunks, skipped = vq.load_corpus([first, second])
            self.assertEqual([c["id"] for c in chunks], ["a1", "b1"])
            self.assertEqual(skipped, [])


if __name__ == "__main__":
    unittest.main()
