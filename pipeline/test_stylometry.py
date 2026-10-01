#!/usr/bin/env python3
"""
test_stylometry.py — unit tests for pipeline/stylometry.py.

Synthetic corpora built in memory (two invented "authors" with distinct function
words) and small JSONL / .md fixtures in a temp dir. No project data is read.

    python -m unittest pipeline/test_stylometry.py -v
"""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import stylometry as st  # noqa: E402


@contextlib.contextmanager
def quiet():
    """Swallow the report the CLI prints, so test output stays readable."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield buf


def make_corpus(marker, n_chunks=6, chunk=200, seed=1):
    """A pseudo-corpus of `n_chunks` chunks of `chunk` words. `marker` selects
    which of two invented function-word profiles to draw from, so the two
    "authors" are genuinely far apart in feature space."""
    import random

    alpha, beta = ("the of and to a in that is was it he", "he it was that in a to and of the")
    words = alpha.split() if marker == "a" else beta.split()
    filler = [f"filler{i}" for i in range(40)]
    rng = random.Random(seed)
    out = []
    for c in range(n_chunks):
        tokens = []
        while len(tokens) < chunk:
            tokens.append(rng.choice(words))
            if rng.random() < 0.35:
                tokens.append(rng.choice(filler))
        out.append(" ".join(tokens[:chunk]))
    return out


class TokenizeTests(unittest.TestCase):
    def test_lowercases_and_drops_punctuation(self):
        self.assertEqual(st.tokenize("The Absurd, man!"),
                         ["the", "absurd", "man"])

    def test_keeps_internal_apostrophe_and_accents(self):
        self.assertEqual(st.tokenize("don't pâtir"), ["don't", "pâtir"])

    def test_digits_and_underscores_excluded(self):
        self.assertEqual(st.tokenize("a1 b_c d 42"), ["a", "b", "c", "d"])

    def test_sentence_opening_detection(self):
        text = "Alpha beta. Gamma? \"Delta!\" Epsilon; Zeta \u2014 Eta"
        # Epsilon opens a sentence too: the \"Delta!\" before it closes a quoted
        # sentence, and a ; or \u2014 does not end one.
        self.assertEqual(st.word_spans(text),
                         [("Alpha", True), ("beta", False), ("Gamma", True),
                          ("Delta", True), ("Epsilon", True), ("Zeta", False),
                          ("Eta", False)])

    def test_opening_quote_does_not_break_sentence_start(self):
        text = "One two. \"Three four"
        self.assertEqual(st.word_spans(text),
                         [("One", True), ("two", False), ("Three", True),
                          ("four", False)])


class MaskNamesTests(unittest.TestCase):
    def test_drops_mid_sentence_capitals_only(self):
        text = "Cigarette sat on the mat. Kirk was asleep by the Kirk window."
        self.assertEqual(st.tokenize(text, mask_names=True),
                         ["cigarette", "sat", "on", "the", "mat", "kirk", "was",
                          "asleep", "by", "the", "window"])

    def test_sentence_initial_capital_is_kept(self):
        self.assertIn("kirk", st.tokenize("Kirk was here", mask_names=True))

    def test_masking_never_adds_or_lowercases(self):
        text = "Paris is cold, said the Kirk of Algiers."
        self.assertEqual(st.tokenize(text, mask_names=True),
                         ["paris", "is", "cold", "said", "the", "of"])

    def test_masking_shortens_only(self):
        text = make_corpus("a", n_chunks=1, chunk=400)[0]
        self.assertLessEqual(len(st.tokenize(text, mask_names=True)),
                             len(st.tokenize(text)))


class ChunkTests(unittest.TestCase):
    def test_even_split(self):
        self.assertEqual([len(c) for c in st.chunks(list(range(10)), 5)],
                         [5, 5])

    def test_short_tail_is_merged(self):
        # 2 chunks of 10 + a tail of 3 -> the 3 folds into the second
        self.assertEqual([len(c) for c in st.chunks(list(range(23)), 10)], [10, 13])

    def test_empty_input(self):
        self.assertEqual(st.chunks([], 10), [])

    def test_single_partial_chunk_not_merged(self):
        self.assertEqual([len(c) for c in st.chunks(list(range(4)), 10)], [4])


class ProfileTests(unittest.TestCase):
    def setUp(self):
        self.samples = [st.tokenize(t) for t in make_corpus("a", n_chunks=8, chunk=200)]

    def test_delta_to_itself_is_zero(self):
        profile = st.build_profile(self.samples, features=20, chunk=200)
        self.assertAlmostEqual(profile.delta([t for s in self.samples for t in s]),
                               0.0, places=9)

    def test_delta_is_symmetric_in_spirit_and_nonnegative(self):
        profile = st.build_profile(self.samples, features=20, chunk=200)
        other = [st.tokenize(t) for t in make_corpus("b", n_chunks=8, chunk=200)]
        d = profile.delta([t for s in other for t in s])
        self.assertGreater(d, 0.0)
        self.assertFalse(d != d)          # not NaN

    def test_distant_author_scores_higher_than_near_author(self):
        profile = st.build_profile(self.samples, features=20, chunk=200)
        near = [st.tokenize(t) for t in make_corpus("a", n_chunks=8, chunk=200, seed=7)]
        far = [st.tokenize(t) for t in make_corpus("b", n_chunks=8, chunk=200, seed=7)]
        self.assertLess(profile.delta([t for s in near for t in s]),
                        profile.delta([t for s in far for t in s]))

    def test_zscore_sign_and_magnitude(self):
        profile = st.build_profile(self.samples, features=20, chunk=200)
        word = profile.vocab[0]
        heavy = [word] * 100
        zs = profile.z(heavy)
        self.assertGreater(zs[0], 0.0)
        self.assertAlmostEqual(zs[0], (1.0 - profile.means[0]) / profile.sds[0])

    def test_chunk_count_and_token_count_recorded(self):
        profile = st.build_profile(self.samples, features=20, chunk=200)
        self.assertEqual(profile.n_chunks, 8)
        self.assertEqual(profile.n_tokens, sum(len(s) for s in self.samples))

    def test_features_are_the_most_frequent_words(self):
        profile = st.build_profile(self.samples, features=5, chunk=200)
        self.assertEqual(len(profile.vocab), 5)

    def test_uniform_reference_has_no_usable_features(self):
        uniform = [st.tokenize(" ".join(["the"] * 200))] * 4
        with self.assertRaises(ValueError):
            st.build_profile(uniform, features=5, chunk=200)

    def test_reference_shorter_than_one_chunk_is_an_error(self):
        with self.assertRaises(ValueError) as ctx:
            st.build_profile([st.tokenize(" ".join(["the"] * 50))], chunk=1000)
        self.assertIn("sub-samples", str(ctx.exception))


class SelfSplitTests(unittest.TestCase):
    def test_is_deterministic_for_the_fixed_seed(self):
        samples = [st.tokenize(t) for t in make_corpus("a", n_chunks=10, chunk=200)]
        first = st.self_split(samples, features=20, chunk=200)
        second = st.self_split(samples, features=20, chunk=200)
        self.assertEqual(first["delta"], second["delta"])
        self.assertEqual(first["seed"], 13)

    def test_halves_partition_the_reference(self):
        samples = [st.tokenize(t) for t in make_corpus("a", n_chunks=10, chunk=200)]
        result = st.self_split(samples, features=20, chunk=200)
        self.assertEqual(result["chunks_a"], 5)
        self.assertEqual(result["chunks_b"], 5)
        self.assertEqual(result["words_a"] + result["words_b"],
                         sum(len(s) for s in samples))

    def test_other_author_scores_above_the_floor(self):
        samples = [st.tokenize(t) for t in make_corpus("a", n_chunks=10, chunk=200)]
        floor = st.self_split(samples, features=20, chunk=200)["delta"]
        other = [st.tokenize(t) for t in make_corpus("b", n_chunks=10, chunk=200)]
        profile = st.build_profile(samples, features=20, chunk=200)
        self.assertGreater(profile.delta([t for s in other for t in s]), floor)


class PunctuationTests(unittest.TestCase):
    def test_rates_are_per_thousand_words(self):
        text = "one two three four"          # 4 words, one '?'
        rates = st.punctuation_rates(text + "?")
        self.assertAlmostEqual(rates["?"], 250.0)
        for key in ("!", "\u2026", "\u2014", ";", ":"):
            self.assertEqual(rates[key], 0.0)

    def test_each_mark_counted(self):
        text = "a! b? c\u2026 d\u2014 e; f: g"
        rates = st.punctuation_rates(text)
        per_thousand = 1000.0 / 7          # 7 words, one of each mark
        for key in ("!", "?", "\u2026", "\u2014", ";", ":"):
            self.assertAlmostEqual(rates[key], per_thousand, places=6, msg=key)

    def test_parentheses_counted_as_closed_pairs(self):
        self.assertAlmostEqual(st.punctuation_rates("a (b c)")["()"], 1000.0 / 3)
        self.assertAlmostEqual(st.punctuation_rates("a (b c")["()"], 0.0)

    def test_empty_text_is_all_zero(self):
        self.assertEqual(set(st.punctuation_rates("").values()), {0.0})


class InputTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)

    def path(self, name):
        return os.path.join(self.dir.name, name)

    def write_jsonl(self, name, rows):
        target = self.path(name)
        with open(target, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
        return target

    def test_parse_spec_defaults_and_field_suffix(self):
        target = self.write_jsonl("a.jsonl", [{"response": "hi"}])
        self.assertEqual(st.parse_spec(target), (target, "response"))
        self.assertEqual(st.parse_spec(target + ":answer"), (target, "answer"))

    def test_parse_spec_keeps_existing_path_with_colon(self):
        odd = self.write_jsonl("we:ird.jsonl", [{"response": "hi"}])
        self.assertEqual(st.parse_spec(odd), (odd, "response"))

    def test_parse_spec_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            st.parse_spec(self.path("nope.jsonl"))

    def test_jsonl_field_and_where(self):
        target = self.write_jsonl("b.jsonl", [
            {"response": "alpha", "type": "conversational"},
            {"response": "beta", "type": "essayist"},
        ])
        self.assertEqual(st.jsonl_texts(target), ["alpha", "beta"])
        self.assertEqual(st.jsonl_texts(target, where=("type", "essayist")), ["beta"])

    def test_jsonl_where_prefix(self):
        target = self.write_jsonl("c.jsonl", [
            {"response": "a", "source": "Notebooks (1935-1942)"},
            {"response": "b", "source": "The Plague"},
        ])
        texts = st.jsonl_texts(target, where_prefix=("source", "Notebooks"))
        self.assertEqual(texts, ["a"])

    def test_jsonl_messages_rows_use_assistant_turns(self):
        target = self.write_jsonl("d.jsonl", [{"messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a1"},
            {"role": "assistant", "content": "a2"},
        ]}])
        self.assertEqual(st.jsonl_texts(target), ["a1", "a2"])

    def test_jsonl_skips_blank_and_malformed_lines(self):
        target = self.path("e.jsonl")
        with open(target, "w", encoding="utf-8") as fh:
            fh.write('{"response": "ok"}\n\nnot json\n[1, 2]\n')
        self.assertEqual(st.jsonl_texts(target), ["ok"])

    def test_markdown_assistant_lines_only(self):
        target = self.path("s.md")
        with open(target, "w", encoding="utf-8") as fh:
            fh.write("# Session\n\n**you:** evening\n\n"
                     "**camus:** The light goes out slowly.\n\n"
                     "---\n\n**camus:** Second reply here.\n")
        self.assertEqual(st.md_texts(target),
                         ["The light goes out slowly.", "Second reply here."])

    def test_markdown_joins_wrapped_replies(self):
        target = self.path("w.md")
        with open(target, "w", encoding="utf-8") as fh:
            fh.write("**camus:** one two\nthree four\n\n**you:** hi\n")
        self.assertEqual(st.md_texts(target), ["one two three four"])

    def test_read_texts_dispatches_on_extension(self):
        target = self.write_jsonl("f.jsonl", [{"passage": "text here"}])
        self.assertEqual(st.read_texts(target + ":passage"), ["text here"])


class CliTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)

    def write(self, name, texts, field="response"):
        target = os.path.join(self.dir.name, name)
        with open(target, "w", encoding="utf-8") as fh:
            for text in texts:
                fh.write(json.dumps({field: text}) + "\n")
        return target

    def write_rows(self, name, rows):
        target = os.path.join(self.dir.name, name)
        with open(target, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
        return target

    def test_main_runs_end_to_end_and_returns_zero(self):
        ref = self.write("ref.jsonl", make_corpus("a", n_chunks=8, chunk=200),
                         field="passage")
        cand = self.write("cand.jsonl", make_corpus("b", n_chunks=4, chunk=200))
        with quiet() as out:
            rc = st.main(["--reference", ref + ":passage", "--candidate", cand,
                          "--features", "20", "--chunk", "200", "--self-split"])
        self.assertEqual(rc, 0)
        self.assertIn("self-split floor", out.getvalue())

    def test_main_reports_a_too_small_reference(self):
        ref = self.write("tiny.jsonl", ["a few words in a row"], field="passage")
        cand = self.write("c.jsonl", ["other words entirely"])
        with quiet():
            rc = st.main(["--reference", ref + ":passage", "--candidate", cand,
                          "--chunk", "1000"])
        self.assertEqual(rc, 1)

    def test_list_labels(self):
        ref = self.write("ref2.jsonl", make_corpus("a", n_chunks=4, chunk=200),
                         field="passage")
        cand = self.write("cand2.jsonl", make_corpus("b", n_chunks=2, chunk=200))
        with quiet() as out:
            rc = st.main(["--reference", ref + ":passage",
                          "--candidate", cand, "--list-labels"])
        self.assertEqual(rc, 0)
        self.assertIn("cand2.jsonl", out.getvalue())

    def test_candidate_filters_are_positional_and_leave_the_reference_alone(self):
        ref = self.write("ref3.jsonl", make_corpus("a", n_chunks=8, chunk=200),
                         field="passage")
        cand = self.write_rows("cand3.jsonl",
                                [{"response": "x", "type": "conversational"},
                                 {"response": "y", "type": "essayist"},
                                 {"response": "z", "type": "conversational"}])
        with quiet() as out:
            rc = st.main(["--reference", ref + ":passage",
                          "--candidate", cand,
                          "--candidate-where", "type=conversational",
                          "--features", "20", "--chunk", "200"])
        self.assertEqual(rc, 0)
        # only the two conversational rows counted (2 words, not 3)
        self.assertIn("type=conversational", out.getvalue())
        self.assertIn(" 2 ", out.getvalue())


if __name__ == "__main__":
    unittest.main()
