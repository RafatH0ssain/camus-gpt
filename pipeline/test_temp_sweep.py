#!/usr/bin/env python3
"""
test_temp_sweep.py — unit tests for pipeline/temp_sweep.py.

Every check is exercised with a hand-written answer. No model is called and no
network is touched: generation goes through a fake camus_rag, the importable
entry points are the ones the CLI uses.

    python -m unittest pipeline/test_temp_sweep.py -v
"""
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import temp_sweep as ts  # noqa: E402


def flags_of(answer, kind, titles=None):
    """titles: raw identity-card text or an already-parsed title set."""
    flags, _ = ts.check_answer(answer, kind, titles if titles is not None else set())
    return flags


class PlanTests(unittest.TestCase):
    def test_conversational_gets_half_the_samples(self):
        jobs = ts.plan([0.45], 8, "both")
        factual = [j for j in jobs if j[0] == "factual"]
        chatty = [j for j in jobs if j[0] == "conversational"]
        self.assertEqual({j[4] for j in factual}, {8})
        self.assertEqual({j[4] for j in chatty}, {4})

    def test_odd_n_rounds_down_but_never_to_zero(self):
        jobs = ts.plan([0.45], 1, "conversational")
        self.assertEqual({j[4] for j in jobs}, {1})
        jobs = ts.plan([0.45], 3, "conversational")
        self.assertEqual({j[4] for j in jobs}, {1})

    def test_all_eight_prompts_per_temperature(self):
        self.assertEqual(len(ts.plan([0.45, 0.8, 1.0, 1.2], 2, "both")), 32)

    def test_prompt_text_is_the_published_set(self):
        prompts = [j[2] for j in ts.plan([0.45], 1, "both")[:8]]
        self.assertIn("what were your dogs' names?", prompts)
        self.assertIn("isn't the absurd just a fancy way of giving up?", prompts)
        self.assertEqual(len(prompts), 8)

    def test_kind_for_maps_prompts_to_check_groups(self):
        self.assertEqual(ts.kind_for("dogs"), "pets")
        self.assertEqual(ts.kind_for("pets"), "pets")
        self.assertEqual(ts.kind_for("novels"), "works")
        self.assertEqual(ts.kind_for("wrote"), "works")
        self.assertEqual(ts.kind_for("hey"), "none")

    def test_default_temps(self):
        self.assertEqual(ts.DEFAULT_TEMPS, (0.45, 0.8, 1.0, 1.2))
        self.assertEqual(ts.DEFAULT_N_FACTUAL, 8)


class PetsTests(unittest.TestCase):
    def test_all_three_dogs_counted(self):
        self.assertEqual(ts.dog_names_named("Pauline, Kirk and Blaise"), 3)
        self.assertEqual(ts.dog_names_named("Pauline only"), 1)
        self.assertEqual(ts.dog_names_named("no names at all"), 0)

    def test_no_flag_on_a_clean_answer(self):
        answer = ("I had three dogs from the kennels. Pauline was the eldest, "
                  "Kirk the quiet one, and Blaise, who never learned to come when "
                  "called. There was a cat too, named Cigarette.")
        flags = flags_of(answer, "pets")
        self.assertFalse(any(flags.values()), flags)

    def test_denial_flagged(self):
        for answer in ("I never kept dogs, if you must know.",
                       "I don't have any pets.",
                       "There were no dogs in my life.",
                       "I had no animals as a boy."):
            self.assertTrue(flags_of(answer, "pets")["pets_denies_dogs"], answer)

    def test_not_a_denial_when_merely_qualified(self):
        answer = "I had dogs. I did not keep them long, and I did not name them all."
        self.assertFalse(flags_of(answer, "pets")["pets_denies_dogs"])

    def test_species_swap_flagged(self):
        self.assertTrue(flags_of("My pets were cats. Pauline slept on the desk.",
                                 "pets")["pets_species_swap"])
        self.assertTrue(flags_of("Only Pauline and Kirk, who were birds.",
                                 "pets")["pets_species_swap"])
        self.assertTrue(flags_of("Blaise was a horse of the first order.",
                                 "pets")["pets_species_swap"])

    def test_naming_a_name_as_a_dog_is_not_a_swap(self):
        answer = "I had cats and dogs: Cigarette the cat, Pauline and Kirk the dogs."
        self.assertFalse(flags_of(answer, "pets")["pets_species_swap"])

    def test_invented_name_flagged_and_reported(self):
        flags, detail = ts.check_answer("My dogs were Pauline, Kirk, Blaise and Fido.",
                                         "pets", set())
        self.assertTrue(flags["pets_invented_name"])
        self.assertIn("Fido", detail["invented_names"])

    def test_known_names_and_sentence_openings_not_flagged(self):
        _, detail = ts.check_answer(
            "Cigarette was a cat. Pauline was a dog. I loved those animals.",
            "pets", set())
        self.assertEqual(detail["invented_names"], [])
        self.assertEqual(detail["dogs_named"], 1)

    def test_english_pronoun_caps_not_flagged(self):
        _, detail = ts.check_answer("I had Pauline. My cats were Cigarette.",
                                    "pets", set())
        self.assertEqual(detail["invented_names"], [])


class WorksTests(unittest.TestCase):
    CARD = (
        "Novels: The Stranger (1942), The Plague (1947), The Fall (1956); "
        "unfinished, posthumous: A Happy Death, The First Man. Stories: Exile and "
        "the Kingdom (1957) \u2014 six: The Adulterous Woman, The Renegade, The "
        "Silent Men, The Guest, Jonas or the Artist at Work, The Growing Stone. "
        "Essays: The Myth of Sisyphus (1942), The Rebel (1951 \u2014 an essay, not a "
        "novel). Plays: Caligula, The Misunderstanding, State of Siege, The Just "
        "Assassins.\n- Born 7 November 1913.")

    def test_card_titles_are_parsed_from_the_card(self):
        titles = ts.card_titles(self.CARD)
        for expected in ("stranger", "plague", "fall", "happy death", "first man",
                         "exile and the kingdom", "myth of sisyphus", "rebel",
                         "caligula", "just assassins", "jonas or the artist at work"):
            self.assertIn(expected, titles)

    def test_card_titles_exclude_the_section_headings(self):
        titles = ts.card_titles(self.CARD)
        for junk in ("novels", "stories", "essays", "plays", "posthumous"):
            self.assertNotIn(junk, titles)

    def test_norm_title_drops_the_leading_article(self):
        self.assertEqual(ts._norm_title("The Plague"), "plague")
        self.assertEqual(ts._norm_title("“The Fall”"), "fall")

    def test_all_three_named_is_clean(self):
        answer = ("The Stranger, The Plague, The Fall \u2014 those three are mine. "
                  "The Myth of Sisyphus and The Rebel are essays, and Exile and the "
                  "Kingdom is stories.")
        flags = flags_of(answer, "works", self.CARD)
        self.assertFalse(flags["works_missing_title"])
        self.assertFalse(any(flags.values()), flags)

    def test_missing_the_fall_is_flagged(self):
        answer = "Two, and only two: The Stranger and The Plague."
        self.assertTrue(flags_of(answer, "works", self.CARD)["works_missing_title"])

    def test_essay_called_a_novel_is_flagged(self):
        answer = ("I wrote four novels: The Stranger, The Plague, The Fall and The "
                  "Myth of Sisyphus.")
        flags = flags_of(answer, "works", self.CARD)
        self.assertTrue(flags["works_essay_as_novel"])
        self.assertFalse(flags["works_invented_title"])

    def test_reverse_order_essay_as_novel_is_flagged(self):
        answer = ("The Rebel is the novel I would defend. The Stranger, The Plague "
                  "and The Fall came earlier.")
        self.assertTrue(flags_of(answer, "works", self.CARD)["works_essay_as_novel"])

    def test_calling_an_essay_an_essay_is_not_flagged(self):
        answer = ("The Stranger, The Plague, The Fall. The Rebel is an essay, not a "
                  "novel.")
        self.assertFalse(flags_of(answer, "works", self.CARD)["works_essay_as_novel"])

    def test_invented_title_flagged_and_reported(self):
        answer = ('I wrote The Stranger, The Plague, The Fall, and \u201cThe Frozen '
                  'Harbour\u201d early on.')
        flags, detail = ts.check_answer(answer, "works", self.CARD)
        self.assertTrue(flags["works_invented_title"])
        self.assertIn("The Frozen Harbour", detail["invented_titles"])

    def test_card_titles_are_not_invented(self):
        answer = ('The Stranger, The Plague, The Fall, "Exile and the Kingdom", '
                  '*The Just Assassins*, "State of Siege".')
        flags, detail = ts.check_answer(answer, "works", self.CARD)
        self.assertFalse(flags["works_invented_title"], detail)

    def test_single_quoted_lines_are_not_titles(self):
        # The card's "these lines are NOT yours" quotes live in single quotes.
        answer = ("The Stranger, The Plague, The Fall. I did not write "
                  "'Should I kill myself, or have a cup of coffee?'")
        flags, _ = ts.check_answer(answer, "works", self.CARD)
        self.assertFalse(flags["works_invented_title"])


class GeneralTests(unittest.TestCase):
    def test_repeated_trigram(self):
        answer = "the same three words the same three words the same three words here"
        self.assertTrue(flags_of(answer, "none")["all_repeat_trigram"])
        self.assertIn("the same three",
                      ts.repeated_trigrams("the same three words " * 3))

    def test_no_trigram_repeat_in_normal_prose(self):
        answer = "One thing leads to another and then to a third, which is different."
        self.assertFalse(flags_of(answer, "none")["all_repeat_trigram"])

    def test_over_220_words(self):
        self.assertTrue(flags_of("word " * 221, "none")["all_over_220_words"])
        self.assertFalse(flags_of("word " * 220, "none")["all_over_220_words"])

    def test_terminal_punctuation(self):
        self.assertTrue(flags_of("no punctuation at the end", "none")["all_no_terminal_punct"])
        for ending in ("end.", "end!", "end?", 'end."', "end…", "end)"):
            self.assertFalse(flags_of(f"an answer that {ending}", "none")
                             ["all_no_terminal_punct"], ending)

    def test_non_english_sentence(self):
        answer = "I think so. Je ne sais pas pourquoi le monde est si cruel aujourd'hui."
        self.assertTrue(flags_of(answer, "none")["all_non_english"])

    def test_english_prose_is_not_flagged(self):
        answer = ("I do not know what to make of it, and I have never been in the "
                  "mood to pretend that I do.")
        self.assertFalse(flags_of(answer, "none")["all_non_english"])

    def test_detail_reports_the_offending_text(self):
        flags, detail = ts.check_answer(
            "the same three words " * 3, "none", set())
        self.assertTrue(flags["all_repeat_trigram"])
        self.assertTrue(detail["repeat_trigrams"])
        self.assertEqual(detail["words"], len(ts._WORD_RE.findall("the same three words " * 3)))

    def test_clean_answer_has_no_flags_at_all(self):
        answer = ("I had three dogs and a cat. Pauline was black and old, Kirk "
                  "never stopped barking, Blaise slept. We were happy enough.")
        flags, _ = ts.check_answer(answer, "pets", set())
        self.assertEqual([k for k, v in flags.items() if v], [])


class AskTests(unittest.TestCase):
    def test_ask_overrides_only_the_temperature(self):
        sent = {}

        class Response:
            def raise_for_status(self_inner):
                pass

            def json(self_inner):
                return {"message": {"content": "answer"}}

        def fake_post(url, json=None, timeout=None):
            sent.update(url=url, json=json, timeout=timeout)
            return Response()

        cr = mock.Mock(OLLAMA="http://localhost:11434", GEN_MODEL="camus")
        with mock.patch.dict(sys.modules, {"requests": mock.Mock(post=fake_post)}):
            out = ts.ask(cr, [{"role": "user", "content": "hey"}],
                         {"temperature": 0.45, "top_k": 40, "num_ctx": 8192}, 1.2)
        self.assertEqual(out, "answer")
        self.assertEqual(sent["url"], "http://localhost:11434/api/chat")
        self.assertEqual(sent["json"]["model"], "camus")
        self.assertFalse(sent["json"]["stream"])
        self.assertEqual(sent["json"]["options"],
                         {"temperature": 1.2, "top_k": 40, "num_ctx": 8192})

    def test_ask_does_not_mutate_the_turn_options(self):
        opts = {"temperature": 0.45}

        class Response:
            def raise_for_status(self_inner):
                pass

            def json(self_inner):
                return {"message": {"content": ""}}

        with mock.patch.dict(sys.modules, {
                "requests": mock.Mock(post=lambda *a, **k: Response())}):
            ts.ask(mock.Mock(OLLAMA="o", GEN_MODEL="m"), [], opts, 1.0)
        self.assertEqual(opts, {"temperature": 0.45})


class SummariseTests(unittest.TestCase):
    def row(self, temperature, key, kind="factual", clean=True, words=50,
            error=None, answer="answer"):
        return {"kind": kind, "key": key, "prompt": key, "temperature": temperature,
                "sample": 0, "model": "camus", "secs": 1.0, "error": error,
                "answer": answer, "flags": {} if clean else {"pets_denies_dogs": True},
                "clean": clean, "words": words}

    def test_per_prompt_clean_rate_and_counts(self):
        rows = [self.row(0.45, "dogs", clean=True),
                self.row(0.45, "dogs", clean=False),
                self.row(0.45, "novels", clean=True)]
        summary = ts.summarise(rows)
        self.assertEqual(len(summary), 1)
        row = summary[0]
        self.assertAlmostEqual(row["per_prompt_clean"]["dogs"], 0.5)
        self.assertEqual(row["per_prompt_n"]["dogs"], 2)
        self.assertAlmostEqual(row["per_prompt_clean"]["novels"], 1.0)
        self.assertNotIn("pets", row["per_prompt_clean"])

    def test_flags_are_counted_across_samples(self):
        rows = [self.row(0.45, "dogs", clean=False), self.row(0.45, "pets", clean=False)]
        self.assertEqual(ts.summarise(rows)[0]["flags"], {"pets_denies_dogs": 2})

    def test_word_count_mean_and_sd(self):
        rows = [self.row(0.45, "dogs", words=10), self.row(0.45, "dogs", words=30)]
        row = ts.summarise(rows)[0]
        self.assertAlmostEqual(row["words_mean"], 20.0)
        self.assertAlmostEqual(row["per_prompt_sd"]["dogs"], 14.142135, places=5)

    def test_temperatures_are_reported_in_order(self):
        rows = [self.row(t, "dogs") for t in (1.2, 0.45, 0.8)]
        self.assertEqual([r["temperature"] for r in ts.summarise(rows)],
                         [0.45, 0.8, 1.2])

    def test_errors_are_counted_not_crashed_on(self):
        rows = [self.row(0.45, "dogs", error="ConnectionError: refused",
                         answer="")]
        self.assertEqual(ts.summarise(rows)[0]["errors"], 1)

    def test_print_summary_writes_a_table(self):
        import contextlib
        import io

        rows = [self.row(0.45, k) for k, _ in ts.FACTUAL_PROMPTS]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ts.print_summary(ts.summarise(rows), 1)
        out = buf.getvalue()
        self.assertIn("dogs clean", out)
        self.assertIn("flags counted", out)


class CliTests(unittest.TestCase):
    def test_dry_run_calls_nothing(self):
        import contextlib
        import io

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            with mock.patch.object(ts, "camus_rag") as fake:
                rc = ts.main(["--temps", "0.45", "1.2", "--n", "1", "--dry-run"])
        self.assertEqual(rc, 0)
        self.assertFalse(fake.called)
        self.assertIn("16 prompts, 16 samples", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
