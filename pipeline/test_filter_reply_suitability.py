#!/usr/bin/env python3
"""
test_filter_reply_suitability.py — unit tests for pipeline/filter_reply_suitability.py.

No network and nothing under data/ read or written: the model call is always an injected
fake, and every file lives in a temporary directory. The fixtures are synthetic rows; the
only strings that look like Camus are one-line stubs written for this file.

    python -m unittest pipeline/test_filter_reply_suitability.py -v
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
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "rag"))

import filter_reply_suitability as frs  # noqa: E402
import judge_opencode as jz  # noqa: E402

KEY = "fake-key-must-never-be-used-0123456789"


def row(prompt, response, src_id="s1", source="Source"):
    return {"prompt": prompt, "response": response, "kind": "primary",
            "source": source, "src_id": src_id}


def decisions(*pairs):
    """A reply the model might send: pairs of (id, keep)."""
    return json.dumps([{"id": i, "keep": keep, "reason": "reads as a reply"}
                       for i, keep in pairs])


def all_keep(batch_size=None, reply=None):
    """A fake call that keeps every row in the batch it is handed, and a tally of what
    it was asked."""
    def call(system, user, key=None):
        items = json.loads(user)
        reply_text = (decisions(*((i, True) for i in range(len(items))))
                      if reply is None else reply(len(items)))
        stats["calls"] += 1
        stats["batches"].append((system, [item["id"] for item in items]))
        return reply_text
    stats = {"calls": 0, "batches": []}
    return call, stats


def quietly(fn, *args, **kwargs):
    """Run something that reports progress, with the report swallowed: a test asserts on
    return values, not on the progress lines."""
    with contextlib.redirect_stdout(io.StringIO()), \
            contextlib.redirect_stderr(io.StringIO()):
        return fn(*args, **kwargs)


class DatedResponseTest(unittest.TestCase):
    """Rule 1: a response that opens on a date is a diary line, not a reply."""

    def test_dated_entries_are_dropped(self):
        for response in ("September 22. The Happy Death.",
                         "May 16. Leave for Paris, heart aching.",
                         "January 4, 1952.",
                         "1942.",
                         "MARCH 17-JUNE But what good does it do to protest?",
                         "NOVEMBER 19-30, concluding that if people like us live",
                         "July 25 Wake up at 7 o'clock.",
                         "April 1946 3: Resignation of FE de Menthon",
                         "August 15, 1945",
                         "May 29, 1958. My job is to write my books"):
            with self.subTest(response=response):
                self.assertTrue(frs.is_dated(response), response)

    def test_case_and_spacing_do_not_matter(self):
        for response in ("  september  22. The Happy Death.", "SEPTEMBER 22.",
                         "Sept. 4.", "Jan. 3.", "oct. 17, '60"):
            with self.subTest(response=response):
                self.assertTrue(frs.is_dated(response), response)

    def test_a_reply_is_not_dropped_for_opening_with_a_number(self):
        for response in ("10 years ago the same thing happened to me.",
                         "3 o'clock in the morning, still awake.",
                         "One thing is certain: we are all going to die.",
                         "Two hundred men refused to move."):
            with self.subTest(response=response):
                self.assertFalse(frs.is_dated(response), response)

    def test_a_reply_mentioning_a_date_is_kept_by_this_rule(self):
        self.assertFalse(frs.is_dated("We met in September and nothing came of it."))
        self.assertFalse(frs.is_dated("They said 1942 was the worst winter."))

    def test_dated_row_is_counted_and_not_sent_to_the_model(self):
        rows = [row("how was your week?", "September 22. The Happy Death."),
                row("tell me something.", "The world does not care about us.")]
        survivors, dropped, _ = frs.apply_rules(rows)
        self.assertEqual(len(survivors), 1)
        self.assertEqual(dropped[frs.RULE_LABELS[0]], 1)


class CitationResponseTest(unittest.TestCase):
    """Rule 2: an opening citation marker means he is carrying someone else's words."""

    def test_citation_openings_are_dropped(self):
        for response in ('Id. "I am not an orphan on the earth."',
                         "Ibid. 12",
                         "Cf. the letter of March 3.",
                         'B.B. "No one realizes some people have to make an effort."',
                         "R.C. The world is not to be taken literally.",
                         "N.R.F. Curious environment intended to inspire writers"):
            with self.subTest(response=response):
                self.assertTrue(frs.is_citation(response), response)

    def test_a_sentence_beginning_with_initials_is_not_dropped(self):
        # One dotted capital is a name or a list marker often enough that treating it as
        # a citation would throw away real rows.
        for response in ("A. He came back that winter without saying why.",
                         "N. Nobody told her, and she never asked.",
                         "J. B. was the one who said it first."):
            with self.subTest(response=response):
                self.assertFalse(frs.is_citation(response), response)

    def test_id_inside_a_sentence_is_not_an_opening_citation(self):
        self.assertFalse(frs.is_citation("He said id. only when he was sure."))
        self.assertFalse(frs.is_citation("She preferred the cf. reference in the margin."))


class CrisisPromptTest(unittest.TestCase):
    """Rule 3: a crisis prompt must never be paired with a notebook fragment."""

    def test_crisis_prompts_are_matched(self):
        for prompt in ("today feels like a good day to stop resisting the thought of dying.",
                       "is calling suicide selfish too simple?",
                       "i want to die.",
                       "some days i think about killing myself.",
                       "I've been cutting myself again.",
                       "i think i might end it this week.",
                       "do you ever feel a death wish?",
                       "i keep dreaming about hanging myself.",
                       "i overdosed last night and nobody noticed.",
                       "dying alone in a hospital sounds like the loneliest ending."):
            with self.subTest(prompt=prompt):
                self.assertTrue(frs.is_crisis(prompt), prompt)

    def test_the_rule_is_the_mention_not_the_felt_state(self):
        # The rule is deliberately blunt: the prompt names suicide, however it does so.
        # A bereaved person asking about a death is not in crisis, and a pairing is not
        # obviously harmful either, but a miss here is the expensive direction — so the
        # mention is what counts and the corpus stays small.
        for prompt in ("after a loved one died by suicide, i felt shock.",
                       "an unarmed uprising sounds more like suicide than strategy.",
                       "what does suicide mean in french?"):
            with self.subTest(prompt=prompt):
                self.assertTrue(frs.is_crisis(prompt), prompt)

    def test_philosophical_suicide_discourse_is_not_a_crisis_message(self):
        # "self-destruction" as an abstraction is not someone asking for help, and the
        # corpus is mostly this: dropping it would empty the one topic Camus wrote about
        # most. What drops a row is a person putting themselves in the sentence.
        for prompt in ("i think treating self-destruction as the central philosophical "
                       "issue is too absolute.",
                       "some people think self-destruction gives an impossible future a "
                       "final shape.",
                       "the fantasy that an artist invents a lawless world can become "
                       "dangerously self-destructive"):
            with self.subTest(prompt=prompt):
                self.assertFalse(frs.is_crisis(prompt), prompt)

    def test_talk_of_death_that_names_no_method_is_not_a_crisis(self):
        for prompt in ("it feels surreal to keep thinking about beauty while everyone "
                       "nearby is dying.",
                       "after decades teaching, he seems less afraid of dying.",
                       "why do people die so young?",
                       "write a desperate plea from a man whose partner is dying.",
                       "beside the sea at night, i could finally accept dying.",
                       "i'm trying to understand how betrayal deepens attachment "
                       "instead of ending it.",
                       "i think i might end the project this week."):
            with self.subTest(prompt=prompt):
                self.assertFalse(frs.is_crisis(prompt), prompt)

    def test_a_crisis_prompt_drops_the_row_whatever_the_response_is(self):
        # Even a perfectly conversational passage is dropped: the pair is the problem,
        # not only the passage.
        rows = [row("today feels like a good day to stop resisting the thought of dying.",
                    "That is the hour when the answer is easiest to live with."),
                row("let's spend tonight talking about the theater.",
                    "Talk on theater.")]
        survivors, dropped, _ = frs.apply_rules(rows)
        self.assertEqual([r["src_id"] for r in survivors], ["s1"])   # the title survives
        self.assertEqual(dropped[frs.RULE_LABELS[2]], 1)             # the crisis row does not

    def test_crisis_is_counted_first_when_a_response_is_also_dated(self):
        rows = [row("i want to die.", "September 22. The Happy Death.")]
        _, dropped, matched = frs.apply_rules(rows)
        self.assertEqual(dropped[frs.RULE_LABELS[2]], 1)
        self.assertEqual(dropped[frs.RULE_LABELS[0]], 0)
        self.assertEqual(matched[frs.RULE_LABELS[0]], 1)   # the overlap stays visible
        self.assertEqual(sum(dropped.values()), 1)


class TitleResponseTest(unittest.TestCase):
    """A bare title survives the rules and is left to the model pass."""

    def test_titles_and_plan_notes_are_not_caught_by_the_rules(self):
        rows = [row("let's spend tonight talking about the theater.", "Talk on theater."),
                row("what did you plan to do next?", "1. Rewrite the ending."),
                row("how do you feel about the plague?", "The plague."),
                row("anything new?", "Chapter II. The revolt.")]
        survivors, dropped, _ = frs.apply_rules(rows)
        self.assertEqual(len(survivors), 4)
        self.assertEqual(sum(dropped.values()), 0)


class RuleCountingTest(unittest.TestCase):
    def test_counts_add_up_to_every_drop(self):
        rows = [row("q1", "September 22. The Happy Death."),
                row("q2", "Id. In industrial cities."),
                row("i want to die.", "Anything at all."),
                row("q4", "The world does not care about us."),
                row("q5", "May 16. Leave for Paris.")]
        survivors, dropped, _ = frs.apply_rules(rows)
        self.assertEqual(sum(dropped.values()), len(rows) - len(survivors))
        self.assertEqual(dropped[frs.RULE_LABELS[0]], 2)
        self.assertEqual(dropped[frs.RULE_LABELS[1]], 1)
        self.assertEqual(dropped[frs.RULE_LABELS[2]], 1)

    def test_every_rule_has_a_label_in_the_report_order(self):
        self.assertEqual(len(frs.RULE_LABELS), 3)
        self.assertTrue(all(label for label in frs.RULE_LABELS))


class NormaliseTest(unittest.TestCase):
    def test_whitespace_and_case_collapse(self):
        self.assertEqual(frs.normalise("  Talk\non   theater. "), "talk on theater.")
        self.assertEqual(frs.normalise("TALK ON THEATER."), "talk on theater.")

    def test_the_key_is_the_response_not_the_src_id(self):
        a = row("q1", "Talk on theater.", src_id="chunk-0001")
        b = row("q2", "Talk on theater.", src_id="chunk-0002")
        self.assertEqual(frs.row_key(a), frs.row_key(b))


class ParseReplyTest(unittest.TestCase):
    def test_plain_array(self):
        items = frs.parse_decisions('[{"id": 0, "keep": true, "reason": "a reply"}]')
        self.assertEqual(items, [{"id": 0, "keep": True, "reason": "a reply"}])

    def test_fenced_array_with_prose_around_it(self):
        text = ("Sure, here you go:\n```json\n[{\"id\": 0, \"keep\": false, "
                "\"reason\": \"a bare title\"}]\n```\nLet me know if you want more.")
        self.assertEqual(frs.parse_decisions(text)[0]["reason"], "a bare title")

    def test_a_single_object_holding_the_array(self):
        text = json.dumps({"decisions": [{"id": 0, "keep": True, "reason": "ok"}]})
        self.assertTrue(frs.parse_decisions(text)[0]["keep"])

    def test_a_reply_with_no_array_is_a_value_error(self):
        for text in ("I cannot do that.", "", "[{\"id\": 0,", "{not json"):
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    frs.parse_decisions(text)

    def test_keep_spelled_as_a_string(self):
        self.assertTrue(frs._keep("true"))
        self.assertFalse(frs._keep("false"))
        self.assertFalse(frs._keep(None))


class AlignTest(unittest.TestCase):
    def test_ids_pair_rows_with_verdicts(self):
        batch = [row("q1", "a"), row("q2", "b"), row("q3", "c")]
        items = [{"id": 2, "keep": False}, {"id": 0, "keep": True}, {"id": 1, "keep": False}]
        pairs = frs.align(batch, items)
        self.assertEqual([p["keep"] for p in pairs], [True, False, False])

    def test_a_reply_of_the_same_length_falls_back_to_order(self):
        batch = [row("q1", "a"), row("q2", "b")]
        pairs = frs.align(batch, [{"id": 1, "keep": True}, {"id": 2, "keep": False}])
        self.assertEqual([p["keep"] for p in pairs], [True, False])

    def test_a_short_reply_leaves_the_rest_undecided(self):
        # Never give a row a verdict the model did not give it: an unmatched row is
        # undecided, and a rerun asks again.
        batch = [row("q1", "a"), row("q2", "b"), row("q3", "c")]
        pairs = frs.align(batch, [{"id": 0, "keep": True}])
        self.assertEqual(pairs[0]["keep"], True)
        self.assertIsNone(pairs[1])
        self.assertIsNone(pairs[2])


class ParseErrorDecisionTest(unittest.TestCase):
    def test_a_reply_that_will_not_parse_is_retried_once_then_left_undecided(self):
        rows = [row(f"q{i}", f"passage {i}") for i in range(2)]
        calls = []

        def call(system, user, key=None):
            calls.append(user)
            return "I am not going to answer in that format."

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "d.jsonl")
            stats, stop = quietly(frs.run, rows, {}, path, batch_size=2, key=KEY, call=call)
        self.assertEqual(len(calls), 2)                     # one retry, not a loop
        self.assertIsNone(stop)
        self.assertEqual(stats["undecided"], 2)              # nothing decided
        self.assertEqual(stats["decided"], 0)
        self.assertFalse(os.path.exists(path) and open(path).read().strip())

    def test_a_truncated_first_reply_is_recovered_by_the_retry(self):
        rows = [row("q1", "a passage")]
        replies = [decisions((0, True))[:20], decisions((0, True))]

        def call(system, user, key=None):
            return replies.pop(0)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "d.jsonl")
            stats, stop = quietly(frs.run, rows, {}, path, batch_size=25, key=KEY, call=call)
        self.assertIsNone(stop)
        self.assertEqual(stats["decided"], 1)
        self.assertEqual(stats["kept"], 1)


class UsageCapTest(unittest.TestCase):
    def test_429_stops_the_run_after_the_first_batch(self):
        rows = [row(f"q{i}", f"passage {i}") for i in range(5)]

        def call(system, user, key=None):
            raise jz.UsageCapReached("opencode judge hit the usage cap (HTTP 429)")

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "d.jsonl")
            stats, stop = quietly(frs.run, rows, {}, path, batch_size=1, key=KEY, call=call)
        self.assertIsNotNone(stop)
        self.assertIn("429", stop)
        self.assertEqual(stats["batches"], 0)     # nothing past the cap was decided
        self.assertEqual(stats["undecided"], 1)   # only the batch that hit it

    def test_a_usage_limit_body_on_another_status_stops_the_run_and_is_not_retried(self):
        rows = [row("q1", "a passage")]
        calls = []

        def call(system, user, key=None):
            calls.append(user)
            raise RuntimeError("opencode judge HTTP 400: monthly limit reached")

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "d.jsonl")
            stats, stop = quietly(frs.run, rows, {}, path, batch_size=1, key=KEY, call=call)
        self.assertIsNotNone(stop)
        self.assertEqual(len(calls), 1)

    def test_the_cap_is_not_swallowed_into_an_undecided_batch(self):
        # UsageCapReached is a SystemExit: a generic except would turn a dead run into a
        # quiet row of failures, so it has its own path out of fetch().
        self.assertTrue(issubclass(jz.UsageCapReached, SystemExit))
        rows = [row("q1", "a passage")]
        with tempfile.TemporaryDirectory() as tmp:
            _, stop = quietly(frs.run, rows, {}, os.path.join(tmp, "d.jsonl"), key=KEY,
                              call=lambda s, u, key=None: (_ for _ in ()).throw(
                                  jz.UsageCapReached("cap")))
        self.assertIn("cap", stop)

    def test_a_cap_in_the_middle_keeps_what_was_decided(self):
        rows = [row(f"q{i}", f"passage {i}") for i in range(4)]
        sent = []

        def call(system, user, key=None):
            sent.append(user)
            if len(sent) == 1:
                return decisions((0, True), (1, False))
            raise jz.UsageCapReached("cap reached")

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "d.jsonl")
            stats, stop = quietly(frs.run, rows, {}, path, batch_size=2, key=KEY, call=call)
            on_disk = frs.load_rows(path)
        self.assertIsNotNone(stop)
        self.assertEqual(len(on_disk), 2)                 # the first batch stands
        self.assertEqual([d["keep"] for d in on_disk], [True, False])
        self.assertEqual(stats["decided"], 2)


class ResumeTest(unittest.TestCase):
    def test_a_rerun_does_not_resend_rows_already_decided(self):
        rows = [row(f"q{i}", f"passage {i}") for i in range(3)]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "d.jsonl")
            call, stats = all_keep(2)
            quietly(frs.run, rows, {}, path, batch_size=2, key=KEY, call=call)
            self.assertEqual(stats["calls"], 2)          # 3 rows -> 2 + 1

            again, second = all_keep(2)
            stats2, _ = quietly(frs.run, rows, frs.load_decisions(path), path,
                                   batch_size=2, key=KEY, call=again)
            self.assertEqual(second["calls"], 0)         # nothing left to do
            self.assertEqual(stats2["todo"], 0)

    def test_duplicate_responses_are_asked_about_once(self):
        rows = [row("q1", "Talk on theater."), row("q2", "Talk on theater.")]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "d.jsonl")
            call, stats = all_keep(25)
            quietly(frs.run, rows, {}, path, batch_size=25, key=KEY, call=call)
            on_disk = frs.load_rows(path)
        self.assertEqual(stats["calls"], 1)
        self.assertEqual(len(on_disk), 1)
        self.assertEqual(stats["batches"][0][1], [0])  # one row sent

    def test_decisions_are_keyed_on_the_normalised_response_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "d.jsonl")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(json.dumps({"key": "talk on theater.", "keep": False,
                                     "reason": "a bare title"}) + "\n")
            decided = frs.load_decisions(path)
        self.assertFalse(decided[frs.row_key(row("q1", "  Talk\non theater. "))]["keep"])

    def test_a_malformed_decisions_line_is_ignored_not_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "d.jsonl")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("{not json\n")
                fh.write(json.dumps({"key": "a.", "keep": True, "reason": "ok"}) + "\n")
            decided = frs.load_decisions(path)
        self.assertEqual(set(decided), {"a."})


class BatchingTest(unittest.TestCase):
    def test_batch_size_is_twenty_five_and_batches_do_not_overlap(self):
        self.assertEqual(frs.BATCH, 25)
        chunks = frs.batches(list(range(60)), 25)
        self.assertEqual([len(c) for c in chunks], [25, 25, 10])
        flat = [n for c in chunks for n in c]
        self.assertEqual(sorted(flat), list(range(60)))
        self.assertEqual(len(flat), len(set(flat)))

    def test_the_model_is_the_one_the_opencode_provider_answers_to(self):
        self.assertEqual(frs.MODEL, jz.DEFAULT_MODEL)
        self.assertEqual(frs.MODEL, "space-bunny-free")

    def test_the_transport_is_judge_opencode(self):
        # Same key loading, same headers, same 429 abort: one transport, not a copy.
        self.assertIs(frs.jz, jz)

    def test_the_batch_payload_carries_id_prompt_and_response(self):
        payload = json.loads(frs.batch_payload([row("q1", "r1"), row("q2", "r2")]))
        self.assertEqual(payload, [{"id": 0, "prompt": "q1", "response": "r1"},
                                   {"id": 1, "prompt": "q2", "response": "r2"}])

    def test_the_instruction_asks_for_keep_flags_and_nothing_else(self):
        self.assertIn("\"keep\"", frs.SYSTEM)
        self.assertIn("ONLY a JSON array", frs.SYSTEM)
        for excluded in ("bare title", "plan", "log", "someone else's words"):
            self.assertIn(excluded, frs.SYSTEM)


class WriteFilteredTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = tmp.name
        self.out = os.path.join(self.tmp, "filtered.jsonl")

    def test_kept_rows_are_written_in_input_order_with_fields_untouched(self):
        rows = [row("q1", "a"), row("q2", "b"), row("q3", "c")]
        decided = {"a": {"keep": True}, "c": {"keep": True}, "b": {"keep": False}}
        kept = frs.write_filtered(rows, decided, self.out)
        self.assertEqual([r["response"] for r in kept], ["a", "c"])
        written = frs.load_rows(self.out)
        self.assertEqual(written, [rows[0], rows[2]])

    def test_rewriting_replaces_the_file_so_two_runs_do_not_double_it(self):
        rows = [row("q1", "a"), row("q2", "b")]
        frs.write_filtered(rows, {"a": {"keep": True}, "b": {"keep": True}}, self.out)
        frs.write_filtered(rows, {"a": {"keep": True}, "b": {"keep": True}}, self.out)
        self.assertEqual(len(frs.load_rows(self.out)), 2)

    def test_extra_fields_such_as_translator_provenance_survive(self):
        r = row("q1", "a passage")
        r["translators"] = ["O'Brien"]
        written = frs.write_filtered([r], {frs.row_key(r): {"keep": True}}, self.out)
        self.assertEqual(written[0]["translators"], ["O'Brien"])

    def test_a_row_with_no_decision_is_not_written(self):
        rows = [row("q1", "a"), row("q2", "b")]
        self.assertEqual(frs.write_filtered(rows, {"a": {"keep": True}}, self.out),
                         [rows[0]])
        self.assertEqual(len(frs.load_rows(self.out)), 1)


class EndToEndTest(unittest.TestCase):
    """Rules, then the model, then the file — on a temporary copy of a tiny corpus."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = tmp.name
        self.src = os.path.join(self.tmp, "rows.jsonl")
        self.decisions = os.path.join(self.tmp, "decisions.jsonl")
        self.out = os.path.join(self.tmp, "filtered.jsonl")
        self.rows = [
            row("let's spend tonight talking about the theater.", "Talk on theater.",
                "n1", "Notebooks"),
            row("today feels like a good day to stop resisting the thought of dying.",
                "September 22. The Happy Death.", "n2", "Notebooks"),
            row("tell me how the town took the news.", 'Id. "I am not an orphan."', "n3"),
            row("what do you make of the crowd?", "The crowd wanted only bread, and was "
                "given a speech about the future of the world.", "n4"),
        ]
        with open(self.src, "w", encoding="utf-8") as fh:
            for r in self.rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    def run_main(self, argv, call):
        with mock.patch.object(frs, "call_model", call), \
                mock.patch.object(frs.jz, "load_key", return_value=KEY), \
                mock.patch("sys.argv", ["filter_reply_suitability.py"] + argv):
            code = quietly(frs.main)
        return code

    def test_dry_run_touches_no_file_and_makes_no_call(self):
        def call(system, user, key=None):
            raise AssertionError("a dry run must not call the model")

        before = os.path.exists(self.out)
        self.assertEqual(self.run_main(["--in", self.src, "--out", self.out,
                                        "--decisions", self.decisions, "--dry-run",
                                        "--samples", "0"], call), 0)
        self.assertEqual(os.path.exists(self.out), before)
        self.assertFalse(os.path.exists(self.decisions))

    def test_the_full_path_keeps_a_reply_and_drops_a_title(self):
        survivors, dropped, _ = frs.apply_rules(frs.load_rows(self.src))
        self.assertEqual(len(survivors), 2)      # the crisis row and the citation are gone
        self.assertEqual(dropped[frs.RULE_LABELS[2]], 1)
        self.assertEqual(dropped[frs.RULE_LABELS[1]], 1)

        def call(system, user, key=None):
            items = json.loads(user)
            self.assertEqual(len(items), 2)
            return json.dumps([{"id": 0, "keep": False, "reason": "a bare title"},
                               {"id": 1, "keep": True, "reason": "reads as a reply"}])

        self.assertEqual(self.run_main(["--in", self.src, "--out", self.out,
                                        "--decisions", self.decisions, "--samples", "0"],
                                       call), 0)
        kept = frs.load_rows(self.out)
        self.assertEqual([r["response"] for r in kept],
                         ["The crowd wanted only bread, and was given a speech about the "
                          "future of the world."])
        records = frs.load_rows(self.decisions)
        self.assertEqual({r["keep"] for r in records}, {True, False})
        self.assertEqual(records[1]["reason"], "reads as a reply")

    def test_a_rerun_after_the_model_stage_makes_no_calls(self):
        def call(system, user, key=None):
            return decisions(*((i, True) for i in range(len(json.loads(user)))))

        self.run_main(["--in", self.src, "--out", self.out, "--decisions", self.decisions,
                       "--samples", "0"], call)
        self.assertEqual(len(frs.load_rows(self.out)), 2)

        def no_calls(system, user, key=None):
            raise AssertionError("a resumed run must not resend decided rows")

        self.assertEqual(self.run_main(["--in", self.src, "--out", self.out,
                                        "--decisions", self.decisions, "--samples", "0"],
                                       no_calls), 0)
        self.assertEqual(len(frs.load_rows(self.out)), 2)

    def test_limit_caps_the_rows_sent_and_the_rest_stay_undecided(self):
        seen = []

        def call(system, user, key=None):
            seen.extend(item["id"] for item in json.loads(user))
            return decisions(*((i, True) for i in range(len(json.loads(user)))))

        self.assertEqual(self.run_main(["--in", self.src, "--out", self.out,
                                        "--decisions", self.decisions, "--limit", "1",
                                        "--samples", "0"], call), 0)
        self.assertEqual(len(seen), 1)
        self.assertEqual(len(frs.load_rows(self.out)), 1)
        self.assertEqual(len(frs.load_rows(self.decisions)), 1)

    def test_the_real_transport_is_never_reached_with_no_network(self):
        # A guard on the whole file: nothing in the module calls requests at import, and
        # the only entry point that talks to the gateway goes through the injected call.
        rows = frs.load_rows(self.src)
        survivors, _, _ = frs.apply_rules(rows)
        with mock.patch.object(frs.jz.requests, "post",
                               side_effect=AssertionError("network in a test")):
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "d.jsonl")
                call, _ = all_keep(25)
                quietly(frs.run, survivors, {}, path, key=KEY, call=call)


class SamplePrintTest(unittest.TestCase):
    def test_samples_are_clipped_and_labelled(self):
        rows = [row("q" + "x" * 200, "r" + "y" * 200), row("q2", "r2")]
        decided = {frs.row_key(rows[0]): {"keep": True, "reason": "reads as a reply"},
                   frs.row_key(rows[1]): {"keep": False, "reason": "a bare title"}}
        with mock.patch("sys.stdout") as out:
            frs.print_samples(rows, decided, 1)
        printed = "".join(call.args[0] for call in out.write.call_args_list)
        self.assertIn("kept pairs", printed)
        self.assertIn("dropped pairs", printed)
        self.assertIn("a bare title", printed)
        self.assertLessEqual(max(len(line) for line in printed.splitlines()), 100)


if __name__ == "__main__":
    unittest.main()