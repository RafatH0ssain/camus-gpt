#!/usr/bin/env python3
"""
test_synthesize_prompts.py — unit tests for pipeline/synthesize_prompts.py.

All passages are synthetic sentences written for this file; nothing under data/ is
read or written. No network: requests.post is mocked and every call is made with
the fake KEY below, so the real opencode auth file is never opened.

    python -m unittest pipeline/test_synthesize_prompts.py -v
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "rag"))

import judge_opencode as jz  # noqa: E402
import synthesize_prompts as sp  # noqa: E402

KEY = "fake-key-must-never-be-printed-0123456789"


def row(chunk_id, passage, source="The Rebel", **extra):
    base = {"source": source, "chunk_id": chunk_id, "passage": passage,
            "words": len(passage.split()), "prompt": None}
    base.update(extra)
    return base


def notebook_row(chunk_id, passage):
    return row(chunk_id, passage, source="Notebooks (1935-1942)")


def words(count, filler="word"):
    return " ".join(f"{filler}{i}" for i in range(count))


def reply(*prompts):
    """A well-formed reply for a batch of len(prompts) passages."""
    return json.dumps([{"id": i, "prompt": p} for i, p in enumerate(prompts)])


class FakeResponse:
    """Enough of requests.Response for judge_opencode.judge()."""

    def __init__(self, status_code=200, payload=None, text=None):
        self.status_code = status_code
        self._payload = payload
        self.text = text if text is not None else (json.dumps(payload) if payload else "")

    def json(self):
        if self._payload is None:
            raise ValueError("Expecting value: line 1 column 1 (char 0)")
        return self._payload


def chat(content):
    return FakeResponse(200, {"choices": [{"message": {"role": "assistant",
                                                       "content": content}}]})


def ratelimited():
    return FakeResponse(429, text="usage limit reached, try again later")


class DropRuleTests(unittest.TestCase):
    def test_leading_quotation_mark(self):
        for passage in ('"The sea was cold that morning.', "\u201cThe sea was cold.",
                        "«The sea was cold."):
            self.assertIsNotNone(sp.drop_rule(passage), passage)

    def test_number_then_paren_or_period(self):
        for passage in ("3) The obvious injustice of the matter",
                        "12. A government that will reform the constitution"):
            self.assertIsNotNone(sp.drop_rule(passage), passage)

    def test_number_without_a_marker_survives(self):
        self.assertIsNone(sp.drop_rule("1942 was the year everything changed."))
        self.assertIsNone(sp.drop_rule("22 men were guillotined that winter."))

    def test_cf_reference(self):
        self.assertIsNotNone(sp.drop_rule("See the note, cf. the winter letters."))
        self.assertIsNotNone(sp.drop_rule("Cf. the entry for the same month."))

    def test_page_reference(self):
        self.assertIsNotNone(sp.drop_rule("The entry resumes on p. 12 of the volume."))
        self.assertIsNotNone(sp.drop_rule("The discussion runs over pp. 30-31."))

    def test_words_containing_cf_are_kept(self):
        self.assertIsNone(sp.drop_rule("The careful reasoning of the passage holds."))
        self.assertIsNone(sp.drop_rule("A page of the manuscript was missing."))

    def test_empty_passage(self):
        self.assertIsNone(sp.drop_rule(""))

    def test_first_rule_wins_so_counts_add_up(self):
        passage = '"Cf. p. 12 of the volume.'
        self.assertEqual(sp.drop_rule(passage),
                         "reported speech (opens on a quotation mark)")


class SelectionTests(unittest.TestCase):
    def test_drop_rules_run_before_the_word_window(self):
        rows = [row("d1", '"' + words(30)),
                row("k1", words(30))]
        selected, stats = sp.select_rows(rows)
        self.assertEqual([r["chunk_id"] for r in selected], ["k1"])
        self.assertEqual(sum(stats["dropped"].values()), 1)

    def test_notebook_word_window_is_three_to_eighty(self):
        rows = [notebook_row("n2", words(2)), notebook_row("n3", words(3)),
                notebook_row("n80", words(80)), notebook_row("n81", words(81))]
        selected, stats = sp.select_rows(rows)
        self.assertEqual(sorted(r["chunk_id"] for r in selected), ["n3", "n80"])
        self.assertEqual(stats["notebook_pool"], 2)

    def test_other_sources_take_twenty_to_one_hundred_and_twenty_words(self):
        rows = [row("a19", words(19)), row("a20", words(20)),
                row("a120", words(120)), row("a121", words(121))]
        selected, _ = sp.select_rows(rows)
        self.assertEqual(sorted(r["chunk_id"] for r in selected), ["a120", "a20"])

    def test_notebook_and_other_rules_do_not_cross(self):
        rows = [notebook_row("n5", words(5)), notebook_row("n95", words(95)),
                row("o25", words(25))]
        selected, _ = sp.select_rows(rows)
        self.assertEqual([r["chunk_id"] for r in selected], ["n5", "o25"])

    def test_notebook_cap_is_fifteen_hundred_in_total(self):
        rows = [notebook_row(f"n{i}", words(10, f"n{i}")) for i in range(2000)]
        selected, stats = sp.select_rows(rows)
        self.assertEqual(len(selected), sp.NOTEBOOK_TOTAL)
        self.assertEqual(stats["notebook_pool"], 2000)

    def test_notebook_cap_spans_the_three_volumes(self):
        rows = []
        for volume in ("1935-1942", "1942-1951", "1951-1959"):
            rows += [row(f"v{volume}-{i}", words(10, f"n{volume}-{i}"),
                         source=f"Notebooks ({volume})") for i in range(700)]
        selected, stats = sp.select_rows(rows)
        self.assertEqual(len(selected), sp.NOTEBOOK_TOTAL)
        self.assertEqual(len({r["source"] for r in selected}), 3)
        self.assertEqual(stats["notebook_pool"], 2100)

    def test_per_source_cap_is_one_hundred_and_fifty(self):
        rows = [row(f"s{i}", words(30, f"s{i}")) for i in range(400)]
        selected, stats = sp.select_rows(rows)
        self.assertEqual(len(selected), sp.OTHER_PER_SOURCE)
        self.assertEqual(stats["other_sources"]["The Rebel"],
                         {"eligible": 400, "already": 0, "kept": 150})


    def test_cap_is_per_source_not_whole_corpus(self):
        rows = ([row(f"a{i}", words(30, f"a{i}"), source="The Rebel") for i in range(200)] +
                [row(f"b{i}", words(30, f"b{i}"), source="American Journals")
                 for i in range(200)])
        selected, stats = sp.select_rows(rows)
        self.assertEqual(len(selected), 300)
        self.assertEqual(stats["other_sources"]["The Rebel"]["kept"], 150)
        self.assertEqual(stats["other_sources"]["American Journals"]["kept"], 150)

    def test_sampling_is_uniform_not_shortest_first(self):
        """The notebook cap must not be the 1500 shortest: a corpus of uniform
        samples keeps the long tail the whole point of the notebooks is."""
        rows = [notebook_row(f"n{i}", words(3 + i % 78, f"n{i}")) for i in range(2000)]
        selected, _ = sp.select_rows(rows)
        lengths = sorted(sp.row_words(r) for r in selected)
        self.assertEqual(len(selected), 1500)
        self.assertGreaterEqual(lengths[-1], 70, "the longest notebooks were dropped")
        self.assertGreaterEqual(lengths[len(lengths) // 2], 30,
                                "the sample is skewed short, so it is not uniform")
        self.assertLessEqual(min(lengths), 5,
                             "three-word entries should survive a uniform sample")

    def test_same_seed_selects_the_same_passages(self):
        rows = ([notebook_row(f"n{i}", words(10, f"n{i}")) for i in range(200)] +
                [row(f"r{i}", words(30, f"r{i}")) for i in range(400)])
        first = [r["chunk_id"] for r in sp.select_rows(rows)[0]]
        second = [r["chunk_id"] for r in sp.select_rows(rows)[0]]
        self.assertEqual(first, second)

    def test_different_seed_changes_the_sample(self):
        rows = [row(f"r{i}", words(30, f"r{i}")) for i in range(400)]
        first = {r["chunk_id"] for r in sp.select_rows(rows, seed=13)[0]}
        second = {r["chunk_id"] for r in sp.select_rows(rows, seed=99)[0]}
        self.assertNotEqual(first, second)

    def test_empty_input(self):
        selected, stats = sp.select_rows([])
        self.assertEqual(selected, [])
        self.assertEqual(stats["input"], 0)


class StableSelectionTests(unittest.TestCase):
    """The sample is a function of the passages, not of the candidate file around
    them: a file re-mined, reordered or carrying new columns selects the same
    passages, or a rerun pays for a different corpus than the run before it."""

    ROWS = ([notebook_row(f"n{i}", words(10, f"n{i}")) for i in range(200)] +
            [row(f"r{i}", words(30, f"r{i}")) for i in range(400)])

    def keys(self, rows, **kwargs):
        return [sp.row_key(r) for r in sp.select_rows(rows, **kwargs)[0]]

    def test_an_extra_field_on_every_row_changes_nothing(self):
        grown = [dict(r, translators="E. Gilson") for r in self.ROWS]
        self.assertEqual(self.keys(self.ROWS), self.keys(grown))

    def test_a_reordered_file_changes_nothing(self):
        self.assertEqual(self.keys(self.ROWS), self.keys(list(reversed(self.ROWS))))

    def test_the_same_seed_picks_the_same_passages(self):
        self.assertEqual(self.keys(self.ROWS), self.keys(self.ROWS))

    def test_the_chunk_id_is_not_part_of_the_selection_key(self):
        """Two rows carrying the same passage are one passage, so a re-mining that
        moved it to another chunk cannot put it in the file twice."""
        rows = self.ROWS + [row("moved-1", self.ROWS[0]["passage"],
                                source=self.ROWS[0]["source"])]
        keys = self.keys(rows)
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(keys, self.keys(self.ROWS))

    def test_a_passage_written_under_another_chunk_id_is_not_offered_again(self):
        already = self.ROWS[:50]
        written = sp.Written([{"response": r["passage"], "source": r["source"],
                               "src_id": "old-" + r["chunk_id"]} for r in already])
        selected, _ = sp.select_rows(self.ROWS, written=written)
        offered = {sp.row_key(r) for r in selected}
        self.assertFalse({sp.row_key(r) for r in already} & offered,
                         "a passage already written must not go out again")
        self.assertEqual(len(selected), 300, "150 notebooks left and 150 of The Rebel")


class CapAccountingTests(unittest.TestCase):
    """The caps describe the finished corpus, so they are counted against the output
    file: a source that already has rows gets the remainder and nothing more."""

    def rows(self, count, source="The Rebel"):
        return [row(f"s{i}", words(30, f"s{i}"), source=source) for i in range(count)]

    def written(self, rows):
        return sp.Written([{"response": r["passage"], "source": r["source"]}
                           for r in rows])

    def test_a_source_with_rows_in_the_file_gets_only_the_remainder(self):
        already = self.rows(100)
        selected, stats = sp.select_rows(self.rows(400), written=self.written(already))
        self.assertEqual(len(selected), sp.OTHER_PER_SOURCE - 100)
        self.assertEqual(stats["other_sources"]["The Rebel"],
                         {"eligible": 400, "already": 100,
                          "kept": sp.OTHER_PER_SOURCE - 100})
        self.assertFalse({sp.row_key(r) for r in selected}
                         & {sp.row_key(r) for r in already},
                         "a passage already written must not go out again")

    def test_a_full_source_is_not_offered_another_row(self):
        selected, stats = sp.select_rows(
            self.rows(400), written=self.written(self.rows(sp.OTHER_PER_SOURCE)))
        self.assertEqual(selected, [])
        self.assertEqual(stats["other_sources"]["The Rebel"]["kept"], 0)

    def test_the_notebook_budget_is_shared_by_the_three_volumes(self):
        rows = []
        for volume in ("1935-1942", "1942-1951", "1951-1959"):
            rows += [row(f"v{volume}-{i}", words(10, f"n{volume}-{i}"),
                         source=f"Notebooks ({volume})") for i in range(900)]
        written = sp.Written([{"response": r["passage"], "source": r["source"]}
                              for r in rows[:1400]])
        selected, stats = sp.select_rows(rows, written=written)
        self.assertEqual(len(selected), sp.NOTEBOOK_TOTAL - 1400,
                         "1400 notebook rows are already written, so 100 may go out")
        self.assertEqual(stats["notebook_kept"], sp.NOTEBOOK_TOTAL - 1400)

    def test_caps_are_counted_on_passages_not_on_rows(self):
        """A file that still holds an old duplicate cannot starve a source: the
        budget is spent once per passage."""
        passage = words(30, "s0")
        written = sp.Written([{"response": passage, "source": "The Rebel"},
                              {"response": passage, "source": "The Rebel"}])
        self.assertEqual(written.spent("The Rebel"), 1)
        selected, _ = sp.select_rows(self.rows(400), written=written)
        self.assertEqual(len(selected), sp.OTHER_PER_SOURCE - 1)


class HttpResumeTests(unittest.TestCase):
    """A run through the real transport with requests.post mocked: the batching, the
    alignment and jz.judge's own reply checks all run, and nothing leaves the
    process. A run is killed the way a real one is — the gateway's cap."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = os.path.join(self.tmp.name, "rows.jsonl")
        self.sent = []
        self.calls = 0
        self.die_after = None
        patcher = mock.patch.object(jz.requests, "post")
        self.post = patcher.start()
        self.addCleanup(patcher.stop)
        self.post.side_effect = self.fake_post

    def fake_post(self, url, **kwargs):
        batch = json.loads(kwargs["json"]["messages"][1]["content"])
        self.calls += 1
        if self.die_after is not None and self.calls > self.die_after:
            raise jz.UsageCapReached("usage cap reached (HTTP 429)")
        self.sent.extend(item["passage"] for item in batch)
        return chat(json.dumps([{"id": item["id"],
                                "prompt": f"a remark, passage {item['id']}, take {self.calls}"}
                               for item in batch]))

    def run_until_done(self, rows, batch_size=20):
        """Rerun as often as it takes, one batch per run, the way a killed run is
        restarted. Returns the number of runs it took."""
        selected, _ = sp.select_rows(rows)
        runs = 0
        while True:
            runs += 1
            self.die_after = self.calls + 1
            stats, stop = sp.run(selected, self.out, batch_size, key=KEY)
            self.die_after = None
            if stats["todo"] == 0 or stop is None:
                return runs

    def written(self):
        return sp.load_rows(self.out) if os.path.exists(self.out) else []

    def per_source(self):
        counts = {}
        for row in self.written():
            counts[row["source"]] = counts.get(row["source"], 0) + 1
        return counts

    def corpus(self, count, source="The Rebel"):
        return [row(f"c{i}", words(30, f"p{i}") + " and a little more here",
                    source=source) for i in range(count)]

    def test_a_run_killed_and_resumed_writes_no_duplicate(self):
        rows = self.corpus(60)
        selected, _ = sp.select_rows(rows)
        self.die_after = 1                      # one batch, then the cap
        first, stop = sp.run(selected, self.out, key=KEY)
        self.assertIn("429", stop)
        self.assertEqual(len(self.written()), 20)
        before = [json.dumps(r, sort_keys=True) for r in self.written()]

        self.die_after = None                   # the cap is gone; the run finishes
        stats, _ = sp.run(selected, self.out, key=KEY)
        self.assertEqual(stats["todo"], 40, "the resumed run re-sent the first batch")
        after = self.written()
        self.assertEqual(len(after), 60)
        self.assertEqual(len({sp.normalise(r["response"]) for r in after}), 60)
        self.assertEqual([json.dumps(r, sort_keys=True) for r in after[:20]], before,
                         "the rows already written must not be touched")
        self.assertEqual(len(self.sent), 60, "no passage went out twice")

    def test_a_resume_after_a_re_mining_writes_no_duplicate(self):
        """The same passages, re-mined into different chunks: a resume keyed on the
        chunk id would pay for all sixty again."""
        rows = self.corpus(60)
        selected, _ = sp.select_rows(rows)
        self.die_after = 1
        sp.run(selected, self.out, key=KEY)
        self.die_after = None

        remined = [row(f"chunk-{i}", r["passage"], source=r["source"])
                   for i, r in enumerate(rows)]
        again, _ = sp.select_rows(remined)
        stats, _ = sp.run(again, self.out, key=KEY)
        self.assertEqual(stats["todo"], 40)
        after = self.written()
        self.assertEqual(len(after), 60)
        self.assertEqual(len({sp.normalise(r["response"]) for r in after}), 60)

    def test_the_caps_hold_across_runs_that_were_all_killed(self):
        rows = self.corpus(400)
        self.run_until_done(rows)
        counts = self.per_source()
        self.assertEqual(counts["The Rebel"], sp.OTHER_PER_SOURCE,
                         "a killed-and-resumed run must not push a source past its cap")
        self.assertEqual(len({sp.normalise(r["response"]) for r in self.written()}),
                         sp.OTHER_PER_SOURCE)

    def test_nothing_more_is_offered_once_the_caps_are_spent(self):
        rows = self.corpus(400)
        self.run_until_done(rows)
        selected, stats = sp.select_rows(rows, written=sp.written_state(self.out))
        self.assertEqual(selected, [])
        calls = self.calls
        sp.run(selected, self.out, key=KEY)
        self.assertEqual(self.calls, calls, "an exhausted corpus makes no call")
        self.assertEqual(len(self.written()), sp.OTHER_PER_SOURCE)

    def test_the_notebook_cap_holds_across_runs(self):
        rows = [row(f"n{i}", words(10, f"n{i}"),
                    source="Notebooks (1935-1942)") for i in range(60)]
        self.run_until_done(rows)
        self.assertEqual(len(self.written()), 60)
        self.assertEqual(len(self.sent), 60)


class PayloadTests(unittest.TestCase):
    def test_payload_carries_every_passage_with_an_id(self):
        batch = [row("a", "first passage"), row("b", "second passage")]
        payload = json.loads(sp.batch_payload(batch))
        self.assertEqual([item["id"] for item in payload], [0, 1])
        self.assertEqual([item["passage"] for item in payload],
                         ["first passage", "second passage"])

    def test_payload_never_carries_the_source_or_chunk_id(self):
        payload = sp.batch_payload([row("chunk-1", "a passage here")])
        self.assertNotIn("chunk-1", payload)
        self.assertNotIn("The Rebel", payload)

    def test_batches_split_by_size(self):
        self.assertEqual([len(b) for b in sp.batches(list(range(45)), 20)],
                         [20, 20, 5])
        self.assertEqual(sp.batches([], 20), [])


class ParseReplyTests(unittest.TestCase):
    def test_bare_array(self):
        items = sp.parse_reply('[{"id": 0, "prompt": "was it cold that morning"}]')
        self.assertEqual(items, [{"id": 0, "prompt": "was it cold that morning"}])

    def test_fenced_array(self):
        items = sp.parse_reply('```json\n[{"id": 0, "prompt": "hello"}]\n```')
        self.assertEqual(items[0]["prompt"], "hello")

    def test_array_wrapped_in_prose(self):
        items = sp.parse_reply('Sure:\n[{"id": 0, "prompt": "hello"}]\nHope that helps.')
        self.assertEqual(items[0]["prompt"], "hello")

    def test_object_wrapping_an_array(self):
        items = sp.parse_reply('{"prompts": [{"id": 0, "prompt": "hello"}]}')
        self.assertEqual(items[0]["prompt"], "hello")

    def test_reply_without_json_raises(self):
        with self.assertRaises(ValueError):
            sp.parse_reply("I cannot do that.")

    def test_align_by_id(self):
        batch = [row("a", "one"), row("b", "two")]
        items = [{"id": 1, "prompt": "second"}, {"id": 0, "prompt": "first"}]
        self.assertEqual(sp.align(batch, items), ["first", "second"])

    def test_align_falls_back_to_order_when_ids_are_wrong(self):
        batch = [row("a", "one"), row("b", "two")]
        items = [{"id": 7, "prompt": "first"}, {"id": 8, "prompt": "second"}]
        self.assertEqual(sp.align(batch, items), ["first", "second"])

    def test_align_leaves_a_gap_when_there_is_nothing_to_match(self):
        batch = [row("a", "one"), row("b", "two")]
        items = [{"id": 0, "prompt": "first"}]
        self.assertEqual(sp.align(batch, items), ["first", None])


class PromptFaultTests(unittest.TestCase):
    PASSAGE = "The sea was cold that morning and nobody spoke on the way back."

    def test_a_plain_prompt_is_accepted(self):
        self.assertIsNone(sp.prompt_fault("was it cold that morning",
                                          self.PASSAGE, set()))

    def test_empty_and_non_string_prompts(self):
        self.assertEqual(sp.prompt_fault("", self.PASSAGE, set()), "empty prompt")
        self.assertEqual(sp.prompt_fault(None, self.PASSAGE, set()), "empty prompt")

    def test_over_twenty_five_words(self):
        self.assertEqual(sp.prompt_fault(words(26), self.PASSAGE, set()),
                         f"over {sp.MAX_PROMPT_WORDS} words")
        self.assertIsNone(sp.prompt_fault(words(25), self.PASSAGE, set()))

    def test_forbidden_names(self):
        for prompt in ("what does camus think of it",
                       "is that in the notebooks too",
                       "which book was that from"):
            self.assertIsNotNone(sp.prompt_fault(prompt, self.PASSAGE, set()), prompt)

    def test_ordinary_words_are_not_forbidden(self):
        self.assertIsNone(sp.prompt_fault("did you read the booking notes",
                                          self.PASSAGE, set()))

    def test_quoting_the_passage_is_caught(self):
        prompt = "the sea was cold that morning and nobody spoke"
        self.assertEqual(sp.prompt_fault(prompt, self.PASSAGE, set()), "quotes the passage")

    def test_short_overlap_is_not_a_quote(self):
        self.assertIsNone(sp.prompt_fault("the sea was cold", self.PASSAGE, set()))

    def test_repeat_of_an_earlier_prompt(self):
        seen = {"was it cold that morning"}
        self.assertEqual(sp.prompt_fault("was it cold that morning", self.PASSAGE, seen),
                         "repeats a prompt already written")

    def test_whitespace_is_normalised_before_comparison(self):
        seen = {"was it cold that morning"}
        self.assertEqual(sp.prompt_fault("  was  it cold that morning ", self.PASSAGE, seen),
                         "repeats a prompt already written")


class QuestionQuotaTests(unittest.TestCase):
    """The instruction asks for at most 5 questions in a batch of 20; synthesise
    enforces it, so the corpus cannot drift into a corpus of questions."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.out = os.path.join(tmp.name, "rows.jsonl")
        self.batch = [row(f"c{i}", f"passage number {i} of some length here")
                      for i in range(20)]

    def questions(self, prompts):
        return [p for p in prompts if p.endswith("?")]

    def run_with(self, prompts):
        stats, _ = sp.run(self.batch, self.out, key=KEY,
                          call=lambda *a, **k: reply(*prompts))
        return stats, [r["prompt"] for r in sp.load_rows(self.out)]

    def test_all_questions_over_the_quota_are_dropped(self):
        prompts = [f"is passage {i} true?" for i in range(20)]
        stats, kept = self.run_with(prompts)
        self.assertEqual(len(self.questions(kept)), sp.MAX_QUESTIONS_PER_BATCH)
        self.assertEqual(stats["faults"]["over the batch question quota"], 15)
        self.assertEqual(stats["written"], 5)

    def test_statements_survive_a_question_heavy_batch(self):
        prompts = [f"is passage {i} true?" if i < 8 else f"passage {i} holds up"
                   for i in range(20)]
        _, kept = self.run_with(prompts)
        self.assertEqual(len(self.questions(kept)), 5)
        self.assertEqual(len(kept), 17)  # 5 of the 8 questions, all 12 statements

    def test_a_batch_of_statements_is_untouched(self):
        prompts = [f"passage {i} holds up" for i in range(20)]
        stats, kept = self.run_with(prompts)
        self.assertEqual(stats["written"], 20)
        self.assertEqual(len(kept), 20)

    def test_the_quota_does_not_reset_mid_reply(self):
        """The first five questions are kept, not the last five."""
        prompts = [f"is this {i} the one?" if i < 7 else f"statement {i}"
                   for i in range(20)]
        _, kept = self.run_with(prompts)
        self.assertEqual(self.questions(kept),
                         ["is this 0 the one?", "is this 1 the one?", "is this 2 the one?",
                          "is this 3 the one?", "is this 4 the one?"])


class UsageLimitTests(unittest.TestCase):
    def test_recognised_cap_wording(self):
        for message in ("opencode judge HTTP 429: usage limit reached",
                        "monthly limit exhausted for this key",
                        "quota exceeded",
                        "you have sent too many requests",
                        "rate_limit exceeded, slow down"):
            self.assertTrue(sp.is_usage_limit_error(RuntimeError(message)), message)

    def test_ordinary_errors_are_not_caps(self):
        for message in ("connection reset by peer", "HTTP 500: internal error",
                        "judge reply was not chat/completions-shaped"):
            self.assertFalse(sp.is_usage_limit_error(RuntimeError(message)), message)


class CallModelTests(unittest.TestCase):
    """call_model hands the work to jz.judge: model, key, headers, budget and the
    429 abort all come from there, so these check the wiring, not a second
    implementation of it. The reply is a real one — an array of objects, which is
    what jz.judge's own reply check looks for."""

    def setUp(self):
        patcher = mock.patch.dict(os.environ, {"OPENCODE_API_KEY": KEY})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.post = mock.patch.object(jz.requests, "post").start()
        self.addCleanup(mock.patch.stopall)

    def test_uses_the_shared_transport_headers_and_model(self):
        self.post.return_value = chat(reply("hello there"))
        self.assertEqual(sp.call_model("SYSTEM", "USER"), reply("hello there"))
        sent = self.post.call_args
        self.assertEqual(sent.args[0], jz.ENDPOINT)
        self.assertEqual(sent.kwargs["headers"]["Authorization"], f"Bearer {KEY}")
        self.assertEqual(sent.kwargs["headers"]["x-opencode-session"], jz.SESSION_ID)
        self.assertEqual(sent.kwargs["json"]["model"], sp.MODEL)
        self.assertEqual(sp.MODEL, jz.DEFAULT_MODEL)
        self.assertEqual(sent.kwargs["json"]["messages"],
                         [{"role": "system", "content": "SYSTEM"},
                          {"role": "user", "content": "USER"}])

    def test_the_token_budget_is_the_shared_one(self):
        self.post.return_value = chat(reply("hello there"))
        sp.call_model("SYSTEM", "USER")
        self.assertEqual(self.post.call_args.kwargs["json"]["max_tokens"],
                         jz.MAX_TOKENS)
        self.assertGreaterEqual(jz.MAX_TOKENS, 20 * 25 * 2,
                                "the budget must fit 20 prompts of 25 words")

    def test_429_raises_the_shared_usage_cap(self):
        self.post.return_value = ratelimited()
        with self.assertRaises(jz.UsageCapReached):
            sp.call_model("SYSTEM", "USER")

    def test_an_empty_reply_is_not_charged_to_the_caller_twice(self):
        """jz.judge retries an unusable reply itself, so one call_model is still
        one batch attempt: the retry budget stays where the shared transport
        keeps it rather than doubling up here."""
        self.post.side_effect = [chat("not json at all"), chat(reply("hello"))]
        self.assertEqual(sp.call_model("SYSTEM", "USER"), reply("hello"))
        self.assertEqual(self.post.call_count, 2)


class RunTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = os.path.join(self.tmp.name, "rows.jsonl")
        self.calls = []

    def fake_call(self, replies):
        """A stand-in for call_model: replies[i] is the i-th reply, and a
        RuntimeError or UsageCapReached in the list is raised instead."""
        pending = list(replies)

        def call(system, user, key=None):
            self.calls.append((system, user))
            item = pending.pop(0) if pending else "[]"
            if isinstance(item, BaseException):
                raise item
            return item

        return call

    def read(self):
        return sp.load_rows(self.out) if os.path.exists(self.out) else []

    def test_rows_carry_the_verbatim_passage_and_the_schema(self):
        batch = [row("c1", "The sea was cold that morning."),
                 row("c2", "Nobody spoke on the way back.")]
        stats, stop = sp.run(batch, self.out, key=KEY,
                             call=self.fake_call([reply("was it cold",
                                                        "what about the silence")]))
        self.assertIsNone(stop)
        rows = self.read()
        self.assertEqual(stats["written"], 2)
        self.assertEqual(rows[0], {"prompt": "was it cold",
                                  "response": "The sea was cold that morning.",
                                  "kind": "primary", "source": "The Rebel",
                                  "src_id": "c1"})
        self.assertEqual([r["response"] for r in rows],
                         ["The sea was cold that morning.", "Nobody spoke on the way back."])

    def test_the_instruction_is_sent_on_every_call(self):
        batch = [row("c1", "one passage here"), row("c2", "another passage here")]
        sp.run(batch, self.out, key=KEY, call=self.fake_call([reply("a", "b")]))
        self.assertEqual(len(self.calls), 1)
        system, user = self.calls[0]
        for phrase in ("25 words", "never quote", "JSON array", "question mark"):
            self.assertIn(phrase, system)
        self.assertIn("one passage here", user)

    def test_the_question_quota_in_the_instruction_matches_the_code(self):
        self.assertIn(str(sp.MAX_QUESTIONS_PER_BATCH), sp.SYSTEM)

    def test_twenty_passages_per_call(self):
        batch = [row(f"c{i}", f"passage number {i} of this batch") for i in range(45)]
        calls = []
        offset = [0]

        def call(system, user, key=None):
            size = len(json.loads(user))
            calls.append(size)
            prompts = [f"p{offset[0] + n}" for n in range(size)]
            offset[0] += size
            return reply(*prompts)

        stats, _ = sp.run(batch, self.out, key=KEY, call=call)
        self.assertEqual(calls, [20, 20, 5])
        self.assertEqual(stats["batches"], 3)
        self.assertEqual(stats["written"], 45)

    def test_a_finished_batch_is_written_before_the_next_call(self):
        batch = [row(f"c{i}", f"passage number {i}") for i in range(40)]
        seen_counts = []

        def call(system, user, key=None):
            seen_counts.append(len(self.read()))
            return reply(*[f"prompt {i}" for i in range(len(json.loads(user)))])

        sp.run(batch, self.out, key=KEY, call=call)
        self.assertEqual(seen_counts, [0, 20])

    def test_a_rerun_skips_the_rows_already_written(self):
        batch = [row(f"c{i}", f"passage number {i} here") for i in range(20)]
        sp.run(batch, self.out, key=KEY,
               call=self.fake_call([reply(*[f"prompt {i}" for i in range(20)])]))
        stats, _ = sp.run(batch, self.out, key=KEY, call=self.fake_call([]))
        self.assertEqual(stats["todo"], 0)
        self.assertEqual(stats["written"], 0)
        self.assertEqual(len(self.read()), 20)

    def test_a_resumed_run_continues_where_it_stopped(self):
        batch = [row(f"c{i}", f"passage number {i} here") for i in range(30)]
        sp.run(batch, self.out, key=KEY, call=self.fake_call([
            reply(*[f"prompt {i}" for i in range(20)]),
            reply(*[f"second {i}" for i in range(10)]),
        ]))
        self.assertEqual(len(self.read()), 30)
        stats, _ = sp.run(batch, self.out, key=KEY, call=self.fake_call([]))
        self.assertEqual(stats["todo"], 0)
        self.assertEqual(len(self.read()), 30)

    def test_a_429_stops_the_run_without_retrying(self):
        batch = [row(f"c{i}", f"passage number {i} here") for i in range(60)]
        stats, stop = sp.run(batch, self.out, batch_size=20, key=KEY, call=self.fake_call([
            reply(*[f"prompt {i}" for i in range(20)]),
            jz.UsageCapReached("usage cap reached (HTTP 429)"),
            reply(*[f"prompt {i}" for i in range(20)]),
            reply(*[f"prompt {i}" for i in range(20)]),
        ]))
        self.assertIsNotNone(stop)
        self.assertIn("429", stop)
        self.assertEqual(stats["calls"], 2, "no batch past the cap may be attempted")
        self.assertEqual(len(self.read()), 20)

    def test_a_usage_limit_error_body_also_stops_the_run(self):
        batch = [row(f"c{i}", f"passage number {i} here") for i in range(40)]
        stats, stop = sp.run(batch, self.out, key=KEY, call=self.fake_call([
            RuntimeError("opencode judge HTTP 400: monthly limit reached"),
            reply(*[f"prompt {i}" for i in range(20)]),
        ]))
        self.assertIsNotNone(stop)
        self.assertEqual(stats["calls"], 1, "a cap must not be retried")
        self.assertEqual(stats["skipped"], [])

    def test_other_errors_are_retried_once(self):
        batch = [row(f"c{i}", f"passage number {i} here") for i in range(20)]
        stats, stop = sp.run(batch, self.out, key=KEY, call=self.fake_call([
            RuntimeError("connection reset by peer"),
            reply(*[f"prompt {i}" for i in range(20)]),
        ]))
        self.assertIsNone(stop)
        self.assertEqual(stats["calls"], 2)
        self.assertEqual(stats["written"], 20)

    def test_a_batch_that_fails_twice_is_skipped_and_its_ids_logged(self):
        batch = [row(f"c{i}", f"passage number {i} here") for i in range(40)]
        stats, stop = sp.run(batch, self.out, key=KEY, call=self.fake_call([
            RuntimeError("connection reset by peer"),
            RuntimeError("connection reset by peer"),
            reply(*[f"prompt {i}" for i in range(20)]),
        ]))
        self.assertIsNone(stop)
        self.assertEqual(stats["skipped"], [["c0", "c1", "c2", "c3", "c4", "c5",
                                             "c6", "c7", "c8", "c9", "c10", "c11",
                                             "c12", "c13", "c14", "c15", "c16",
                                             "c17", "c18", "c19"]])
        self.assertEqual(stats["written"], 20)

    def test_a_reply_that_will_not_parse_skips_the_batch(self):
        batch = [row(f"c{i}", f"passage number {i} here") for i in range(20)]
        stats, _ = sp.run(batch, self.out, key=KEY, call=self.fake_call([
            "I am afraid I cannot help with that."]))
        self.assertEqual(stats["written"], 0)
        self.assertEqual(len(stats["skipped"]), 1)
        self.assertEqual(stats["calls"], 1, "an unparseable reply is not retried")

    def test_rejected_prompts_are_counted_and_left_out(self):
        passage = "The sea was cold that morning and nobody spoke on the way back."
        batch = [row("c1", passage),
                 row("c2", "Another passage entirely, of equal length here."),
                 row("c3", "A third passage, also of a perfectly normal length."),
                 row("c4", "A fourth passage, of a perfectly ordinary length too.")]
        stats, _ = sp.run(batch, self.out, key=KEY, call=self.fake_call([reply(
            "the sea was cold that morning and nobody spoke",  # quotes the passage
            words(30),  # too long
            "which book is that from",  # forbidden
            "was it cold that morning",  # fine
        )]))
        self.assertEqual(stats["written"], 1)
        self.assertEqual(stats["rejected"], 3)
        self.assertEqual(stats["faults"]["quotes the passage"], 1)
        self.assertEqual(stats["faults"][f"over {sp.MAX_PROMPT_WORDS} words"], 1)
        self.assertEqual(stats["faults"]["mentions a book, notebook or Camus"], 1)
        self.assertEqual([r["src_id"] for r in self.read()], ["c4"])

    def test_the_same_prompt_twice_in_one_batch_is_kept_once(self):
        batch = [row("c1", "A passage of an ordinary length, nothing special."),
                 row("c2", "Another passage of an ordinary length, nothing special.")]
        stats, _ = sp.run(batch, self.out, key=KEY,
                          call=self.fake_call([reply("why then", "why then")]))
        self.assertEqual(stats["written"], 1)
        self.assertEqual(stats["faults"]["repeats a prompt already written"], 1)

    def test_an_empty_selection_writes_nothing(self):
        stats, stop = sp.run([], self.out, key=KEY, call=self.fake_call([]))
        self.assertEqual((stats["calls"], stats["written"], stop), (0, 0, None))
        self.assertEqual(self.read(), [])


if __name__ == "__main__":
    unittest.main()