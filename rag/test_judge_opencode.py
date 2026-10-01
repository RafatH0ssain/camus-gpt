#!/usr/bin/env python3
"""
test_judge_opencode.py — unit tests for the OpenCode Zen judge backend used by
rag/eval_camus.py and rag/eval_memory.py.

No network: requests.post is mocked, and the key is only ever the fake KEY below — the real
opencode auth file is never read, because every call is made with an explicit key or with
OPENCODE_API_KEY set. eval_memory is not imported (it reads the untracked
memory_fixtures.json at import time); it shares this backend, so it is covered from here.

    python -m unittest rag/test_judge_opencode.py -v
"""
import argparse, json, os, sys, tempfile, unittest
from unittest import mock

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import eval_camus as ec  # noqa: E402  (imports camus_rag; no KB or Ollama access at import)
import judge_opencode as jz  # noqa: E402

KEY = "fake-key-must-never-be-printed-0123456789"
PROBE = dict(id="pet_cat", cat="identity_pets", turns=["do you have a cat"],
             expect="Affirms having a cat and names Cigarette.", forbid="Other cat names.")
CONVO = [{"role": "user", "content": "do you have a cat"},
         {"role": "assistant", "content": "I had a cat. Cigarette."}]
VALID_JSON = ('{"voice": 4, "factuality": 5, "engagement": 3, "rationale": "terse and correct"}')
VALID_SCORES = {"voice": 4, "factuality": 5, "engagement": 3, "rationale": "terse and correct"}
CUT_OFF = '{"voice": 4, "rationale": "The response is terse, first-pe'   # ran out of tokens mid-object


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


def reasoning_only(reasoning):
    """A reasoning model that spent its whole budget thinking: content null, thoughts present."""
    return FakeResponse(200, {"choices": [{"message": {"role": "assistant", "content": None,
                                                       "reasoning_content": reasoning}}]})


def judge_a_probe():
    """eval_camus' judge, called the way its loop calls it (key from the patched env)."""
    return ec.judge_opencode(jz.DEFAULT_MODEL, PROBE, CONVO, "I had a cat. Cigarette.")


def run_like_harness(fn):
    """The catch both harnesses use: row["scores"] on success, row["judge_error"] on
    Exception, and nothing else — a failure never becomes a score."""
    row = {}
    try:
        row["scores"] = fn()
    except Exception as e:  # noqa: BLE001 - mirrors the harness
        row["judge_error"] = str(e)[:200]
    return row


class JudgeOpencodeTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = tmp.name
        self.auth = os.path.join(tmp.name, "auth.json")
        self.absent = os.path.join(tmp.name, "absent.json")
        self.set_env(**{"OPENCODE_API_KEY": KEY})
        self.post = mock.patch.object(jz.requests, "post").start()
        self.addCleanup(mock.patch.stopall)

    def set_env(self, **values):
        patcher = mock.patch.dict(os.environ, values)
        patcher.start()
        self.addCleanup(patcher.stop)

    def write_auth(self, key=KEY, payload=None):
        with open(self.auth, "w", encoding="utf-8") as fh:
            json.dump(payload if payload is not None else
                      {"opencode-go": {"type": "api", "key": key}}, fh)
        return self.auth

    # ------------------------------------------------------------------ headers ----
    def test_all_four_required_headers_are_sent(self):
        self.post.return_value = chat("{}")
        jz.judge("space-bunny-free", "sys", "prompt")
        sent = self.post.call_args
        self.assertEqual(sent.args[0], "https://opencode.ai/zen/go/v1/chat/completions")
        h = sent.kwargs["headers"]
        self.assertEqual(h["Authorization"], f"Bearer {KEY}")
        self.assertEqual(h["User-Agent"], "camus-gpt-eval/1.0")
        self.assertEqual(h["Content-Type"], "application/json")
        self.assertTrue(h["x-opencode-session"].startswith("ses_"), h)
        self.assertEqual(len(h["x-opencode-session"]), len("ses_") + 32)

    def test_request_shape_and_timeout(self):
        self.post.return_value = chat("{}")
        jz.judge("some-model", "SYSTEM", "USER")
        sent = self.post.call_args
        self.assertEqual(sent.kwargs["timeout"], 120)
        self.assertEqual(sent.kwargs["json"], {
            "model": "some-model",
            "messages": [{"role": "system", "content": "SYSTEM"},
                         {"role": "user", "content": "USER"}],
            "max_tokens": 4000, "temperature": 0})

    def test_max_tokens_leaves_room_for_a_reasoning_model_to_finish_its_answer(self):
        self.assertEqual(jz.MAX_TOKENS, 4000)

    def test_session_id_is_one_hex_id_reused_for_the_whole_process(self):
        self.post.return_value = chat("{}")
        jz.judge("m", "s", "u")
        jz.judge("m", "s", "u")
        seen = [c.kwargs["headers"]["x-opencode-session"] for c in self.post.call_args_list]
        self.assertEqual(seen, [jz.SESSION_ID, jz.SESSION_ID])
        self.assertEqual(len(jz.SESSION_ID), len("ses_") + 32)
        int(jz.SESSION_ID[len("ses_"):], 16)  # hex id, not an arbitrary string

    # ---------------------------------------------------------------------- key ----
    def test_key_from_env(self):
        self.write_auth()  # a different key on disk must not be used
        self.post.return_value = chat("{}")
        jz.judge("m", "s", "u", auth_path=self.absent)
        self.assertEqual(self.post.call_args.kwargs["headers"]["Authorization"],
                         f"Bearer {KEY}")

    def test_key_from_auth_json(self):
        self.set_env(OPENCODE_API_KEY="")
        self.write_auth()
        self.post.return_value = chat("{}")
        jz.judge("m", "s", "u", auth_path=self.auth)
        self.assertEqual(self.post.call_args.kwargs["headers"]["Authorization"],
                         f"Bearer {KEY}")

    def test_env_key_wins_over_auth_json(self):
        self.write_auth("file-key-must-lose")
        self.post.return_value = chat("{}")
        jz.judge("m", "s", "u", auth_path=self.auth)
        self.assertEqual(self.post.call_args.kwargs["headers"]["Authorization"],
                         f"Bearer {KEY}")

    def test_blank_env_key_falls_back_to_auth_json(self):
        self.set_env(OPENCODE_API_KEY="   ")
        self.write_auth()
        self.post.return_value = chat("{}")
        jz.judge("m", "s", "u", auth_path=self.auth)
        self.assertEqual(self.post.call_args.kwargs["headers"]["Authorization"],
                         f"Bearer {KEY}")

    def test_missing_key_everywhere_exits_and_names_the_env_var(self):
        self.set_env(OPENCODE_API_KEY="")
        with self.assertRaises(SystemExit) as cm:
            jz.load_key(self.absent)
        self.assertIn("OPENCODE_API_KEY", str(cm.exception))

    def test_auth_json_without_opencode_go_exits(self):
        self.set_env(OPENCODE_API_KEY="")
        self.write_auth(payload={"other-provider": {"key": KEY}})
        with self.assertRaises(SystemExit):
            jz.load_key(self.auth)

    def test_unreadable_auth_json_exits(self):
        self.set_env(OPENCODE_API_KEY="")
        with open(self.auth, "w", encoding="utf-8") as fh:
            fh.write("{not json")
        with self.assertRaises(SystemExit):
            jz.load_key(self.auth)

    # ------------------------------------------------------ key never in errors ----
    def test_key_never_appears_in_an_exception(self):
        failures = [FakeResponse(500, text=f"gateway rejected {KEY}"),
                    FakeResponse(429, text=f"capped, key {KEY}"),
                    requests.Timeout(f"timed out carrying {KEY}"),
                    requests.ConnectionError(f"refused with {KEY}")]
        for failure in failures:
            with self.subTest(failure=type(failure).__name__):
                self.post.side_effect = failure if isinstance(failure, Exception) else None
                self.post.return_value = None if isinstance(failure, Exception) else failure
                with self.assertRaises((RuntimeError, jz.UsageCapReached)) as cm:
                    jz.judge("m", "s", "u")
                self.assertNotIn(KEY, str(cm.exception))
                self.assertIn("[redacted]", str(cm.exception))

    def test_key_never_reaches_the_harness_judge_error_field(self):
        self.post.return_value = FakeResponse(503, text=f"upstream down, key {KEY}")
        row = run_like_harness(lambda: jz.judge("m", "s", "u"))
        self.assertNotIn(KEY, json.dumps(row))
        self.assertIn("[redacted]", row["judge_error"])

    # --------------------------------------------------------------- reply parse ----
    def test_fenced_json_reply_is_parsed_to_the_harness_schema(self):
        content = ('```json\n{"voice": 4, "factuality": 5, "engagement": 3,\n'
                   ' "rationale": "terse and correct"}\n```')
        self.post.return_value = chat(content)
        row = run_like_harness(judge_a_probe)
        self.assertEqual(row["scores"], {"voice": 4, "factuality": 5, "engagement": 3,
                                         "rationale": "terse and correct"})

    def test_bare_fence_without_a_language_tag(self):
        self.assertEqual(ec.parse_judge('```\n{"voice": 2, "factuality": 2, '
                                        '"engagement": 2}\n```')["voice"], 2)

    def test_first_object_wins_with_nested_braces_and_trailing_prose(self):
        got = ec.parse_judge(
            "Sure!\n{\"voice\": 3, \"factuality\": 9, \"engagement\": 0, "
            "\"rationale\": \"names {Cigarette} once\", \"detail\": {\"n\": 1}}\n"
            "Let me know if you want more.")
        self.assertEqual(got["voice"], 3)
        self.assertEqual(got["factuality"], 5)  # clamped to 1-5
        self.assertEqual(got["engagement"], 1)  # clamped to 1-5
        self.assertEqual(got["rationale"], "names {Cigarette} once")

    def test_reply_without_json_is_a_judge_error_not_a_score(self):
        self.post.return_value = chat("I'm afraid I can't do that.")
        row = run_like_harness(judge_a_probe)
        self.assertNotIn("scores", row)
        self.assertIn("no JSON", row["judge_error"])

    def test_reply_missing_a_score_is_a_judge_error_not_a_score(self):
        self.post.return_value = chat('{"voice": 3, "engagement": 3}')
        row = run_like_harness(judge_a_probe)
        self.assertNotIn("scores", row)
        self.assertTrue(row["judge_error"])

    # ------------------------------------------------ empty / truncated replies ----
    # The judge model reasons before it answers, so a reply can come back with nothing in it
    # or cut off mid-object. One identical retry, then the failure stands.
    def test_empty_reply_then_a_good_one_scores(self):
        self.post.side_effect = [chat(""), chat(VALID_JSON)]
        row = run_like_harness(judge_a_probe)
        self.assertEqual(row.get("judge_error"), None)
        self.assertEqual(row["scores"], VALID_SCORES)
        self.assertEqual(self.post.call_count, 2)

    def test_truncated_reply_then_a_good_one_scores(self):
        self.post.side_effect = [chat(CUT_OFF), chat(VALID_JSON)]
        row = run_like_harness(judge_a_probe)
        self.assertEqual(row.get("judge_error"), None)
        self.assertEqual(row["scores"], VALID_SCORES)
        self.assertEqual(self.post.call_count, 2)

    def test_null_content_beside_reasoning_is_treated_as_empty_and_retried(self):
        self.post.side_effect = [reasoning_only("Weighing voice against factuality..."),
                                 chat(VALID_JSON)]
        row = run_like_harness(judge_a_probe)
        self.assertEqual(row.get("judge_error"), None)
        self.assertEqual(row["scores"], VALID_SCORES)
        self.assertEqual(self.post.call_count, 2)

    def test_null_content_with_no_reasoning_beside_it_is_also_a_judge_error(self):
        self.post.side_effect = [reasoning_only("..."), reasoning_only("...")]
        row = run_like_harness(judge_a_probe)
        self.assertNotIn("scores", row)
        self.assertIn("no JSON", row["judge_error"])
        self.assertEqual(self.post.call_count, 2)

    def test_reply_missing_the_message_is_retried_then_a_judge_error(self):
        shapeless = FakeResponse(200, {"id": "gen-1", "object": "chat.completion", "created": 0})
        self.post.side_effect = [shapeless, chat(VALID_JSON)]
        self.assertEqual(run_like_harness(judge_a_probe)["scores"], VALID_SCORES)
        self.assertEqual(self.post.call_count, 2)
        self.post.reset_mock()
        self.post.side_effect = [shapeless, shapeless]
        row = run_like_harness(judge_a_probe)
        self.assertNotIn("scores", row)
        self.assertIn("chat/completions-shaped", row["judge_error"])
        self.assertEqual(self.post.call_count, 2)

    def test_the_retry_resends_the_identical_request(self):
        self.post.side_effect = [chat(CUT_OFF), chat(VALID_JSON)]
        run_like_harness(judge_a_probe)
        first, second = self.post.call_args_list
        self.assertEqual(second.args, first.args)
        self.assertEqual(second.kwargs["json"], first.kwargs["json"])
        self.assertEqual(second.kwargs["headers"], first.kwargs["headers"])
        self.assertEqual(second.kwargs["timeout"], first.kwargs["timeout"])

    def test_two_empty_replies_are_a_judge_error_not_a_score(self):
        self.post.side_effect = [chat(""), chat("")]
        row = run_like_harness(judge_a_probe)
        self.assertNotIn("scores", row)
        self.assertIn("no JSON", row["judge_error"])
        self.assertEqual(self.post.call_count, 2)  # one retry, not a loop

    def test_two_truncated_replies_are_a_judge_error_carrying_todays_message(self):
        self.post.side_effect = [chat(CUT_OFF), chat(CUT_OFF)]
        row = run_like_harness(judge_a_probe)
        self.assertNotIn("scores", row)
        self.assertIn("unterminated JSON", row["judge_error"])
        self.assertEqual(self.post.call_count, 2)

    def test_a_good_first_reply_is_never_retried(self):
        self.post.return_value = chat(VALID_JSON)
        row = run_like_harness(judge_a_probe)
        self.assertEqual(row["scores"], VALID_SCORES)
        self.assertEqual(self.post.call_count, 1)

    # --------------------------------------------------------------- usage cap ----
    def test_429_raises_usage_cap_not_a_judge_error(self):
        self.post.return_value = FakeResponse(429, text="usage limit reached")
        with self.assertRaises(jz.UsageCapReached) as cm:
            judge_a_probe()
        self.assertIn("429", str(cm.exception))
        self.assertNotIn(KEY, str(cm.exception))

    def test_429_is_not_retried_not_even_after_an_empty_reply(self):
        self.post.side_effect = [chat(""), FakeResponse(429, text="usage limit reached")]
        with self.assertRaises(jz.UsageCapReached):
            judge_a_probe()
        self.assertEqual(self.post.call_count, 2)  # the retry hit the cap; no third call

    def test_usage_cap_is_not_an_exception_so_a_harness_cannot_swallow_it(self):
        self.assertTrue(issubclass(jz.UsageCapReached, SystemExit))
        self.assertFalse(issubclass(jz.UsageCapReached, Exception))

    def test_429_stops_the_run_after_the_first_call(self):
        self.post.return_value = FakeResponse(429, text="usage limit reached")
        attempted, rows = [], []
        with self.assertRaises(jz.UsageCapReached):
            for i in range(3):
                attempted.append(i)
                rows.append(run_like_harness(judge_a_probe))
        self.assertEqual(attempted, [0])
        self.assertEqual(rows, [])       # nothing recorded as a score or a judge error
        self.assertEqual(self.post.call_count, 1)

    # ------------------------------------------------------- other judge errors ----
    def test_other_failures_become_judge_errors_and_the_run_continues(self):
        cases = {"http 500": FakeResponse(500, text="internal error"),
                 "http 401": FakeResponse(401, text="unauthorized"),
                 "timeout": requests.Timeout("timed out"),
                 "connection error": requests.ConnectionError("connection refused"),
                 "unparseable reply": chat("no json here"),
                 "reply without choices": FakeResponse(200, {"choices": []}),
                 "reply not json at all": FakeResponse(200, None, text="<html>502</html>")}
        for name, failure in cases.items():
            with self.subTest(case=name):
                self.post.side_effect = failure if isinstance(failure, Exception) else None
                self.post.return_value = None if isinstance(failure, Exception) else failure
                row = run_like_harness(judge_a_probe)
                self.assertNotIn("scores", row)
                self.assertTrue(row["judge_error"], name)


class DefaultsTest(unittest.TestCase):
    def test_opencode_is_the_default_backend(self):
        self.assertEqual(jz.DEFAULT_JUDGE, "opencode")
        self.assertEqual(jz.JUDGE_CHOICES, ("opencode", "anthropic", "ollama", "none"))
        args = jz.add_judge_args(argparse.ArgumentParser()).parse_args([])
        self.assertEqual(args.judge, "opencode")
        self.assertIsNone(args.judge_model)

    def test_judge_model_defaults_and_overrides(self):
        self.assertEqual(jz.DEFAULT_MODEL, "space-bunny-free")
        self.assertEqual(jz.resolve_model("opencode", None), "space-bunny-free")
        self.assertEqual(jz.resolve_model("opencode", "other-model"), "other-model")
        self.assertEqual(jz.resolve_model("anthropic", None), "claude-sonnet-4-6")
        self.assertIsNone(jz.resolve_model("none", None))
        with self.assertRaises(SystemExit):
            jz.resolve_model("ollama", None)


if __name__ == "__main__":
    unittest.main()