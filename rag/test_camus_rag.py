#!/usr/bin/env python3
"""
test_camus_rag.py — unit tests for the chat pipeline (rag/camus_rag.py and its callers).

No Ollama, network, GPU or knowledge base is touched: requests.post is patched and
sentence-transformers is never needed.

    python -m unittest discover -s rag -p "test_*.py"
"""
import ast
import contextlib
import importlib
import io
import json
import os
import sys
import types
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (HERE, os.path.join(ROOT, "pipeline")):
    if p not in sys.path:
        sys.path.insert(0, p)

# The eval scripts load .env at import; a stub keeps the tests independent of python-dotenv.
sys.modules.setdefault("dotenv", types.SimpleNamespace(load_dotenv=lambda *a, **k: None))

import camus_rag as cr  # noqa: E402


def load_variant(name):
    """Import a fresh copy of camus_rag with CORE_VARIANT set."""
    with mock.patch.dict(os.environ, {"CORE_VARIANT": name}):
        spec = importlib.util.spec_from_file_location(f"camus_rag_{name}", os.path.join(HERE, "camus_rag.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return mod


def import_eval_memory():
    """eval_memory loads the gitignored memory_fixtures.json at import; give it the template."""
    if "eval_memory" in sys.modules:
        return sys.modules["eval_memory"]
    import shutil, tempfile
    with tempfile.TemporaryDirectory() as d:
        shutil.copy(os.path.join(ROOT, "memory_fixtures.example.json"), os.path.join(d, "memory_fixtures.json"))
        old = os.getcwd()
        os.chdir(d)
        try:
            import eval_memory
        finally:
            os.chdir(old)
    return eval_memory


def fake_response(content="ok"):
    r = mock.Mock()
    r.json.return_value = {"message": {"content": content}}
    r.raise_for_status.return_value = None
    r.iter_lines.return_value = [json.dumps({"message": {"content": content}, "done": True}).encode()]
    return r


def hit(text, score, type_="fact", **kw):
    return dict(text=text, score=score, type=type_, **kw)


class GateTests(unittest.TestCase):
    def test_is_task(self):
        self.assertTrue(cr.is_task("Please analyze this letter I found."))
        self.assertTrue(cr.is_task("x" * 300))
        self.assertFalse(cr.is_task("Where were you born?"))
        self.assertFalse(cr.is_task("What was your cat called?"))

    def test_is_task_keeps_biographical_questions(self):
        for q in ("What can you tell me about Louis Germain?",
                  "What can you tell me about your mother?",
                  "Who is the author of The Plague?",
                  "What do you make of your time at Combat?"):
            with self.subTest(q=q):
                self.assertFalse(cr.is_task(q))
        for q in ("What can you tell from this letter?", "What do you make of this poem?",
                  "Guess the author of this passage."):
            with self.subTest(q=q):
                self.assertTrue(cr.is_task(q))

    def test_is_followup(self):
        self.assertTrue(cr.is_followup("and why?"))
        self.assertTrue(cr.is_followup(""))
        self.assertTrue(cr.is_followup("What about the dogs, and how they behaved around the house?"))
        self.assertFalse(cr.is_followup("Tell me about the village where you grew up as a boy."))

    def test_is_self_ref(self):
        self.assertTrue(cr.is_self_ref("What was I working on again?"))
        self.assertTrue(cr.is_self_ref("Where did we leave off?"))
        self.assertTrue(cr.is_self_ref("I'm stuck again, same as last time."))
        self.assertFalse(cr.is_self_ref("Where were you born?"))
        self.assertFalse(cr.is_self_ref("I like your novels."))
        self.assertFalse(cr.is_self_ref(""))
        self.assertFalse(cr.is_self_ref(None))


class BuildSystemTests(unittest.TestCase):
    def test_facts_shown_when_not_suppressed(self):
        s = cr.build_system([hit("UNIQUEFACT he kept a cat", 0.9)])
        self.assertIn("UNIQUEFACT", s)
        self.assertIn("Facts about your life:", s)

    def test_facts_absent_when_suppressed(self):
        # Suppression happens upstream (hits emptied); the flag must also avoid the
        # "no memory matched" disclaimer.
        s = cr.build_system([], kb_suppressed=True)
        self.assertEqual(s, cr.CORE)
        self.assertNotIn("No memory closely matched", s)

    def test_no_hits_adds_disclaimer(self):
        self.assertIn("No memory closely matched", cr.build_system([]))

    def test_confidence_tiers(self):
        self.assertIn("Possibly-related", cr.build_system([hit("f", 0.63)]))
        self.assertIn("probably NOT relevant", cr.build_system([hit("f", 0.56)]))

    def test_views_with_quote(self):
        s = cr.build_system([hit("a view", 0.9, "view", quote="a quote")])
        self.assertIn('(your words: "a quote")', s)

    def test_identity_card_always_present(self):
        for hits, sup in (([], False), ([], True), ([hit("f", 0.9)], False), ([hit("f", 0.56)], False)):
            self.assertIn(cr._IDENTITY_CARD, cr.build_system(hits, sup))

    def test_core_variants_share_identity_card(self):
        cores = {}
        for name in ("current", "lean", "persona"):
            mod = load_variant(name)
            self.assertEqual(mod._IDENTITY_CARD, cr._IDENTITY_CARD)
            self.assertTrue(mod.CORE.endswith(mod._IDENTITY_CARD))
            cores[name] = mod.CORE
        prose = {k: v[: -len(cr._IDENTITY_CARD)] for k, v in cores.items()}
        self.assertEqual(len(set(prose.values())), 3)

    def test_unknown_variant_exits(self):
        with self.assertRaises(SystemExit):
            load_variant("nonsense")


class SpaceSyncTests(unittest.TestCase):
    def test_space_identity_card_matches(self):
        with open(os.path.join(ROOT, "space", "app.py"), encoding="utf-8") as f:
            tree = ast.parse(f.read())
        core = None
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "CORE" for t in node.targets):
                core = ast.literal_eval(node.value)
        self.assertIsNotNone(core, "space/app.py has no literal CORE")
        self.assertTrue(core.endswith(cr._IDENTITY_CARD),
                        "space/app.py identity card has drifted from rag/camus_rag.py")


class ThinkFalseTests(unittest.TestCase):
    """Every /api/chat payload must carry "think": False (Gemma otherwise returns empty content)."""

    def assertThinkFalse(self, post):
        self.assertTrue(post.called, "requests.post was never called")
        for call in post.call_args_list:
            body = call.kwargs.get("json")
            self.assertIsNotNone(body)
            self.assertIs(body.get("think"), False, f"missing think:false in {call}")

    def test_stream_chat(self):
        with mock.patch.object(cr.requests, "post", return_value=fake_response("hi")) as post, \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cr.stream_chat([{"role": "user", "content": "x"}]), "hi")
        self.assertThinkFalse(post)

    def test_util_chat(self):
        with mock.patch.object(cr, "UTIL_MODEL", "some-model"), \
             mock.patch.object(cr.requests, "post", return_value=fake_response("sum")) as post:
            cr.util_chat([{"role": "user", "content": "x"}])
            cr.util_chat([{"role": "user", "content": "x"}], fmt="json", system="s")
        self.assertThinkFalse(post)

    def test_eval_camus_ask(self):
        import eval_camus as ec
        turn = types.SimpleNamespace(messages=[], opts={}, hits=[])
        with mock.patch.object(ec.cr, "build_turn", return_value=turn), \
             mock.patch.object(ec.requests, "post", return_value=fake_response("a")) as post:
            ec.ask.kb, ec.ask.bm25, ec.ask.ce = ([], None), None, None
            ec.ask(["one", "two"])
        self.assertEqual(post.call_count, 2)
        self.assertThinkFalse(post)

    def test_eval_memory_run_probe(self):
        em = import_eval_memory()
        fake_cr = types.SimpleNamespace(
            OLLAMA="http://x", GEN_MODEL="m", embed=None, MemoryCtx=lambda **k: None,
            build_turn=lambda *a, **k: types.SimpleNamespace(messages=[], opts={}, memories=[]))
        fake_mem = types.SimpleNamespace(MemoryStore=lambda *a: None, load_profile=lambda p: "")
        em.run_probe.facts = em.run_probe.vecs = em.run_probe.bm25 = em.run_probe.ce = None
        probe = dict(id="p", family="f", seed="none", turns=["hello"])
        with mock.patch.dict(sys.modules, {"memory": fake_mem}), \
             mock.patch.object(em, "seed_store", return_value=("a", "b", "c", "d")), \
             mock.patch.object(em.requests, "post", return_value=fake_response("a")) as post:
            row = em.run_probe(probe, "tmp", fake_cr)
        self.assertEqual(row["answer"], "a")
        self.assertThinkFalse(post)

    def test_camus_client_generate(self):
        import camus_client as cc
        client = cc.CamusClient.__new__(cc.CamusClient)
        with mock.patch.object(cc.requests, "post", return_value=fake_response("a")) as post:
            self.assertEqual(client._generate([{"role": "user", "content": "x"}], {}), "a")
        self.assertThinkFalse(post)

    def test_temp_sweep_ask(self):
        import temp_sweep as ts
        fake = types.SimpleNamespace(OLLAMA="http://x", GEN_MODEL="m")
        with mock.patch("requests.post", return_value=fake_response("a")) as post:
            self.assertEqual(ts.ask(fake, [], {"temperature": 1}, 0.3), "a")
        self.assertThinkFalse(post)
        self.assertEqual(post.call_args.kwargs["json"]["options"]["temperature"], 0.3)

    def test_eval_camus_judge_ollama(self):
        import eval_camus as ec
        with mock.patch.object(ec.requests, "post", return_value=fake_response("{}")) as post, \
             mock.patch.object(ec, "parse_judge", side_effect=lambda text: text), \
             mock.patch.object(ec, "judge_prompt", return_value="p"):
            ec.judge_ollama("judge", {}, [], "a")
        self.assertThinkFalse(post)

    def test_eval_memory_judge_ollama(self):
        em = import_eval_memory()
        fake_cr = types.SimpleNamespace(OLLAMA="http://x")
        with mock.patch.object(em.requests, "post", return_value=fake_response("{}")) as post, \
             mock.patch.object(em, "parse_judge", side_effect=lambda text: text), \
             mock.patch.object(em, "judge_prompt", return_value="p"):
            em.judge_ollama("judge", {}, [], "a", cr=fake_cr)
        self.assertThinkFalse(post)


class HistoryTests(unittest.TestCase):
    def test_recent_user(self):
        h = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"},
             {"role": "user", "content": "c"}, {"role": "assistant", "content": "d"}]
        self.assertEqual(cr.recent_user(h), "c")
        self.assertEqual(cr.recent_user(h, 2), "a c")
        self.assertEqual(cr.recent_user([]), "")
        self.assertEqual(cr.recent_user([{"role": "assistant", "content": "z"}]), "")

    def test_history_window_trimming(self):
        history = []
        for i in range(40):
            history += [{"role": "user", "content": f"u{i}"}, {"role": "assistant", "content": f"a{i}"}]
        with mock.patch.object(cr, "retrieve", return_value=[]):
            turn = cr.build_turn("Where were you born?", history, [], None)
        msgs = turn.messages
        self.assertEqual(msgs[0]["role"], "system")
        self.assertEqual(len(msgs), 1 + cr.HIST_WINDOW + 1)
        self.assertEqual(msgs[1], history[-cr.HIST_WINDOW])
        self.assertEqual(msgs[-2], history[-1])
        self.assertEqual(msgs[-1], {"role": "user", "content": "Where were you born?"})

    def test_short_history_untrimmed_and_temperature(self):
        with mock.patch.object(cr, "retrieve", return_value=[]):
            t = cr.build_turn("Where were you born?", [], [], None)
            task = cr.build_turn("Please analyze this letter.", [], [], None)
        self.assertEqual(len(t.messages), 2)
        self.assertEqual(t.opts["temperature"], cr.TEMP_FACTUAL)
        self.assertEqual(task.opts["temperature"], cr.TEMP_TASK)
        self.assertEqual(task.hits, [])

    def test_followup_folds_previous_user_turn(self):
        history = [{"role": "user", "content": "Tell me about Cigarette"},
                   {"role": "assistant", "content": "A cat."}]
        with mock.patch.object(cr, "retrieve", return_value=[]) as ret:
            cr.build_turn("and the dogs?", history, [], None)
        self.assertEqual(ret.call_args.args[0], "Tell me about Cigarette and the dogs?")


if __name__ == "__main__":
    unittest.main()
