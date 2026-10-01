"""Static checks on the v3 (Gemma 4) notebooks.

These are the mistakes that cannot be caught by reading the diff, because they are
silent: a wrong base model id, a chat template applied without thinking disabled, a
response mask that spans the prompt instead of the reply. None of them raise at import
time — they just quietly train the wrong thing. A notebook is a script that only runs
on Colab, so the checks that CAN run locally run here instead.

Run:
    python3 -m unittest discover -s training -p "test_v3_notebooks.py"
    python3 training/test_v3_notebooks.py
"""

import json
import os
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))

NOTEBOOKS = [
    "CamusGPT_v3_Phase1_VoiceSFT_Gemma4.ipynb",
    "CamusGPT_v3_Phase2_SFT_Gemma4.ipynb",
    "CamusGPT_v3_Export_GGUF_Gemma4.ipynb",
]

# The older architecture's chat markup and base-model id. None may appear anywhere in
# a v3 notebook: a single leftover "<start_of_turn>" is enough to mis-mask the loss.
FORBIDDEN = ["<start_of_turn>", "<end_of_turn>", "gemma-3"]

# Gemma 4 turned these on: the base id, the loader its model card uses, thinking
# disabled at render time, and the model-turn marker the mask is anchored on.
# The turn markers are raw strings: what a notebook has to contain is the literal
# source text "<|turn>model\n" (backslash + n), not a real newline.
REQUIRED = [
    "gemma-4-12b-it",
    "FastModel",
    "enable_thinking=False",
    r"<|turn>model\n",
]


def load(name):
    with open(os.path.join(HERE, name), encoding="utf-8") as f:
        return json.load(f)


def cell_text(nb):
    return "".join("".join(c.get("source", [])) for c in nb["cells"])


class TestV3Notebooks(unittest.TestCase):
    def test_every_notebook_exists(self):
        for name in NOTEBOOKS:
            with self.subTest(notebook=name):
                self.assertTrue(os.path.exists(os.path.join(HERE, name)),
                                f"{name} is missing")

    def test_parses_as_json(self):
        for name in NOTEBOOKS:
            with self.subTest(notebook=name):
                nb = load(name)
                self.assertIsInstance(nb, dict)
                self.assertEqual(nb.get("nbformat"), 4)
                self.assertIn("cells", nb)
                self.assertTrue(nb["cells"], f"{name} has no cells")

    def test_outputs_and_execution_counts_cleared(self):
        """A committed notebook with stale outputs misleads the next reader: they look
        like a successful run of code that has since changed."""
        for name in NOTEBOOKS:
            nb = load(name)
            with self.subTest(notebook=name):
                for i, cell in enumerate(nb["cells"]):
                    if cell["cell_type"] == "code":
                        self.assertEqual(cell.get("outputs", []), [],
                                         f"{name} cell {i} still has outputs")
                        self.assertIsNone(cell.get("execution_count"),
                                          f"{name} cell {i} still has an execution_count")
                    else:
                        self.assertNotIn("outputs", cell,
                                         f"{name} cell {i} is not code but has outputs")

    def test_required_gemma4_markers_present(self):
        for name in NOTEBOOKS:
            text = cell_text(load(name))
            for needle in REQUIRED:
                with self.subTest(notebook=name, needle=needle):
                    self.assertIn(needle, text,
                                  f"{name} does not mention {needle!r}")

    def test_no_legacy_chat_markup(self):
        for name in NOTEBOOKS:
            text = cell_text(load(name))
            for needle in FORBIDDEN:
                with self.subTest(notebook=name, needle=needle):
                    self.assertNotIn(needle, text,
                                     f"{name} still mentions {needle!r}")

    def test_end_of_turn_token_retargeted(self):
        """The mask is anchored on the model-turn marker; the generation stop token is
        the separate '<turn|>'. If one is renamed, the other must be too."""
        for name in NOTEBOOKS:
            text = cell_text(load(name))
            with self.subTest(notebook=name):
                self.assertIn("<turn|>", text, f"{name} does not use the Gemma 4 EOT token")

    def test_tokenizer_unwrap_kept(self):
        """Unsloth may return a wrapper; the plain tokenizer underneath is the one whose
        decode()/convert_tokens_to_ids() speak Gemma 4 markup."""
        for name in NOTEBOOKS:
            text = cell_text(load(name))
            with self.subTest(notebook=name):
                if "convert_tokens_to_ids" in text or "decode(" in text:
                    self.assertIn('getattr(tokenizer, "tokenizer", tokenizer)', text,
                                  f"{name} uses the tokenizer without unwrapping it")

    def test_training_notebooks_probe_token_type_ids(self):
        """Gemma 3 needed zero-filled token_type_ids; that was architecture-specific.
        The wrapper must be conditional, not unconditional."""
        for name in NOTEBOOKS[:2]:
            text = cell_text(load(name))
            with self.subTest(notebook=name):
                self.assertIn("token_type_ids", text)
                self.assertIn("TOKEN_TYPE_IDS_PATH", text,
                              f"{name} does not report which token_type_ids path it took")
                self.assertIn('if "token_type_ids" not in str(e)', text,
                              f"{name} installs the wrapper without checking the error")

    def test_training_notebooks_keep_trainer_settings(self):
        """TRL/accelerate concerns, not architecture ones: do not lose them in a port."""
        for name in NOTEBOOKS[:2]:
            text = cell_text(load(name))
            for needle in ("average_tokens_across_devices = False",
                           "remove_unused_columns",
                           "train_on_responses_only"):
                with self.subTest(notebook=name, needle=needle):
                    self.assertIn(needle, text, f"{name} lost {needle!r}")

    def test_training_notebooks_check_thought_channel(self):
        for name in NOTEBOOKS[:2]:
            text = cell_text(load(name))
            with self.subTest(notebook=name):
                self.assertIn("assert_no_thoughts", text,
                              f"{name} never checks for a thought channel")
                self.assertIn("<|channel>thought", text)

    def test_training_notebooks_read_the_shared_corpus(self):
        """The per-corpus loading is gone; the two runs are selected by one variable."""
        p1 = cell_text(load(NOTEBOOKS[0]))
        with self.subTest(notebook=NOTEBOOKS[0]):
            self.assertIn('PHASE1_CORPUS = "new"', p1)
            self.assertIn("phase1_{PHASE1_CORPUS}.jsonl", p1)
            self.assertIn("data_v3", p1)
            # the adapter folder name has to carry the corpus, or the two runs collide
            self.assertIn("camus3_voice_lora_{PHASE1_CORPUS}", p1)
            for old in ("camus_sft.jsonl", "camus_conversational.jsonl"):
                self.assertNotIn(old, p1, f"{NOTEBOOKS[0]} still loads {old} directly")

        p2 = cell_text(load(NOTEBOOKS[1]))
        with self.subTest(notebook=NOTEBOOKS[1]):
            self.assertIn("data_v3/phase2.jsonl", p2)
            for old in ("camus_refusals.jsonl", "camus_epistemic.jsonl",
                        "camus_analysis.jsonl", "camus_multiturn.jsonl",
                        "camus_phase3.jsonl"):
                self.assertNotIn(old, p2, f"{NOTEBOOKS[1]} still loads {old} directly")

    def test_training_notebooks_keep_system_role_separate(self):
        """Gemma 4 has a native system role. A system message must not be pasted onto
        the front of the first user turn any more."""
        for name in NOTEBOOKS[:2]:
            text = cell_text(load(name))
            with self.subTest(notebook=name):
                self.assertIn("<|turn>system", text,
                              f"{name} never renders a system turn")

    def test_adapter_is_saved_before_the_behavioural_gate(self):
        """The gate is a manual read. Persisting first means a run that fails the gate
        still leaves a diagnosable adapter instead of a discarded one."""
        for name in NOTEBOOKS[:2]:
            nb = load(name)
            save_at, gate_at = None, None
            for i, cell in enumerate(nb["cells"]):
                if cell["cell_type"] != "code":
                    continue
                text = "".join(cell["source"])
                if save_at is None and "save_pretrained(" in text and "merged" not in text:
                    save_at = i
                if gate_at is None and ("CHECK 3" in text or "behavioural GATE" in text):
                    gate_at = i
            with self.subTest(notebook=name):
                self.assertIsNotNone(save_at, f"{name} never persists the adapter")
                self.assertIsNotNone(gate_at, f"{name} has no behavioural gate")
                self.assertLess(save_at, gate_at,
                                f"{name} runs its behavioural gate before saving")

    def test_notebooks_report_wall_clock_and_point_at_the_roadmap(self):
        """No Gemma-3 run time is recorded anywhere in this repo, which is why the time
        estimate in docs/gemma4_assessment.md is a range. Do not repeat that."""
        for name in NOTEBOOKS:
            text = cell_text(load(name))
            with self.subTest(notebook=name):
                self.assertIn("T0 = time.time()", text)
                self.assertIn("time.time() - T0", text)
                self.assertIn("docs/ROADMAP.md", text)

    def test_export_notebook_uses_the_gguf4_ggml_path(self):
        text = cell_text(load(NOTEBOOKS[2]))
        for needle in ("convert_hf_to_gguf.py", "llama-quantize", "q4_k_m",
                       "camus3-12b.Q4_K_M.gguf", "RENDERER gemma4", "PARSER gemma4"):
            with self.subTest(needle=needle):
                self.assertIn(needle, text, f"export notebook lost {needle!r}")
        self.assertIn("mmproj", text,
                      "export notebook does not mention the projector it does not need")


if __name__ == "__main__":
    unittest.main(verbosity=2)
