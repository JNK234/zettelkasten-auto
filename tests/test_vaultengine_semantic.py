"""The experimental suggestion ranking (vaultengine/experimental/semantic.py) and `vault suggest`.

Lexical results are pinned (they need nothing extra). Semantic ones run only when
WordLlama is installed, and check behaviour, not model-specific numbers.
"""

import json
import subprocess
import sys
import unittest
from pathlib import Path

from vaultengine import Vault
from vaultengine.experimental.paraphrases import PARAPHRASES
from vaultengine.experimental.semantic import (
    Ranker, evaluate, evaluate_paraphrases, related_to_text, semantic_available, suggest_sources, suggest_targets,
)
from vaultengine.formats import TypedVault, load_format

TESTS = Path(__file__).parent
REPO_ROOT = TESTS.parent
VAULT = TESTS / "fixtures" / "vault"
BRAIN = load_format(REPO_ROOT / "formats" / "brain.yaml")
D = "50_MOC/discourse/rl-environment-design"
EVD = f"{D}/EVD - Reward is a deterministic outcome check 1 only if the final database equals the data.md"


class LexicalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.typed = TypedVault(BRAIN, Vault.load(VAULT))
        cls.ranker = Ranker(cls.typed)

    def test_evaluation_on_the_sample_vault(self):
        results = evaluate(self.typed, methods=("lexical",))
        self.assertEqual(results["A_source_of_note"], {"queries": 33, "candidates": 28, "lexical": {"top1": 33, "top3": 33}})
        self.assertEqual(results["B_claim_of_evidence"], {"queries": 26, "candidates": 8, "lexical": {"top1": 16, "top3": 24}})
        self.assertEqual(evaluate_paraphrases(self.typed, methods=("lexical",)), {"queries": 8, "lexical": {"top1": 7, "top3": 8}})

    def test_text_leaves_out_links_and_metadata(self):
        text = self.ranker.text(EVD)
        self.assertNotIn("[[", text)
        self.assertNotIn("Strength: B", text)
        self.assertIn("deterministic outcome check", text)

    def test_targets_are_of_the_relation_s_types_and_not_linked_yet(self):
        suggestions = suggest_targets(self.ranker, EVD, "supports", limit=10)
        paths = [s.path for s in suggestions]
        self.assertTrue(paths and all(self.typed.type_of(p) == "claim" for p in paths))
        already = {e.target_path for e in self.typed.relations(EVD) if e.relation == "supports"}
        self.assertFalse(already & set(paths))
        self.assertEqual([s.rank for s in suggestions], list(range(1, len(suggestions) + 1)))
        self.assertEqual(paths[0], f"{D}/CLM - State graders should check what must not change as well as what must.md")

    def test_sources_not_cited_yet(self):
        note = "20_ZETTLEKASTEN/cot-monitor-beats-action-monitor.md"
        cited = {l.edge.target_path for l in self.typed.links(note) if l.edge.target_path}
        suggestions = suggest_sources(self.ranker, note, limit=28)
        self.assertTrue(all(self.typed.type_of(s.path) == "source" for s in suggestions))
        self.assertFalse(cited & {s.path for s in suggestions})

    def test_a_paraphrase_finds_its_paper_where_the_duplicate_gate_sees_nothing(self):
        from vaultengine.formats import similarity

        truth = "10_SOURCES/2411.00640 - Adding Error Bars to Evals.md"
        text = PARAPHRASES[truth]
        self.assertEqual(similarity.content_overlap(similarity.shingles(text), similarity.shingles(self.ranker.text(truth))).score, 0.0)
        self.assertEqual(related_to_text(self.ranker, text, "source")[0].path, truth)

    def test_unknown_relation(self):
        with self.assertRaises(ValueError):
            suggest_targets(self.ranker, EVD, "endorses")


@unittest.skipUnless(semantic_available(), "WordLlama is not installed (pip install wordllama)")
class SemanticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.typed = TypedVault(BRAIN, Vault.load(VAULT))
        cls.ranker = Ranker(cls.typed)

    def test_semantic_and_fused_rank_the_same_candidates(self):
        for method in ("semantic", "fused"):
            with self.subTest(method=method):
                suggestions = suggest_targets(self.ranker, EVD, "supports", method=method, limit=8)
                self.assertEqual(len(suggestions), len(suggest_targets(self.ranker, EVD, "supports", limit=8)))
                self.assertTrue(all(s.semantic is not None and -1.0 <= s.semantic <= 1.0 for s in suggestions))
        fused = suggest_targets(self.ranker, EVD, "supports", method="fused", limit=1)[0]
        self.assertIsNotNone(fused.lexical)

    def test_the_model_loads_offline(self):
        import socket
        from unittest import mock

        from vaultengine.experimental import semantic

        semantic._MODEL = None
        with mock.patch.object(socket, "create_connection", side_effect=OSError("no network in this test")):
            self.assertIsNotNone(semantic.load_model())

    def test_evaluation_runs_for_every_method(self):
        results = evaluate(self.typed)
        for task in ("A_source_of_note", "B_claim_of_evidence"):
            for method in ("lexical", "semantic", "fused"):
                self.assertLessEqual(results[task][method]["top1"], results[task][method]["top3"])


class SuggestCliTests(unittest.TestCase):
    def run_vault(self, *args):
        return subprocess.run([sys.executable, "-m", "vaultengine", *args], capture_output=True, text=True, cwd=REPO_ROOT)

    def test_suggest(self):
        result = self.run_vault("suggest", str(VAULT), "-f", "brain", "--note", EVD, "--as", "supports", "--json")
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertEqual((data["method"], len(data["suggestions"])), ("lexical", 3))
        text = self.run_vault("suggest", str(VAULT), "-f", "brain", "--note", EVD, "--as", "supports").stdout
        self.assertIn("to link the first: vault link", text)
        self.assertEqual(self.run_vault("suggest", str(VAULT), "-f", "brain").returncode, 2)

    def test_coverage_suggest(self):
        data = json.loads(self.run_vault("coverage", str(VAULT), "-f", "brain", "--suggest", "--json").stdout)
        suggested = [n["suggested_source"] for n in data["citing"]["notes"]]
        self.assertEqual(len(suggested), 110)
        self.assertTrue(all(s and s["path"].startswith("10_SOURCES/") for s in suggested))


if __name__ == "__main__":
    unittest.main()
