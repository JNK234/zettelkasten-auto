"""Related notes: personalized PageRank against an exact solution, the link graph, signals, reasons, CLI."""

import io
from datetime import datetime
import json
import os
import random
import unittest
from contextlib import redirect_stderr, redirect_stdout
from fractions import Fraction
from unittest.mock import patch

from vaultengine.cli import main as cli_main
from vaultengine.graphrank import ALPHA, build_graph, component, hop_paths, personalized_pagerank
from vaultengine.index import VaultIndex, default_db_path
from vaultengine.querylang import QueryError
from vaultengine.related import WEIGHTS, Relatedness

from test_vaultengine_search import TempVault

EXACT_ALPHA = Fraction(3, 20)
assert float(EXACT_ALPHA) == ALPHA


def exact_ppr(adjacency, seed):
    """Solve (I - (1 - a) M) r = a e_s exactly, with fractions, by Gaussian elimination.

    M[j][i] = w_ij / W_i; a node without edges sends its mass back to the seed.
    Only the seed's component, as in personalized_pagerank (other nodes get 0 anyway).
    """
    nodes = component(adjacency, seed)
    index = {node: k for k, node in enumerate(nodes)}
    n = len(nodes)
    a = EXACT_ALPHA
    matrix = [[Fraction(int(i == j)) for j in range(n)] + [a if i == index[seed] else Fraction(0)] for i in range(n)]
    for node in nodes:
        i = index[node]
        total = sum(Fraction(w) for w in adjacency[node].values())
        if total == 0:
            matrix[index[seed]][i] -= 1 - a
            continue
        for other, weight in adjacency[node].items():
            matrix[index[other]][i] -= (1 - a) * Fraction(weight) / total
    for col in range(n):
        pivot = next(r for r in range(col, n) if matrix[r][col] != 0)
        matrix[col], matrix[pivot] = matrix[pivot], matrix[col]
        lead = matrix[col][col]
        matrix[col] = [x / lead for x in matrix[col]]
        for r in range(n):
            if r != col and matrix[r][col] != 0:
                factor = matrix[r][col]
                matrix[r] = [x - factor * y for x, y in zip(matrix[r], matrix[col])]
    return {node: matrix[index[node]][n] for node in nodes}


def undirected(edges, n):
    adjacency = [dict() for _ in range(n)]
    for a, b, w in edges:
        adjacency[a][b] = adjacency[a].get(b, 0) + w
        adjacency[b][a] = adjacency[b].get(a, 0) + w
    return adjacency


# 0-1-2 triangle with a double edge 0-1, a pendant 3 on 2, a bridge 2-4, 4-5;
# a separate pair 6-7; 8 has no edges.
SMALL = undirected([(0, 1, 2), (1, 2, 1), (0, 2, 1), (2, 3, 1), (2, 4, 1), (4, 5, 3), (6, 7, 1)], 9)


class ExactPageRankTests(unittest.TestCase):
    def assert_matches_exact(self, adjacency, seed):
        exact = exact_ppr(adjacency, seed)
        approx = personalized_pagerank(adjacency, seed, tol=1e-13)
        self.assertEqual(set(approx), set(exact))
        for node, value in exact.items():
            self.assertAlmostEqual(approx[node], float(value), delta=1e-9, msg=f"seed {seed}, node {node}")
        self.assertEqual(sum(exact.values()), 1)

    def test_small_graph_every_seed(self):
        for seed in range(9):
            with self.subTest(seed=seed):
                self.assert_matches_exact(SMALL, seed)

    def test_exact_values_by_hand_for_a_single_edge(self):
        # Two nodes, one edge: r0 = a + (1-a) r1 and r1 = (1-a) r0, so r0 = 1/(2-a) = 20/37.
        pair = undirected([(0, 1, 1)], 2)
        self.assertEqual(exact_ppr(pair, 0), {0: Fraction(20, 37), 1: Fraction(17, 37)})
        self.assertAlmostEqual(personalized_pagerank(pair, 0)[0], 20 / 37, delta=1e-9)

    def test_isolated_seed_keeps_all_mass(self):
        self.assertEqual(personalized_pagerank(SMALL, 8), {8: 1.0})
        self.assertEqual(exact_ppr(SMALL, 8), {8: 1})

    def test_random_weighted_graphs(self):
        rng = random.Random(4)
        for trial in range(40):
            n = rng.randint(2, 12)
            edges = [(a, b, rng.randint(1, 5)) for a in range(n) for b in range(a + 1, n) if rng.random() < 0.3]
            adjacency = undirected(edges, n)
            seed = rng.randrange(n)
            with self.subTest(trial=trial):
                self.assert_matches_exact(adjacency, seed)

    def test_degree_normalised_scores_are_symmetric(self):
        """r_s(i) / d_i = r_i(s) / d_s on an undirected graph (why related.py divides by degree)."""
        degree = [sum(Fraction(w) for w in row.values()) for row in SMALL]
        for s in range(6):
            exact_s = exact_ppr(SMALL, s)
            for i in range(6):
                if i != s:
                    self.assertEqual(exact_s[i] / degree[i], exact_ppr(SMALL, i)[s] / degree[s])


# ----- the graph of a vault --------------------------------------------------------

GRAPH_NOTES = {
    "A.md": "[[B]] [[B]] [[Missing]] [[A]] ![[pic.png]] [[Dup]]",
    "B.md": "[[C]] [[Missing]]",
    "C.md": "back to [[A]]",
    "x/Dup.md": "one",
    "y/Dup.md": "two",
    "Alone.md": "no links",
}


class VaultGraphTests(unittest.TestCase):
    def setUp(self):
        self.vault = TempVault(GRAPH_NOTES)
        (self.vault.root / "pic.png").write_bytes(b"\x89PNG")
        self.ix = VaultIndex(default_db_path(self.vault.root))
        self.ix.update(self.vault.root)
        self.graph = build_graph(self.ix.conn)

    def tearDown(self):
        self.ix.close()
        self.vault.cleanup()

    def node(self, label):
        return self.graph.labels.index(label)

    def test_nodes_and_weights(self):
        g = self.graph
        a, b, c, m = self.node("A.md"), self.node("B.md"), self.node("C.md"), self.node("Missing")
        self.assertTrue(g.is_missing(m))
        self.assertEqual(g.adjacency[a], {b: 2.0, m: 1.0, c: 1.0})  # [[B]] twice; no self-link, attachment or ambiguous [[Dup]]
        self.assertEqual(g.adjacency[m], {a: 1.0, b: 1.0})
        self.assertEqual(g.adjacency[self.node("Alone.md")], {})
        self.assertEqual(g.direct, {(a, b), (b, c), (c, a)})

    def test_pagerank_on_the_built_graph_matches_exact(self):
        for label in ("A.md", "Missing", "Alone.md"):
            exact = exact_ppr(self.graph.adjacency, self.node(label))
            approx = personalized_pagerank(self.graph.adjacency, self.node(label), tol=1e-13)
            for node, value in exact.items():
                self.assertAlmostEqual(approx[node], float(value), delta=1e-9)

    def test_hop_paths(self):
        paths = hop_paths(self.graph, self.node("C.md"))
        self.assertEqual(paths[self.node("A.md")].hops, 1)
        two = paths[self.node("Missing")]
        self.assertEqual((two.hops, sorted(self.graph.labels[v] for v in two.via)), (2, ["A.md", "B.md"]))


# ----- related notes -----------------------------------------------------------------
#
# Seed links [[Linked]] twice and the missing [[topic-x]]; Linked links [[Two hops]];
# Same topic also links [[topic-x]]. Similar text shares the seed's words but no links.
# #baby is on 6 of 10 notes (too common to count); #rl is on 3. "Related zettels" is a
# heading in 3 notes (a template).

RELATED_NOTES = {
    "Seed.md": "Status: #baby\n\n# Seed\nThe kl penalty keeps the policy close to the reference model. See [[Linked]] and [[topic-x]]. #rl\n\n## Related zettels\n- [[Linked]]\n",
    "Linked.md": "Status: #baby\n\nKl penalty tuning for ppo. Links [[Two hops]].\n\n## Related zettels\n- none\n",
    "Two hops.md": "Status: #baby\n\nPpo clipping and trust regions. #rl\n\n## Related zettels\n- none\n",
    "Same topic.md": "Status: #baby\n\nAnother note on [[topic-x]] about gardening.\n",
    "Similar text.md": "Status: #baby\n\nA kl penalty on the policy keeps it close to the reference model.\n",
    "Tagged.md": "Tags: #rl\n\nNothing in common here, bananas.\n",
    "Far.md": "Status: #baby\n\nUnrelated bananas and apples.\n",
    "Other.md": "Bananas apples gardening.\n",
    "Filler 1.md": "Apples only.\n",
    "Filler 2.md": "Apples again.\n",
}


class RelatedVaultCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vault = TempVault(RELATED_NOTES)
        cls.ix = VaultIndex(default_db_path(cls.vault.root))
        cls.ix.update(cls.vault.root)
        cls.engine = Relatedness(cls.ix.conn)

    @classmethod
    def tearDownClass(cls):
        cls.ix.close()
        cls.vault.cleanup()


class RelatedTests(RelatedVaultCase):
    def hits(self, ref, **kwargs):
        engine = self.engine
        return {hit.title: hit for hit in engine.related(engine.seed(ref), limit=None, **kwargs).hits}

    def test_reasons(self):
        hits = self.hits("Seed")
        self.assertNotIn("Seed", hits)
        self.assertEqual(hits["Linked"].reason.split(" · ")[0], "linked from it")
        self.assertTrue(hits["Linked"].linked)
        self.assertEqual(hits["Two hops"].reason, "2 hops via Linked · shares #rl")
        self.assertEqual(hits["Same topic"].reason, "2 hops via topic-x (no note yet)")
        self.assertFalse(hits["Similar text"].linked)
        self.assertTrue(hits["Similar text"].reason.startswith("similar text ("), hits["Similar text"].reason)
        self.assertEqual(hits["Similar text"].text, 1.0)  # the closest text of all
        words = lambda text: set(text.lower().replace(".", " ").split())
        for term in hits["Similar text"].shared_terms:  # the terms named are in both notes
            self.assertIn(term, words(RELATED_NOTES["Seed.md"]) & words(RELATED_NOTES["Similar text.md"]))
        self.assertEqual(hits["Tagged"].reason, "shares #rl")
        self.assertEqual(self.hits("Linked")["Seed"].reason.split(" · ")[0], "links to it")

    def test_blend_and_scaling(self):
        hits = self.hits("Seed").values()
        for hit in hits:
            self.assertAlmostEqual(hit.score, WEIGHTS["graph"] * hit.graph + WEIGHTS["text"] * hit.text + WEIGHTS["tags"] * hit.tags, delta=1e-12)
            self.assertTrue(all(0 <= v <= 1 for v in (hit.graph, hit.text, hit.tags)))
        for signal in ("graph", "text", "tags"):
            self.assertEqual(max(getattr(h, signal) for h in hits), 1.0)
        self.assertEqual(max(hits, key=lambda h: h.graph).title, "Linked")

    def test_graph_signal_is_degree_normalised_pagerank(self):
        engine = self.engine
        g = engine.graph
        seed = engine.seed("Seed")
        rank = personalized_pagerank(g.adjacency, seed.node)
        raw = {g.labels[n]: r / g.degree(n) for n, r in rank.items() if not g.is_missing(n) and n != seed.node}
        top = max(raw.values())
        for title, hit in self.hits("Seed").items():
            self.assertAlmostEqual(hit.graph, raw.get(f"{title}.md", 0) / top, delta=1e-12)

    def test_common_tags_and_template_headings_do_not_count(self):
        seed = self.engine.seed("Seed")
        self.assertEqual(seed.tags, {"rl"})  # #baby is on 6 of 10 notes
        self.assertNotIn("zettels", seed.terms)
        self.assertNotIn("related", seed.terms)
        self.assertNotIn("baby", seed.terms)
        self.assertNotIn("status", seed.terms)  # a property key, not text
        self.assertIn("kl", seed.terms)
        self.assertEqual(self.hits("Seed")["Far"].tags if "Far" in self.hits("Seed") else 0, 0)

    def test_unlinked_only(self):
        hits = self.hits("Seed", unlinked_only=True)
        self.assertNotIn("Linked", hits)
        self.assertIn("Similar text", hits)

    def test_missing_note_as_seed(self):
        seed = self.engine.seed("topic-x")
        self.assertEqual((seed.path, sorted(seed.linkers)), (None, ["Same topic.md", "Seed.md"]))
        hits = self.hits("topic-x")
        self.assertEqual(hits["Seed"].reason.split(" · ")[0], "links to it")
        self.assertTrue(hits["Seed"].linked)
        self.assertEqual(hits["Linked"].reason.split(" · ")[0], "2 hops via Seed")

    def test_unknown_and_ambiguous_references(self):
        with self.assertRaisesRegex(QueryError, 'no note or link target called "Nope"'):
            self.engine.seed("Nope")
        vault = TempVault({"a/Same.md": "x", "b/Same.md": "y"})
        try:
            with VaultIndex(default_db_path(vault.root)) as ix:
                ix.update(vault.root)
                with self.assertRaisesRegex(QueryError, "could be 2 notes"):
                    Relatedness(ix.conn).seed("Same")
        finally:
            vault.cleanup()


class RelatedCliTests(RelatedVaultCase):
    def run_cli(self, *args):
        out, err = io.StringIO(), io.StringIO()
        with patch.dict(os.environ, {"COLUMNS": "80"}), redirect_stdout(out), redirect_stderr(err):
            code = cli_main(["related", str(self.vault.root), *args])
        return code, out.getvalue(), err.getvalue()

    def test_output(self):
        code, out, err = self.run_cli("Seed", "--color", "never")
        self.assertEqual((code, err), (0, ""))
        lines = out.splitlines()
        self.assertIn("  score = 0.5 × graph + 0.3 × text + 0.2 × tags · 9 notes compared", lines)
        linked = next(i for i, line in enumerate(lines) if line.startswith("   1. Linked"))
        self.assertTrue(lines[linked].endswith("linked"))
        self.assertEqual(lines[linked + 1].split(" · ")[0], "      linked from it")
        self.assertIn("      2 hops via topic-x (no note yet)", lines)
        self.assertTrue(any(line.startswith("   ") and "Similar text" in line and line.endswith("not linked") for line in lines))

    def test_options_and_errors(self):
        _, out, _ = self.run_cli("Seed", "--unlinked", "--color", "never")
        self.assertFalse([line for line in out.splitlines() if line[6:].startswith("Linked ")])
        self.assertTrue(all(line.endswith("not linked") for line in out.splitlines() if line[4:6] == ". "))
        _, out, _ = self.run_cli("Seed", "--explain", "--color", "never", "--limit", "1")
        self.assertIn("      graph 1.00 · text ", out)
        _, out, _ = self.run_cli("topic-x", "--color", "never")
        self.assertIn("Related to topic-x (no note yet; 2 notes link to it)", out)
        _, out, _ = self.run_cli("Seed", "--json")
        data = json.loads(out)
        self.assertEqual(data["seed"]["path"], "Seed.md")
        self.assertEqual(data["hits"][0]["title"], "Linked")
        code, out, err = self.run_cli("Nope")
        self.assertEqual((code, out), (2, ""))
        self.assertIn('error: no note or link target called "Nope"', err)


if __name__ == "__main__":
    unittest.main()


# ----- notes that do not exist yet -------------------------------------------------------
#
# [[reward-hump]] is linked from 3 notes (one only on its Tags: line, one only in a field list) and
# mentioned without a link in one more. [[Reward Hump]] is the same name spelled differently;
# [[reward-hump-in-practice]] is a narrower name; [[llm-judge]] < [[llm-judge-evaluation-methodology]]
# > [[evaluation-methodology]] must not chain into one group. [[on-policy-vs-off-policy]] is the
# existing "on-policy versus off-policy" note; [[policy]] is too short to say so.

SUGGEST_NOTES = {
    "Sources/Paper.md": "Tags: [[topic]]\n\nThe reward climbs while quality falls: that is the [[reward-hump]] in RLHF.\n",
    "Notes/KL.md": "Tags: [[topic]]\n\nThe KL penalty exists to prevent the [[reward-hump]], so the policy stays near its reference.\n",
    "Notes/Only tags.md": "Tags: [[reward-hump]] [[topic]]\n\nNothing else here.\n",
    "Notes/Fields.md": "supports:: [[reward-hump]]\n",
    "Notes/Mention.md": "People call it the reward hump without linking anything.\n",
    "Notes/Other.md": "See [[Reward Hump]] and [[reward-hump-in-practice]] and [[llm-judge-evaluation-methodology]].\n"
                      "Also [[reward-hump-seen-across-rlhf-practice]] (three more words: its own topic).\n",
    "Notes/Judge.md": "[[llm-judge]] [[evaluation-methodology]] [[evaluation-methodology]] [[llm-judge]] ![[missing.png]]\n",
    "Notes/Judge two.md": "[[llm-judge]] and [[evaluation-methodology]]\n",
    "Notes/on-policy versus off-policy.md": "The existing note.\n",
    "Notes/Links.md": "[[on-policy-vs-off-policy]] [[on-policy-vs-off-policy]] [[policy]] [[off-policy-evaluation]]\n",
    "Notes/Links two.md": "[[on-policy-vs-off-policy]] [[policy]] [[off-policy-evaluation]]\n",
    "Notes/Off-policy evaluation with importance sampling and doubly robust estimators.md": "A long title.\n",
}


class SuggestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from vaultengine.suggest import Suggester

        cls.vault = TempVault(SUGGEST_NOTES)
        cls.ix = VaultIndex(default_db_path(cls.vault.root))
        cls.ix.update(cls.vault.root)
        cls.suggester = Suggester(cls.ix.conn)
        cls.result = cls.suggester.suggestions(limit=None)

    @classmethod
    def tearDownClass(cls):
        cls.ix.close()
        cls.vault.cleanup()

    def test_wanted_notes_ranked_by_linking_notes(self):
        wanted = {m.name: m for m in self.result.notes}
        self.assertEqual(self.result.notes[0].name, "reward-hump")
        hump = wanted["reward-hump"]
        # Mentions: Mention.md says "reward hump"; so does Other.md, in [[Reward Hump]], a different link target
        # (names match case-insensitively, but not "-" against " "), which the overlapping names show.
        self.assertEqual((hump.linking_notes, hump.links, hump.mentions), (4, 4, 2))
        self.assertEqual(hump.folders, [("Notes", 3), ("Sources", 1)])
        self.assertNotIn("missing.png", wanted)  # a missing attachment is not a note to write
        self.assertNotIn("Reward Hump", wanted)  # linked from one note only
        self.assertEqual(wanted["topic"].linking_notes, 3)
        self.assertTrue(all(m.linking_notes >= 2 for m in self.result.notes))

    def test_overlapping_names_do_not_chain(self):
        groups = [[name for name, _ in group] for group in self.result.same_idea]
        self.assertFalse([g for g in groups if "llm-judge" in g and "evaluation-methodology" in g])  # no chaining
        holders = [g for g in groups if "llm-judge-evaluation-methodology" in g]
        self.assertEqual(len(holders), 1)  # it joins one core (a tie here: the first core alphabetically)
        self.assertEqual(holders[0][0], "evaluation-methodology")
        self.assertTrue(any(group[0] == "reward-hump" and "reward-hump-in-practice" in group for group in groups))
        self.assertFalse([g for g in groups if "reward-hump-seen-across-rlhf-practice" in g])  # 3 extra words

    def test_probably_existing(self):
        # [[off-policy-evaluation]] is inside a much longer title (3 of 9 words): a narrower topic, not that note.
        self.assertEqual(self.result.existing, [("on-policy-vs-off-policy", 2, "Notes/on-policy versus off-policy.md")])

    def test_draft(self):
        path, text = self.suggester.draft("reward-hump", now=datetime(2026, 11, 7, 9, 30))
        self.assertEqual(path, "Notes/reward-hump.md")
        lines = text.splitlines()
        self.assertEqual(lines[:5], ["Created: 07-11-2026 09:30", "", "Tags: [[topic]]", "", "# reward-hump"])
        quotes = [line for line in lines if line.startswith("- [[") and ": " in line and '"' in line]
        self.assertEqual(len(quotes), 2)  # Paper and KL; the Tags: line and the field list are not quoted
        self.assertTrue(any('"The KL penalty exists to prevent the [[reward-hump]]' in q for q in quotes))
        self.assertIn("- also linked from: [[Fields]], [[Only tags]]", lines)
        self.assertEqual(lines[-1], "1. [[Paper]]")  # references: linking notes in a sources folder

    def test_write_creates_the_note_once(self):
        vault = TempVault(SUGGEST_NOTES)
        try:
            VaultIndex(default_db_path(vault.root)).update(vault.root)
            out, err = io.StringIO(), io.StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                code = cli_main(["suggest", str(vault.root), "--draft", "reward-hump", "--write"])
            self.assertEqual(code, 0, err.getvalue())
            self.assertTrue((vault.root / "Notes/reward-hump.md").exists())
            with VaultIndex(default_db_path(vault.root)) as ix:
                ix.update(vault.root)
                unresolved = ix.conn.execute("SELECT COUNT(*) FROM links WHERE target_key = 'reward-hump' AND status != 'resolved'").fetchone()[0]
            self.assertEqual(unresolved, 0)  # every link to it now resolves
            with redirect_stdout(io.StringIO()), redirect_stderr(err := io.StringIO()):
                code = cli_main(["suggest", str(vault.root), "--draft", "reward-hump-in-practice", "--write", "--folder", "Notes"])
            self.assertEqual(code, 0)
            (vault.root / "Notes/again.md").write_text("x")
            with redirect_stdout(io.StringIO()), redirect_stderr(err := io.StringIO()):
                code = cli_main(["suggest", str(vault.root), "--draft", "reward-hump-in-practice", "--write", "--folder", "Notes"])
            self.assertEqual(code, 2)
            self.assertIn("already exists; nothing written", err.getvalue())
        finally:
            vault.cleanup()

    def test_cli(self):
        out = io.StringIO()
        with patch.dict(os.environ, {"COLUMNS": "80"}), redirect_stdout(out):
            code = cli_main(["suggest", str(self.vault.root), "--color", "never"])
        lines = out.getvalue().splitlines()
        self.assertEqual(code, 0)
        self.assertIn("    1. reward-hump", lines)
        self.assertIn("       4 notes link here · 2 more mention it", lines)
        self.assertIn("    [[on-policy-vs-off-policy]] (2 notes) → Notes/on-policy versus off-policy.md", lines)
        err = io.StringIO()
        with redirect_stdout(io.StringIO()), redirect_stderr(err):
            self.assertEqual(cli_main(["suggest", str(self.vault.root), "--draft", "nope"]), 2)
        self.assertIn('no link points to a missing note called "nope"', err.getvalue())
