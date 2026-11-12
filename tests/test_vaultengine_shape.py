"""The shape of the vault (`vault map`) and the deepen lenses, on graphs whose answers are known.

Two planted communities joined by one note: Louvain must find exactly the two groups, and the
joining note must come first as a bridge. Every expected number is computed here by hand (or
from the test's own tables), not by calling the code under test.
"""

import io
import json
import math
import os
import unittest
from contextlib import redirect_stderr, redirect_stdout
from itertools import combinations
from unittest.mock import patch

from vaultengine.cli import main as cli_main
from vaultengine.embedding import EmbeddingError
from vaultengine.graphrank import build_graph
from vaultengine.index import VaultIndex, default_db_path
from vaultengine.lenses import Lenses
from vaultengine.related import Relatedness
from vaultengine.shape import VaultShape, connected_parts, hits, louvain, modularity, percentile, percentile_rank

from test_vaultengine_embedding import NO_ENGINE_ENV
from test_vaultengine_search import TempVault

A = [f"a{i}" for i in range(1, 9)]
B = [f"b{i}" for i in range(1, 9)]


def planted():
    """Every pair inside A linked once, the same in B; X links a1, a2, b1 and b2; nothing else."""
    notes = {}
    for group in (A, B):
        for i, name in enumerate(group):
            notes[f"{name}.md"] = " ".join(f"[[{other}]]" for other in group[i + 1:]) + "\n"
    notes["X.md"] = "[[a1]] [[a2]] [[b1]] [[b2]]\n"
    return notes


def off():
    raise EmbeddingError("no model in this test")


class PlantedCommunitiesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vault = TempVault(planted())
        cls.ix = VaultIndex(default_db_path(cls.vault.root))
        cls.ix.update(cls.vault.root)
        cls.shape = VaultShape(cls.ix.conn)

    @classmethod
    def tearDownClass(cls):
        cls.ix.close()
        cls.vault.cleanup()

    def groups(self, shape):
        return {frozenset(shape.sg.name(v) for v in c.notes) for c in shape.communities}

    def test_exactly_the_two_planted_groups(self):
        groups = self.groups(self.shape)
        self.assertEqual({g - {"X"} for g in groups}, {frozenset(A), frozenset(B)})
        self.assertEqual(sum("X" in g for g in groups), 1)  # X joins one side
        self.assertEqual(self.shape.summary()["alone"], 0)

    def test_the_same_every_run_and_for_other_seeds(self):
        again = VaultShape(self.ix.conn)
        self.assertEqual(again.membership, self.shape.membership)
        for seed in (1, 7, 123):
            with self.subTest(seed=seed):
                self.assertEqual({g - {"X"} for g in self.groups(VaultShape(self.ix.conn, seed=seed))},
                                 {frozenset(A), frozenset(B)})

    def test_modularity_by_hand(self):
        edges = [(x, y) for g in (A, B) for x, y in combinations(g, 2)] + [("X", t) for t in ("a1", "a2", "b1", "b2")]
        m = len(edges)
        degree = {n: sum(n in e for e in edges) for n in A + B + ["X"]}
        side = {n: ("A" if n in A else "B") for n in A + B}
        side["X"] = "A" if any("X" in g and "a1" in g for g in self.groups(self.shape)) else "B"
        q = 0.0
        for c in ("A", "B"):
            inside = sum(1 for x, y in edges if side[x] == side[y] == c)
            total = sum(d for n, d in degree.items() if side[n] == c)
            q += inside / m - (total / (2 * m)) ** 2
        self.assertAlmostEqual(self.shape.modularity, q, places=12)

    def test_the_joining_note_is_the_first_bridge(self):
        bridges = self.shape.bridges()
        self.assertEqual(bridges[0]["path"], "X.md")
        b = {r["path"]: r["betweenness"] for r in bridges}
        pairs = 16 * 15 / 2  # pairs of the other notes
        self.assertAlmostEqual(b["X.md"], 64 / pairs, places=12)  # every A-B pair
        # a1: half the paths from a3..a8 to X and to every b go through it (the other half through a2).
        self.assertAlmostEqual(b["a1.md"], (6 * 0.5 + 6 * 8 * 0.5) / pairs, places=12)
        self.assertEqual(b["a5.md"], 0.0)  # A is complete: no shortest path needs a5
        self.assertTrue(self.shape.betweenness.exact)
        self.assertEqual(bridges[0]["spans"], sorted(bridges[0]["spans"], key=lambda c: c))  # two sides, 2 links each
        self.assertEqual(len(bridges[0]["spans"]), 2)

    def test_constraint_by_hand(self):
        c = self.shape.constraint
        x = next(v for v in self.shape.sg.notes if self.shape.sg.labels[v] == "X.md")
        # X's four contacts weigh 1/4 each; a1 and a2 are tied (a2 gives 1/8 of its weight to a1), so are b1, b2.
        self.assertAlmostEqual(c[x], 4 * (1 / 4 + (1 / 4) * (1 / 8)) ** 2, places=12)
        self.assertEqual(min(c, key=c.get), x)  # the least constrained note

    def test_a_sampled_betweenness_says_so_and_keeps_the_bridge_first(self):
        sampled = VaultShape(self.ix.conn, path_budget=1)
        b = sampled.betweenness
        self.assertFalse(b.exact)
        self.assertEqual(b.sources, 1)
        full = VaultShape(self.ix.conn, path_budget=10**9).betweenness
        self.assertTrue(full.exact)
        half = VaultShape(self.ix.conn, path_budget=(len(self.shape.sg.adj) + self.shape.sg.edges) * 9)
        self.assertEqual(half.betweenness.sources, 9)
        self.assertEqual(half.bridges()[0]["path"], "X.md")


class AlgorithmTests(unittest.TestCase):
    def test_louvain_on_two_triangles(self):
        adj = [dict() for _ in range(6)]
        for x, y in [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3)]:
            adj[x][y] = adj[y][x] = 1.0
        membership = louvain(adj, seed=3)
        self.assertEqual(len({membership[v] for v in (0, 1, 2)}), 1)
        self.assertEqual(len({membership[v] for v in (3, 4, 5)}), 1)
        self.assertNotEqual(membership[0], membership[3])
        # Q = 2 * (3/7 - (7/14)^2) = 6/7 - 1/2
        self.assertAlmostEqual(modularity(adj, membership), 6 / 7 - 0.5, places=12)

    def test_a_community_in_pieces_is_split(self):
        adj = [dict() for _ in range(4)]
        adj[0][1] = adj[1][0] = 1.0
        adj[2][3] = adj[3][2] = 1.0
        parts = connected_parts(adj, [0, 0, 0, 0])  # one "community" that is two pieces
        self.assertEqual(parts[0], parts[1])
        self.assertEqual(parts[2], parts[3])
        self.assertNotEqual(parts[0], parts[2])

    def test_hits_by_hand(self):
        # H1 -> A1, A2; H2 -> A1. Authorities: the top eigenvector of [[2, 1], [1, 1]]: (1, (sqrt 5 - 1) / 2).
        vault = TempVault({"H1.md": "[[A1]] [[A2]]", "H2.md": "[[A1]]", "A1.md": "a", "A2.md": "b"})
        try:
            with VaultIndex(default_db_path(vault.root)) as ix:
                ix.update(vault.root)
                graph = build_graph(ix.conn)
                hub, auth = hits(graph)
                by = lambda scores: {graph.labels[v]: x for v, x in scores.items()}
                golden = (math.sqrt(5) - 1) / 2
                self.assertAlmostEqual(by(auth)["A1.md"], 1.0, places=9)
                self.assertAlmostEqual(by(auth)["A2.md"], golden, places=9)
                self.assertAlmostEqual(by(hub)["H1.md"], 1.0, places=9)
                self.assertAlmostEqual(by(hub)["H2.md"], golden, places=9)
        finally:
            vault.cleanup()

    def test_percentiles(self):
        self.assertEqual(percentile([1, 2, 3, 4], 50), 2.5)
        self.assertEqual(percentile([1, 2, 3, 4, 5], 90), 4.6)
        self.assertEqual(percentile([], 90), 0.0)
        self.assertEqual(percentile_rank([1, 2, 3, 4], 2), 50.0)


# ----- the lenses ---------------------------------------------------------------------------------


class LensTests(unittest.TestCase):
    """On the vault of test_vaultengine_relations: its tables give the expected values."""

    @classmethod
    def setUpClass(cls):
        from test_vaultengine_relations import NOTES, note_file

        cls.vault = TempVault({f"{n}.md": note_file(n) for n in NOTES})
        cls.ix = VaultIndex(default_db_path(cls.vault.root))
        cls.ix.update(cls.vault.root)
        engine = Relatedness(cls.ix.conn, off, cls.vault.root)
        cls.lenses = Lenses(engine, VaultShape(cls.ix.conn, graph=engine.graph), 50.0)

    @classmethod
    def tearDownClass(cls):
        cls.ix.close()
        cls.vault.cleanup()

    def test_cited_together_and_foundations_cut_at_the_vault_percentile(self):
        from test_vaultengine_relations import LINKS, NOTES, expected_cocitation, expected_coupling

        for kind, lens, expected in (("cocitation", self.lenses.cited_together, expected_cocitation),
                                     ("coupling", self.lenses.foundations, expected_coupling)):
            with self.subTest(kind=kind):
                values = {}
                for a, b in combinations(NOTES, 2):
                    if a in LINKS or b in LINKS or kind == "cocitation":
                        value = expected(a, b)[0]
                        if value > 0:
                            values[(a, b)] = value
                cut = percentile(list(values.values()), 50)
                result = lens(self.lenses.node("A"), 0)
                self.assertAlmostEqual(result["cutoff"], cut, places=12)
                self.assertEqual(result["pairs_in_vault"], len(values))
                mine = {(b if a == "A" else a): v for (a, b), v in values.items() if "A" in (a, b)}
                self.assertEqual([r["path"].removesuffix(".md") for r in result["notes"]],
                                 sorted((n for n, v in mine.items() if v >= cut), key=lambda n: (-mine[n], n)))

    def test_neighbourhood_cuts_at_the_percentile_of_this_notes_scores(self):
        result = self.lenses.neighbourhood(self.lenses.node("A"), 0)
        self.assertEqual(result["compared"], 11)
        self.assertEqual(result["text_unavailable"], "no model in this test")
        self.assertTrue(all(h["score"] >= result["cutoff"] for h in result["hits"]))
        self.assertEqual(result["kept"], len(result["hits"]))

    def test_refuses_a_missing_note(self):
        from vaultengine.querylang import QueryError

        with self.assertRaisesRegex(QueryError, "not a note yet"):
            self.lenses.node("gone")


class BackboneTests(unittest.TestCase):
    """A and B: complete groups of six. In B the chain b1 -> b2 -> ... -> b6 is linked three times a step
    (weight 3) and every other pair once; leaf links b4 only; X links a1 and b3."""

    @classmethod
    def setUpClass(cls):
        notes = {}
        for group in ([f"a{i}" for i in range(1, 7)], [f"b{i}" for i in range(1, 7)]):
            for i, name in enumerate(group):
                links = [f"[[{o}]]" for o in group[i + 1:]]
                if name.startswith("b") and i + 1 < len(group):
                    links += [f"[[{group[i + 1]}]]"] * 2  # the chain: two more links to the next one
                notes[f"{name}.md"] = " ".join(links) + "\n"
        notes["X.md"] = "[[a1]] [[b3]]\n"
        notes["leaf.md"] = "[[b4]]\n"
        cls.vault = TempVault(notes)
        cls.ix = VaultIndex(default_db_path(cls.vault.root))
        cls.ix.update(cls.vault.root)
        engine = Relatedness(cls.ix.conn, off, cls.vault.root)
        cls.lenses = Lenses(engine, VaultShape(cls.ix.conn, graph=engine.graph))

    @classmethod
    def tearDownClass(cls):
        cls.ix.close()
        cls.vault.cleanup()

    def members(self, note):
        shape = self.lenses.shape
        c = shape.community_of(self.lenses.node(note))
        return {shape.sg.name(v) for v in c.notes}

    def test_the_groups(self):
        self.assertTrue({f"b{i}" for i in range(1, 7)} | {"leaf"} <= self.members("b1"))
        self.assertTrue({f"a{i}" for i in range(1, 7)} <= self.members("a1"))
        self.assertNotEqual(self.members("a1"), self.members("b1"))

    def test_the_chain_is_the_backbone(self):
        # The strongest tree over B takes the five chain steps (weight 3) first; leaf (and X, if it is in B) hang off
        # b4 and b3 with weight 1. The longest path in that tree is the chain: 5 steps, against 4 from leaf to b1.
        result = self.lenses.backbone(self.lenses.node("b2"))
        names = [p.removesuffix(".md") for p in result["chain"]]
        self.assertIn(names, (["b1", "b2", "b3", "b4", "b5", "b6"], ["b6", "b5", "b4", "b3", "b2", "b1"]))
        self.assertEqual(result["note_on_chain"], names.index("b2") + 1)
        self.assertEqual(result["steps"], ["links"] * 5)
        off_chain = self.lenses.backbone(self.lenses.node("leaf"))
        self.assertEqual((off_chain["steps_off"], off_chain["joins_at"]), (1, "b4.md"))

    def test_cluster_view(self):
        result = self.lenses.cluster(self.lenses.node("b3"), 3)
        self.assertEqual(result["notes"], len(self.members("b3")))
        # b3 is where X comes in: some of its weight goes to another community, through b3 (or X) on this side.
        neighbour = result["neighbours"][0]
        self.assertTrue({"b3.md", "X.md"} & set(neighbour["through"]), neighbour)
        self.assertEqual(result["topic_pages"], [])


class CliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vault = TempVault(planted())
        VaultIndex(default_db_path(cls.vault.root)).update(cls.vault.root)

    @classmethod
    def tearDownClass(cls):
        cls.vault.cleanup()

    def run_cli(self, *args):
        out, err = io.StringIO(), io.StringIO()
        env = {**NO_ENGINE_ENV, "VAULTENGINE_EMBEDDING__MODEL_PATH": "/no/model", "COLUMNS": "100"}
        with patch.dict(os.environ, env, clear=True), redirect_stdout(out), redirect_stderr(err):
            code = cli_main(list(args))
        return code, out.getvalue(), err.getvalue()

    def test_map(self):
        code, out, err = self.run_cli("map", str(self.vault.root), "--color", "never")
        self.assertEqual((code, err), (0, ""))
        self.assertIn("  communities  Louvain, seed 42, resolution 1 (the same every run)", out)
        self.assertIn("2 communities hold 17 notes; 0 notes in none", out)
        lines = out.splitlines()
        first = next(i for i, line in enumerate(lines) if line.startswith("         betweenness"))
        self.assertTrue(lines[first + 1].startswith("      1. 0.533 (p100)"), lines[first + 1])
        self.assertIn(" X  → ", lines[first + 1])
        code, out, _ = self.run_cli("map", str(self.vault.root), "--json")
        data = json.loads(out)
        self.assertEqual(data["bridges"][0]["path"], "X.md")
        self.assertEqual(data["communities"][0]["topic_pages"], [])

    def test_topic_pages_are_the_notes_the_format_types_as_topics(self):
        # MOC/t.md is typed "topic"; MOC/other.md sits in the same folder with another type: it is not a page.
        files = {**planted(), "MOC/t.md": "[[a3]] [[a4]] [[a5]]\n", "MOC/other.md": "[[b3]] [[b4]] [[b5]]\n"}
        vault = TempVault(files)
        fmt = vault.root.parent / "f.yaml"
        fmt.write_text("format: 1\ntypes:\n  topic:\n    match: {folder: MOC, name: 't'}\n"
                       "  overview:\n    match: {folder: MOC}\n  note:\n    match: {name: '*'}\n")
        try:
            VaultIndex(default_db_path(vault.root)).update(vault.root)
            code, out, err = self.run_cli("map", str(vault.root), "--json", "--format", str(fmt))
            self.assertEqual(code, 0, err)
            data = json.loads(out)
            self.assertEqual(data["topic_pages"], "notes the format f types as topic")
            pages = {p["path"] for c in data["communities"] for p in c["topic_pages"]}
            self.assertEqual(pages, {"MOC/t.md"})
            code, out, _ = self.run_cli("map", str(vault.root), "--color", "never")
            self.assertIn("  topic pages  no format: the format's note types say which notes are topic pages", out)
        finally:
            vault.cleanup()

    def test_deepen(self):
        code, out, err = self.run_cli("deepen", str(self.vault.root), "X", "--color", "never")
        self.assertEqual((code, err), (0, ""))
        for title in ("  Neighbourhood", "  Cited together", "  Same foundations", "  Backbone", "  Cluster"):
            self.assertIn(title, out)
        self.assertIn("text signal unavailable: embedding model folder not found: /no/model", out)
        code, out, _ = self.run_cli("deepen", str(self.vault.root), "a3", "--lens", "cluster", "--json")
        self.assertEqual(list(json.loads(out)["lenses"]), ["cluster"])
        code, _, err = self.run_cli("deepen", str(self.vault.root), "nope")
        self.assertEqual(code, 2)
        self.assertIn('no note or link target called "nope"', err)


if __name__ == "__main__":
    unittest.main()


class ReferenceTests(unittest.TestCase):
    """The numpy betweenness and HITS and the reworked constraint give the numbers the pure-Python loops gave
    (tests/shape_reference.py), bit for bit: on the sample vault (exact betweenness) and on random graphs,
    with a budget small enough to sample source notes."""

    def graphs(self):
        import random
        from pathlib import Path

        from vaultengine.shape import shape_graph
        from vaultengine.tagrule import rule_for_format

        sample = Path(__file__).resolve().parent / "fixtures" / "vault"
        vault = TempVault({str(p.relative_to(sample)): p.read_text(encoding="utf-8") for p in sample.rglob("*.md")})
        self.addCleanup(vault.cleanup)
        with VaultIndex(default_db_path(vault.root)) as ix:
            ix.update(vault.root)
            yield "sample", shape_graph(ix.conn, rule_for_format("brain"))
        rng = random.Random(11)
        for trial in range(4):
            n = rng.randint(40, 120)
            files = {}
            for i in range(n):
                targets = {rng.randrange(n) for _ in range(rng.randint(0, 6))} - {i}
                missing = [f"[[gone-{rng.randrange(8)}]]"] if rng.random() < 0.2 else []
                tags = [f"#t{rng.randrange(5)}"] if rng.random() < 0.5 else []
                files[f"n{i}.md"] = " ".join([f"[[n{t}]]" for t in sorted(targets)] + missing + tags) + "\n"
            vault = TempVault(files)
            self.addCleanup(vault.cleanup)
            with VaultIndex(default_db_path(vault.root)) as ix:
                ix.update(vault.root)
                yield f"random {trial}", shape_graph(ix.conn)

    def test_same_numbers_bit_for_bit(self):
        import shape_reference as ref
        from vaultengine import shape

        for name, sg in self.graphs():
            for budget in (20_000_000, 2_000):  # exact, and a seeded sample of sources
                with self.subTest(graph=name, budget=budget):
                    new, old = shape.betweenness(sg, budget, 42), ref.betweenness(sg, budget, 42)
                    self.assertEqual((new.sources, new.exact), (old.sources, old.exact))
                    self.assertEqual(new.values, old.values)
            with self.subTest(graph=name, measure="constraint"):
                self.assertEqual(shape.constraint(sg), ref.constraint(sg))
            with self.subTest(graph=name, measure="hits"):
                self.assertEqual(shape.hits(sg.graph), ref.hits(sg.graph))
