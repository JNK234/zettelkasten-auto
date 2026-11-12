"""Discover lenses and meaning edges, on planted fixtures whose answers the tests compute themselves.

Hidden link and unwritten: planted notes, missing notes and names; expected scores, merges and spreads are
computed here from the fixture's own tables. Wander: replayed by an independent implementation of the rule.
Gap, surprise and meaning edges: a tiny Model2Vec model written by hand (test_vaultengine_embedding), so every
vector, centroid and cosine is computed here with numpy, not by the code under test.
"""

import io
import json
import math
import os
import random
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from itertools import combinations
from pathlib import Path
from unittest.mock import patch

import numpy as np

from vaultengine.cli import main as cli_main
from vaultengine.discover import Discover
from vaultengine.embedding import EmbeddingError
from vaultengine.embedding.model2vec import Model2VecProvider
from vaultengine.index import VaultIndex, default_db_path
from vaultengine.related import Relatedness
from vaultengine.shape import VaultShape, percentile

from test_vaultengine_embedding import NO_ENGINE_ENV, WORDS, make_model
from test_vaultengine_search import TempVault


def off():
    raise EmbeddingError("no model in this test")


class Fixture:
    """A vault (dict of files) indexed, with Relatedness, VaultShape and Discover over it."""

    def __init__(self, files, provider=off, percentile_cut=90.0, seed=42, meaning_percentile=None):
        self.vault = TempVault(files)
        self.ix = VaultIndex(default_db_path(self.vault.root))
        self.ix.update(self.vault.root)
        self.engine = Relatedness(self.ix.conn, provider, self.vault.root)
        meaning = self.engine.text if meaning_percentile is not None else None
        self.shape = VaultShape(self.ix.conn, graph=self.engine.graph, meaning=meaning,
                                meaning_percentile=meaning_percentile or 50.0)
        self.discover = Discover(self.engine, self.shape, percentile_cut, seed)

    def node(self, name):
        return self.engine.seed(name).node

    def name(self, node):
        return self.shape.sg.name(node)

    def close(self):
        self.ix.close()
        self.vault.cleanup()


# ----- hidden link ---------------------------------------------------------------------------------
# A links B1..B4. C is linked from B1..B4 (no link with A): four middles. D is linked from B1 only. E links B1 and
# B2 and shares the rare word "zymurgy" with A. F and G only fill the vault. Names are letters and digits, so the
# only alphabetic key term is "zymurgy" (on two notes).

HIDDEN = {
    "A.md": "[[B1]] [[B2]] [[B3]] [[B4]] zymurgy\n",
    "B1.md": "[[C]] [[D]]\n", "B2.md": "[[C]]\n", "B3.md": "[[C]]\n", "B4.md": "[[C]]\n",
    "C.md": "c\n", "D.md": "d\n", "E.md": "[[B1]] [[B2]] zymurgy\n", "F.md": "f\n", "G.md": "[[F]]\n",
}
HIDDEN_LINKS = {"A": ["B1", "B2", "B3", "B4"], "B1": ["C", "D"], "B2": ["C"], "B3": ["C"], "B4": ["C"], "E": ["B1", "B2"], "G": ["F"]}
WORDS_ON = {"zymurgy": ["A", "E"]}


def hidden_by_hand():
    """score of every unlinked pair with 2+ middles, from the tables: Adamic-Adar over shared neighbours and words."""
    notes = sorted({n.removesuffix(".md") for n in HIDDEN})
    near = {n: set() for n in notes}
    for a, targets in HIDDEN_LINKS.items():
        for b in targets:
            near[a].add(b)
            near[b].add(a)
    middles = {n: {("node", m) for m in near[n]} for n in notes}
    for word, on in WORDS_ON.items():
        for n in on:
            middles[n].add(("term", word))
    members = {}
    for n, ms in middles.items():
        for m in ms:
            members.setdefault(m, set()).add(n)
    scores = {}
    for a, c in combinations(notes, 2):
        if c in near[a]:
            continue
        shared = middles[a] & middles[c]
        if len(shared) >= 2:
            scores[(a, c)] = (sum(1 / math.log(1 + len(members[m])) for m in shared), shared)
    return scores


class HiddenLinkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.f = Fixture(HIDDEN, percentile_cut=50.0)

    @classmethod
    def tearDownClass(cls):
        cls.f.close()

    def test_the_planted_hidden_link_with_its_middles(self):
        expected = hidden_by_hand()
        result = self.f.discover.hidden_link(self.f.node("A"), 0)
        self.assertEqual(result["pairs_in_vault"], len(expected))
        self.assertAlmostEqual(result["cutoff"], percentile([s for s, _ in expected.values()], 50), places=12)
        top = result["results"][0]
        self.assertEqual(top["note"], "C.md")
        score, shared = expected[("A", "C")]
        self.assertAlmostEqual(top["score"], score, places=12)
        # B1 is attached to A, C, D, E: 1/ln 5; B2..B4 to A, C (B2 also E) : the rarer middles weigh more.
        self.assertEqual(sorted(m["middle"] for m in top["middles"]), ["B1", "B2", "B3", "B4"])
        self.assertTrue(top["reason"].startswith("no link either way; 4 middles in common: "))

    def test_every_score_matches_the_tables(self):
        expected = hidden_by_hand()
        for (a, c), (score, shared) in expected.items():
            with self.subTest(pair=(a, c)):
                got = self.f.discover.hidden_scores(self.f.node(a))[self.f.node(c)]
                self.assertAlmostEqual(got[0], score, places=12)
                self.assertEqual(len(got[1]), len(shared))
        # A and E share B1, B2 and the word: three middles, one of them a key term.
        e = self.f.discover.hidden_scores(self.f.node("A"))[self.f.node("E")]
        self.assertIn(("term", "zymurgy"), e[1])

    def test_linked_notes_and_single_middles_are_never_hidden_links(self):
        found = {r["note"] for r in self.f.discover.hidden_link(self.f.node("A"), 0)["results"]}
        self.assertNotIn("B1.md", found)  # linked
        self.assertNotIn("D.md", found)  # one middle (B1) only


# ----- unwritten -------------------------------------------------------------------------------------
# Three groups of four notes, each group fully linked inside (three communities). [[topic-x]] is linked from one
# note of each group; [[topic-y]] from two notes of group P; [[llm-judge]] from P and [[llm-judge-method]] from Q:
# the same idea by its name words (a core of two words and one more), so together they reach two communities.


def group(prefix):
    names = [f"{prefix}{i}" for i in range(1, 5)]
    return {f"{n}.md": " ".join(f"[[{o}]]" for o in names[i + 1:]) for i, n in enumerate(names)}


UNWRITTEN = {**group("p"), **group("q"), **group("r")}
UNWRITTEN["p1.md"] += " [[topic-x]] [[topic-y]] [[llm-judge]]"
UNWRITTEN["p2.md"] += " [[topic-y]]"
UNWRITTEN["q1.md"] += " [[topic-x]] [[llm-judge-method]]"
UNWRITTEN["r1.md"] += " [[topic-x]]"


class UnwrittenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.f = Fixture(UNWRITTEN, percentile_cut=50.0)

    @classmethod
    def tearDownClass(cls):
        cls.f.close()

    def community(self, name):
        return self.f.shape.membership[self.f.node(name)]

    def test_three_groups(self):
        self.assertEqual(len({self.community(f"{g}1") for g in "pqr"}), 3)

    def test_spread_by_name_words(self):
        result = self.f.discover.unwritten(False, None, 0)
        rows = {tuple(r["names"]): r for r in result["results"]}
        # Spreads by hand: topic-x 3 communities, the llm-judge pair 2 (merged), topic-y 1.
        spreads = [3, 2, 1]  # one per missing note or merged group
        self.assertEqual(result["groups"], 3)
        self.assertEqual(result["cutoff"], percentile(spreads, 50))
        self.assertEqual(list(rows), [("topic-x",), ("llm-judge", "llm-judge-method")])
        self.assertEqual(rows[("topic-x",)]["spread"], 3)
        self.assertEqual(rows[("topic-x",)]["communities"],
                         {self.community("p1"): 1, self.community("q1"): 1, self.community("r1"): 1})
        self.assertIn("merged by name words: [[llm-judge]], [[llm-judge-method]]", rows[("llm-judge", "llm-judge-method")]["reason"])
        self.assertEqual(result["merged"], 1)

    def test_without_a_model_the_embedding_merge_says_so(self):
        self.assertEqual(self.f.discover.unwritten(True, None, 0), {"unavailable": "no model in this test"})


class UnwrittenByEmbeddingTests(unittest.TestCase):
    """Missing names in the tiny model's words: the merge follows mutual nearest neighbours computed here."""

    NAMES = ["alpha beta", "beta alpha alpha", "gamma reward", "reward reward", "gamma"]

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.model = Path(cls.tmp.name) / "tiny"
        cls.matrix = make_model(cls.model).astype(np.float64)
        files = {**group("p"), **group("q")}
        for i, name in enumerate(cls.NAMES):  # each name linked from one note of each group: spread 2 each
            files[f"p{i % 4 + 1}.md"] += f" [[{name}]]"
            files[f"q{i % 4 + 1}.md"] += f" [[{name}]]"
        cls.f = Fixture(files, Model2VecProvider(cls.model), percentile_cut=50.0)

    @classmethod
    def tearDownClass(cls):
        cls.f.close()
        cls.tmp.cleanup()

    def vector(self, text):
        ids = {w: i + 2 for i, w in enumerate(WORDS)}
        rows = []
        for word in text.split():
            rows += [ids["re"], ids["##ward"]] if word == "reward" else [ids[word]]
        v = self.matrix[rows].mean(axis=0)
        return v / np.linalg.norm(v)

    def test_merges_are_the_mutual_nearest_neighbours_above_the_cutoff(self):
        names = sorted(self.NAMES)  # the engine works on lookup keys, which sort the same here
        vectors = np.array([self.vector(n) for n in names])
        centered = vectors - vectors.mean(axis=0)
        centered /= np.linalg.norm(centered, axis=1, keepdims=True)
        sims = centered @ centered.T
        np.fill_diagonal(sims, -np.inf)
        nearest, best = sims.argmax(axis=1), sims.max(axis=1)
        cut = percentile(list(best), 50)
        pairs = {frozenset((names[i], names[j])) for i, j in enumerate(nearest) if nearest[j] == i and best[i] >= cut}
        groups, how = self.f.discover.missing_groups(True)
        self.assertAlmostEqual(how["cutoff"], cut, places=6)
        merged = {frozenset(g) for g in groups if len(g) > 1}
        self.assertEqual(merged, pairs)
        self.assertIn(frozenset(("alpha beta", "beta alpha alpha")), merged)  # the same words in another order


# ----- wander --------------------------------------------------------------------------------------------

WANDER = {f"w{i}.md": " ".join(f"[[w{j}]]" for j in range(8) if j > i and (i * j + i + j) % 3 != 1) for i in range(8)}


class WanderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.f = Fixture(WANDER)

    @classmethod
    def tearDownClass(cls):
        cls.f.close()

    def replay(self, start, steps, seed):
        """The documented rule, written again: neighbours either way sorted by path, unvisited first."""
        near = {n: set() for n in WANDER}
        for source, text in WANDER.items():
            for j in range(8):
                if f"[[w{j}]]" in text:
                    near[source].add(f"w{j}.md")
                    near[f"w{j}.md"].add(source)
        rng = random.Random(seed)
        walk, seen = [start], {start}
        for _ in range(steps):
            around = sorted(near[walk[-1]] - {walk[-1]})
            if not around:
                break
            fresh = [n for n in around if n not in seen]
            walk.append(rng.choice(fresh or around))
            seen.add(walk[-1])
        return walk

    def test_replays_exactly(self):
        for seed in (1, 42, 99):
            with self.subTest(seed=seed):
                got = self.f.discover.wander(self.f.node("w0"), 12, seed)
                self.assertEqual(got["walk"], self.replay("w0.md", 12, seed))
                self.assertEqual(self.f.discover.wander(self.f.node("w0"), 12, seed), got)  # and again

    def test_a_different_seed_can_take_another_walk(self):
        walks = {tuple(self.f.discover.wander(self.f.node("w0"), 12, seed)["walk"]) for seed in range(10)}
        self.assertGreater(len(walks), 1)


# ----- gap and surprise, with a hand-made model --------------------------------------------------------------
# Three groups of four, fully linked inside. X says "alpha beta", Y "alpha beta gamma" (close to X in meaning),
# Z "reward reward" (far). No link between X and Y; one link x1 -> z1. So X-Y is a gap, x1 -> z1 a surprise.


def words_group(prefix, text):
    files = group(prefix)
    return {name: body + f"\n{text}\n" for name, body in files.items()}


GAPS = {**words_group("x", "alpha beta"), **words_group("y", "alpha beta gamma"), **words_group("z", "reward reward")}
GAPS["x1.md"] += "[[z1]]\n"


class GapSurpriseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.model = Path(cls.tmp.name) / "tiny"
        cls.matrix = make_model(cls.model).astype(np.float64)
        cls.f = Fixture(GAPS, Model2VecProvider(cls.model), percentile_cut=90.0)

    @classmethod
    def tearDownClass(cls):
        cls.f.close()
        cls.tmp.cleanup()

    def vector(self, text):
        ids = {w: i + 2 for i, w in enumerate(WORDS)}
        rows = []
        for word in text.split():
            rows += [ids["re"], ids["##ward"]] if word == "reward" else [ids[word]]
        v = self.matrix[rows].mean(axis=0)
        return v / np.linalg.norm(v)

    def by_hand(self):
        texts = {g: t for g, t in (("x", "alpha beta"), ("y", "alpha beta gamma"), ("z", "reward reward"))}
        notes = {f"{g}{i}": self.vector(t) for g, t in texts.items() for i in range(1, 5)}
        mean = np.mean(list(notes.values()), axis=0)
        centered = {n: (v - mean) / np.linalg.norm(v - mean) for n, v in notes.items()}
        centroid = {}
        for g in "xyz":
            c = np.mean([centered[f"{g}{i}"] for i in range(1, 5)], axis=0)
            centroid[g] = c / np.linalg.norm(c)
        return centered, centroid

    def community(self, name):
        return self.f.shape.membership[self.f.node(name)]

    def test_the_gap_is_x_and_y(self):
        _, centroid = self.by_hand()
        closeness = {frozenset(p): float(centroid[p[0]] @ centroid[p[1]]) for p in combinations("xyz", 2)}
        result = self.f.discover.gaps(True, None, 0)
        self.assertAlmostEqual(result["cutoff"], percentile(list(closeness.values()), 90), places=6)
        self.assertEqual(len(result["results"]), 1)
        gap = result["results"][0]
        self.assertEqual(set(gap["communities"]), {self.community("x1"), self.community("y1")})
        self.assertAlmostEqual(gap["closeness"], closeness[frozenset("xy")], places=6)
        self.assertEqual((gap["links"], gap["ratio"]), (0, 0.0))

    def test_the_surprise_is_x1_to_z1(self):
        centered, _ = self.by_hand()
        linked = [(a, b) for g in "xyz" for a, b in combinations([f"{g}{i}" for i in range(1, 5)], 2)] + [("x1", "z1")]
        sims = [float(centered[a] @ centered[b]) for a, b in linked]
        result = self.f.discover.surprises(True, None, 0)
        self.assertEqual(result["linked_pairs"], len(linked))
        self.assertAlmostEqual(result["cutoff"], percentile(sims, 10), places=6)
        self.assertEqual([(r["a"], r["b"]) for r in result["results"]], [("x1.md", "z1.md")])
        self.assertAlmostEqual(result["results"][0]["similarity"], float(centered["x1"] @ centered["z1"]), places=6)
        line = GAPS["x1.md"].splitlines().index("[[z1]]") + 1
        self.assertIn(f"x1 links it (line {line})", result["results"][0]["reason"])

    def test_without_embeddings_they_run_on_key_terms_or_say_why_not(self):
        for lens in (self.f.discover.gaps, self.f.discover.surprises):
            with self.subTest(lens=lens.__name__):
                self.assertIn("results", lens(False, None, 0))
        no_model = Fixture(GAPS)
        try:
            self.assertEqual(no_model.discover.gaps(True, None, 0), {"unavailable": "no model in this test"})
            self.assertEqual(no_model.discover.surprises(True, None, 0), {"unavailable": "no model in this test"})
        finally:
            no_model.close()


# ----- meaning edges in the map ---------------------------------------------------------------------------
# P: four notes fully linked, three say "alpha beta", p4 "gamma gamma". lone has no link or tag and says
# "gamma gamma gamma": p4 and lone are each other's nearest neighbour.

MEANING = {**{f"p{i}.md": body + ("\ngamma gamma\n" if i == 4 else "\nalpha beta\n") for i, (name, body) in
              enumerate(sorted(group("p").items()), 1)},
           **words_group("q", "reward reward"), "lone.md": "gamma gamma gamma\n"}


class MeaningEdgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.model = Path(cls.tmp.name) / "tiny"
        make_model(cls.model)
        cls.plain = Fixture(MEANING, Model2VecProvider(cls.model))
        cls.meaning = Fixture(MEANING, Model2VecProvider(cls.model), meaning_percentile=10.0)

    @classmethod
    def tearDownClass(cls):
        cls.plain.close()
        cls.meaning.close()
        cls.tmp.cleanup()

    def test_the_lone_note_lands_with_its_nearest_neighbour(self):
        self.assertEqual([self.plain.name(v) for v in self.plain.shape.stranded], ["lone"])
        edges = {(self.meaning.name(a), self.meaning.name(b)) for a, b, _ in self.meaning.shape.meaning_edges}
        self.assertIn(("lone", "p4"), edges)
        self.assertEqual(self.meaning.shape.stranded, [])
        m = self.meaning.shape.membership
        self.assertEqual(m[self.meaning.node("lone")], m[self.meaning.node("p4")])
        from vaultengine.shape import compare

        changes = compare(self.plain.shape, self.meaning.shape)
        self.assertEqual([p["note"] for p in changes["placed"]], ["lone.md"])

    def test_meaning_edges_leave_bridges_on_links(self):
        self.assertEqual(self.meaning.shape.betweenness.values, self.plain.shape.betweenness.values)


# ----- bridge and same role -----------------------------------------------------------------------------------


class BridgeAndRoleTests(unittest.TestCase):
    def test_the_joining_note_carries_every_path_between_the_groups(self):
        from test_vaultengine_shape import planted

        f = Fixture(planted())
        try:
            x = f.node("X")
            result = f.discover.bridge(x, 0)
            side = {f.shape.membership[f.node("a3")], f.shape.membership[f.node("b3")]}
            across = next(b for b in result["brokers"] if set(b["communities"]) == side)
            # Every shortest path between a note of one group and a note of the other runs through X.
            self.assertEqual(across["share"], 1.0)
            self.assertEqual(across["pairs"], 8 * 8)  # X is the note itself, never an end of a pair
            self.assertEqual(result["betweenness_percentile"], 100.0)
            self.assertTrue(result["exact"])
        finally:
            f.close()

    def test_the_same_position_in_another_community(self):
        # Two separate groups built the same way: a hub linked from three notes that also link each other in a chain.
        files = {}
        for g in "gh":
            files[f"{g}hub.md"] = "hub\n"
            files[f"{g}1.md"] = f"[[{g}hub]] [[{g}2]]\n"
            files[f"{g}2.md"] = f"[[{g}hub]] [[{g}3]]\n"
            files[f"{g}3.md"] = f"[[{g}hub]]\n"
        f = Fixture(files, percentile_cut=50.0)
        try:
            result = f.discover.same_role(f.node("ghub"), 0)
            self.assertEqual(result["results"][0]["note"], "hhub.md")
            self.assertEqual(result["results"][0]["distance"], 0.0)  # every feature the same
            self.assertNotIn("g1.md", [r["note"] for r in result["results"]])  # same community: never
        finally:
            f.close()


# ----- the CLI: the same output every run ---------------------------------------------------------------------


class DiscoverCliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vault = TempVault(HIDDEN)
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

    def test_every_lens_and_the_same_output_twice(self):
        code, out, err = self.run_cli("discover", str(self.vault.root), "A", "--color", "never",
                                      "--setting", "explore.percentile=50")
        self.assertEqual((code, err), (0, ""))
        for title in ("  Hidden link", "  Same role elsewhere", "  Bridge", "  Gap", "  Surprise", "  Unwritten", "  Wander"):
            self.assertIn(title, out)
        self.assertIn("    with embeddings: unavailable (embedding model folder not found: /no/model)", out)
        self.assertIn("      1. C\n", out)
        again = self.run_cli("discover", str(self.vault.root), "A", "--color", "never", "--setting", "explore.percentile=50")
        self.assertEqual(again[1], out)
        code, out, _ = self.run_cli("discover", str(self.vault.root), "--json")
        self.assertEqual(list(json.loads(out)["lenses"]), ["gap", "surprise", "unwritten"])
        code, _, err = self.run_cli("discover", str(self.vault.root), "--lens", "wander")
        self.assertEqual(code, 2)
        self.assertIn("wander start from a note", err)


if __name__ == "__main__":
    unittest.main()
