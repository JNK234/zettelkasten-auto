"""How two notes relate (`vault explain`): every signal checked on a small hand-made vault.

The expected values are computed here, from the tables below (what links to what, which tags
and words each note has), with math and numpy only: nothing from graphrank.py, related.py or
relations.py is called to get them. The text signal uses a tiny Model2Vec model written by
hand (test_vaultengine_embedding.make_model), so every vector is known.
"""

import io
import json
import math
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np

from vaultengine.cli import main as cli_main
from vaultengine.embedding import EmbeddingError
from vaultengine.embedding.model2vec import Model2VecProvider
from vaultengine.index import VaultIndex, default_db_path
from vaultengine.querylang import QueryError
from vaultengine.related import Relatedness
from vaultengine.relations import Relations

from test_vaultengine_embedding import NO_ENGINE_ENV, WORDS, make_model
from test_vaultengine_search import TempVault

# ----- the vault, as tables -------------------------------------------------------------------
# Links (source -> targets, in order; "gone" has no note). Words: the body text the tiny model
# knows (alpha, beta, gamma, reward = re + ##ward); titles (single letters), [[links]] and tags
# are not in its vocabulary or not embedded. #common is on 5 of 12 notes: more than a third.

LINKS = {
    "A": ["B", "X", "Y", "gone"],
    "B": ["X", "gone"],
    "C": ["A", "B"],
    "D": ["A", "B", "X", "Y"],
    "E": ["A"],
    "F": ["X"],
}
TAGS = {"A": ["rl", "rare", "common"], "B": ["rl", "rare/deep", "common"], "C": ["common"], "D": ["common"],
        "E": ["common"], "F": ["rl"]}
WORDS_OF = {"A": "alpha beta alpha", "B": "alpha gamma reward", "C": "gamma", "D": "beta beta", "E": "reward",
            "F": "beta", "X": "alpha", "Y": "gamma gamma", "G": "beta gamma", "H": "reward alpha", "I": "", "J": "gamma"}
NOTES = sorted(WORDS_OF)  # 12 notes


def note_file(name: str) -> str:
    links = " ".join(f"[[{t}]]" for t in LINKS.get(name, []))
    tags = " ".join(f"#{t}" for t in TAGS.get(name, []))
    return f"{WORDS_OF[name]}\n\n{links}\n\n{tags}\n".replace("\n\n\n", "\n\n")


def citers(node):  # notes that link to node
    return {s for s, targets in LINKS.items() if node in targets}


def references(node):  # what node links to
    return set(LINKS.get(node, []))


def expected_cocitation(a, b):
    w = lambda c: 1 / math.log(1 + len(set(LINKS[c])))
    both = (citers(a) & citers(b)) - {a, b}
    total = lambda x: sum(w(c) for c in citers(x))
    return sum(w(c) for c in both) / math.sqrt((total(a) + 1) * (total(b) + 1)), {c: w(c) for c in both}


def expected_coupling(a, b):
    w = lambda z: 1 / math.log(1 + len(citers(z)))
    both = references(a) & references(b)
    total = lambda x: sum(w(z) for z in references(x))
    return sum(w(z) for z in both) / math.sqrt((total(a) + 1) * (total(b) + 1)), {z: w(z) for z in both}


def tag_sets():
    expanded = {n: {"/".join(t.split("/")[:d]) for t in TAGS.get(n, []) for d in range(1, t.count("/") + 2)} for n in NOTES}
    counts = {}
    for tags in expanded.values():
        for t in tags:
            counts[t] = counts.get(t, 0) + 1
    idf = {t: math.log(len(NOTES) / n) for t, n in counts.items()}
    kept = {n: {t for t in tags if counts[t] <= len(NOTES) / 3} for n, tags in expanded.items()}
    return kept, idf


def expected_tags(a, b):
    kept, idf = tag_sets()
    union = sum(idf[t] for t in kept[a] | kept[b])
    return (sum(idf[t] for t in kept[a] & kept[b]) / union if union else 0.0), kept


class TinyText:
    """Vectors and word credits by hand, from the tiny model's matrix."""

    def __init__(self, matrix):
        self.matrix = matrix.astype(np.float64)
        self.ids = {w: i + 2 for i, w in enumerate(WORDS)}

    def tokens(self, text):
        out = []
        for word in text.split():
            out += [self.ids["re"], self.ids["##ward"]] if word == "reward" else [self.ids[word]]
        return out

    def vector(self, name):
        ids = self.tokens(WORDS_OF[name])
        if not ids:
            return None
        v = self.matrix[ids].mean(axis=0)
        return v / np.linalg.norm(v)

    def mean(self):
        return np.mean([v for v in (self.vector(n) for n in NOTES) if v is not None], axis=0)

    def cosine(self, a, b):
        mu = self.mean()
        u, v = self.vector(a) - mu, self.vector(b) - mu
        return float(u @ v / np.linalg.norm(u) / np.linalg.norm(v))

    def credits(self, a, b):
        """word -> (alpha_a E_w . u_b / (|raw_a| Z_a) + the same from b) / 2, words as the model splits them."""
        mu = self.mean()

        def side(x, y):
            ids = self.tokens(WORDS_OF[x])
            raw = self.matrix[ids].mean(axis=0)
            u_x = raw / np.linalg.norm(raw) - mu
            z_x = np.linalg.norm(u_x)
            u_y = self.vector(y) - mu
            u_y /= np.linalg.norm(u_y)
            out = {}
            words = WORDS_OF[x].split()
            for word in set(words):
                e = self.matrix[self.tokens(word)].sum(axis=0)
                alpha = words.count(word) / len(ids)  # each occurrence is 1/n of the mean, all its pieces together
                out[word] = alpha * float(e @ u_y) / (np.linalg.norm(raw) * z_x) / 2
            return out

        one, two = side(a, b), side(b, a)
        return {w: one.get(w, 0.0) + two.get(w, 0.0) for w in set(one) | set(two)}


WEIGHTS = {"direct": 0.3, "cocitation": 0.15, "coupling": 0.15, "tags": 0.1, "text": 0.3}


class HandMadeVaultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.model = Path(cls.tmp.name) / "tiny"
        cls.tiny = TinyText(make_model(cls.model))
        cls.vault = TempVault({f"{n}.md": note_file(n) for n in NOTES})
        cls.ix = VaultIndex(default_db_path(cls.vault.root))
        cls.ix.update(cls.vault.root)
        cls.relations = Relations(cls.ix.conn, Model2VecProvider(cls.model), cls.vault.root)

    @classmethod
    def tearDownClass(cls):
        cls.ix.close()
        cls.vault.cleanup()
        cls.tmp.cleanup()

    def explain(self, a, b):
        return self.relations.explain(a, b, words=20)

    def test_the_vault_is_what_the_tables_say(self):
        rows = self.ix.conn.execute(
            "SELECT s.path, l.target FROM links l JOIN files s ON s.id = l.source_id ORDER BY s.path, l.ordinal").fetchall()
        self.assertEqual([tuple(r) for r in rows], [(f"{s}.md", t) for s in sorted(LINKS) for t in LINKS[s]])
        self.assertEqual(len(NOTES), 12)

    def test_direct_links(self):
        rel = self.explain("A", "B")
        d = rel.signal("direct")
        self.assertEqual((d.value, d.evidence), (1.0, {"a_to_b": [3], "b_to_a": [], "a_to_b_tag_lines": [], "b_to_a_tag_lines": []}))
        self.assertEqual(self.explain("B", "A").signal("direct").evidence,
                         {"a_to_b": [], "b_to_a": [3], "a_to_b_tag_lines": [], "b_to_a_tag_lines": []})  # [[B]]: A's line 3
        self.assertEqual(self.explain("X", "Y").signal("direct").value, 0.0)

    def test_cocitation(self):
        for a, b in (("A", "B"), ("X", "Y"), ("A", "X"), ("X", "G")):
            with self.subTest(pair=(a, b)):
                value, both = expected_cocitation(a, b)
                signal = self.explain(a, b).signal("cocitation")
                self.assertAlmostEqual(signal.value, value, places=12)
                self.assertEqual({x["note"]: x["weight"] for x in signal.evidence["both"]},
                                 {f"{c}.md": pytest_approx(w) for c, w in both.items()})
        # A and B: C (links 2 notes) and D (links 4) link to both; A's own link to B is a direct link, not a co-citation.
        value, _ = expected_cocitation("A", "B")
        by_hand = (1 / math.log(3) + 1 / math.log(5)) / math.sqrt(
            (1 / math.log(3) + 1 / math.log(5) + 1 / math.log(2) + 1) * (1 / math.log(5) + 1 / math.log(3) + 1 / math.log(5) + 1))
        self.assertAlmostEqual(value, by_hand, places=12)
        self.assertEqual([x["note"] for x in self.explain("A", "B").signal("cocitation").evidence["both"]], ["C.md", "D.md"])

    def test_coupling(self):
        for a, b in (("A", "B"), ("A", "D"), ("B", "F"), ("C", "E"), ("X", "G")):
            with self.subTest(pair=(a, b)):
                value, both = expected_coupling(a, b)
                signal = self.explain(a, b).signal("coupling")
                self.assertAlmostEqual(signal.value, value, places=12)
                self.assertEqual({x["target"].removesuffix(".md"): x["weight"] for x in signal.evidence["both"]},
                                 {z: pytest_approx(w) for z, w in both.items()})
        # A and B both link X (linked from A, B, D, F) and the missing "gone" (A, B): the rarer one weighs more.
        both = self.explain("A", "B").signal("coupling").evidence["both"]
        self.assertEqual([(x["target"], x["missing"], x["linked_from"]) for x in both], [("gone", True, 2), ("X.md", False, 4)])
        self.assertAlmostEqual(both[0]["weight"], 1 / math.log(3), places=12)

    def test_shared_tags(self):
        for a, b in (("A", "B"), ("A", "F"), ("C", "D"), ("X", "G")):
            with self.subTest(pair=(a, b)):
                value, kept = expected_tags(a, b)
                self.assertAlmostEqual(self.explain(a, b).signal("tags").value, value, places=12)
        e = self.explain("A", "B").signal("tags").evidence
        # #rare counts for B through #rare/deep; #common is on 5 of 12 notes, more than a third: left out.
        self.assertEqual([x["tag"] for x in e["shared"]], ["rare", "rl"])
        self.assertEqual([x["tag"] for x in e["only_b"]], ["rare/deep"])
        self.assertEqual(e["left_out"], [{"tag": "common", "notes": 5}])
        self.assertAlmostEqual(e["shared"][0]["weight"], math.log(12 / 2), places=12)
        self.assertAlmostEqual(self.explain("A", "B").signal("tags").value,
                               (math.log(4) + math.log(6)) / (math.log(4) + math.log(6) + math.log(12)), places=12)

    def test_text(self):
        for a, b in (("A", "B"), ("A", "H"), ("C", "Y"), ("D", "F")):
            with self.subTest(pair=(a, b)):
                signal = self.explain(a, b).signal("text")
                cosine = self.tiny.cosine(a, b)
                self.assertAlmostEqual(signal.evidence["cosine"], cosine, places=5)
                self.assertAlmostEqual(signal.value, max(0.0, cosine), places=5)
                e = signal.evidence
                self.assertAlmostEqual(e["shared_total"] + e["one_note_total"] + e["baseline"], cosine, places=5)
                credits = self.tiny.credits(a, b)
                for item in e["words"]:
                    self.assertAlmostEqual(item["share"], credits[item["word"]], places=5)
        # A (alpha beta alpha) and B (alpha gamma reward) share one word.
        words = self.explain("A", "B").signal("text").evidence["words"]
        self.assertEqual([x["word"] for x in words], ["alpha"])
        # A note with no word the model knows has no direction: no text signal with anything.
        self.assertEqual(self.explain("I", "A").signal("text").value, 0.0)

    def test_score_is_the_weighted_sum(self):
        rel = self.explain("A", "B")
        expected = (WEIGHTS["direct"] * 1.0 + WEIGHTS["cocitation"] * expected_cocitation("A", "B")[0]
                    + WEIGHTS["coupling"] * expected_coupling("A", "B")[0] + WEIGHTS["tags"] * expected_tags("A", "B")[0]
                    + WEIGHTS["text"] * max(0.0, self.tiny.cosine("A", "B")))
        self.assertAlmostEqual(rel.score, expected, places=5)
        for s in rel.signals:
            self.assertEqual(s.weight, WEIGHTS[s.name])
            self.assertAlmostEqual(s.contribution, s.weight * s.value, places=12)
        self.assertEqual(rel.signal("text").evidence["model_id"], "tiny")

    def test_other_weights(self):
        weights = {"direct": 0.0, "cocitation": 1.0, "coupling": 1.0, "tags": 0.0, "text": 0.0}
        rel = Relations(self.ix.conn, Model2VecProvider(self.model), self.vault.root, weights).explain("A", "B")
        self.assertAlmostEqual(rel.score, (expected_cocitation("A", "B")[0] + expected_coupling("A", "B")[0]) / 2, places=12)

    def test_without_the_model(self):
        def broken():
            raise EmbeddingError("model.safetensors: not a safetensors file")

        rel = Relations(self.ix.conn, broken, self.vault.root).explain("A", "B")
        text = rel.signal("text")
        self.assertEqual((text.available, text.unavailable, text.weight, text.value),
                         (False, "model.safetensors: not a safetensors file", 0.0, 0.0))
        rest = {k: w / 0.7 for k, w in WEIGHTS.items() if k != "text"}  # the other four share the weight
        self.assertAlmostEqual(rel.score, rest["direct"] * 1.0 + rest["cocitation"] * expected_cocitation("A", "B")[0]
                               + rest["coupling"] * expected_coupling("A", "B")[0] + rest["tags"] * expected_tags("A", "B")[0],
                               places=12)
        self.assertIsNone(rel.model_id)

    def test_refusals(self):
        with self.assertRaisesRegex(QueryError, '"gone" is not a note yet'):
            self.relations.explain("A", "gone")
        with self.assertRaisesRegex(QueryError, "are the same note"):
            self.relations.explain("A", "A.md")
        with self.assertRaisesRegex(QueryError, 'no note or link target called "nope"'):
            self.relations.explain("A", "nope")

    def test_cli(self):
        def run(*args, env=None):
            out, err = io.StringIO(), io.StringIO()
            with patch.dict(os.environ, {**NO_ENGINE_ENV, "VAULTENGINE_EMBEDDING__MODEL_PATH": str(self.model), **(env or {})},
                            clear=True), redirect_stdout(out), redirect_stderr(err):
                code = cli_main(["explain", str(self.vault.root), *args])
            return code, out.getvalue(), err.getvalue()

        code, out, err = run("A", "B", "--color", "never")
        self.assertEqual((code, err), (0, ""))
        lines = out.splitlines()
        self.assertEqual(lines[0], "How A and B relate")
        self.assertIn("  direct links    1.00 ×   0.30 =        0.300", lines)
        self.assertIn("    A → B: 1 link (line 3)", lines)
        self.assertIn("    B → A: none", lines)
        self.assertIn("    C  0.91  (links to 2)", lines)
        self.assertIn("    gone (no note yet)  0.91  (linked from 2)", lines)
        self.assertIn("    left out (on more than a third of the notes): #common (5 notes)", lines)
        self.assertTrue(any(line.startswith("    words in both notes, by their share of the cosine: alpha +") for line in lines))
        self.assertIn("12 notes had no cached vector: embedded in memory for this run only", out)
        code, out, _ = run("A", "B", "--json")
        data = json.loads(out)
        self.assertEqual([s["name"] for s in data["signals"]], ["direct", "cocitation", "coupling", "tags", "text"])
        self.assertEqual(data["configured_weights"], WEIGHTS)
        code, out, err = run("A", "B", "--color", "never", env={"VAULTENGINE_EMBEDDING__MODEL_PATH": "/nope"})
        self.assertEqual(code, 0, err)
        self.assertIn("  text               —        —    unavailable", out.splitlines())
        self.assertIn("  direct links    1.00 ×   0.43 =        0.429", out.splitlines())  # 0.3 / 0.7
        self.assertIn("  text  unavailable  (embedding model folder not found: /nope)", out.splitlines())
        code, _, err = run("A", "gone")
        self.assertEqual(code, 2)
        self.assertIn('error: "gone" is not a note yet', err)


# ----- tag links ----------------------------------------------------------------------------
# P and Q are both tagged [[topic]] (a missing note) on their Tags: line, and both link R in the body;
# P is also tagged with the note Q. Five filler notes make N = 8.

TAGGED = {
    "P.md": "Tags: [[topic]] [[Q]]\n\nSee [[R]].\n",
    "Q.md": "Tags: [[topic]]\n\nSee [[R]].\n",
    "R.md": "r\n", **{f"Z{i}.md": "z\n" for i in range(5)},
}


class TagLinkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vault = TempVault(TAGGED)
        cls.ix = VaultIndex(default_db_path(cls.vault.root))
        cls.ix.update(cls.vault.root)

    @classmethod
    def tearDownClass(cls):
        cls.ix.close()
        cls.vault.cleanup()

    def explain(self, a, b, rule=None):
        def off():
            raise EmbeddingError("no model in this test")

        return Relations(self.ix.conn, off, self.vault.root, tag_rule=rule).explain(a, b)

    def test_tags_line_links_are_tags_not_coupling(self):
        rel = self.explain("P", "Q")
        # Coupling: only R (a body link of both), linked from 2 notes: w = 1/ln 3 on each side.
        w = 1 / math.log(3)
        self.assertAlmostEqual(rel.signal("coupling").value, w / math.sqrt((w + 1) * (w + 1)), places=12)
        self.assertEqual([x["target"] for x in rel.signal("coupling").evidence["both"]], ["R.md"])
        # Tags: P {[[topic]], [[q]]}, Q {[[topic]]}; [[topic]] on 2 of 8 notes, [[q]] on 1.
        self.assertAlmostEqual(rel.signal("tags").value, math.log(8 / 2) / (math.log(8 / 2) + math.log(8 / 1)), places=12)
        self.assertEqual([x["tag"] for x in rel.signal("tags").evidence["shared"]], ["[[topic]]"])
        # P's tag link to Q still links the two (line 1), marked as a tag link.
        d = rel.signal("direct").evidence
        self.assertEqual((rel.signal("direct").value, d["a_to_b"], d["a_to_b_tag_lines"]), (1.0, [1], [1]))
        # Co-citation leaves tag links out too: nothing links P and Q together in a body.
        self.assertEqual(rel.signal("cocitation").value, 0.0)

    def test_the_index_records_the_field_of_each_link(self):
        rows = self.ix.conn.execute(
            """SELECT f.path, l.target, l.field_source, l.field_key FROM links l JOIN files f ON f.id = l.source_id
               WHERE f.path IN ('P.md', 'Q.md') ORDER BY f.path, l.ordinal""").fetchall()
        self.assertEqual([tuple(r) for r in rows], [("P.md", "topic", "header", "Tags"), ("P.md", "Q", "header", "Tags"),
                                                    ("P.md", "R", None, None), ("Q.md", "topic", "header", "Tags"),
                                                    ("Q.md", "R", None, None)])

    def test_related_names_a_shared_tag_link(self):
        engine = Relatedness(self.ix.conn, lambda: (_ for _ in ()).throw(EmbeddingError("off")), self.vault.root)
        hits = {h.title: h for h in engine.related(engine.seed("Q"), limit=None).hits}
        self.assertIn("shares [[topic]]", hits["P"].reason)
        self.assertTrue(hits["P"].linked)  # P's tag link to Q links them

    def test_a_format_decides_which_fields_are_tags(self):
        from vaultengine.formats.spec import parse_format
        from vaultengine.tagrule import format_rule

        spec = parse_format("format: 1\ntypes:\n  note:\n    match: {name: '*'}\n    metadata:\n"
                            "      Tags: {kind: links}\n      Topics: {kind: links, link_role: tag}\n")
        rule = format_rule(spec)
        self.assertTrue(rule.description.endswith("(Topics)"), rule.description)
        self.assertFalse(rule.is_tag("P.md", "header", "Tags"))  # this format gives Tags no tag role
        self.assertTrue(rule.is_tag("P.md", "header", "topics"))
        rel = self.explain("P", "Q", rule)
        self.assertEqual(rel.signal("tags").value, 0.0)  # so [[topic]] is a shared reference again
        self.assertEqual(sorted(x["target"] for x in rel.signal("coupling").evidence["both"]), ["R.md", "topic"])

    def test_brain_gives_every_type_a_tag_field(self):
        from vaultengine.tagrule import rule_for_format

        rule = rule_for_format("brain")
        self.assertEqual(rule.description, "format brain (Tags)")
        for path in ("20_ZETTLEKASTEN/x.md", "10_SOURCES/x.md", "50_MOC/topics/x.md", "50_MOC/discourse/d/EVD - x.md"):
            self.assertTrue(rule.is_tag(path, "header", "Tags"), path)
            self.assertFalse(rule.is_tag(path, "inline", "Tags"), path)  # a header field, not tags::
        self.assertFalse(rule.is_tag("untyped/x.md", "header", "Tags"))


def pytest_approx(value):
    """Float equality to 12 places inside assertEqual on dicts."""

    class Approx(float):
        def __eq__(self, other):
            return abs(float(self) - other) < 1e-12

        __hash__ = float.__hash__

    return Approx(value)


if __name__ == "__main__":
    unittest.main()
