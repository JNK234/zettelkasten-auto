"""`vault experiment`: the statistics by hand, the harness on synthetic vaults with a planted answer, and the
proof that a hidden link cannot leak into the scores that are meant to find it."""

import io
import json
import random
import re
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np

from vaultengine import experiment as ex
from vaultengine.cli import main as cli_main
from vaultengine.index import VaultIndex, default_db_path
from vaultengine.tagrule import default_rule

from embedding_model import requires_model

CLUSTERS = 6
PER = 10
TOPIC = [[f"topic{k}word{j}" for j in range(6)] for k in range(CLUSTERS)]
NOISE = [f"noise{j}" for j in range(40)]
VOCAB = [w for words in TOPIC for w in words] + NOISE
SEEDS = [1, 2, 3, 4, 5]


class BagProvider:
    """A stand-in embedding model: the counts of the VOCAB words, normalised. Other words are unknown."""

    model_id = name = "bag"
    dim = len(VOCAB)
    index = {w: i for i, w in enumerate(VOCAB)}

    def embed(self, texts):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                if word in self.index:
                    out[row, self.index[word]] += 1
            norm = np.linalg.norm(out[row])
            if norm:
                out[row] /= norm
        return out


def note_name(k, i):
    return f"note-{k}-{i}"


def plant(root: Path, *, text="topic", links="sparse", boilerplate=False, tags=False, seed=1):
    """A vault of CLUSTERS × PER notes with one thing planted.

    text: "topic" (four of the cluster's words) or "noise" (four noise words: text says nothing).
    links: "dense" (three cluster mates each: links find the hidden partner), "sparse" (one cluster mate and
    one note anywhere: links say little) or "pairs" (disjoint pairs inside each cluster).
    boilerplate: a ``Kind:`` metadata line of 20 noise words and a short noise first section before the idea,
    what ``no_template`` leaves out and ``first_section`` keeps. tags: ``Tags: [[cluster-k]]``.
    """
    rng = random.Random(seed)
    others = [(k, i) for k in range(CLUSTERS) for i in range(PER)]
    for k in range(CLUSTERS):
        for i in range(PER):
            mates = [j for j in range(PER) if j != i]
            if links == "dense":
                targets = [note_name(k, j) for j in rng.sample(mates, 3)]
            elif links == "sparse":
                targets = [note_name(k, rng.choice(mates)), note_name(*rng.choice([o for o in others if o != (k, i)]))]
            else:
                targets = [note_name(k, i + 1)] if i % 2 == 0 else []
            words = rng.sample(TOPIC[k], 4) if text == "topic" else rng.sample(NOISE, 4)
            lines = [f"Tags: [[cluster-{k}]]"] if tags else []
            if boilerplate:
                lines += ["Kind: " + " ".join(rng.sample(NOISE, 20)), "", "## Intro", "", " ".join(rng.sample(NOISE, 2))]
            lines += ["", "## Idea", "", " ".join(words) + ". " + " ".join(f"See [[{t}]]." for t in targets), ""]
            (root / f"{note_name(k, i)}.md").write_text("\n".join(lines), encoding="utf-8")


class PlantedVault:
    def __init__(self, **kwargs):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "vault"
        self.root.mkdir()
        plant(self.root, **kwargs)
        self.ix = VaultIndex(default_db_path(self.root))
        self.ix.update(self.root)

    def run(self, today, **kwargs):
        args = dict(tag_rule=default_rule(), spec=None, provider=BagProvider(), related_weights=today,
                    text_variant="whole", resolution=1.0, without=[], seeds=SEEDS)
        args.update(kwargs)
        return ex.run(self.ix.conn, self.root, **args)

    def close(self):
        self.ix.close()
        self._tmp.cleanup()


def decision(out, setting):
    return next(d for d in out["decisions"] if d["setting"] == setting)


class StatisticsTests(unittest.TestCase):
    def test_nmi_and_ari_by_hand(self):
        same, crossed = ([0, 0, 1, 1], [5, 5, 7, 7]), ([0, 0, 1, 1], [0, 1, 0, 1])
        self.assertAlmostEqual(ex.nmi(*same), 1.0)
        self.assertAlmostEqual(ex.ari(*same), 1.0)
        self.assertAlmostEqual(ex.nmi(*crossed), 0.0)
        # ARI: no pair together in both; expected 2 × 2 / 6; max (2 + 2) / 2: (0 - 2/3) / (2 - 2/3) = -1/2.
        self.assertAlmostEqual(ex.ari(*crossed), -0.5)
        # fine refines coarse (x, y inside a; z is b), so MI = H(coarse) and NMI = 2 H(c) / (H(c) + H(f)).
        coarse, fine = list("aaaabb"), list("xxyyzz")
        h = lambda *p: -sum(q * np.log(q) for q in p)
        hc, hf = h(4 / 6, 2 / 6), h(1 / 3, 1 / 3, 1 / 3)
        self.assertAlmostEqual(ex.nmi(coarse, fine), 2 * hc / (hc + hf))

    def test_nmi_and_ari_agree_with_scikit_learn(self):
        try:
            from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
        except ImportError:
            self.skipTest("scikit-learn is not installed")
        rng = random.Random(3)
        for trial in range(30):
            n = rng.randint(2, 60)
            a = [rng.randrange(rng.randint(1, 6)) for _ in range(n)]
            b = [rng.randrange(rng.randint(1, 8)) for _ in range(n)]
            with self.subTest(trial=trial):
                self.assertAlmostEqual(ex.ari(a, b), adjusted_rand_score(a, b), places=10)
                self.assertAlmostEqual(ex.nmi(a, b), normalized_mutual_info_score(a, b), places=10)

    def test_clear_means_ahead_on_every_seed_and_beyond_twice_the_standard_error(self):
        self.assertTrue(ex.compare([0.5, 0.6, 0.55], [0.4, 0.5, 0.44]).clear)
        self.assertFalse(ex.compare([0.5, 0.6, 0.55], [0.4, 0.61, 0.44]).clear)  # behind on one seed
        self.assertFalse(ex.compare([0.5, 0.9, 0.41], [0.4, 0.5, 0.4]).clear)  # ahead, but the spread is wide
        self.assertFalse(ex.compare([0.5], [0.1]).clear)  # one seed says nothing about the noise

    def test_expected_rank_counts_ties_at_half(self):
        scores = {1: 0.9, 2: 0.5, 3: 0.5, 4: 0.5, 5: 0.1}
        self.assertEqual(ex.expected_rank(scores, 3, [1, 2, 3, 4, 5]), 3.0)  # one above, two tied
        self.assertEqual(ex.expected_rank(scores, 9, [1, 2, 3, 4, 5, 9]), 6.0)  # no score: below every scored note


class PlantedAnswerTests(unittest.TestCase):
    """The harness must find what was planted, clearly, and change nothing where nothing was planted."""

    def test_text_planted_text_weighted_mixes_win(self):
        vault = PlantedVault(text="topic", links="sparse")
        try:
            out = vault.run({"graph": 1.0}, questions=("relatedness",))
        finally:
            vault.close()
        results = out["datasets"][0]["relatedness"]["results"]
        self.assertTrue(ex.compare(results["text only"].mrr, results["links only"].mrr).clear)
        chosen = decision(out, "related.unlinked_weights")
        self.assertTrue(chosen["change"])
        self.assertGreaterEqual(chosen["to"]["text"], 0.5)

    def test_links_planted_links_win(self):
        vault = PlantedVault(text="noise", links="dense")
        try:
            out = vault.run({"text": 1.0}, questions=("relatedness",))
        finally:
            vault.close()
        results = out["datasets"][0]["relatedness"]["results"]
        self.assertTrue(ex.compare(results["links only"].mrr, results["text only"].mrr).clear)
        chosen = decision(out, "related.unlinked_weights")
        self.assertTrue(chosen["change"])
        self.assertEqual(chosen["to"]["text"], 0.0)

    def test_boilerplate_planted_no_template_wins(self):
        vault = PlantedVault(text="topic", links="sparse", boilerplate=True)
        try:
            out = vault.run({"graph": 1.0}, questions=("text",))
        finally:
            vault.close()
        results = out["datasets"][0]["text"]["results"]
        whole, bare, first = (results[f"text only · {v}"] for v in ("whole", "no_template", "first_section"))
        self.assertTrue(ex.compare(bare.mrr, whole.mrr).clear)
        self.assertTrue(ex.compare(bare.mrr, first.mrr).clear)  # the first section is the noise intro
        chosen = decision(out, "embedding.text")
        self.assertEqual((chosen["change"], chosen["to"]), (True, "no_template"))

    def test_nothing_planted_nothing_changes(self):
        # No template, no metadata, one section: the three texts are the same, so are their scores.
        vault = PlantedVault(text="topic", links="sparse")
        try:
            out = vault.run({"graph": 1.0}, questions=("text",))
        finally:
            vault.close()
        results = out["datasets"][0]["text"]["results"]
        self.assertEqual(results["text only · whole"].mrr, results["text only · no_template"].mrr)
        self.assertEqual(decision(out, "embedding.text")["change"], False)

    def test_meaning_edges_planted_they_win(self):
        # Links only join pairs; the clusters are in the text and in the (held out) tags.
        vault = PlantedVault(text="topic", links="pairs", tags=True)
        try:
            out = vault.run({"graph": 1.0}, questions=("communities",))
        finally:
            vault.close()
        q = out["datasets"][0]["communities"]
        self.assertEqual((q["tag_labelled"], q["tag_labels"]), (CLUSTERS * PER, CLUSTERS))
        plain = q["results"]["no meaning edges · resolution 1"]
        self.assertLess(max(plain.ari_tags), 0.3)  # 30 pairs, not 6 clusters
        chosen = decision(out, "explore (meaning edges, resolution)")
        self.assertTrue(chosen["change"])
        self.assertTrue(chosen["to"]["meaning"])
        best = q["results"][chosen["option"]]
        self.assertGreater(min(best.ari_tags), 0.6)

    def test_same_vault_and_seeds_give_the_same_numbers(self):
        vault = PlantedVault(text="topic", links="sparse", boilerplate=True, tags=True)
        try:
            runs = [vault.run({"graph": 0.5, "tags": 0.2, "text": 0.3}) for _ in range(2)]
        finally:
            vault.close()

        def numbers(out):
            got = []
            for d in out["datasets"]:
                for q in ("relatedness", "text"):
                    got += [(n, r.mrr, r.r5, r.r10) for n, r in d[q]["results"].items()]
                got += [(n, r.ari_tags, r.nmi_tags, r.stranded, r.stability) for n, r in d["communities"]["results"].items()]
            return got, out["decisions"]

        self.assertEqual(numbers(runs[0]), numbers(runs[1]))
        strip = lambda md: re.sub(r"\d+\.\d\d \|", "t |", re.sub(r"Whole run: \d+ s|Time: .*", "", md))  # timings aside
        self.assertEqual(strip(ex.markdown(runs[0], "v", "d", 1.0)), strip(ex.markdown(runs[1], "v", "d", 2.0)))


class NoLeakTests(unittest.TestCase):
    """Hiding a link must leave exactly the vault in which it was never written: every signal from the note
    that wrote it, to every note, is the same as in a vault whose files never had the link."""

    LINKED = {
        "a.md": "Tags: [[b]], [[topic]]\n\n# A\n\ntopic0word1 topic0word2 noise1. See [[b]] and [[c]]. #shared\n",
        "b.md": "Tags: [[topic]]\n\ntopic0word1 noise2, back to [[a|noise5 noise6 topic0word4]]. Also [[d]].\n",
        "c.md": "topic0word3 noise1 [[d]] [[b]] #shared\n",
        "d.md": "topic1word1 [[e]]\n",
        "e.md": "topic0word2 noise3 [[a]] [[b|noise5]]\n",
        "f.md": "noise4 [[b]] [[c]]\n",
    }

    def write(self, files):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        for path, text in files.items():
            (root / path).write_text(text, encoding="utf-8")
        ix = VaultIndex(default_db_path(root))
        self.addCleanup(ix.close)
        ix.update(root)
        return root, ix

    def signals(self, files, hidden_paths):
        root, ix = self.write(files)
        data = ex.load_dataset(ix.conn, root, "v")
        by_path = {p: f for f, p in data.notes}
        hidden = {frozenset(by_path[p] for p in pair) for pair in hidden_paths}
        held = ex.held_out(data, hidden, default_rule())
        texts = ex.Texts(data, BagProvider(), ex.template_rule(ix.conn, data))
        vectors = {v: texts.matrix(v, held.blanked) for v in ex.TEXT_VARIANTS}
        raw, candidates = ex.query_signals(data, held, by_path["a.md"], {"graph", "cocitation", "coupling"}, vectors)
        named = {k: {data.path_of[f]: v for f, v in values.items()} for k, values in raw.items()}
        words = {v: texts.text(by_path["a.md"], v, held.blanked.get(by_path["a.md"], ())) for v in ex.TEXT_VARIANTS}
        return named, sorted(data.path_of[f] for f in candidates), words, data

    def never_written(self):
        # The same files with every link between a and b replaced by spaces (what blanking does).
        out = dict(self.LINKED)
        for path, link in (("a.md", "[[b]]"), ("b.md", "[[a|noise5 noise6 topic0word4]]")):
            out[path] = out[path].replace(link, " " * len(link))
        return out

    def test_hidden_link_scores_equal_a_vault_without_it(self):
        hidden, hidden_candidates, hidden_words, data = self.signals(self.LINKED, [("a.md", "b.md")])
        spans = data.spans[frozenset(f for f, p in data.notes if p in ("a.md", "b.md"))]
        self.assertEqual(len(spans), 3)  # a -> b twice (Tags: and body), b -> a once: all three hidden
        never, never_candidates, never_words, _ = self.signals(self.never_written(), [])
        self.assertEqual(hidden_words, never_words)
        self.assertEqual(hidden_candidates, never_candidates)
        self.assertIn("b.md", hidden_candidates)
        self.assertEqual(sorted(hidden), sorted(never))
        for signal in sorted(never):
            with self.subTest(signal=signal):
                self.assertEqual(sorted(hidden[signal]), sorted(never[signal]))
                for path, value in never[signal].items():
                    self.assertAlmostEqual(hidden[signal][path], value, delta=1e-12)

    def test_the_link_does_count_when_it_is_not_hidden(self):
        # Not a vacuous proof: with the link in place, b is no candidate and its signals differ.
        shown, candidates, _, _ = self.signals(self.LINKED, [])
        never, _, _, _ = self.signals(self.never_written(), [])
        self.assertNotIn("b.md", candidates)
        self.assertNotEqual(shown["text:whole"].get("b.md"), never["text:whole"].get("b.md"))  # the alias's words
        self.assertNotEqual(shown["graph"].get("b.md"), never["graph"].get("b.md"))
        self.assertNotEqual(shown["tags"].get("b.md"), never["tags"].get("b.md"))


class MathAndTextVariantTests(unittest.TestCase):
    FILES = {
        "Policy.md": "# Policy\n\nThe gradient $\\nabla_\\theta \\log \\pi_\\theta(a|s)$ weights the advantage.\n\n"
                     "$$\n\\mathrm{KL} = \\frac{p}{q}\n$$\n\nAdvantage estimates cost $5 and $10.\n",
        "Other 1.md": "Advantage and gradient notes, no math. mathrm is written here as a word.\n",
        "Other 2.md": "Policy words: frac.\n",
        "Other 3.md": "Nothing here.\n",
        "Other 4.md": "Nothing here either.\n",
    }

    def index(self, files):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        for path, text in files.items():
            (root / path).write_text(text, encoding="utf-8")
        ix = VaultIndex(default_db_path(root))
        self.addCleanup(ix.close)
        ix.update(root)
        return root, ix

    def test_math_is_not_a_key_term(self):
        from vaultengine.suggest import KeyTerms

        _, ix = self.index(self.FILES)
        policy = ix.conn.execute("SELECT id FROM files WHERE path = 'Policy.md'").fetchone()[0]
        math = {t for (t,) in ix.conn.execute("SELECT term FROM math_terms WHERE file_id = ?", (policy,))}
        self.assertTrue({"mathrm", "frac", "nabla_", "theta"} <= math, math)  # terms as search tokenizes them
        self.assertFalse({"advantage", "cost"} & math)  # "$5 and $10" is prices, not math
        terms = KeyTerms(ix.conn, 5).of(policy)
        self.assertFalse(math & set(terms), terms)
        self.assertIn("advantage", terms)
        # In a note where they are words, they stay terms (only the math of each note is taken out).
        other = ix.conn.execute("SELECT id FROM files WHERE path = 'Other 1.md'").fetchone()[0]
        self.assertIn("mathrm", KeyTerms(ix.conn, 5).of(other))

    def test_embedding_text_variants(self):
        from vaultengine.embedding.text import embedding_text
        from vaultengine.parser import parse_note_with_layout

        text = ("Status: draft words\nTags: [[topic]]\n\n# Title heading\n\nFirst section text.\n\n"
                "## Related zettels\n\n- [[other]]\n\n## Idea\n\nThe idea itself.\nkey:: value words\n")
        note, layout = parse_note_with_layout(text, path="Note.md")
        whole = embedding_text("Note.md", note, layout)
        self.assertEqual(whole, embedding_text("Note.md", note, layout, "whole", lambda h: True))  # whole ignores it
        for kept in ("Note", "Title heading", "Related zettels", "draft words", "First section text.", "The idea itself."):
            self.assertIn(kept, whole)
        bare = embedding_text("Note.md", note, layout, "no_template", lambda h: h == "Related zettels")
        self.assertNotIn("Related zettels", bare)  # the template heading
        for gone in ("draft", "value words"):  # metadata lines, values too
            self.assertNotIn(gone, bare)
        for kept in ("Note", "Title heading", "First section text.", "The idea itself.", "[[other]]"):
            self.assertIn(kept, bare)
        first = embedding_text("Note.md", note, layout, "first_section")
        self.assertEqual(first, "Note First section text.")
        with self.assertRaisesRegex(ValueError, "unknown embedding text variant"):
            embedding_text("Note.md", note, layout, "nope")


@requires_model
class CommandTests(unittest.TestCase):
    def test_vault_experiment_writes_the_report_and_the_decisions(self):
        vault = PlantedVault(text="topic", links="sparse", tags=True)
        try:
            report = vault.root.parent / "out" / "report.md"
            stdout, stderr = io.StringIO(), io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                code = cli_main(["experiment", str(vault.root), "--seeds", "2", "--question", "relatedness",
                                 "--output", str(report), "--json", "--quiet"])
        finally:
            text = report.read_text() if report.exists() else ""
            vault.close()
        self.assertEqual(code, 0, stderr.getvalue())
        got = json.loads(stdout.getvalue())
        self.assertEqual([d["setting"] for d in got["decisions"]], ["related.unlinked_weights"])
        self.assertIn("## Which signals rank related notes", text)
        self.assertIn("| links only |", text)


if __name__ == "__main__":
    unittest.main()
