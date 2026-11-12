"""Connecting notes: unlinked mentions and links to add, and the rules neither may break."""

import io
import math
import os
import re
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

from vaultengine.cli import main as cli_main
from vaultengine.connect import WEIGHTS, LinkSuggester, MentionFinder
from vaultengine.index import VaultIndex, default_db_path

from embedding_model import requires_model
from test_vaultengine_search import FIXTURES, TempVault

TARGET = "Notes/reward-hacking.md"

# Each source note says what must (or must not) be found in it.
MENTION_NOTES = {
    TARGET: "---\naliases: [spec gaming]\n---\nreward hacking is when a policy games its reward.\n",  # itself: never; "reward": yes
    "Notes/reward.md": "Reward.\n",
    "Notes/hacking.md": "Hacking.\n",
    "Plain.md": "We saw reward hacking in training.\nAlso Reward-Hacking again, and REWARD HACKING.\n",  # 3
    "Wrapped.md": "The reward\nhacking story.\n",  # 1: a line break is not a boundary
    "Heading.md": "# About reward hacking\nbody\n",  # 1: headings are text
    "Alias.md": "This is spec gaming, plainly.\n",  # 1: through the alias
    "Code.md": "`reward hacking` inline\n```\nreward hacking\n```\n%% reward hacking %%\n",  # 0
    "In links.md": (
        "See [[reward-hacking-and-more|reward hacking]] and [[other reward hacking note]]. ![[reward hacking.png]]\n"
        "[reward hacking](https://x.y) and https://example.com/reward-hacking and www.reward-hacking.com\n"
    ),  # 0
    "Tags and frontmatter.md": "---\nsummary: reward hacking\n---\n#reward-hacking #reward\n",  # 0
    "Linked.md": "[[reward-hacking]] is covered; reward hacking again here.\n",  # 0: already links it
    "Linked by heading.md": "[[reward-hacking#Section]] and reward hacking.\n",  # 0
    "Embedded.md": "![[reward-hacking]] then reward hacking.\n",  # 0
    "Linked by alias.md": "[[spec gaming]] and reward hacking.\n",  # 0
    "Not words.md": "reward hackings, rewards hacking, reward_hacking.\n",  # 0 of reward-hacking; 1 of reward
    "Across.md": "reward\n## Split\nhacking\n",  # 0 of reward-hacking (a heading between); 1 of reward
    "Broken.md": "reward [[x]] hacking\n",  # 0 of reward-hacking (a link between); 1 of reward
}


class MentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vault = TempVault(MENTION_NOTES)
        cls.ix = VaultIndex(default_db_path(cls.vault.root))
        cls.ix.update(cls.vault.root)
        cls.finder = MentionFinder(cls.ix.conn, cls.vault.root)
        cls.mentions = cls.finder.all()

    @classmethod
    def tearDownClass(cls):
        cls.ix.close()
        cls.vault.cleanup()

    def found(self, target):
        return sorted((m.source, m.line, m.text) for m in self.mentions if m.target == target)

    def test_exactly_the_plain_text_mentions(self):
        self.assertEqual(
            self.found(TARGET),
            [
                ("Alias.md", 1, "spec gaming"),
                ("Heading.md", 1, "reward hacking"),
                ("Plain.md", 1, "reward hacking"),
                ("Plain.md", 2, "REWARD HACKING"),
                ("Plain.md", 2, "Reward-Hacking"),
                ("Wrapped.md", 1, "reward\nhacking"),
            ],
        )

    def test_longest_name_wins_and_single_words_still_count(self):
        # "reward hacking" is never also a mention of the note "reward"; a lone "reward" is.
        self.assertEqual(
            self.found("Notes/reward.md"),
            [("Across.md", 1, "reward"), ("Broken.md", 1, "reward"), ("Not words.md", 1, "reward"), (TARGET, 4, "reward")],
        )

    def test_a_longer_match_takes_its_words_even_when_not_reported(self):
        # "reward hacking" is never also "hacking" - not even in Linked.md, where the longer match is
        # dropped because that note already links reward-hacking.
        self.assertEqual(
            self.found("Notes/hacking.md"),
            [("Across.md", 3, "hacking"), ("Broken.md", 1, "hacking"), ("Not words.md", 1, "hacking")],
        )

    def test_one_note(self):
        target = next(f for f, p in self.finder.paths.items() if p == TARGET)
        self.assertEqual(sorted((m.source, m.line, m.text) for m in self.finder.of(target)), self.found(TARGET))

    def test_never_linked_notes_self_code_or_links(self):
        sources = {m.source for m in self.mentions}
        for never in ("Code.md", "In links.md", "Tags and frontmatter.md", "Linked.md", "Linked by heading.md",
                      "Embedded.md", "Linked by alias.md"):
            self.assertNotIn(never, sources)
        self.assertFalse([m for m in self.mentions if m.source == m.target])

    def test_min_words(self):
        finder = MentionFinder(self.ix.conn, self.vault.root, min_words=2)
        self.assertEqual({m.target for m in finder.all()}, {TARGET})


# ----- links to add ----------------------------------------------------------------------
#
# A and B both link X and Y and the missing [[topic]]; C links D and E; F and G say the same
# things in words. Linked pairs with plenty of evidence: H->I (plain), J->K#heading, L embeds M,
# N links O by its alias, P has an ambiguous link that could be Q.

PAIR_NOTES = {
    "A.md": "[[X]] [[Y]] [[topic]]",
    "B.md": "[[X]] [[Y]] [[topic]]",
    "X.md": "x", "Y.md": "y",
    "C.md": "[[D]] and [[E]]",
    "D.md": "d", "E.md": "e",
    "F.md": "Gradient penalty schedules stabilise adversarial critics.",
    "G.md": "Adversarial critics need gradient penalty schedules.",
    "H.md": "[[I]] [[X]] [[Y]]", "I.md": "[[X]] [[Y]]",
    "J.md": "[[K#Part]] [[X]] [[Y]]", "K.md": "# Part\n[[X]] [[Y]]",
    "L.md": "![[M]] [[X]] [[Y]]", "M.md": "[[X]] [[Y]]",
    "N.md": "[[nickname]] [[X]] [[Y]]", "O.md": "---\naliases: [nickname]\n---\n[[X]] [[Y]]",
    "P.md": "[[Q]] [[X]] [[Y]]", "one/Q.md": "[[X]] [[Y]]", "two/Q.md": "q",
    "Z1.md": "filler words here", "Z2.md": "more filler words", "Z3.md": "other filler text",
}


@requires_model
class LinkSuggestionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vault = TempVault(PAIR_NOTES)
        cls.ix = VaultIndex(default_db_path(cls.vault.root))
        cls.ix.update(cls.vault.root)
        cls.suggester = LinkSuggester(cls.ix.conn)
        cls.pairs = {(s.a, s.b): s for s in cls.suggester.suggestions(limit=None)}

    @classmethod
    def tearDownClass(cls):
        cls.ix.close()
        cls.vault.cleanup()

    def test_never_existing_links_self_or_missing_notes(self):
        linked = [("H.md", "I.md"), ("J.md", "K.md"), ("L.md", "M.md"), ("N.md", "O.md"), ("P.md", "one/Q.md"), ("P.md", "two/Q.md")]
        for a, b in linked:
            self.assertNotIn((min(a, b), max(a, b)), self.pairs)
        self.assertTrue(all(a != b for a, b in self.pairs))
        self.assertTrue(all(a.endswith(".md") and b.endswith(".md") for a, b in self.pairs))
        self.assertTrue(all(a < b for a, b in self.pairs))  # each pair once
        self.assertNotIn("topic", {p for pair in self.pairs for p in pair})

    def test_shared_neighbours_by_hand(self):
        pair = self.pairs[("A.md", "B.md")]
        # Neighbours of A and of B: X, Y (each linked by 8 notes: A B H I J K L M N O P Q... count below), topic (2).
        degree = {n: self.ix.conn.execute("SELECT COUNT(DISTINCT source_id) FROM links WHERE target = ?", (n,)).fetchone()[0]
                  for n in ("X", "Y", "topic")}
        w = {n: 1 / math.log(1 + d) for n, d in degree.items()}
        total = sum(w.values())  # A's and B's neighbours are the same three
        self.assertAlmostEqual(pair.shared, total / math.sqrt((total + 1) * (total + 1)), delta=1e-12)
        self.assertEqual(pair.cited, 0.0)
        self.assertEqual(pair.shared_with[0], "topic (no note yet)")  # the rarest neighbour first
        self.assertTrue(pair.reason.startswith("shared neighbours: topic (no note yet), "))

    def test_cited_together_by_hand(self):
        pair = self.pairs[("D.md", "E.md")]
        weight = 1 / math.log(1 + 2)  # C links two notes
        self.assertAlmostEqual(pair.cited, weight / math.sqrt((weight + 1) * (weight + 1)), delta=1e-12)
        self.assertEqual(pair.shared, 0.0)  # C is their only neighbour, and it cites both: not counted twice
        self.assertEqual(pair.reason, "cited together by C")

    def test_similar_text(self):
        # F and G say the same thing in other word order, and have no link evidence at all.
        pair = self.pairs[("F.md", "G.md")]
        self.assertEqual((pair.shared, pair.cited), (0.0, 0.0))
        self.assertGreater(pair.text, 0.8)  # centered cosine of their embeddings
        self.assertTrue(pair.reason.startswith("similar text ("), pair.reason)
        self.assertAlmostEqual(pair.score, WEIGHTS["text"] * pair.text, delta=1e-12)
        both = set(PAIR_NOTES["F.md"].lower().rstrip(".").split()) & set(PAIR_NOTES["G.md"].lower().rstrip(".").split())
        self.assertTrue(pair.terms and set(pair.terms) <= both, pair.terms)
        # G is F's closest note by text.
        text = self.suggester.text_similarity()
        f = next(i for i, p in self.suggester.paths.items() if p == "F.md")
        closest = max((c, self.suggester.paths[b if a == f else a]) for (a, b), c in text.items() if f in (a, b))
        self.assertEqual(closest[1], "G.md")

    def test_near_empty_notes_are_not_similar_text(self):
        # "d" and "e": a positive cosine, but no word in common to name, so no "similar text" reason.
        self.assertNotIn("similar text", self.pairs[("D.md", "E.md")].reason)

    def test_path_filter(self):
        pairs = self.suggester.suggestions(limit=None, path="one/")
        self.assertEqual(pairs, [])  # one/Q.md is in no pair with another note under one/
        pairs = self.suggester.suggestions(limit=None, path=".md")
        self.assertEqual(len(pairs), len(self.pairs))

    def test_for_one_note(self):
        d = next(f for f, p in self.suggester.paths.items() if p == "D.md")
        pairs = self.suggester.suggestions(limit=None, for_note=d)
        self.assertTrue(pairs and all("D.md" in (s.a, s.b) for s in pairs))


@requires_model
class SampleVaultRules(unittest.TestCase):
    """The rules on the whole sample vault, checked against the raw links table and the raw files."""

    @classmethod
    def setUpClass(cls):
        from test_vaultengine_search import SampleVaultReferenceTests

        cls.ix = SampleVaultReferenceTests.shared_index()
        cls.links = set()  # (source, target) as written in the files
        cls.linked = set()  # either direction
        for a, b in cls.ix.conn.execute(
            """SELECT s.path, t.path FROM links l JOIN files s ON s.id = l.source_id JOIN files t ON t.id = l.target_id
               WHERE l.status = 'resolved'"""
        ):
            cls.links.add((a, b))
            cls.linked |= {(a, b), (b, a)}

    def test_no_suggested_pair_is_linked(self):
        pairs = LinkSuggester(self.ix.conn).suggestions(limit=None)
        self.assertGreater(len(pairs), 100)
        for s in pairs:
            self.assertNotIn((s.a, s.b), self.linked)
            self.assertNotEqual(s.a, s.b)

    def test_no_mention_is_linked_self_code_or_link(self):
        mentions = MentionFinder(self.ix.conn, FIXTURES / "vault").all()
        self.assertGreater(len(mentions), 50)
        texts = {}
        for m in mentions:
            self.assertNotEqual(m.source, m.target)
            self.assertNotIn((m.source, m.target), self.links)  # the mentioning note does not link it yet
            text = texts.setdefault(m.source, (FIXTURES / "vault" / m.source).read_text(encoding="utf-8"))
            # Independent checks on the raw text: not inside [[...]], (...) of a markdown link, `code`, or a URL.
            for pattern in (r"!?\[\[[^\]]*\]\]", r"\]\([^)]*\)", r"`[^`\n]*`", r"https?://\S+"):
                for span in re.finditer(pattern, text):
                    self.assertFalse(span.start() < m.end and m.start < span.end(), (m.source, m.line, pattern))


class ConnectCliTests(unittest.TestCase):
    def setUp(self):
        self.vault = TempVault(MENTION_NOTES)
        VaultIndex(default_db_path(self.vault.root)).update(self.vault.root)

    def tearDown(self):
        self.vault.cleanup()

    def run_cli(self, *args):
        out, err = io.StringIO(), io.StringIO()
        with patch.dict(os.environ, {"COLUMNS": "100"}), redirect_stdout(out), redirect_stderr(err):
            code = cli_main(list(args))
        return code, out.getvalue(), err.getvalue()

    def test_unlinked_for_a_note(self):
        code, out, err = self.run_cli("unlinked", str(self.vault.root), "reward-hacking", "--color", "never")
        self.assertEqual((code, err), (0, ""))
        lines = out.splitlines()
        self.assertIn("  6 plain-text mentions of reward-hacking in 4 notes that do not link to it yet", lines)
        self.assertIn("    Wrapped.md:1  The «reward hacking» story.", lines)  # the line break shown as a space
        self.assertIn("    Plain.md:1  We saw «reward hacking» in training.", lines)
        self.assertIn("    Alias.md:1  This is «spec gaming», plainly.", lines)

    def test_unlinked_everything_and_errors(self):
        _, out, _ = self.run_cli("unlinked", str(self.vault.root), "--color", "never")
        self.assertIn("  reward  (4 notes)", out.splitlines())
        self.assertIn("one-word titles match every use of the word; --min-words 2 leaves them out", out)
        code, _, err = self.run_cli("unlinked", str(self.vault.root), "nope")
        self.assertEqual(code, 2)
        self.assertIn('no note or link target called "nope"', err)

    def test_suggest_sections(self):
        code, out, _ = self.run_cli("suggest", str(self.vault.root), "--color", "never")
        self.assertEqual(code, 0)
        self.assertIn("  Links to add  (notes with no link between them;", out)
        self.assertIn("  Notes to write  (", out)
        _, out, _ = self.run_cli("suggest", str(self.vault.root), "--links", "--color", "never")
        self.assertNotIn("Notes to write", out)


if __name__ == "__main__":
    unittest.main()


def tearDownModule():
    from test_vaultengine_search import tearDownModule as close_shared_index

    close_shared_index()


PLANTED = """
## Planted
GRPO has a related failure mode: zero gradient prompt filtering drops groups whose samples all score the same.
Decoys: `zero-gradient-prompt-filtering` in inline code,
```
zero gradient prompt filtering in a fenced block
```
%% zero gradient prompt filtering in a comment %%
[[zero-gradient-prompt-filtering-v2|zero gradient prompt filtering]] as link text,
[zero gradient prompt filtering](https://example.com/zero-gradient-prompt-filtering) as a markdown link,
https://example.com/zero-gradient-prompt-filtering as a bare URL, #zero-gradient-prompt-filtering as a tag,
zero gradient [[x]] prompt filtering broken by a link, and zero gradients prompt filtering (not the words).
"""


class PlantedMentionInSampleVault(unittest.TestCase):
    """A plain-text mention planted in a copy of the sample vault is found, and none of the decoys."""

    def test_planted_mention(self):
        import shutil
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = os.path.join(tmp, "vault")
            shutil.copytree(FIXTURES / "vault", root, ignore=shutil.ignore_patterns(".vaultengine"))
            target = "20_ZETTLEKASTEN/zero-gradient-prompt-filtering.md"
            with open(os.path.join(root, "20_ZETTLEKASTEN/kl-penalty-in-rlhf.md"), "a", encoding="utf-8") as f:
                f.write(PLANTED)
            with open(os.path.join(root, "10_SOURCES/2601.22511 - Mock Worlds Real Skills (SynthAgent).md"), "a", encoding="utf-8") as f:
                f.write("\nWe return to zero gradient prompt filtering later.\n")  # this note already links it
            with open(os.path.join(root, target), "a", encoding="utf-8") as f:
                f.write("\nZero gradient prompt filtering, restated in its own note.\n")  # the note itself
            with VaultIndex(default_db_path(root)) as ix:
                ix.update(root)
                finder = MentionFinder(ix.conn, root)
                target_id = next(f for f, p in finder.paths.items() if p == target)
                found = [(m.source, m.text) for m in finder.of(target_id)]
                everywhere = [(m.source, m.text) for m in finder.all() if m.target == target]
        expected = [("20_ZETTLEKASTEN/kl-penalty-in-rlhf.md", "zero gradient prompt filtering")]
        self.assertEqual(found, expected)
        self.assertEqual(everywhere, expected)
