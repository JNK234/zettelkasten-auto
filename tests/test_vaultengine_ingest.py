"""vault ingest (conversion, the duplicate gate) and vault coverage.

The duplicate gate is proved three ways: title pairs and content pairs that should and
should not match (with the scores pinned, including both sides of each threshold),
and whole ingests into a copy of the sample vault, where a refused ingest must leave
every file untouched.
"""

import difflib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from vaultengine import Vault
from vaultengine.formats import TypedVault, load_format, parse_format, similarity
from vaultengine.formats.coverage import coverage
from vaultengine.formats.ingest import DuplicateFound, fit_body, plan_ingest, read_document
from vaultengine.formats.write import WriteError, run_write
from vaultengine.index import VaultIndex

TESTS = Path(__file__).parent
VAULT = TESTS / "fixtures" / "vault"
REPO_ROOT = TESTS.parent
BRAIN = load_format(REPO_ROOT / "formats" / "brain.yaml")
ERROR_BARS = "10_SOURCES/2411.00640 - Adding Error Bars to Evals.md"


def body_of(path: Path) -> str:
    return path.read_text(encoding="utf-8").split("\n# Notes", 1)[1]


def files_state(root: Path) -> dict:
    state = {}
    for directory, _, names in os.walk(root):
        for name in names:
            path = Path(directory) / name
            rel = path.relative_to(root).as_posix()
            if not rel.startswith(".vaultengine/"):
                state[rel] = (path.read_bytes(), path.stat().st_mtime_ns)
    return state


# ----- title similarity ----------------------------------------------------------------------------------

TITLE_SHOULD_MATCH = [
    # (a, b, exact score)
    ("Understanding R1-Zero-Like Training", "understanding r1 zero-like training", 1.0),            # case, punctuation
    ("Understanding R1-Zero-Like Training", "Understanding R1-Zero-Like Trainng", 1.0),             # typo
    ("Tülu 3: Pushing Frontiers in Open Language Model Post-Training",
     "Tulu 3 Pushing Frontiers in Open Language Model Post Training", 1.0),                          # accents
    ("Adding Error Bars to Evals", "Adding Error Bars to Evals (v2)", 10 / 11),                     # a suffix
    ("The Case for Ensuring Powerful AIs are Controlled",
     "The case for ensuring that powerful AIs are controlled", 16 / 17),                            # a filler word
    ("one two three four five six seven eight nine ten",
     "one two three four five six seven eight nine eleven", 0.9),                                    # exactly 0.9
]

TITLE_SHOULD_NOT_MATCH = [
    ("Measuring AI Ability to Complete Long Tasks", "Measuring AI Ability to Complete Short Tasks", 12 / 14),  # one word
    ("Tulu 3 RLVR Post-Training", "Tulu 2 RLVR Post-Training", 0.8),                                   # version number
    ("GPT-4 Technical Report", "GPT-4o Technical Report", 0.75),
    ("Rubrics as Rewards (RaR)", "Rubrics as Rewards", 6 / 7),    # a dropped acronym: content has to catch this one
    ("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi",
     "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu omicron pi", 26 / 29),     # just below 0.9
    ("Alignment Faking in Large Language Models", "Why Do Some Language Models Fake Alignment While Others Don't", None),
]


class TitleSimilarityTests(unittest.TestCase):
    def test_should_match(self):
        for a, b, score in TITLE_SHOULD_MATCH:
            with self.subTest(a=a, b=b):
                self.assertAlmostEqual(similarity.title_similarity(a, b), score, places=9)
                self.assertTrue(similarity.is_title_duplicate(similarity.title_similarity(a, b)))

    def test_should_not_match(self):
        for a, b, score in TITLE_SHOULD_NOT_MATCH:
            with self.subTest(a=a, b=b):
                got = similarity.title_similarity(a, b)
                if score is not None:
                    self.assertAlmostEqual(got, score, places=9)
                self.assertFalse(similarity.is_title_duplicate(got))

    def test_symmetric_enough_and_bounded(self):
        for a, b, _ in TITLE_SHOULD_MATCH + TITLE_SHOULD_NOT_MATCH:
            with self.subTest(a=a):
                self.assertLessEqual(similarity.title_similarity(a, b), 1.0)
                self.assertEqual(similarity.title_similarity(a, a), 1.0)
        self.assertEqual(similarity.title_similarity("", "x"), 0.0)

    def test_why_words_not_letters(self):
        # Letter-level similarity would call these two different papers the same.
        a, b = "measuring ai ability to complete long tasks", "measuring ai ability to complete short tasks"
        self.assertGreaterEqual(difflib.SequenceMatcher(None, a, b).ratio(), 0.9)
        self.assertLess(similarity.title_similarity(a, b), 0.9)

    def test_distinct_sample_sources_are_far_apart(self):
        typed = TypedVault(BRAIN, Vault.load(VAULT))
        from vaultengine.formats.ingest import note_titles
        source = BRAIN.type_named("source")
        titles = {p: note_titles(typed, p, source) for p in typed.notes_of_type("source")}
        worst = max(
            similarity.title_similarity(x, y)
            for a in titles for b in titles if a < b for x in titles[a] for y in titles[b]
        )
        self.assertLess(worst, 0.5)


# ----- content overlap -------------------------------------------------------------------------------------


class ContentOverlapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = body_of(VAULT / ERROR_BARS)  # about 6,800 words of real prose
        cls.other = body_of(VAULT / "10_SOURCES/2507.17746 - Rubrics as Rewards (RaR).md")
        cls.paragraphs = [p for p in cls.base.split("\n\n") if len(p.split()) > 20]

    def score(self, a: str, b: str):
        return similarity.content_overlap(similarity.shingles(a), similarity.shingles(b)).score

    def edit_every(self, text: str, n: int, seed: int = 1) -> str:
        words = text.split(" ")
        for i in range(0, len(words), n):
            words[i] = "edited" + str(i)
        return " ".join(words)

    def test_should_match(self):
        rewrapped = "\n".join(textwrap.wrap(re.sub(r"\*\*|__|`", "", self.base), 60))
        shuffled = self.paragraphs[:]
        random.Random(3).shuffle(shuffled)
        cases = {
            "identical": (self.base, self.base),
            "reformatted (rewrapped, bold and code marks removed)": (self.base, rewrapped),
            "links turned into plain text": (self.base, re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", self.base)),
            "paragraphs in another order": ("\n\n".join(self.paragraphs), "\n\n".join(shuffled)),
            "new text inside a longer note": ("\n\n".join(self.paragraphs[:20]), self.base + "\n\nMy own commentary follows. " * 3),
            "light edits (one word in 60)": (self.base, self.edit_every(self.base, 60)),
        }
        for label, (a, b) in cases.items():
            with self.subTest(label):
                score = self.score(a, b)
                self.assertTrue(similarity.is_content_duplicate(score), f"{label}: {score}")

    def test_should_not_match(self):
        quote = self.paragraphs[5]
        cases = {
            "heavy edits (one word in 12)": (self.base, self.edit_every(self.base, 12)),
            "one shared paragraph": ("\n\n".join(self.paragraphs[:12]), self.other[:9000] + "\n\n" + quote),
            "a different paper": (self.base, self.other),
        }
        for label, (a, b) in cases.items():
            with self.subTest(label):
                score = self.score(a, b)
                self.assertFalse(similarity.is_content_duplicate(score), f"{label}: {score}")

    def test_pinned_scores(self):
        self.assertEqual(self.score(self.base, self.base), 1.0)
        self.assertLess(self.score(self.base, self.other), 0.02)
        light = self.score(self.base, self.edit_every(self.base, 60))
        heavy = self.score(self.base, self.edit_every(self.base, 12))
        self.assertGreater(light, 0.9)
        self.assertLess(heavy, 0.65)

    def test_too_short_is_not_compared(self):
        short = "This sentence is far too short to say anything about duplicates at all, identical or not."
        result = similarity.content_overlap(similarity.shingles(short), similarity.shingles(short))
        self.assertIsNone(result.score)
        self.assertFalse(similarity.is_content_duplicate(result.score))

    def test_threshold_is_inclusive(self):
        a = frozenset((f"w{i}",) for i in range(100))
        at = frozenset(list(a)[:80]) | frozenset((f"x{i}",) for i in range(20))
        below = frozenset(sorted(a)[:79]) | frozenset((f"x{i}",) for i in range(21))
        self.assertEqual(similarity.content_overlap(a, at).score, 0.8)
        self.assertTrue(similarity.is_content_duplicate(0.8))
        self.assertAlmostEqual(similarity.content_overlap(a, below).score, 0.79)
        self.assertFalse(similarity.is_content_duplicate(similarity.content_overlap(a, below).score))

    def test_sample_sources_do_not_overlap(self):
        bodies = {p.name: similarity.shingles(body_of(p)) for p in (VAULT / "10_SOURCES").glob("*.md") if "\n# Notes" in p.read_text()}
        names = sorted(bodies)
        worst = max(
            (similarity.content_overlap(bodies[a], bodies[b]).score or 0.0)
            for i, a in enumerate(names) for b in names[i + 1:]
        )
        self.assertLess(worst, 0.05)


# ----- conversion -----------------------------------------------------------------------------------------


class ConversionTests(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.dir = Path(self._dir.name)

    def tearDown(self):
        self._dir.cleanup()

    def write(self, name: str, content: str | bytes) -> Path:
        path = self.dir / name
        path.write_bytes(content if isinstance(content, bytes) else content.encode("utf-8"))
        return path

    def test_html(self):
        doc = read_document(self.write("page.html", """<html><head>
            <title>A Post About Graders | Example Blog</title><link rel="canonical" href="https://ex.org/p/graders/">
            <script>x()</script><style>p{}</style></head><body><header>site</header><nav><a href="/">home</a></nav>
            <article><h1>A Post About Graders</h1><p>Text with a <a href="../other/">relative link</a> and
            <a href="#top">an anchor</a>.</p><img src="/i.png" alt="a chart"><img src="data:x" alt="inline">
            <h2>Part</h2><p>More.</p><form><input></form></article><footer>foot</footer></body></html>"""))
        self.assertEqual((doc.kind, doc.title, doc.url), ("html", "A Post About Graders", "https://ex.org/p/graders/"))
        self.assertIn("[relative link](https://ex.org/p/other/)", doc.body)
        self.assertIn("an anchor.", doc.body)
        self.assertIn("![a chart](https://ex.org/i.png)", doc.body)
        self.assertIn("inline", doc.body)
        for junk in ("x()", "p{}", "home", "foot", "site"):
            self.assertNotIn(junk, doc.body)
        body = fit_body(doc.body, doc.title, 1, doc.warnings)
        self.assertFalse(body.startswith("#"))  # the H1 repeating the title is dropped
        self.assertIn("\n## Part\n", body)

    def test_html_without_canonical_url_keeps_relative_links_as_text(self):
        doc = read_document(self.write("p.html", "<html><body><p>See <a href='/x'>this page</a>.</p></body></html>"))
        self.assertEqual(doc.title, "p")
        self.assertIn("See this page.", doc.body)
        self.assertEqual(doc.warnings, ["1 relative links kept as text (the page has no canonical URL)"])

    def test_markdown(self):
        doc = read_document(self.write("n.md", "---\ntitle: From Frontmatter\nurl: https://ex.org/a\n---\n# Heading\nbody\n"))
        self.assertEqual((doc.title, doc.url), ("From Frontmatter", "https://ex.org/a"))
        doc = read_document(self.write("m.md", "# The Title\n\ntext\n\n# Part\n\n## Sub\n"))
        self.assertEqual(doc.title, "The Title")
        warnings = []
        self.assertEqual(fit_body(doc.body, doc.title, 1, warnings), "text\n\n## Part\n\n### Sub")
        self.assertEqual(warnings, ["headings moved down 1 level to sit under the body heading"])

    def test_text(self):
        doc = read_document(self.write("t.txt", "A Plain Title\n\n# not a heading\nsupports:: [[Something]]\nbody\n"))
        self.assertEqual(doc.title, "A Plain Title")
        warnings = []
        body = fit_body(doc.body, doc.title, 1, warnings)
        self.assertIn("\\# not a heading", body)
        self.assertIn("supports\\:: [[Something]]", body)
        self.assertEqual(warnings, ["1 `key::` line escaped so they add no relations"])
        from vaultengine import parse_note
        self.assertEqual((parse_note(body).headings, parse_note(body).inline_fields), ([], []))

    def test_errors(self):
        with self.assertRaises(WriteError):
            read_document(self.write("x.pdf", b"%PDF-1.4"))
        with self.assertRaises(WriteError):
            read_document(self.write("e.txt", "   \n"))
        with self.assertRaises(WriteError):
            read_document(self.dir / "missing.md")
        doc = read_document(self.write("latin.txt", "Caf\xe9 title\n\nbody".encode("cp1252")))
        self.assertEqual((doc.title, doc.warnings), ("Café title", ["not UTF-8; read as Windows-1252"]))


# ----- the gate on the sample vault --------------------------------------------------------------------------

NEW_ARTICLE = """<html><head><title>Why Reward Model Ensembles Delay Overoptimization | Field Notes</title>
<link rel="canonical" href="https://fieldnotes.example.org/posts/ensembles/"></head><body><article>
<h1>Why Reward Model Ensembles Delay Overoptimization</h1>
<p>Optimizing a policy against a learned reward model works until it does not. Past a certain amount of optimization
pressure the proxy reward keeps rising while the true objective falls. An ensemble of reward models trained from
different seeds pushes that turning point further out, and this post explains why that happens.</p>
<h2>The proxy is right on average and wrong at the edges</h2>
<p>A single reward model makes idiosyncratic errors: regions of response space where it is confidently and wrongly
enthusiastic. The policy finds those regions, because they pay. Models trained from other seeds make different errors
in different places, so the mean of several models is less enthusiastic exactly where any one of them is wrong.</p>
</article></body></html>"""


class IngestGateTests(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.root = Path(self._dir.name) / "vault"
        self.inbox = Path(self._dir.name) / "inbox"
        self.inbox.mkdir()
        shutil.copytree(VAULT, self.root)
        self.db = self.root / ".vaultengine" / "index.sqlite"
        with VaultIndex(self.db) as index:
            index.update(self.root)

    def tearDown(self):
        self._dir.cleanup()

    def ingest(self, path: Path, title=None):
        document = read_document(path, title=title)
        return run_write(BRAIN, self.root, self.db, lambda t: plan_ingest(t, document, "source"))

    def assertDuplicate(self, path: Path, expected_path: str, reasons_like: list[str], title=None):
        before = files_state(self.root)
        with self.assertRaises(DuplicateFound) as caught:
            self.ingest(path, title)
        self.assertEqual(files_state(self.root), before)  # nothing written
        first = caught.exception.duplicates[0]
        self.assertEqual(first.path, expected_path)
        for reason in reasons_like:
            self.assertTrue(any(r.startswith(reason) for r in first.reasons), (reason, first.reasons))
        return first

    def test_exact_copy_of_a_sample_source(self):
        copy = self.inbox / "error-bars-copy.md"
        shutil.copy(self.root / ERROR_BARS, copy)
        found = self.assertDuplicate(copy, ERROR_BARS, ["same file", "content overlap 1.000"])
        self.assertFalse(similarity.is_title_duplicate(found.title))  # renamed: only content and hash catch it

    def test_edited_copy_with_a_new_title(self):
        text = (self.root / ERROR_BARS).read_text(encoding="utf-8")
        words = text.split(" ")
        for i in range(0, len(words), 40):
            words[i] = words[i].upper()  # case changes do not matter
        for i in range(7, len(words), 80):
            words[i] = "changed"
        edited = self.inbox / "stats-for-evals.md"
        edited.write_text("# Statistics for LLM evaluations\n\n" + " ".join(words), encoding="utf-8")
        found = self.assertDuplicate(edited, ERROR_BARS, ["content overlap"])
        self.assertFalse(found.same_hash)
        self.assertGreater(found.overlap, 0.8)

    def test_saved_page_of_a_paper_already_in_the_vault(self):
        page = self.inbox / "abs.html"
        page.write_text("<html><head><meta property='og:title' content='Adding Error Bars to Evals: A Statistical Approach "
                        "to Language Model Evaluations'></head><body><main><p>A short abstract, in other words than the note.</p>"
                        "</main></body></html>", encoding="utf-8")
        found = self.assertDuplicate(page, ERROR_BARS, ["title similarity 1.000"])
        self.assertEqual(found.title_pair[1], "Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations")

    def test_title_near_misses_are_not_duplicates(self):
        page = self.inbox / "short-tasks.html"
        page.write_text(NEW_ARTICLE.replace("Why Reward Model Ensembles Delay Overoptimization",
                                            "Measuring AI Ability to Complete Short Tasks"), encoding="utf-8")
        outcome = self.ingest(page)  # the vault has "... Long Tasks": 0.857 by title, different content
        self.assertEqual(outcome.status, "written")

    def test_genuinely_new_then_the_same_again(self):
        page = self.inbox / "ensembles.html"
        page.write_text(NEW_ARTICLE, encoding="utf-8")
        outcome = self.ingest(page)
        self.assertEqual(outcome.status, "written")
        path = "10_SOURCES/Why Reward Model Ensembles Delay Overoptimization.md"
        text = (self.root / path).read_text(encoding="utf-8")
        import hashlib
        digest = hashlib.sha256(page.read_bytes()).hexdigest()
        self.assertTrue(text.startswith("Created: "))
        self.assertIn(f"\n\nOriginal: ensembles.html\n\nContent hash: sha256:{digest}\n\nIngested: ", text)
        self.assertIn("## References\n\n1. [Why Reward Model Ensembles Delay Overoptimization](https://fieldnotes.example.org/posts/ensembles/)"
                      "\n\n# Notes\n\nOptimizing a policy", text)
        self.assertIn("\n## The proxy is right on average and wrong at the edges\n", text)
        self.assertNotIn("\n# Why Reward", text)
        typed = TypedVault(BRAIN, Vault.load(self.root))
        self.assertEqual(typed.type_of(path), "source")
        self.assertEqual(typed.issues(path), [])
        self.assertEqual(outcome.index["synced"], [path])

        found = self.assertDuplicate(page, path, ["same file", "title similarity 1.000", "content overlap 1.000"])
        self.assertTrue(found.same_hash)
        # the same article saved again with other page chrome is still the same content
        variant = self.inbox / "ensembles-again.html"
        variant.write_text(NEW_ARTICLE.replace("| Field Notes", "— Field Notes (mirror)").replace("<article>",
                           "<nav><a href='/'>Home</a> <a href='/tags'>Tags</a></nav><article>"), encoding="utf-8")
        found = self.assertDuplicate(variant, path, ["content overlap"])
        self.assertFalse(found.same_hash)
        self.assertFalse(similarity.is_title_duplicate(found.title))  # "… — Field Notes (mirror)" is 0.8 by title
        self.assertGreater(found.overlap, 0.95)

    def test_relation_lines_in_a_document_are_neutralised(self):
        text_file = self.inbox / "fields.txt"
        text_file.write_text("Notes From a Talk on Graders\n\nsupports:: [[CLM - Grade the observed end state rather than the trajectory]]\n"
                             + "Graders should be tested before training starts. " * 12, encoding="utf-8")
        outcome = self.ingest(text_file)
        self.assertEqual(outcome.status, "written")
        typed = TypedVault(BRAIN, Vault.load(self.root))
        path = "10_SOURCES/Notes From a Talk on Graders.md"
        self.assertEqual(typed.relations(path), [])

    def test_type_without_an_ingest_block(self):
        page = self.inbox / "x.html"
        page.write_text(NEW_ARTICLE, encoding="utf-8")
        document = read_document(page)
        with self.assertRaises(WriteError) as caught:
            run_write(BRAIN, self.root, self.db, lambda t: plan_ingest(t, document, "zettel"))
        self.assertIn("has no `ingest:` block", str(caught.exception))


# ----- coverage ----------------------------------------------------------------------------------------------


class CoverageTests(unittest.TestCase):
    def test_sample(self):
        data = coverage(TypedVault(BRAIN, Vault.load(VAULT)))
        self.assertEqual((data["sources"]["total"], data["sources"]["linked"]), (28, 23))
        self.assertEqual(data["sources"]["unlinked"], [
            "10_SOURCES/2501.10970 - The Alternative Annotator Test for LLM-as-a-Judge.md",
            "10_SOURCES/Benchmark Contamination - Detection and Defeat.md",
            "10_SOURCES/Cao et al - SkyRL-v0 Train Real-World Long-Horizon Agents via Reinforcement Learning.md",
            "10_SOURCES/DeepSeek R1 - Paper Notes.md",
            "10_SOURCES/Hamel Husain - AI Evals FAQ (Everything You Need to Know).md",
        ])
        citing = data["citing"]
        self.assertEqual((citing["total"], citing["cite_a_source"], citing["without_source"]), (143, 33, 110))
        self.assertEqual(citing["by_reason"], {"missing": 106, "not_a_source": 0, "none": 4})
        none = sorted(n["path"] for n in citing["notes"] if n["reason"] == "none")
        self.assertEqual(none, ["20_ZETTLEKASTEN/q-value-function.md", "20_ZETTLEKASTEN/reward.md",
                                "20_ZETTLEKASTEN/value-function.md", "50_MOC/comparisons/proactive-vs-reactive-memory-management.md"])

    def test_unlinked_sources_are_the_index_orphans(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch) / "v"
            shutil.copytree(VAULT, root)
            with VaultIndex(root / "i.sqlite") as index:
                index.update(root)
                from vaultengine import queries
                orphans = {p for p in queries.health(index.conn)["orphans"] if p.startswith("10_SOURCES/")}
        self.assertEqual(orphans, set(coverage(TypedVault(BRAIN, Vault.load(VAULT)))["sources"]["unlinked"]))

    def test_reasons(self):
        spec = parse_format(textwrap.dedent("""
            format: 1
            link_roles: {reference: [References]}
            coverage: {sources: [source], citing: [note], citation_roles: [reference]}
            types:
              source: {match: {folder: S}}
              note: {match: {folder: N}}
        """))
        vault = Vault.from_files({
            "S/paper.md": "", "S/unread.md": "",
            "N/good.md": "# References\n[[paper]]\n",
            "N/missing.md": "# References\n[[not here]]\n",
            "N/wrong.md": "# References\n[[good]]\n",
            "N/none.md": "text [[good]]\n",
        })
        data = coverage(TypedVault(spec, vault))
        self.assertEqual(data["sources"]["unlinked"], ["S/unread.md"])
        reasons = {n["path"]: n["reason"] for n in data["citing"]["notes"]}
        self.assertEqual(reasons, {"N/missing.md": "missing", "N/wrong.md": "not-a-source", "N/none.md": "none"})

    def test_spec_errors(self):
        from vaultengine.formats import FormatError
        base = "format: 1\ntypes: {s: {match: {folder: S}}}\n"
        for text, message in (
            (base + "coverage: {sources: [x], citing: [s]}", "coverage.sources[0]: unknown type 'x'"),
            (base + "coverage: {sources: [s], citing: [s], citation_roles: [nope]}", "'nope' is not a link role or relation"),
            (base + "coverage: {sources: [s]}", "missing required key 'citing'"),
        ):
            with self.subTest(text=text):
                with self.assertRaises(FormatError) as caught:
                    parse_format(text)
                self.assertIn(message, str(caught.exception))


# ----- CLI -------------------------------------------------------------------------------------------------


def run_vault(*args) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "vaultengine", *args], capture_output=True, text=True, cwd=REPO_ROOT)


class IngestCliTests(unittest.TestCase):
    def test_duplicate_and_new(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch) / "vault"
            shutil.copytree(VAULT, root)
            copy = Path(scratch) / "copy.md"
            shutil.copy(root / ERROR_BARS, copy)
            result = run_vault("ingest", str(root), str(copy), "--as", "source", "-f", "brain")
            self.assertEqual(result.returncode, 1)
            self.assertIn("✗ not ingested: copy.md duplicates 1 existing note; nothing was written", result.stdout)
            self.assertIn("content overlap   1.000  duplicate (threshold 0.8)", result.stdout)
            data = json.loads(run_vault("ingest", str(root), str(copy), "--as", "source", "-f", "brain", "--json").stdout)
            self.assertEqual((data["status"], data["duplicates"][0]["path"], data["duplicates"][0]["same_hash"]),
                             ("duplicate", ERROR_BARS, True))
            page = Path(scratch) / "new.html"
            page.write_text(NEW_ARTICLE, encoding="utf-8")
            result = run_vault("ingest", str(root), str(page), "--as", "source", "-f", "brain")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("+ 10_SOURCES/Why Reward Model Ensembles Delay Overoptimization.md", result.stdout)
            self.assertEqual(run_vault("ingest", str(root), str(Path(scratch) / "nope.md"), "--as", "source", "-f", "brain").returncode, 2)

    def test_coverage(self):
        result = run_vault("coverage", str(VAULT), "-f", "brain")
        self.assertEqual(result.returncode, 0)
        self.assertIn("sources           28   23 linked from other notes, 5 that nothing links to", result.stdout)
        self.assertIn("106 cite only missing notes, 0 cite only non-sources, 4 have no citation at all", result.stdout)


if __name__ == "__main__":
    unittest.main()
