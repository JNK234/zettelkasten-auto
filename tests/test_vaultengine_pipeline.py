"""`vault pipeline` and `vault note` under formats/core.yaml.

The synthetic vault below is built so that each source, zettel and page exercises one
case; the expected stages and gaps are written out by hand next to it.
"""

import json
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from vaultengine import Vault
from vaultengine.formats import FormatError, TypedVault, load_format, parse_format
from vaultengine.formats.pipeline import Pipeline, PipelineNotConfigured

TESTS = Path(__file__).parent
REPO_ROOT = TESTS.parent
VAULT = TESTS / "fixtures" / "vault"
CORE = REPO_ROOT / "formats" / "core.yaml"

HEAD = "Created: 01-01-2026 10:00\nTags: [[ml]]\n"
SOURCE = "Created: 01-01-2026 10:00\nTags: [[ml]]\nSource: https://example.com\n\n# S\n"

FILES = {
    # Sources, and the stage each should be in.
    "10_SOURCES/S compiled.md": SOURCE,         # z1 and z8 link it; page P links both    -> compiled
    "10_SOURCES/S extracted.md": SOURCE,        # z2 links it, z7 embeds it; no page       -> extracted
    "10_SOURCES/S idea only.md": SOURCE,        # z6 links it; only an idea links z6       -> extracted
    "10_SOURCES/S ingested.md": SOURCE,         # nothing links it                         -> ingested
    "10_SOURCES/S page direct.md": SOURCE,      # P links it directly, no zettel does      -> ingested
    "10_SOURCES/S tags only.md": SOURCE,        # z3 has it only in Tags:                  -> ingested
    # Zettels.
    "20_ZETTLEKASTEN/z1.md": HEAD + "\nFrom [[S compiled]].\n",
    "20_ZETTLEKASTEN/z2.md": HEAD + "\nFrom [[S extracted]], see [[z1]].\n",
    "20_ZETTLEKASTEN/z3.md": "Created: 01-01-2026 10:00\nTags: [[S tags only]]\n\nNo citation in the body.\n",
    "20_ZETTLEKASTEN/z4.md": HEAD + "\n# References\n1. [[S never ingested]]\n",
    "20_ZETTLEKASTEN/z5.md": HEAD + "\n# References\n1. [The paper](https://arxiv.org/abs/2401.00001)\n",
    "20_ZETTLEKASTEN/z6.md": HEAD + "\n[[S idea only]]\n",
    "20_ZETTLEKASTEN/z7.md": HEAD + "\n![[S extracted]]\n",
    "20_ZETTLEKASTEN/z8.md": HEAD + "\n[see](../10_SOURCES/S%20compiled.md)\n",
    # Pages, an idea, and an ignored note.
    "50_MOC/P.md": HEAD + "\n[[z1]] [[z4]] [[z8]] and the paper itself: [[S page direct]]\n",
    "50_MOC/topics/Q.md": HEAD + "\nNothing yet, and a tag link does not count either.\n",
    "50_MOC/ideas/I.md": HEAD + "Status: #open\n\n[[z6]]\n",
    "50_MOC/discourse/old.md": "[[z7]] [[z2]] [[S ingested]]\n",
}

STAGES = {
    "10_SOURCES/S compiled.md": "compiled",
    "10_SOURCES/S extracted.md": "extracted",
    "10_SOURCES/S idea only.md": "extracted",
    "10_SOURCES/S ingested.md": "ingested",
    "10_SOURCES/S page direct.md": "ingested",
    "10_SOURCES/S tags only.md": "ingested",
}


def run_vault(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "vaultengine", *args], capture_output=True, text=True, cwd=REPO_ROOT)


def pipeline(files=FILES) -> Pipeline:
    return Pipeline(TypedVault(load_format(CORE), Vault.from_files(files)))


class StageTests(unittest.TestCase):
    def test_each_source(self):
        report = pipeline().report()
        self.assertEqual({row["path"]: row["stage"] for row in report["sources"]}, STAGES)
        self.assertEqual(report["stages"], {"ingested": 3, "extracted": 2, "compiled": 1})

    def test_what_each_stage_rests_on(self):
        rows = {row["path"]: row for row in pipeline().report()["sources"]}
        compiled = rows["10_SOURCES/S compiled.md"]
        self.assertEqual(compiled["zettels"], ["20_ZETTLEKASTEN/z1.md", "20_ZETTLEKASTEN/z8.md"])
        self.assertEqual(compiled["pages"], [{"path": "50_MOC/P.md", "via": ["20_ZETTLEKASTEN/z1.md", "20_ZETTLEKASTEN/z8.md"]}])
        # The discourse note links z2 and z7, but it is ignored: no page.
        self.assertEqual(rows["10_SOURCES/S extracted.md"]["zettels"], ["20_ZETTLEKASTEN/z2.md", "20_ZETTLEKASTEN/z7.md"])
        self.assertEqual(rows["10_SOURCES/S extracted.md"]["pages"], [])
        self.assertEqual(rows["10_SOURCES/S idea only.md"]["pages"], [])  # an idea is not a page

    def test_sources_listed_furthest_along_first(self):
        order = [row["path"] for row in pipeline().report()["sources"]]
        self.assertEqual(order, [
            "10_SOURCES/S compiled.md", "10_SOURCES/S extracted.md", "10_SOURCES/S idea only.md",
            "10_SOURCES/S ingested.md", "10_SOURCES/S page direct.md", "10_SOURCES/S tags only.md",
        ])

    def test_a_stage_moves_when_the_links_change(self):
        # Nothing is stored: a page linking z2 compiles S extracted on the next run.
        self.assertEqual(pipeline().stage("10_SOURCES/S extracted.md"), "extracted")
        moved = {**FILES, "50_MOC/topics/Q.md": HEAD + "\n[[z2]]\n"}
        self.assertEqual(pipeline(moved).stage("10_SOURCES/S extracted.md"), "compiled")
        back = {k: v for k, v in FILES.items() if k != "20_ZETTLEKASTEN/z1.md"}
        self.assertEqual(pipeline(back).stage("10_SOURCES/S compiled.md"), "compiled")  # z8 still carries it
        gone = {k: v for k, v in back.items() if k != "20_ZETTLEKASTEN/z8.md"}
        self.assertEqual(pipeline(gone).stage("10_SOURCES/S compiled.md"), "ingested")


class GapTests(unittest.TestCase):
    def test_the_two_gaps(self):
        gaps = pipeline().report()["gaps"]
        self.assertEqual(gaps["not_extracted"], [
            "10_SOURCES/S ingested.md", "10_SOURCES/S page direct.md", "10_SOURCES/S tags only.md",
        ])
        self.assertEqual(gaps["not_compiled"], [
            "20_ZETTLEKASTEN/z2.md", "20_ZETTLEKASTEN/z3.md", "20_ZETTLEKASTEN/z5.md",
            "20_ZETTLEKASTEN/z6.md", "20_ZETTLEKASTEN/z7.md",
        ])

    def test_counts_and_why_zettels_rest_on_no_source(self):
        report = pipeline().report()
        self.assertEqual({k: report["extract"][k] for k in ("total", "in_a_compile_note", "resting_on_a_source")},
                         {"total": 8, "in_a_compile_note": 3, "resting_on_a_source": 5})
        unsourced = report["extract"]["unsourced"]
        self.assertEqual(unsourced["by_reason"], {"missing": 1, "not-a-source": 0, "url-only": 1, "none": 1})
        self.assertEqual([(n["path"], n["reason"]) for n in unsourced["notes"]], [
            ("20_ZETTLEKASTEN/z3.md", "none"),      # its source is only in Tags:
            ("20_ZETTLEKASTEN/z4.md", "missing"),   # cites a source note that is not in the vault
            ("20_ZETTLEKASTEN/z5.md", "url-only"),  # a URL is not a source
        ])
        self.assertEqual(unsourced["notes"][1]["citations"][0]["raw"], "[[S never ingested]]")
        self.assertEqual(unsourced["notes"][2]["citations"][0]["url"], "https://arxiv.org/abs/2401.00001")
        self.assertEqual(report["compile"], {"total": 2, "using_an_extract_note": 1})
        self.assertEqual(report["ignored"], 1)

    def test_agrees_with_the_violations(self):
        # The same "links to" is behind min_links: zettels resting on no source are exactly the too-few-links zettels.
        p = pipeline()
        unsourced = [n["path"] for n in p.report()["extract"]["unsourced"]["notes"]]
        flagged = [i.path for i in p.typed.issues() if i.code == "too-few-links" and p.typed.type_of(i.path) == "zettel"]
        self.assertEqual(unsourced, flagged)

    def test_not_configured(self):
        spec = parse_format("format: 1\ntypes: {t: {match: {folder: A}}}\n")
        with self.assertRaises(PipelineNotConfigured):
            Pipeline(TypedVault(spec, Vault.from_files({"A/x.md": "x"})))

    def test_validation(self):
        cases = {
            "pipeline: {sources: [s], extract: [nope], compile: [p]}": "pipeline.extract[0]: unknown type 'nope'",
            "pipeline: {sources: [s], extract: [s], compile: [p]}": "pipeline: 's' cannot be in both sources and extract",
            "pipeline: {sources: [s], extract: [z]}": "pipeline: missing required key 'compile'",
            "pipeline: {sources: [s], extract: [z], compile: [p], stages: 3}": "pipeline: unknown key 'stages'",
        }
        types = "types: {s: {match: {folder: S}}, z: {match: {folder: Z}}, p: {match: {folder: P}}}\n"
        for line, error in cases.items():
            with self.subTest(line=line):
                with self.assertRaises(FormatError) as raised:
                    parse_format(f"format: 1\n{line}\n{types}")
                self.assertIn(error, str(raised.exception))


class NoteFactsTests(unittest.TestCase):
    def setUp(self):
        self.p = pipeline()

    def paths(self, items):
        return [(item["path"], item.get("direct"), item.get("via")) for item in items]

    def test_a_compiled_source(self):
        facts = self.p.note("10_SOURCES/S compiled.md")
        self.assertEqual((facts["type"], facts["stage"]), ("source", "compiled"))
        self.assertEqual(self.paths(facts["used_by"]["extract"]),
                         [("20_ZETTLEKASTEN/z1.md", True, []), ("20_ZETTLEKASTEN/z8.md", True, [])])
        self.assertEqual(self.paths(facts["used_by"]["compile"]),
                         [("50_MOC/P.md", False, ["20_ZETTLEKASTEN/z1.md", "20_ZETTLEKASTEN/z8.md"])])
        self.assertEqual(facts["rests_on"], {"sources": [], "extract": []})

    def test_a_page(self):
        facts = self.p.note("50_MOC/P.md")
        self.assertIsNone(facts["stage"])
        self.assertEqual(self.paths(facts["rests_on"]["sources"]), [
            ("10_SOURCES/S compiled.md", False, ["20_ZETTLEKASTEN/z1.md", "20_ZETTLEKASTEN/z8.md"]),
            ("10_SOURCES/S page direct.md", True, []),
        ])
        self.assertEqual([i["path"] for i in facts["rests_on"]["extract"]],
                         ["20_ZETTLEKASTEN/z1.md", "20_ZETTLEKASTEN/z4.md", "20_ZETTLEKASTEN/z8.md"])
        self.assertEqual(facts["violations"], [])

    def test_zettels(self):
        z1 = self.p.note("20_ZETTLEKASTEN/z1.md")
        self.assertEqual(z1["stage"], "compiled")
        self.assertEqual([i["path"] for i in z1["used_by"]["compile"]], ["50_MOC/P.md"])
        self.assertEqual([i["path"] for i in z1["used_by"]["extract"]], ["20_ZETTLEKASTEN/z2.md"])  # z2 links z1
        z2 = self.p.note("20_ZETTLEKASTEN/z2.md")
        self.assertEqual(z2["stage"], "extracted")
        self.assertEqual(self.paths(z2["rests_on"]["sources"]), [
            ("10_SOURCES/S compiled.md", False, ["20_ZETTLEKASTEN/z1.md"]),  # through z1
            ("10_SOURCES/S extracted.md", True, []),
        ])
        self.assertEqual(z2["linked_from"], [{"path": "50_MOC/discourse/old.md", "type": None, "ignored": True}])
        z4 = self.p.note("20_ZETTLEKASTEN/z4.md")
        self.assertIsNone(z4["stage"])  # in page P, but extracted from nothing
        self.assertEqual(z4["links_to_nothing"], [{"line": 5, "raw": "[[S never ingested]]", "status": "unresolved"}])
        self.assertEqual([v["code"] for v in z4["violations"]], ["too-few-links"])

    def test_tag_links_are_not_facts(self):
        z3 = self.p.note("20_ZETTLEKASTEN/z3.md")
        self.assertEqual((z3["links_to"], z3["rests_on"]["sources"]), ([], []))
        self.assertEqual(self.p.note("10_SOURCES/S tags only.md")["linked_from"], [])


class CliTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory()
        self.root = Path(self.scratch.name)
        for path, text in FILES.items():
            (self.root / path).parent.mkdir(parents=True, exist_ok=True)
            (self.root / path).write_text(text)

    def tearDown(self):
        self.scratch.cleanup()

    def snapshot(self):
        return sorted((str(p.relative_to(self.root)), p.read_bytes()) for p in self.root.rglob("*") if p.is_file())

    def test_pipeline(self):
        before = self.snapshot()
        result = run_vault("pipeline", str(self.root), "-f", "core")
        self.assertEqual(result.returncode, 0, result.stderr)
        out = result.stdout
        for line in (
            "  source notes              6",
            "    compiled                1   a page links to a zettel that links to it",
            "    extracted               2   a zettel links to it, no page uses those yet",
            "    ingested                3   no zettel links to it",
            "  zettel notes              8",
            "    in a page               3",
            "    resting on a source     5",
            "    on no source note       3   1 cite source notes that are not in the vault, 1 cite only URLs, 1 cite nothing",
            "  page notes                2   1 link to a zettel",
            "  compiled     2 z   1 p   10_SOURCES/S compiled.md",
            "Source notes nothing has been extracted from (3)\n  10_SOURCES/S ingested.md",
            "Zettel notes no page includes (5)\n  20_ZETTLEKASTEN/z2.md",
        ):
            self.assertIn(line, out)
        data = json.loads(run_vault("pipeline", str(self.root), "-f", "core", "--json").stdout)
        self.assertEqual(data["stages"], {"ingested": 3, "extracted": 2, "compiled": 1})
        self.assertEqual(self.snapshot(), before)  # computed, never stored
        self.assertFalse((self.root / ".vaultengine").exists())

    def test_note(self):
        result = run_vault("note", str(self.root), "S compiled", "-f", "core")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("10_SOURCES/S compiled.md  (source)\n  stage      compiled\n", result.stdout)
        self.assertIn("  pages      1 page\n      50_MOC/P.md  (via z1, z8)", result.stdout)
        page = run_vault("note", str(self.root), "50_MOC/P", "-f", "core").stdout
        self.assertIn("      10_SOURCES/S compiled.md  (via z1, z8)\n      10_SOURCES/S page direct.md\n", page)
        data = json.loads(run_vault("note", str(self.root), "z2", "-f", "core", "--json").stdout)
        self.assertEqual((data["path"], data["stage"]), ("20_ZETTLEKASTEN/z2.md", "extracted"))
        self.assertTrue(data["modified"])
        self.assertIn("20_ZETTLEKASTEN/z4.md  (zettel)\n  stage      none: it rests on no source note\n",
                      run_vault("note", str(self.root), "z4", "-f", "core").stdout)
        missing = run_vault("note", str(self.root), "nope", "-f", "core")
        self.assertEqual(missing.returncode, 2)
        self.assertTrue(missing.stderr.strip().endswith("error: no note 'nope'"), missing.stderr)  # after the no-index note
        brain = run_vault("pipeline", str(self.root), "-f", "brain")
        self.assertEqual(brain.returncode, 2)
        self.assertIn("format 'brain' has no `pipeline:` block", brain.stderr)


class SampleVaultTests(unittest.TestCase):
    """What the pipeline says about the sample vault (a slice of a bigger vault: most cited sources are not in it)."""

    def test_numbers(self):
        report = Pipeline(TypedVault(load_format(CORE), Vault.load(VAULT))).report()
        self.assertEqual(report["stages"], {"ingested": 13, "extracted": 12, "compiled": 3})
        self.assertEqual({k: report["extract"][k] for k in ("total", "in_a_compile_note", "resting_on_a_source")},
                         {"total": 109, "in_a_compile_note": 5, "resting_on_a_source": 28})
        self.assertEqual(report["extract"]["unsourced"]["by_reason"], {"missing": 78, "not-a-source": 0, "url-only": 0, "none": 3})
        self.assertEqual(report["compile"], {"total": 3, "using_an_extract_note": 2})
        self.assertEqual((len(report["gaps"]["not_extracted"]), len(report["gaps"]["not_compiled"])), (13, 104))
        self.assertEqual(report["ignored"], 40)


if __name__ == "__main__":
    unittest.main()
