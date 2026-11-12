"""`vault findings`, `vault idea new` and `vault idea set` under formats/core.yaml.

Proved on a small vault built so that each method has one clear finding: ids are the same on
every run and survive unrelated edits (and new evidence); reading writes nothing; nothing is written
without --write; an idea from a finding passes `vault check`; an idea without a finding block fails
it and is refused; only open, dropped and done are statuses.

And on the sample vault, the findings and link suggestions are those a pair-by-pair implementation of
the same scoring gives (tests/fixtures/findings/sample_reference.json, recorded with it; the text
evidence comes from the embedding model, so those tests need models/potion-base-8M).
"""

import hashlib
import json
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from embedding_model import requires_model
from vaultengine.findings import PREFIX, finding_id

TESTS = Path(__file__).parent
REPO_ROOT = TESTS.parent
SAMPLE = TESTS / "fixtures" / "vault"
REFERENCE = TESTS / "fixtures" / "findings" / "sample_reference.json"


def run_vault(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "vaultengine", *args], capture_output=True, text=True, cwd=REPO_ROOT)


def source() -> str:
    return "Created: 01-01-2026 10:00\nTags: [[ml]]\nSource: https://example.com\n\n# S\n"


def zettel(tags: str, body: str) -> str:
    return f"Created: 01-01-2026 10:00\nTags: {tags}\n\n# Title\n\n{body}\n"


FILES = {
    "10_SOURCES/S1.md": source(),
    "10_SOURCES/S2.md": source(),
    "10_SOURCES/S3.md": source(),
    # missing-note: [[grader-hacking]] is linked from four notes and does not exist
    "20_ZETTLEKASTEN/z1.md": zettel("[[rl]] [[grading]]", "From [[S1]]. See [[grader-hacking]] and [[llm-judge]]."),
    "20_ZETTLEKASTEN/z2.md": zettel("[[rl]] [[grading]]", "From [[S2]]. See [[grader-hacking]] and [[llm-judge]]."),
    "20_ZETTLEKASTEN/z3.md": zettel("[[rl]]", "From [[S3]]. See [[grader-hacking]] and [[llm-judge-position-bias]]."),
    "20_ZETTLEKASTEN/z4.md": zettel("[[grading]]", "From [[S3]]. See [[grader-hacking]], [[llm-judge-position-bias]]."),
    # existing-name: [[on-policy-vs-off-policy]] is probably on-policy-versus-off-policy
    "20_ZETTLEKASTEN/on-policy-versus-off-policy.md": zettel("[[rl]]", "From [[S1]]."),
    "20_ZETTLEKASTEN/z5.md": zettel("[[rl]]", "From [[S2]]. Compare [[on-policy-vs-off-policy]]."),
    "20_ZETTLEKASTEN/z6.md": zettel("[[rl]]", "From [[S3]]. Compare [[on-policy-vs-off-policy]]."),
    # link: a and b are not linked, but both link S1, S2 and hub, c, d and e link both of them, and only they have
    # the tag [[end-state-checks]] (tags count as evidence for a link, like shared neighbours and text)
    "20_ZETTLEKASTEN/a.md": zettel("[[grading]] [[end-state-checks]]", "Graders must check end state. From [[S1]] and [[S2]], see [[hub]]."),
    "20_ZETTLEKASTEN/b.md": zettel("[[grading]] [[end-state-checks]]", "End state graders ignore the path. From [[S1]] and [[S2]], see [[hub]]."),
    "20_ZETTLEKASTEN/hub.md": zettel("[[grading]]", "From [[S3]]."),
    "20_ZETTLEKASTEN/c.md": zettel("[[grading]]", "From [[S3]]. Both [[a]] and [[b]]."),
    "20_ZETTLEKASTEN/d.md": zettel("[[grading]]", "From [[S3]]. Both [[a]] and [[b]]."),
    "20_ZETTLEKASTEN/e.md": zettel("[[grading]]", "From [[S3]]. Both [[a]] and [[b]]."),
    "50_MOC/discourse/old.md": "Links from an ignored note count for nothing: [[grader-hacking]] [[ignored-only]] [[ignored-only]]\n",
    "50_MOC/discourse/old2.md": "[[ignored-only]]\n",
    "50_MOC/discourse/old3.md": "[[ignored-only]]\n",
}


class Vault:
    def __init__(self, files=FILES):
        self.dir = tempfile.TemporaryDirectory()
        self.root = Path(self.dir.name)
        for path, text in files.items():
            self.write(path, text)
        self.index()

    def write(self, path, text):
        (self.root / path).parent.mkdir(parents=True, exist_ok=True)
        (self.root / path).write_text(text, encoding="utf-8")

    def index(self):
        result = run_vault("index", str(self.root))
        assert result.returncode == 0, result.stderr

    def findings(self, *extra) -> dict[str, dict]:
        result = run_vault("findings", str(self.root), "-f", "core", "--json", *extra)
        assert result.returncode == 0, result.stderr
        return {f["id"]: f for f in json.loads(result.stdout)["findings"]}

    def snapshot(self):
        return {str(p.relative_to(self.root)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in self.root.rglob("*") if p.is_file() and not p.name.endswith(("-wal", "-shm"))}

    def close(self):
        self.dir.cleanup()


class FindingsTests(unittest.TestCase):
    def setUp(self):
        self.vault = Vault()

    def tearDown(self):
        self.vault.close()

    def test_one_of_each_method(self):
        found = self.vault.findings()
        by_method = {}
        for f in found.values():
            by_method.setdefault(f["method"], []).append(f)
        missing = {f["notes"][0]["name"]: f for f in by_method["missing-note"]}
        # llm-judge* are linked from two notes each: not "many". [[rl]], [[grading]], [[ml]] are only in Tags lines:
        # tags, not missing notes.
        self.assertEqual(sorted(missing), ["grader-hacking"])
        self.assertEqual(missing["grader-hacking"]["notes"][0]["linking_notes"], 4)  # not the ignored note
        self.assertNotIn("ignored-only", str(found))  # linked only from an ignored folder
        same = by_method["same-idea"]
        self.assertEqual([[n["name"] for n in f["notes"]] for f in same], [["llm-judge", "llm-judge-position-bias"]])
        name = by_method["existing-name"]
        self.assertEqual([[n["name"] for n in f["notes"]] for f in name], [["on-policy-vs-off-policy", "on-policy-versus-off-policy"]])
        links = [sorted(n["name"] for n in f["notes"]) for f in by_method["link"]]
        self.assertIn(["a", "b"], links)
        link = next(f for f in by_method["link"] if sorted(n["name"] for n in f["notes"]) == ["a", "b"])
        self.assertEqual(link["id"], finding_id("link", ["20_ZETTLEKASTEN/a.md", "20_ZETTLEKASTEN/b.md"]))
        self.assertTrue(any(line.startswith("cited together by: c, d, e") for line in link["evidence"]), link["evidence"])
        for f in found.values():
            self.assertRegex(f["id"], rf"^{PREFIX[f['method']]}-[0-9a-f]{{10}}$")

    def test_thresholds_are_percentiles_of_the_vault(self):
        data = json.loads(run_vault("findings", str(self.vault.root), "-f", "core", "--json").stdout)
        # Linking notes of the missing names (tags and ignored notes do not count): 4, 2, 2, 2. The 90th percentile
        # is 4, so only grader-hacking is a missing-note finding; the 75th is 2, the existing-name floor.
        self.assertEqual(data["thresholds"]["missing-note"],
                         {"linking_notes": 4, "percentile": 90, "floor": 2, "percentile_value": 4})
        self.assertEqual(data["thresholds"]["existing-name"]["linking_notes"], 2)
        self.assertGreaterEqual(data["thresholds"]["link"]["score"], 0.10)
        # Ten names linked once each: the 90th percentile drops to 2, and the names two notes link become findings.
        self.vault.write("20_ZETTLEKASTEN/many.md", zettel("[[other]]", "From [[S3]]. " + " ".join(f"[[once-{i}]]" for i in range(10))))
        self.vault.index()
        data = json.loads(run_vault("findings", str(self.vault.root), "-f", "core", "--json").stdout)
        self.assertEqual(data["thresholds"]["missing-note"]["linking_notes"], 2)
        names = sorted(f["notes"][0]["name"] for f in data["findings"] if f["method"] == "missing-note")
        self.assertEqual(names, ["grader-hacking", "llm-judge", "llm-judge-position-bias", "on-policy-vs-off-policy"])

    def test_ids_are_derived_from_method_and_notes(self):
        self.assertEqual(finding_id("link", ["A.md", "b.md"]), finding_id("link", ["b.md", "a.md"]))  # order, case
        self.assertNotEqual(finding_id("link", ["a.md", "b.md"]), finding_id("link", ["a.md", "c.md"]))
        self.assertNotEqual(finding_id("missing-note", ["x"]), finding_id("same-idea", ["x"]))
        self.assertEqual(finding_id("missing-note", ["grader-hacking"]), "missing-" + hashlib.sha256(
            "missing-note\0grader-hacking".encode()).hexdigest()[:10])

    def test_stable_across_runs(self):
        self.assertEqual(self.vault.findings(), self.vault.findings())

    def test_ids_survive_unrelated_edits(self):
        before = self.vault.findings()
        # Unrelated: a new source and zettel of their own, an edit to a note's text, a deleted ignored note.
        self.vault.write("10_SOURCES/S9.md", source())
        self.vault.write("20_ZETTLEKASTEN/unrelated.md", zettel("[[other]]", "About something else. From [[S9]]."))
        self.vault.write("20_ZETTLEKASTEN/hub.md", zettel("[[grading]]", "From [[S3]]. More words about the hub."))
        (self.vault.root / "50_MOC/discourse/old3.md").unlink()
        self.vault.index()
        after = self.vault.findings()
        self.assertEqual({i: f["notes"] for i, f in before.items()}, {i: f["notes"] for i, f in after.items()})

    def test_new_evidence_keeps_the_id(self):
        before = self.vault.findings()
        missing = finding_id("missing-note", ["grader-hacking"])
        link = finding_id("link", ["20_ZETTLEKASTEN/a.md", "20_ZETTLEKASTEN/b.md"])
        self.assertEqual(before[missing]["notes"][0]["linking_notes"], 4)
        # A fifth note links the missing note; a fourth note cites a and b together.
        self.vault.write("10_SOURCES/S9.md", source())
        self.vault.write("20_ZETTLEKASTEN/z7.md", zettel("[[rl]]", "From [[S9]]. Also [[grader-hacking]]."))
        self.vault.write("20_ZETTLEKASTEN/f.md", zettel("[[grading]]", "From [[S3]]. Both [[a]] and [[b]]."))
        self.vault.index()
        after = self.vault.findings()
        self.assertEqual(after[missing]["notes"][0]["linking_notes"], 5)  # the evidence changed, the id did not
        self.assertIn("cited together by: c, d, e, f", after[link]["evidence"])

    def test_a_finding_goes_when_it_is_resolved(self):
        finding = self.vault.findings()[finding_id("missing-note", ["grader-hacking"])]
        self.vault.write("20_ZETTLEKASTEN/grader-hacking.md", zettel("[[grading]]", "From [[S1]]."))
        self.vault.index()
        self.assertNotIn(finding["id"], self.vault.findings())

    def test_reading_writes_nothing(self):
        before = self.vault.snapshot()
        self.vault.findings()
        run_vault("findings", str(self.vault.root), "-f", "core")
        self.assertEqual(self.vault.snapshot(), before)

    def test_one_finding_and_errors(self):
        one = next(iter(self.vault.findings()))
        shown = run_vault("findings", str(self.vault.root), "-f", "core", "--id", one)
        self.assertIn(one, shown.stdout)
        missing = run_vault("findings", str(self.vault.root), "-f", "core", "--id", "link-0000000000")
        self.assertEqual(missing.returncode, 2)
        (self.vault.root / ".vaultengine/index.sqlite").unlink()
        no_index = run_vault("findings", str(self.vault.root), "-f", "core")
        self.assertEqual(no_index.returncode, 2)
        self.assertIn("run `vault index", no_index.stderr)


@requires_model
class SameAsPairByPairTests(unittest.TestCase):
    """The accumulating implementation gives the pair-by-pair one's results on the sample: the same findings (ids,
    titles, notes, evidence, in the same order), thresholds, and every link suggestion with the same scores."""

    @classmethod
    def setUpClass(cls):
        cls.reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
        cls.dir = tempfile.TemporaryDirectory()
        cls.root = Path(cls.dir.name) / "vault"
        shutil.copytree(SAMPLE, cls.root)
        assert run_vault("index", str(cls.root)).returncode == 0

    @classmethod
    def tearDownClass(cls):
        cls.dir.cleanup()

    def test_findings_and_thresholds(self):
        result = run_vault("findings", str(self.root), "-f", "core", "--json")
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertEqual(data["thresholds"], self.reference["thresholds"])
        found = [{k: f[k] for k in ("id", "method", "title", "notes", "evidence")} for f in data["findings"]]
        self.assertEqual(len(found), 94)
        self.assertEqual(found, self.reference["findings"])

    def test_link_cap_and_floor(self):
        """engine.findings.link_limit keeps the best links (the same, in the same order); link_floor raises the bar."""
        links = [f["id"] for f in self.reference["findings"] if f["method"] == "link"]
        capped = run_vault("findings", str(self.root), "-f", "core", "--json", "--method", "link", "--limit", "0",
                           "--setting", "findings.link_limit=3")
        self.assertEqual(capped.returncode, 0, capped.stderr)
        data = json.loads(capped.stdout)
        self.assertEqual([f["id"] for f in data["findings"]], links[:3])
        self.assertEqual((data["thresholds"]["link"]["limit"], data["thresholds"]["link"]["capped"]), (3, True))
        high = run_vault("findings", str(self.root), "-f", "core", "--json", "--method", "link", "--limit", "0",
                         "--setting", "findings.link_floor=0.9")
        data = json.loads(high.stdout)
        self.assertEqual((data["findings"], data["thresholds"]["link"]["score"]), ([], 0.9))

    def test_every_link_suggestion(self):
        from vaultengine.connect import LinkSuggester
        from vaultengine.findings import counts_for
        from vaultengine.formats import load_format

        counts = counts_for(load_format(REPO_ROOT / "formats" / "core.yaml"))
        conn = sqlite3.connect(self.root / ".vaultengine" / "index.sqlite")
        conn.row_factory = sqlite3.Row
        self.addCleanup(conn.close)
        for_note = self.reference["links_for_note"][0]
        note_path = ({for_note[0], for_note[1]} & {self.reference["links_for_note"][1][0], self.reference["links_for_note"][1][1]}).pop()
        cases = {"links_core_all": (counts, {}), "links_plain_all": (None, {}),
                 "links_path_zettels": (None, {"path": "20_ZETTLEKASTEN"}), "links_for_note": (None, {"for_note": note_path})}
        for name, (counting, kwargs) in cases.items():
            with self.subTest(name):
                suggester = LinkSuggester(conn, None, counting)
                if "for_note" in kwargs:
                    kwargs = {"for_note": next(i for i, p in suggester.paths.items() if p == note_path)}
                got = [[s.a, s.b, s.score, s.shared, s.cited, s.tags, s.text, s.shared_with, s.cited_by, s.shared_tags,
                        s.terms, s.reason] for s in suggester.suggestions(limit=None, **kwargs)]
                want = self.reference[name]
                self.assertEqual(len(got), len(want))
                for mine, theirs in zip(got, want):
                    self.assertEqual(mine[:2] + mine[7:], theirs[:2] + theirs[7:])
                    for value, expected in zip(mine[2:7], theirs[2:7]):
                        self.assertAlmostEqual(value, expected, places=6)


class IdeaTests(unittest.TestCase):
    def setUp(self):
        self.vault = Vault()
        self.link = next(f for f in self.vault.findings().values()
                         if f["method"] == "link" and sorted(n["name"] for n in f["notes"]) == ["a", "b"])

    def tearDown(self):
        self.vault.close()

    def idea(self, *extra):
        return run_vault("idea", "new", str(self.vault.root), "--from", self.link["id"], "-f", "core", *extra)

    def test_nothing_written_without_write(self):
        before = self.vault.snapshot()
        result = self.idea()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"- id: {self.link['id']}", result.stdout)
        self.assertIn("Not written: add --write to write it.", result.stdout)
        self.assertEqual(self.vault.snapshot(), before)
        self.assertEqual(json.loads(self.idea("--json").stdout)["written"], False)
        self.assertEqual(self.vault.snapshot(), before)

    def test_an_idea_from_a_finding_passes_vault_check(self):
        self.assertEqual(run_vault("check", str(self.vault.root), "-f", "core").returncode, 0)
        result = self.idea("--write")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        path = self.vault.root / "50_MOC/ideas/a and b.md"
        text = path.read_text()
        lines = text.splitlines()
        self.assertRegex(lines[0], r"^Created: \d\d-\d\d-\d{4} \d\d:\d\d$")
        self.assertEqual(lines[1:5], ["Tags: [[grading]] [[end-state-checks]]", "Status: #open", "", "# a and b"])
        block = text[text.index("<!-- vault:finding -->"):text.index("<!-- /vault:finding -->")]
        for line in (f"- id: {self.link['id']}", "- method: link (two notes that probably belong linked)",
                     "- notes: [[a]], [[b]]", "- evidence:", "  - cited together by: c, d, e"):
            self.assertIn(line, block)
        self.assertTrue(text.endswith("<!-- /vault:finding -->\n\n## Claim\n\n"))
        check = run_vault("check", str(self.vault.root), "-f", "core")
        self.assertEqual(check.returncode, 0, check.stdout)
        # The idea is not a finding of its own, and the finding it came from keeps its id.
        after = self.vault.findings()
        self.assertIn(self.link["id"], after)
        self.assertFalse([f for f in after.values() if any("ideas/" in (n["path"] or "") for n in f["notes"])])

    def test_an_idea_without_a_finding_is_refused_and_fails_check(self):
        before = self.vault.snapshot()
        refused = run_vault("new", str(self.vault.root), "-f", "core", "--type", "idea", "--title", "A thought",
                            "--set", "Tags=[[x]]", "--body", "No finding here.")
        self.assertEqual(refused.returncode, 1)
        self.assertIn("missing-block        no finding block (<!-- vault:finding --> … <!-- /vault:finding -->)", refused.stdout)
        self.assertEqual(self.vault.snapshot(), before)
        head = "Created: 01-01-2026 10:00\nTags: [[x]]\nStatus: #open\n\n# T\n\n"
        cases = {
            "none.md": (head + "Just a claim.\n", "no finding block"),
            "empty.md": (head + "<!-- vault:finding -->\n\n<!-- /vault:finding -->\n", "the finding block is empty"),
            "unclosed.md": (head + "<!-- vault:finding -->\n- id: x\n", "markers are not one opening followed by one closing"),
            "in code.md": (head + "```\n<!-- vault:finding -->\n- id: x\n<!-- /vault:finding -->\n```\n", "no finding block"),
        }
        for name, (text, message) in cases.items():
            self.vault.write(f"50_MOC/ideas/{name}", text)
        data = json.loads(run_vault("check", str(self.vault.root), "-f", "core", "--json").stdout)
        found = {n["path"].split("/")[-1]: [(v["code"], v["message"]) for v in n["violations"]] for n in data["notes"]}
        for name, (_, message) in cases.items():
            with self.subTest(name=name):
                self.assertEqual([code for code, _ in found[name]], ["missing-block"])
                self.assertIn(message, found[name][0][1])

    def test_status(self):
        self.assertEqual(self.idea("--write").returncode, 0)
        idea = "50_MOC/ideas/a and b.md"
        before = self.vault.snapshot()
        shown = run_vault("idea", "set", str(self.vault.root), idea, "done", "-f", "core")
        self.assertIn("+Status: #done", shown.stdout)
        self.assertEqual(self.vault.snapshot(), before)  # nothing without --write
        for status in ("dropped", "#done", "open"):
            with self.subTest(status=status):
                result = run_vault("idea", "set", str(self.vault.root), idea, status, "-f", "core", "--write")
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn(f"Status: #{status.lstrip('#')}\n", (self.vault.root / idea).read_text())
        refused = run_vault("idea", "set", str(self.vault.root), idea, "parked", "-f", "core", "--write")
        self.assertEqual(refused.returncode, 1)
        self.assertIn("Status: #parked is not an allowed value (open, dropped, done)", refused.stdout)
        not_idea = run_vault("idea", "set", str(self.vault.root), "z1", "done", "-f", "core", "--write")
        self.assertEqual(not_idea.returncode, 2)
        self.assertIn("20_ZETTLEKASTEN/z1.md is a zettel, not a idea", not_idea.stderr)

    def test_unknown_finding_and_other_methods(self):
        result = run_vault("idea", "new", str(self.vault.root), "--from", "missing-0000000000", "-f", "core")
        self.assertEqual(result.returncode, 2)
        self.assertIn("no finding 'missing-0000000000' now", result.stderr)
        titles = {}
        for finding in self.vault.findings().values():
            data = json.loads(run_vault("idea", "new", str(self.vault.root), "--from", finding["id"], "-f", "core",
                                        "--json").stdout)
            self.assertEqual(data["status"], "dry-run", data)
            titles[finding["method"]] = data["idea"]["path"]
        self.assertEqual(titles["missing-note"], "50_MOC/ideas/Missing note grader-hacking.md")  # not grader-hacking.md
        self.assertEqual(titles["same-idea"], "50_MOC/ideas/llm-judge or llm-judge-position-bias.md")
        self.assertEqual(titles["existing-name"], "50_MOC/ideas/on-policy-vs-off-policy means on-policy-versus-off-policy.md")


if __name__ == "__main__":
    unittest.main()
