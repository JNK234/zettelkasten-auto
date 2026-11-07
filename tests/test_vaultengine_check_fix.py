import difflib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path

from vaultengine import Vault
from vaultengine.formats import TypedVault, load_format, parse_format
from vaultengine.formats.fix import apply_fixes, plan_fixes, unified_diff

TESTS = Path(__file__).parent
VAULT = TESTS / "fixtures" / "vault"
REPO_ROOT = TESTS.parent
DISCOURSE = "50_MOC/discourse/rl-environment-design"
CLEAN_CLAIM = f"{DISCOURSE}/CLM - Without a real end state, grade the trajectory against path-agnostic subgoals.md"
ITS_EVIDENCE = f"{DISCOURSE}/EVD - Where there is no real state to inspect, SynthAgent grades the trajectory against a.md"

FORMAT = """
format: 1
relations:
  supports: {inverse: supported-by, targets: [claim]}
  supported-by: {targets: [evidence]}
  opposes: {inverse: opposed-by}
types:
  claim:
    match: {name: "CLM - *"}
    relations_heading: Relations
    headings: [Relations, Scope]
    ordered_headings: true
    relations: {optional: [supported-by, opposed-by]}
  evidence:
    match: {name: "EVD - *"}
    relations: {optional: [supports, opposes]}
  zettel:
    match: {folder: Z}
    headings: [What This Is, Tensions, References]
  loose:
    match: {folder: L}
    headings: ["/^conn/"]
"""


def run_vault(*args: str, cwd: Path = REPO_ROOT) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "vaultengine", *args], capture_output=True, text=True, cwd=cwd)


class TempVault:
    """A vault on disk; files given as text are written as UTF-8 bytes exactly as given."""

    def __init__(self, files: dict[str, str | bytes]) -> None:
        self.dir = tempfile.TemporaryDirectory()
        self.root = Path(self.dir.name)
        for path, content in files.items():
            target = self.root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content if isinstance(content, bytes) else content.encode("utf-8"))

    def typed(self, format_text: str = FORMAT) -> TypedVault:
        return TypedVault(parse_format(textwrap.dedent(format_text)), Vault.load(self.root))

    def read(self, path: str) -> bytes:
        return (self.root / path).read_bytes()

    def close(self) -> None:
        self.dir.cleanup()


def only_insertions(old: bytes, new: bytes) -> bool:
    ops = difflib.SequenceMatcher(None, old.splitlines(True), new.splitlines(True), autojunk=False).get_opcodes()
    return all(op[0] in ("equal", "insert") for op in ops)


# ----- fix: planning and writing ---------------------------------------------------------------


class MirrorFixTests(unittest.TestCase):
    def setUp(self):
        self.vault = None

    def tearDown(self):
        if self.vault:
            self.vault.close()

    def fix(self, files, **kwargs):
        self.vault = TempVault(files)
        typed = self.vault.typed()
        plan = plan_fixes(typed, **kwargs)
        return plan, {f.path: f.new for f in plan.files}

    def test_appends_to_the_existing_field(self):
        plan, new = self.fix({
            "CLM - c.md": "# C\n## Relations\nsupported-by:: [[EVD - a]]  \n## Scope\n",
            "EVD - a.md": "supports:: [[CLM - c]]\n",
            "EVD - b.md": "supports:: [[CLM - c]]\n",
            "EVD - d.md": "- supports:: [[CLM - c]]\n",
        })
        self.assertEqual(new["CLM - c.md"], "# C\n## Relations\nsupported-by:: [[EVD - a]], [[EVD - b]], [[EVD - d]]  \n## Scope\n")
        self.assertEqual(plan.fixes, 2)
        self.assertEqual(plan.files[0].changes, ["line 3: added [[EVD - b]], [[EVD - d]] to `supported-by::`"])

    def test_empty_field_gets_a_value(self):
        _, new = self.fix({"CLM - c.md": "## Relations\nsupported-by::\n## Scope\n", "EVD - a.md": "supports:: [[CLM - c]]\n"})
        self.assertEqual(new["CLM - c.md"], "## Relations\nsupported-by:: [[EVD - a]]\n## Scope\n")

    def test_new_line_after_the_last_relation_field_keeps_its_prefix(self):
        _, new = self.fix({
            "CLM - c.md": "## Relations\n- opposed-by:: [[EVD - x]]\n\n## Scope\n",
            "EVD - a.md": "supports:: [[CLM - c]]\n",
            "EVD - x.md": "opposes:: [[CLM - c]]\n",
        })
        self.assertEqual(new["CLM - c.md"], "## Relations\n- opposed-by:: [[EVD - x]]\n- supported-by:: [[EVD - a]]\n\n## Scope\n")

    def test_under_the_relations_heading(self):
        _, new = self.fix({"CLM - c.md": "# C\n\n## Relations\n\n## Scope\n", "EVD - a.md": "supports:: [[CLM - c]]\n"})
        self.assertEqual(new["CLM - c.md"], "# C\n\n## Relations\nsupported-by:: [[EVD - a]]\n\n## Scope\n")

    def test_reverse_direction(self):
        _, new = self.fix({"CLM - c.md": "## Relations\nsupported-by:: [[EVD - a]]\n## Scope\n", "EVD - a.md": "# E\n"})
        self.assertEqual(new, {})  # no relation field and no relations_heading for evidence
        _, new = self.fix({"CLM - c.md": "## Relations\nsupported-by:: [[EVD - a]]\n## Scope\n",
                           "EVD - a.md": "opposes:: [[CLM - x]]\n", "CLM - x.md": "## Relations\nopposed-by:: [[EVD - a]]\n## Scope\n"})
        self.assertEqual(new["EVD - a.md"], "opposes:: [[CLM - x]]\nsupports:: [[CLM - c]]\n")

    def test_no_place_or_not_allowed_is_skipped(self):
        # Evidence has no relation field yet and its type sets no relations_heading.
        plan, new = self.fix({"CLM - c.md": "## Relations\nsupported-by:: [[EVD - a]]\n## Scope\n", "EVD - a.md": "# E\n"})
        self.assertEqual(new, {})
        self.assertEqual([(i.path, i.code, reason) for i, reason in plan.skipped],
                         [("EVD - a.md", "missing-mirror", "no relation field to extend and no relations_heading to put one under")])
        strict = FORMAT.replace("relations: {optional: [supported-by, opposed-by]}", "relations: {optional: [opposed-by]}")
        self.vault.close()
        self.vault = TempVault({"CLM - c.md": "## Relations\n## Scope\n", "EVD - a.md": "supports:: [[CLM - c]]\n"})
        plan = plan_fixes(self.vault.typed(strict))
        self.assertEqual([reason for _, reason in plan.skipped], ["`supported-by::` is not a relation of claim"])

    def test_name_that_is_not_unique_is_written_as_a_path(self):
        _, new = self.fix({"CLM - c.md": "## Relations\nsupported-by::\n## Scope\n",
                           "A/EVD - a.md": "supports:: [[CLM - c]]\n", "B/EVD - a.md": "# other\n"})
        self.assertEqual(new["CLM - c.md"], "## Relations\nsupported-by:: [[A/EVD - a]]\n## Scope\n")


class HeadingFixTests(unittest.TestCase):
    def fix(self, files, format_text=FORMAT):
        vault = TempVault(files)
        try:
            plan = plan_fixes(vault.typed(format_text))
            return plan, {f.path: f.new for f in plan.files}
        finally:
            vault.close()

    def test_before_the_next_required_heading_at_the_usual_level(self):
        others = {f"Z/n{i}.md": "## What This Is\n## Tensions\n# References\n" for i in range(3)}
        _, new = self.fix({**others, "Z/a.md": "# Title\n## Definition\ntext\n# References\n1. x\n"})
        self.assertEqual(new["Z/a.md"], "# Title\n## Definition\ntext\n\n## What This Is\n\n## Tensions\n\n# References\n1. x\n")

    def test_at_the_end_with_and_without_a_final_newline(self):
        others = {"Z/n.md": "## What This Is\n## Tensions\n# References\n"}
        _, new = self.fix({**others, "Z/a.md": "## What This Is\n## Tensions\ntext", "Z/b.md": "## What This Is\n## Tensions\n\n"})
        self.assertEqual(new["Z/a.md"], "## What This Is\n## Tensions\ntext\n\n# References\n")
        self.assertEqual(new["Z/b.md"], "## What This Is\n## Tensions\n\n# References\n")

    def test_neighbour_level_when_no_note_has_it(self):
        _, new = self.fix({"Z/a.md": "### What This Is\n"})
        self.assertEqual(new["Z/a.md"], "### What This Is\n\n### Tensions\n\n### References\n")
        _, new = self.fix({"Z/a.md": "text\n"})
        self.assertEqual(new["Z/a.md"], "text\n\n## What This Is\n\n## Tensions\n\n## References\n")

    def test_skipped_when_out_of_order_or_a_pattern(self):
        ordered = FORMAT.replace("headings: [What This Is, Tensions, References]",
                                 "headings: [What This Is, Tensions, References]\n    ordered_headings: true")
        plan, new = self.fix({"Z/a.md": "## Tensions\n## What This Is\n", "L/x.md": "# x\n"}, ordered)
        self.assertEqual(new, {})
        reasons = sorted(reason for _, reason in plan.skipped)
        self.assertEqual(reasons, ["a required heading is a pattern, not a name to write",
                                   "the note's headings are out of order; fix by hand"])

    def test_repair_that_would_not_hold_is_rejected(self):
        # The note ends inside an unclosed code fence, so a heading appended there is code, not a heading.
        plan, new = self.fix({"Z/a.md": "## What This Is\n## Tensions\n```\ncode\n"})
        self.assertEqual(new, {})
        self.assertEqual([reason for _, reason in plan.skipped], ["the repair would not clear its violations in Z/a.md"])


class ByteExactTests(unittest.TestCase):
    def test_crlf_bom_and_trailing_spaces_survive(self):
        raw = "\ufeffStatus: x\r\n\r\n## Relations\r\nsupported-by:: [[EVD - a]] \r\nkeep  \r\n## Scope\r\n".encode("utf-8")
        vault = TempVault({"CLM - c.md": raw, "EVD - a.md": "supports:: [[CLM - c]]\n", "EVD - b.md": "supports:: [[CLM - c]]\n"})
        try:
            plan = plan_fixes(vault.typed())
            written, refused = apply_fixes(plan, vault.root)
            self.assertEqual((len(written), refused), (1, []))
            self.assertEqual(
                vault.read("CLM - c.md"),
                "\ufeffStatus: x\r\n\r\n## Relations\r\nsupported-by:: [[EVD - a]], [[EVD - b]] \r\nkeep  \r\n## Scope\r\n".encode("utf-8"),
            )
        finally:
            vault.close()

    def test_inserted_lines_use_the_file_newline(self):
        vault = TempVault({"Z/a.md": b"## What This Is\r\n## Tensions\r\n", "Z/b.md": b"## What This Is\n## Tensions\n# References\n"})
        try:
            apply_fixes(plan_fixes(vault.typed()), vault.root)
            self.assertEqual(vault.read("Z/a.md"), b"## What This Is\r\n## Tensions\r\n\r\n# References\r\n")
        finally:
            vault.close()

    def test_fixture_vault_only_gains_lines(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch) / "vault"
            shutil.copytree(VAULT, root)
            before = {p: p.read_bytes() for p in root.rglob("*.md")}
            typed = TypedVault(load_format(REPO_ROOT / "formats" / "brain.yaml"), Vault.load(root))
            plan = plan_fixes(typed)
            self.assertEqual(len(plan.files), 28)  # 27 zettels + RLHF.md; the fixture has no missing mirror
            written, _ = apply_fixes(plan, root)
            changed = [p for p, old in before.items() if p.read_bytes() != old]
            self.assertEqual(len(changed), len(written))
            for path in changed:
                old, new = before[path], path.read_bytes()
                self.assertTrue(only_insertions(old, new) or (new.startswith(old) and not old.endswith(b"\n")), path)
            after = TypedVault(typed.spec, Vault.load(root))
            self.assertEqual([i for i in after.issues() if i.code == "missing-heading"], [])
            self.assertEqual(plan_fixes(after).files, [])  # idempotent

    def test_changed_on_disk_is_not_written(self):
        vault = TempVault({"Z/a.md": "## What This Is\n## Tensions\n"})
        try:
            plan = plan_fixes(vault.typed())
            (vault.root / "Z/a.md").write_text("edited meanwhile\n")
            written, refused = apply_fixes(plan, vault.root)
            self.assertEqual(written, [])
            self.assertIn("changed while the write was being prepared", refused[0][1])
            self.assertEqual(vault.read("Z/a.md"), b"edited meanwhile\n")
        finally:
            vault.close()

    def test_not_utf8_is_not_touched(self):
        vault = TempVault({"Z/a.md": b"## What This Is\n\xff\n"})
        try:
            plan = plan_fixes(vault.typed())
            self.assertEqual(plan.files, [])
            self.assertEqual([reason for _, reason in plan.skipped], ["file is not valid UTF-8; not touched"])
        finally:
            vault.close()

    def test_diff(self):
        vault = TempVault({"Z/a.md": "## What This Is\n## Tensions\ntext"})
        try:
            diff = unified_diff(plan_fixes(vault.typed()))
            self.assertIn("--- a/Z/a.md\n+++ b/Z/a.md\n", diff)
            self.assertIn("-text\n\\ No newline at end of file\n+text\n+\n+## References\n", diff)
        finally:
            vault.close()


# ----- CLI: check and fix ---------------------------------------------------------------------


class CheckCliTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory()
        self.root = Path(self.scratch.name) / "vault"
        shutil.copytree(VAULT, self.root)

    def tearDown(self):
        self.scratch.cleanup()

    def check(self, *extra):
        return run_vault("check", str(self.root), "--format", "brain", *extra)

    def edit(self, path: str, old: str, new: str) -> None:
        target = self.root / path
        text = target.read_text(encoding="utf-8")
        self.assertIn(old, text)
        target.write_text(text.replace(old, new), encoding="utf-8")

    def index(self):
        self.assertEqual(run_vault("index", str(self.root)).returncode, 0)
        time.sleep(0.01)

    def test_full_check(self):
        result = self.check()
        self.assertEqual(result.returncode, 1)
        self.assertIn("  checked 180 notes\n", result.stdout)
        self.assertIn("10_SOURCES/RLHF.md  (source)\n    138  missing-heading      missing heading 'Notes' (expected at the end of the note)", result.stdout)
        self.assertIn("✗ 102 violations in 77 notes\n      59  unresolved-target\n      28  missing-heading\n      15  missing-key", result.stdout)
        data = json.loads(self.check("--json").stdout)
        self.assertFalse(data["ok"])
        self.assertEqual(data["summary"]["violations"], 102)
        self.assertTrue(all(v["line"] >= 1 for note in data["notes"] for v in note["violations"]))

    def test_changed_needs_an_index(self):
        result = self.check("--changed")
        self.assertEqual(result.returncode, 2)
        self.assertIn("--changed needs an index", result.stderr)

    def test_changed_catches_only_the_broken_note(self):
        self.index()
        clean = self.check("--changed")
        self.assertEqual(clean.returncode, 0, clean.stdout)
        self.assertIn("✓ no violations in 0 notes checked", clean.stdout)

        self.edit(CLEAN_CLAIM, "Status: #supported", "Status: #suported")
        result = self.check("--changed", "--json")
        self.assertEqual(result.returncode, 1)
        data = json.loads(result.stdout)
        self.assertEqual(data["scope"]["changed"], [CLEAN_CLAIM])
        self.assertEqual([(n["path"], [v["code"] for v in n["violations"]]) for n in data["notes"]], [(CLEAN_CLAIM, ["bad-value"])])
        self.assertEqual(data["notes"][0]["violations"][0]["line"], 5)

    def test_changed_reports_what_a_change_breaks_elsewhere(self):
        self.index()
        other_claim = f"{DISCOURSE}/CLM - Correctness must gate all shaping credit.md"
        self.edit(ITS_EVIDENCE, "supports:: [[", "supports:: [[CLM - Correctness must gate all shaping credit]], [[")
        data = json.loads(self.check("--changed", "--json").stdout)
        notes = {n["path"]: n for n in data["notes"]}
        self.assertIn(other_claim, notes)
        self.assertFalse(notes[other_claim]["checked"])
        self.assertEqual(notes[other_claim]["violations"][0]["code"], "missing-mirror")
        self.assertEqual(notes[other_claim]["violations"][0]["related"], ITS_EVIDENCE)
        text = self.check("--changed").stdout
        self.assertIn("(claim, not changed: affected by a changed or deleted note)", text)

    def test_changed_reports_links_to_deleted_notes(self):
        self.index()
        (self.root / ITS_EVIDENCE).unlink()
        data = json.loads(self.check("--changed", "--json").stdout)
        self.assertEqual(data["scope"]["deleted"], [ITS_EVIDENCE])
        self.assertEqual([(n["path"], [v["code"] for v in n["violations"]]) for n in data["notes"]],
                         [(CLEAN_CLAIM, ["unresolved-target"])])

    def test_fix_dry_run_apply_recheck(self):
        self.edit(CLEAN_CLAIM, "supported-by:: [[EVD - Where there is no real state to inspect, SynthAgent grades the trajectory against a]]\n", "")
        before = {p: p.read_bytes() for p in self.root.rglob("*.md")}
        dry = run_vault("fix", str(self.root), "-f", "brain", "--dry-run")
        self.assertEqual(dry.returncode, 0, dry.stderr)
        self.assertIn("+supported-by:: [[EVD - Where there is no real state to inspect, SynthAgent grades the trajectory against a]]", dry.stdout)
        self.assertIn("Would fix 29 notes:", dry.stdout)
        self.assertIn("Dry run: nothing was written.", dry.stdout)
        self.assertEqual({p: p.read_bytes() for p in self.root.rglob("*.md")}, before)

        applied = run_vault("fix", str(self.root), "-f", "brain")
        self.assertIn("Fixed 29 notes:", applied.stdout)
        self.assertIn("Violations left after this fix, for you to fix: 59 unresolved-target, 15 missing-key", applied.stdout)
        after = self.check()
        self.assertIn("✗ 74 violations in 49 notes", after.stdout)
        self.assertNotIn("missing-mirror", after.stdout)
        self.assertNotIn("missing-heading", after.stdout)
        self.assertIn("Nothing to fix automatically.", run_vault("fix", str(self.root), "-f", "brain").stdout)

    def test_fix_only(self):
        data = json.loads(run_vault("fix", str(self.root), "-f", "brain", "--dry-run", "--only", "missing-mirror", "--json").stdout)
        self.assertEqual(data["files"], [])
        self.assertTrue(data["dry_run"])


if __name__ == "__main__":
    unittest.main()
