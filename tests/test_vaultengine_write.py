"""The safe write path: vault new / link / set, and the commit machinery under them.

Refusals are tested as carefully as successes: every refusal test checks that no
file's bytes or mtime changed (dot-files included), that the index is logically
unchanged, and that no journal or temporary file was left behind.
"""

import datetime as dt
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from vaultengine import Vault
from vaultengine.formats import TypedVault, load_format, parse_format
from vaultengine.formats import write as write_module
from vaultengine.formats.write import (
    FileChange, WriteError, WriteFailed, WritePlan, commit, note_name, plan_link, plan_new, plan_set, recover,
    run_write, write_lock,
)
from vaultengine.index import VaultIndex

TESTS = Path(__file__).parent
VAULT = TESTS / "fixtures" / "vault"
REPO_ROOT = TESTS.parent
BRAIN = load_format(REPO_ROOT / "formats" / "brain.yaml")
D = "50_MOC/discourse/rl-environment-design"
EVD_REWARD = "EVD - Reward is a deterministic outcome check 1 only if the final database equals the data"
CLM_END_STATE = "CLM - Grade the observed end state rather than the trajectory"
NOW = dt.datetime(2026, 11, 6, 9, 30)


def run_vault(*args, env=None) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "vaultengine", *args], capture_output=True, text=True, cwd=REPO_ROOT,
                          env={**os.environ, **(env or {})})


def files_state(root: Path) -> dict[str, tuple[bytes, int]]:
    """Every file under the vault (dot-files too) with its bytes and mtime, except the index and lock."""
    state = {}
    for directory, _, names in os.walk(root):
        for name in names:
            path = Path(directory) / name
            rel = path.relative_to(root).as_posix()
            if rel.startswith(".vaultengine/index.sqlite"):  # compared separately, logically (index_dump)
                continue
            state[rel] = (path.read_bytes(), path.stat().st_mtime_ns)
    return state


def index_dump(root: Path, with_mtimes: bool = True) -> dict:
    db = root / ".vaultengine" / "index.sqlite"
    if not db.exists():
        return {}
    conn = sqlite3.connect(db)
    try:
        mtime = ", f.mtime_ns" if with_mtimes else ""
        return {
            "meta": sorted(conn.execute("SELECT key, value FROM meta WHERE key NOT IN ('last_indexed_at', 'generation')")
                           if not with_mtimes else conn.execute("SELECT key, value FROM meta")),
            "files": sorted(conn.execute(f"SELECT f.path, f.kind, f.size, f.content_hash{mtime} FROM files f")),
            "headings": sorted(conn.execute("SELECT f.path, h.ordinal, h.level, h.text FROM headings h JOIN files f ON f.id = h.file_id")),
            "tags": sorted(conn.execute("SELECT f.path, t.name, t.source, IFNULL(t.line, 0) FROM tags t JOIN files f ON f.id = t.file_id")),
            "links": sorted(conn.execute(
                """SELECT s.path, l.ordinal, l.raw, l.line, l.status, IFNULL(t.path, ''), IFNULL(l.subpath_status, '')
                   FROM links l JOIN files s ON s.id = l.source_id LEFT JOIN files t ON t.id = l.target_id""")),
        }
    finally:
        conn.close()


class Scratch(unittest.TestCase):
    """A copy of the sample vault, indexed."""

    index = True

    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.root = Path(self._dir.name) / "vault"
        shutil.copytree(VAULT, self.root)
        if self.index:
            with VaultIndex(self.root / ".vaultengine" / "index.sqlite") as index:
                index.update(self.root)
        self.db = self.root / ".vaultengine" / "index.sqlite"

    def tearDown(self):
        self._dir.cleanup()

    def typed(self) -> TypedVault:
        return TypedVault(BRAIN, Vault.load(self.root))

    def write(self, build, dry_run=False):
        return run_write(BRAIN, self.root, self.db, build, dry_run=dry_run)

    def text(self, path: str) -> str:
        return (self.root / path).read_text(encoding="utf-8")

    def edit(self, path: str, old: str, new: str) -> None:
        text = self.text(path)
        self.assertIn(old, text)
        (self.root / path).write_text(text.replace(old, new), encoding="utf-8")

    def pending(self) -> set[str]:
        with VaultIndex.open_existing(self.db) as index:
            stale = index.staleness(self.root)
        return set(stale.added) | set(stale.changed) | set(stale.deleted)

    def assertNoLeftovers(self):
        journal = self.root / ".vaultengine" / "journal"
        self.assertEqual(list(journal.glob("*")) if journal.exists() else [], [])
        strays = [str(p) for p in self.root.rglob(".vault-write-*")]
        self.assertEqual(strays, [])

    def assertRefusedUntouched(self, attempt, expected_codes):
        """``attempt()`` must be refused (or raise WriteError); nothing may change anywhere."""
        files, index = files_state(self.root), index_dump(self.root)
        try:
            outcome = attempt()
        except WriteError as exc:
            outcome, error = None, exc
        else:
            error = None
        if expected_codes is None:
            self.assertIsNotNone(error, "expected a WriteError")
        else:
            self.assertIsNone(error)
            self.assertEqual(outcome.status, "refused")
            self.assertEqual(sorted(i.code for i in outcome.verdict.new_violations), sorted(expected_codes))
        self.assertEqual(files_state(self.root), files)
        self.assertEqual(index_dump(self.root), index)
        self.assertNoLeftovers()
        return outcome or error


# ----- happy paths ---------------------------------------------------------------------------------


class NewTests(Scratch):
    def test_new_zettel(self):
        outcome = self.write(lambda t: plan_new(
            t, "zettel", "Verifier Hacking Needs a Do-Nothing Baseline", now=NOW,
            values={"Tags": "[[evaluation-methodology]]"},
            sections={"What This Is": "A grader a do-nothing agent passes pays for inaction."},
        ))
        self.assertEqual(outcome.status, "written")
        path = "20_ZETTLEKASTEN/verifier-hacking-needs-a-do-nothing-baseline.md"
        self.assertEqual(self.text(path), (
            "Created: 06-11-2026 09:30\n\nStatus: #baby\n\nTags: [[evaluation-methodology]]\n\n"
            "# Verifier Hacking Needs a Do-Nothing Baseline\n\n"
            "## What This Is\nA grader a do-nothing agent passes pays for inaction.\n\n"
            "## Why It Matters\n\n## Key Details\n\n## Tensions\n\n# References\n"
        ))
        self.assertEqual(outcome.index, {"status": "updated", "synced": [path], "pending": []})
        self.assertEqual(self.pending(), set())
        self.assertEqual(self.typed().issues(path), [])
        self.assertNoLeftovers()

    def test_new_evidence_writes_the_mirrors(self):
        claim = f"{D}/{CLM_END_STATE}.md"
        outcome = self.write(lambda t: plan_new(
            t, "evidence", "Fresh finding: the DB diff is the grade", folder=D, now=NOW,
            values={"Strength": "C", "Tags": "[[discourse-graph]] [[rl-environment-design]]"},
            relations=[("from", "2504.03601 - APIGen-MT Multi-Turn Agent Data Generation"), ("supports", CLM_END_STATE)],
        ))
        self.assertEqual(outcome.status, "written", outcome.verdict and outcome.verdict.new_violations)
        new_path = f"{D}/EVD - Fresh finding the DB diff is the grade.md"
        self.assertEqual(sorted(f.path for f in outcome.plan.files), sorted([new_path, claim]))
        self.assertIn("supports:: [[CLM - Grade the observed end state rather than the trajectory]]", self.text(new_path))
        self.assertIn("[[EVD - Fresh finding the DB diff is the grade]]\nopposed-by::", self.text(claim))
        self.assertEqual(self.pending(), set())

    def test_names(self):
        claim = BRAIN.type_named("claim")
        long_title = "Shaping terms get exploited: so when a task has a hard correctness requirement, partial credit should be gated"
        name = note_name(claim, long_title)
        self.assertEqual(name, "CLM - Shaping terms get exploited so when a task has a hard correctness requirement, parti")  # cut like the vault cuts
        self.assertLessEqual(len(name), 90)
        self.assertEqual(note_name(BRAIN.type_named("zettel"), "KL Penalty in RLHF (β > 0)"), "kl-penalty-in-rlhf-0")
        self.assertEqual(note_name(BRAIN.type_named("source"), "A/B: tests [v2]"), "A B tests v2")


class LinkTests(Scratch):
    def test_link_writes_both_sides_and_is_idempotent(self):
        evd, clm = f"{D}/{EVD_REWARD}.md", f"{D}/{CLM_END_STATE}.md"
        before_evd, before_clm = self.text(evd), self.text(clm)
        outcome = self.write(lambda t: plan_link(t, EVD_REWARD, CLM_END_STATE, "supports"))
        self.assertEqual(outcome.status, "written")
        self.assertEqual(
            self.text(evd),
            before_evd.replace("[[CLM - Anchor reward in a deterministic check of the outcome]]",
                               f"[[CLM - Anchor reward in a deterministic check of the outcome]], [[{CLM_END_STATE}]]"),
        )
        self.assertEqual(
            self.text(clm),
            before_clm.replace("booke]]\nopposed-by", f"booke]], [[{EVD_REWARD}]]\nopposed-by"),
        )
        self.assertEqual(sorted(outcome.index["synced"]), sorted([evd, clm]))
        self.assertEqual(self.pending(), set())
        again = self.write(lambda t: plan_link(t, EVD_REWARD, CLM_END_STATE, "supports"))
        self.assertEqual(again.status, "unchanged")
        self.assertEqual(len(again.plan.notes), 2)

    def test_one_way_relation_writes_one_side(self):
        claim, question = f"{D}/CLM - Correctness must gate all shaping credit.md", f"{D}/QUE 03 - Grade end state, trajectory or text.md"
        question_before = (self.root / question).read_bytes()
        outcome = self.write(lambda t: plan_link(t, claim, "QUE 03 - Grade end state, trajectory or text", "informs"))
        self.assertEqual(outcome.status, "written")
        self.assertEqual([f.path for f in outcome.plan.files], [claim])  # informs has no inverse
        self.assertIn("[[QUE 25 - Weighting scores by frequency and cost of error]], [[QUE 03 - Grade end state, trajectory or text]]\n",
                      self.text(claim))
        self.assertEqual((self.root / question).read_bytes(), question_before)
        again = self.write(lambda t: plan_link(t, claim, "QUE 03 - Grade end state, trajectory or text", "informs"))
        self.assertEqual(again.status, "unchanged")

    def test_new_field_under_the_relations_heading(self):
        evd = f"{D}/{EVD_REWARD}.md"
        self.edit(evd, "from:: [[2504.03601 - APIGen-MT Multi-Turn Agent Data Generation]]\n", "")
        with VaultIndex(self.db) as index:
            index.update(self.root)
        # the EVD now lacks `from::`; link it back: the field goes right under ## Relations
        outcome = self.write(lambda t: plan_link(t, EVD_REWARD, "2504.03601 - APIGen-MT Multi-Turn Agent Data Generation", "from"))
        self.assertEqual(outcome.status, "written")
        self.assertIn("supports:: [[CLM - Anchor reward in a deterministic check of the outcome]]\n"
                      "from:: [[2504.03601 - APIGen-MT Multi-Turn Agent Data Generation]]\n", self.text(evd))


class SetTests(Scratch):
    def test_replace_a_value(self):
        clm = f"{D}/{CLM_END_STATE}.md"
        before = self.text(clm)
        outcome = self.write(lambda t: plan_set(t, CLM_END_STATE, "status", "#contested"))
        self.assertEqual(outcome.status, "written")
        self.assertEqual(self.text(clm), before.replace("Status: #supported", "Status: #contested"))
        self.assertEqual(self.write(lambda t: plan_set(t, CLM_END_STATE, "Status", "#contested")).status, "unchanged")

    def test_add_a_missing_key_repairs_the_note(self):
        path = "20_ZETTLEKASTEN/cot-monitor-beats-action-monitor.md"
        before = self.text(path)
        outcome = self.write(lambda t: plan_set(t, path, "Created", "{now}", now=NOW))
        self.assertEqual(outcome.status, "written")
        self.assertEqual(self.text(path), "Created: 06-11-2026 09:30\n" + before)
        self.assertEqual(self.typed().issues(path), [])

    def test_insert_in_key_order(self):
        path = "20_ZETTLEKASTEN/reward.md"
        self.edit(path, "Status: #baby \n\n", "")
        with VaultIndex(self.db) as index:
            index.update(self.root)
        self.write(lambda t: plan_set(t, path, "Status", "#baby"))
        self.assertTrue(self.text(path).startswith("\nCreated: 12-07-2025 22:12\n\nStatus: #baby\n\nTags: "))


class FrontmatterSetTests(unittest.TestCase):
    FORMAT = parse_format("""
format: 1
metadata_source: either
types:
  note:
    match: {folder: N}
    metadata:
      status: {required: true, kind: tags, values: [draft, done]}
      up: {kind: links}
      aliases: {kind: text, multiple: false}
""")

    def run_set(self, text, key, value):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            (root / "N").mkdir()
            (root / "N/a.md").write_text(text, encoding="utf-8")
            outcome = run_write(self.FORMAT, root, root / ".vaultengine/index.sqlite",
                                lambda t: plan_set(t, "N/a", key, value))
            return outcome, (root / "N/a.md").read_text(encoding="utf-8")

    def test_replace_insert_quote(self):
        outcome, text = self.run_set("---\nstatus: draft  # note\ntitle: T\n---\nbody\n", "status", "done")
        self.assertEqual(text, "---\nstatus: done\ntitle: T\n---\nbody\n")
        self.assertEqual(outcome.index["status"], "no-index")
        _, text = self.run_set("---\nstatus: draft\n---\nbody\n", "up", "[[Parent]]")
        self.assertEqual(text, "---\nstatus: draft\nup: '[[Parent]]'\n---\nbody\n")

    def test_multi_line_value_is_refused(self):
        with self.assertRaises(WriteError) as caught:
            self.run_set("---\nstatus: draft\naliases:\n  - A\n---\n", "aliases", "B")
        self.assertIn("spans several lines", str(caught.exception))


# ----- the index stays in sync, and only for what was written ---------------------------------------------------


class IndexSyncTests(Scratch):
    def test_other_pending_edits_stay_pending(self):
        other = "20_ZETTLEKASTEN/reward.md"
        self.edit(other, "# Reward", "# Reward (edited by hand)")
        self.write(lambda t: plan_link(t, EVD_REWARD, CLM_END_STATE, "supports"))
        self.assertEqual(self.pending(), {other})

    def test_a_written_note_with_unchecked_violations_stays_pending(self):
        clm = f"{D}/{CLM_END_STATE}.md"
        self.edit(clm, "Status: #supported", "Status: #suported")  # a hand edit the hook has not seen
        outcome = self.write(lambda t: plan_link(t, EVD_REWARD, CLM_END_STATE, "supports"))
        self.assertEqual(outcome.status, "written")  # the link adds no violation
        self.assertEqual(outcome.index["pending"], [clm])
        self.assertEqual(self.pending(), {clm})  # so `vault check --changed` still reports the typo

    def test_partial_updates_equal_a_full_rebuild(self):
        self.write(lambda t: plan_link(t, EVD_REWARD, CLM_END_STATE, "supports"))
        self.write(lambda t: plan_set(t, CLM_END_STATE, "Status", "#contested"))
        self.write(lambda t: plan_new(t, "zettel", "Index Check Note", now=NOW, values={"Tags": "[[reward]]"}))
        self.write(lambda t: plan_new(
            t, "evidence", "EVD that resolves nothing yet", folder=D, now=NOW, values={"Strength": "D"},
            relations=[("from", "2504.03601 - APIGen-MT Multi-Turn Agent Data Generation"), ("supports", CLM_END_STATE)],
        ))
        incremental = index_dump(self.root, with_mtimes=False)
        fresh = Path(self._dir.name) / "fresh"
        shutil.copytree(self.root, fresh, ignore=shutil.ignore_patterns(".vaultengine"))
        with VaultIndex(fresh / ".vaultengine" / "index.sqlite") as index:
            index.update(fresh)
        rebuilt = index_dump(fresh, with_mtimes=False)
        rebuilt["meta"] = [row for row in rebuilt["meta"] if row[0] != "vault_root"]
        incremental["meta"] = [row for row in incremental["meta"] if row[0] != "vault_root"]
        self.assertEqual(incremental, rebuilt)

    def test_index_that_needs_a_rebuild_is_left_alone(self):
        conn = sqlite3.connect(self.db)
        conn.execute("UPDATE meta SET value = '0' WHERE key = 'parser_version'")
        conn.commit()
        conn.close()
        before = index_dump(self.root)
        outcome = self.write(lambda t: plan_set(t, CLM_END_STATE, "Status", "#contested"))
        self.assertEqual(outcome.status, "written")
        self.assertEqual(outcome.index["status"], "needs-rebuild")
        self.assertEqual(index_dump(self.root), before)


class NoIndexTests(Scratch):
    index = False

    def test_writes_without_creating_an_index(self):
        outcome = self.write(lambda t: plan_set(t, CLM_END_STATE, "Status", "#contested"))
        self.assertEqual((outcome.status, outcome.index["status"]), ("written", "no-index"))
        self.assertFalse(self.db.exists())


# ----- refusals: nothing changes, anywhere ------------------------------------------------------------------


class RefusalTests(Scratch):
    def test_link_to_the_wrong_type(self):
        outcome = self.assertRefusedUntouched(
            lambda: self.write(lambda t: plan_link(t, EVD_REWARD, "kl-penalty-in-rlhf", "supports")), ["relation-target-type"])
        self.assertEqual(outcome.verdict.new_violations[0].message, "`supports::` [[kl-penalty-in-rlhf]] is a zettel; expected claim")

    def test_link_relation_not_allowed_on_the_type(self):
        self.assertRefusedUntouched(
            lambda: self.write(lambda t: plan_link(t, CLM_END_STATE, f"{EVD_REWARD}", "supports")),
            ["unexpected-relation", "relation-target-type", "unexpected-relation", "relation-target-type"],
        )

    def test_link_with_an_undeclared_relation(self):
        self.assertRefusedUntouched(lambda: self.write(lambda t: plan_link(t, EVD_REWARD, CLM_END_STATE, "endorses")),
                                    ["unknown-field"])

    def test_set_a_value_that_is_not_allowed(self):
        self.assertRefusedUntouched(lambda: self.write(lambda t: plan_set(t, CLM_END_STATE, "Status", "#settled")), ["bad-value"])
        self.assertRefusedUntouched(lambda: self.write(lambda t: plan_set(t, CLM_END_STATE, "Created", "yesterday")), ["bad-value"])
        self.assertRefusedUntouched(lambda: self.write(lambda t: plan_set(t, EVD_REWARD, "Strength", "")), ["missing-value"])

    def test_set_errors(self):
        for args, message in (
            ((CLM_END_STATE, "Colour", "red"), "'Colour' is not a metadata key of claim"),
            ((CLM_END_STATE, "Status", "#a\n#b"), "the value must be one line"),
            (("no such note", "Status", "#x"), "no note 'no such note'"),
        ):
            with self.subTest(args=args):
                error = self.assertRefusedUntouched(lambda: self.write(lambda t: plan_set(t, *args)), None)
                self.assertIn(message, str(error))

    def test_new_claim_without_evidence(self):
        self.assertRefusedUntouched(lambda: self.write(lambda t: plan_new(
            t, "claim", "Graders must be tested against a known-bad trajectory", folder=D, now=NOW,
            values={"Status": "#supported"}, relations=[("informs", "QUE 03 - Grade end state, trajectory or text")],
        )), ["missing-relation"])

    def test_new_zettel_without_tags(self):
        self.assertRefusedUntouched(lambda: self.write(lambda t: plan_new(t, "zettel", "No tags", now=NOW)), ["missing-value"])

    def test_new_note_that_completes_a_dangling_relation(self):
        # CLM - Grade the observed end state... says supported-by:: [[EVD - tau2 official score never uses an LLM judge]],
        # which does not exist. Creating it without `supports::` back would leave the pair half written.
        outcome = self.assertRefusedUntouched(lambda: self.write(lambda t: plan_new(
            t, "evidence", "tau2 official score never uses an LLM judge", folder=D, now=NOW, values={"Strength": "B"},
            relations=[("from", "2504.03601 - APIGen-MT Multi-Turn Agent Data Generation")],
        )), ["missing-mirror", "missing-relation"])
        mirror = next(i for i in outcome.verdict.new_violations if i.code == "missing-mirror")
        self.assertIn(f"missing `supports:: [[{CLM_END_STATE}]]`", mirror.message)
        written = self.write(lambda t: plan_new(
            t, "evidence", "tau2 official score never uses an LLM judge", folder=D, now=NOW, values={"Strength": "B"},
            relations=[("from", "2504.03601 - APIGen-MT Multi-Turn Agent Data Generation"), ("supports", CLM_END_STATE)],
        ))
        self.assertEqual(written.status, "written")
        # The claim already names the new note (the link was dangling), so it is not written twice.
        self.assertEqual([f.path for f in written.plan.files], [f"{D}/EVD - tau2 official score never uses an LLM judge.md"])
        self.assertEqual(self.text(f"{D}/{CLM_END_STATE}.md").count("[[EVD - tau2 official score never uses an LLM judge]]"), 1)
        self.assertEqual(written.plan.notes, [f"{D}/{CLM_END_STATE}.md already has `supported-by:: "
                                              "[[EVD - tau2 official score never uses an LLM judge]]`"])
        self.assertEqual(self.typed().issues(f"{D}/EVD - tau2 official score never uses an LLM judge.md"), [])

    def test_new_errors(self):
        cases = (
            (dict(type_name="zettel", title="KL Penalty in RLHF"), "already exists"),
            (dict(type_name="claim", title="x", folder="20_ZETTLEKASTEN"), "20_ZETTLEKASTEN/CLM - x.md would be a zettel, not a claim"),
            (dict(type_name="zettel", title="x", name="../escape"), "cannot be a note name"),
            (dict(type_name="nope", title="x"), "has no type 'nope'"),
            (dict(type_name="zettel", title="x", values={"Colour": "red"}), "not a metadata key"),
            (dict(type_name="zettel", title="x", sections={"Intro": "t"}), "not a required heading"),
            (dict(type_name="zettel", title="x", relations=[("refines", "no such note")]), "no note 'no such note'"),
            (dict(type_name="zettel", title=""), "the title is empty"),
        )
        for kwargs, message in cases:
            with self.subTest(kwargs=kwargs):
                kwargs = dict(kwargs)
                type_name, title = kwargs.pop("type_name"), kwargs.pop("title")
                error = self.assertRefusedUntouched(lambda: self.write(lambda t: plan_new(t, type_name, title, now=NOW, **kwargs)), None)
                self.assertIn(message, str(error))

    def test_dry_run_writes_nothing(self):
        outcome = self.assertRefusedUntouched(
            lambda: self.write(lambda t: plan_link(t, EVD_REWARD, "kl-penalty-in-rlhf", "supports"), dry_run=True),
            ["relation-target-type"])
        self.assertEqual(outcome.status, "refused")
        files, index = files_state(self.root), index_dump(self.root)
        ok = self.write(lambda t: plan_link(t, EVD_REWARD, CLM_END_STATE, "supports"), dry_run=True)
        self.assertEqual(ok.status, "dry-run")
        self.assertEqual((files_state(self.root), index_dump(self.root)), (files, index))


# ----- the commit: atomic as a group, even when it fails or the process dies ------------------------------


class CommitTests(Scratch):
    def two_file_plan(self):
        typed = self.typed()
        return plan_link(typed, EVD_REWARD, CLM_END_STATE, "supports").files

    def test_failure_in_the_middle_restores_everything(self):
        files = self.two_file_plan() + [FileChange("New Folder/Sub/new.md", None, "Tags: x\n")]
        before = files_state(self.root)
        real_replace = os.replace
        calls = []

        def failing_replace(src, dst):
            calls.append(dst)
            if len(calls) == 3:
                raise OSError(28, "No space left on device")
            return real_replace(src, dst)

        with mock.patch.object(write_module.os, "replace", failing_replace):
            with self.assertRaises(WriteFailed) as caught:
                commit(self.root, files)
        self.assertIn("every file was restored", str(caught.exception))
        self.assertEqual(files_state(self.root), before)  # bytes and mtimes of the two renamed files too
        self.assertFalse((self.root / "New Folder").exists())
        self.assertNoLeftovers()

    def test_read_back_mismatch_restores_everything(self):
        files = self.two_file_plan()
        before = files_state(self.root)
        real_read = Path.read_bytes
        reads = {"n": 0}

        def corrupting(self_path):
            data = real_read(self_path)
            if self_path.name.startswith("CLM - Grade") and reads["n"] >= 1:
                return data + b"corrupt"
            if self_path.name.startswith("CLM - Grade"):
                reads["n"] += 1
            return data

        with mock.patch.object(Path, "read_bytes", corrupting):
            with self.assertRaises(WriteFailed):
                commit(self.root, files)
        self.assertEqual(files_state(self.root), before)
        self.assertNoLeftovers()

    def test_file_changed_after_planning_is_not_overwritten(self):
        clm = f"{D}/{CLM_END_STATE}.md"

        def build(typed):
            plan = plan_link(typed, EVD_REWARD, CLM_END_STATE, "supports")
            (self.root / clm).write_text("edited by someone else meanwhile\n", encoding="utf-8")
            return plan

        before_evd = (self.root / D / f"{EVD_REWARD}.md").read_bytes()
        with self.assertRaises(WriteFailed) as caught:
            self.write(build)
        self.assertIn("changed while the write was being prepared", str(caught.exception))
        self.assertEqual(self.text(clm), "edited by someone else meanwhile\n")
        self.assertEqual((self.root / D / f"{EVD_REWARD}.md").read_bytes(), before_evd)
        self.assertNoLeftovers()

    def test_crash_between_renames_is_rolled_back_by_the_next_write(self):
        before = files_state(self.root)
        index = index_dump(self.root)
        crashed = run_vault("link", str(self.root), "-f", "brain", EVD_REWARD, CLM_END_STATE, "--as", "supports",
                            env={"VAULTENGINE_TEST_CRASH_AFTER_RENAMES": "1"})
        self.assertEqual(crashed.returncode, 70)
        after_crash = files_state(self.root)
        self.assertNotEqual(after_crash, before)  # half written: one file renamed, journal + backup present
        self.assertTrue(list((self.root / ".vaultengine" / "journal").glob("*.json")))
        self.assertEqual(index_dump(self.root), index)  # the index is only updated after a commit

        result = run_vault("set", str(self.root), "-f", "brain", CLM_END_STATE, "Status", "#supported")
        self.assertIn("recovered an interrupted write: restored", result.stdout)
        self.assertEqual(files_state(self.root), before)
        self.assertNoLeftovers()

    def test_recover_removes_leftovers_of_finished_writes(self):
        stray = self.root / "20_ZETTLEKASTEN" / ".vault-write-abc-0.bak"
        stray.write_text("x")
        with write_lock(self.root):
            self.assertEqual(recover(self.root), [])
        self.assertFalse(stray.exists())

    def test_permissions_are_kept(self):
        clm = self.root / D / f"{CLM_END_STATE}.md"
        os.chmod(clm, 0o640)
        self.write(lambda t: plan_set(t, CLM_END_STATE, "Status", "#contested"))
        self.assertEqual(clm.stat().st_mode & 0o777, 0o640)

    def test_a_refused_write_adds_nothing_to_a_vault_without_an_engine_folder(self):
        shutil.rmtree(self.root / ".vaultengine")
        before = files_state(self.root)
        outcome = self.write(lambda t: plan_set(t, CLM_END_STATE, "Status", "#settled"))
        self.assertEqual(outcome.status, "refused")
        self.assertEqual(files_state(self.root), before)
        self.assertFalse((self.root / ".vaultengine").exists())
        dry = self.write(lambda t: plan_set(t, CLM_END_STATE, "Status", "#contested"), dry_run=True)
        self.assertEqual(dry.status, "dry-run")
        self.assertFalse((self.root / ".vaultengine").exists())

    def test_one_writer_at_a_time(self):
        with write_lock(self.root):
            with self.assertRaises(WriteError) as caught:
                with write_lock(self.root, timeout=0.2):
                    pass
        self.assertIn("another vault write is in progress", str(caught.exception))


# ----- CLI ----------------------------------------------------------------------------------------------


class WriteCliTests(Scratch):
    def test_new_link_set_refuse(self):
        new = run_vault("new", str(self.root), "-f", "brain", "--type", "zettel", "--title", "CLI Zettel",
                        "--set", "Tags=[[reward]]", "--section", "Tensions", "None yet.")
        self.assertEqual(new.returncode, 0, new.stderr)
        self.assertIn("+ 20_ZETTLEKASTEN/cli-zettel.md", new.stdout)
        self.assertIn("index    updated 1 note", new.stdout)

        link = run_vault("link", str(self.root), "-f", "brain", EVD_REWARD, CLM_END_STATE, "--as", "supports", "--json")
        data = json.loads(link.stdout)
        self.assertEqual((data["status"], len(data["files"]), data["new_violations"]), ("written", 2, []))

        before = files_state(self.root)
        refused = run_vault("set", str(self.root), "-f", "brain", CLM_END_STATE, "Status", "#settled")
        self.assertEqual(refused.returncode, 1)
        self.assertIn("✗ refused: the change would add 1 violation; nothing was written", refused.stdout)
        self.assertIn("-Status: #supported\n+Status: #settled", refused.stdout)
        self.assertEqual(files_state(self.root), before)

        usage = run_vault("link", str(self.root), "-f", "brain", "nope", CLM_END_STATE, "--as", "supports")
        self.assertEqual(usage.returncode, 2)
        self.assertIn("error: no note 'nope'", usage.stderr)
        self.assertEqual(files_state(self.root), before)

        check = run_vault("check", str(self.root), "-f", "brain", "--changed")
        self.assertEqual(check.returncode, 0, check.stdout)


if __name__ == "__main__":
    unittest.main()


class RandomWritesTests(Scratch):
    """Random links and sets over the sample vault. After each one, checked from disk:
    written => no new violation anywhere, written notes in sync with the index;
    refused or rejected => not one byte, mtime or index row changed."""

    def test_random_sequence(self):
        import random
        from collections import Counter
        from vaultengine.formats.write import violation_keys

        rng = random.Random(20261106)
        typed = self.typed()
        notes = sorted(typed.vault.notes)
        relations = sorted(BRAIN.relations) + ["endorses"]
        values = {"Status": ["#baby", "#supported", "#evidence", "#contested", "#bogus", "baby"],
                  "Strength": ["A", "C", "E", ""], "Created": ["24-09-2026 19:20", "2026-09-24"],
                  "Tags": ["[[reward]]", "plain", ""]}
        outcomes = Counter()
        for step in range(36):
            before_keys = Counter(k for i in typed.issues() for k in violation_keys(i))
            files, index = files_state(self.root), index_dump(self.root)
            if rng.random() < 0.6:
                a, b = rng.sample(notes, 2)
                relation = rng.choice(relations)
                build = lambda t, a=a, b=b, r=relation: plan_link(t, a, b, r)
            else:
                path = rng.choice(notes)
                note_type = BRAIN.type_for(path)
                key = rng.choice([k.name for k in note_type.metadata if k.name in values] or ["Status"])
                build = lambda t, p=path, k=key, v=rng.choice(values.get(key, ["x"])): plan_set(t, p, k, v)
            try:
                outcome = self.write(build)
                status = outcome.status
            except WriteError:
                status = "rejected"
            outcomes[status] += 1
            typed = self.typed()  # from disk
            after_keys = Counter(k for i in typed.issues() for k in violation_keys(i))
            if status == "written":
                self.assertEqual(after_keys - before_keys, Counter(), f"step {step}: a write added violations")
                self.assertEqual(self.pending() & {f.path for f in outcome.plan.files}, set(), f"step {step}")
            else:
                self.assertEqual(files_state(self.root), files, f"step {step}: {status} but files changed")
                self.assertEqual(index_dump(self.root), index, f"step {step}: {status} but the index changed")
            self.assertNoLeftovers()
        print("\nrandom writes:", dict(outcomes))
        self.assertGreater(outcomes["written"], 3, outcomes)
        self.assertGreater(outcomes["refused"], 10, outcomes)
        self.assertEqual(self.pending(), set())  # nothing but safe writes happened: index and disk agree
