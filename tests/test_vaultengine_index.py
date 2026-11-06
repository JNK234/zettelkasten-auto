import io
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from vaultengine import LinkGraph, Vault
from vaultengine import index as index_module
from vaultengine.cli import main as cli_main
from vaultengine.index import RACY_MTIME, VaultIndex, default_db_path
from vaultengine.queries import health, stats

FIXTURES = Path(__file__).parent / "fixtures"
REPO_ROOT = Path(__file__).parent.parent
OLD_TIME = 1_600_000_000  # well outside the racy window


def snapshot(conn):
    """Every indexed fact keyed by paths (not row ids), for comparing two indexes."""
    path_of = {row[0]: row[1] for row in conn.execute("SELECT id, path FROM files")}

    def rows(sql):
        return sorted(tuple(path_of.get(v, v) if i == 0 else v for i, v in enumerate(row)) for row in conn.execute(sql))

    return {
        "files": rows("SELECT id, kind, content_hash FROM files"),
        "notes": rows("SELECT file_id, has_frontmatter, frontmatter_error, body_start_line, body_chars, line_count FROM notes"),
        "properties": rows("SELECT file_id, key, value_json FROM properties"),
        "aliases": rows("SELECT file_id, alias, alias_key FROM aliases"),
        "tags": rows("SELECT file_id, name, name_key, source, line FROM tags"),
        "headings": rows("SELECT file_id, ordinal, level, text, line FROM headings"),
        "blocks": rows("SELECT file_id, block_id, line FROM blocks"),
        "links": sorted(
            (path_of[r["source_id"]], r["ordinal"], r["raw"], r["target"], r["heading_path"], r["block_id"], r["display"],
             r["line"], r["target_key"], r["target_name_key"], r["status"], path_of.get(r["target_id"]),
             r["matched_by"], r["candidates"], r["subpath_status"])
            for r in conn.execute("SELECT * FROM links")
        ),
    }


class TempVault:
    def __init__(self, files=None):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "vault"
        self.root.mkdir()
        self.clock = OLD_TIME
        for path, text in (files or {}).items():
            self.write(path, text)

    def write(self, path, text, old_mtime=True):
        full = self.root / path
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_bytes(text.encode("utf-8") if isinstance(text, str) else text)
        if old_mtime:
            self.clock += 10
            os.utime(full, (self.clock, self.clock))

    def touch(self, path):
        self.clock += 10
        os.utime(self.root / path, (self.clock, self.clock))

    def delete(self, path):
        (self.root / path).unlink()

    def rename(self, old, new):
        (self.root / new).parent.mkdir(parents=True, exist_ok=True)
        (self.root / old).rename(self.root / new)

    def index(self):
        return VaultIndex(default_db_path(self.root))

    def fresh_snapshot(self):
        with tempfile.TemporaryDirectory() as other:
            with VaultIndex(Path(other) / "full.sqlite") as fresh:
                fresh.update(self.root)
                return snapshot(fresh.conn)

    def cleanup(self):
        self._tmp.cleanup()


class IncrementalTestCase(unittest.TestCase):
    FILES: dict = {}

    def setUp(self):
        self.vault = TempVault(self.FILES)
        self.ix = self.vault.index()
        self.first = self.ix.update(self.vault.root)

    def tearDown(self):
        self.ix.close()
        self.vault.cleanup()

    def update(self):
        report = self.ix.update(self.vault.root)
        self.assertEqual(snapshot(self.ix.conn), self.vault.fresh_snapshot(), "incremental index differs from full build")
        return report

    def link(self, source, raw):
        row = self.ix.conn.execute(
            """SELECT l.status, t.path AS target, l.matched_by, l.subpath_status, l.candidates
               FROM links l JOIN files s ON s.id = l.source_id LEFT JOIN files t ON t.id = l.target_id
               WHERE s.path = ? AND l.raw = ?""",
            (source, raw),
        ).fetchone()
        return dict(row)


class ChangeDetectionTests(IncrementalTestCase):
    FILES = {
        "A.md": "# A\n[[B]] [[C]]",
        "B.md": "# B\n## Part\n[[A#Nope]]",
        "C.md": "# C",
        "img.png": b"\x89PNG",
    }

    def test_first_run_parses_every_note(self):
        self.assertEqual(self.first.parsed, ["A.md", "B.md", "C.md"])
        self.assertEqual(self.first.attachments_added, ["img.png"])

    def test_second_run_reads_and_parses_nothing(self):
        report = self.update()
        self.assertEqual((report.parsed, report.hashed, report.deleted, report.links_checked), ([], 0, [], 0))

    def test_editing_one_note_parses_exactly_that_note(self):
        self.vault.write("B.md", "# B\n## Part\nedited [[A]]")
        report = self.update()
        self.assertEqual(report.parsed, ["B.md"])
        self.assertEqual(report.changed, ["B.md"])
        self.assertEqual(report.hashed, 1)

    def test_touch_without_content_change_hashes_but_does_not_parse(self):
        self.vault.touch("C.md")
        report = self.update()
        self.assertEqual((report.hashed, report.parsed), (1, []))
        self.assertEqual(self.update().hashed, 0)  # new mtime was stored

    def test_recently_written_files_are_rehashed_next_run(self):
        self.vault.write("C.md", "# C\nchanged just now", old_mtime=False)
        self.assertEqual(self.update().parsed, ["C.md"])
        mtime = self.ix.conn.execute("SELECT mtime_ns FROM files WHERE path = 'C.md'").fetchone()[0]
        self.assertEqual(mtime, RACY_MTIME)
        report = self.update()
        self.assertEqual((report.hashed, report.parsed), (1, []))

    def test_delete_removes_all_rows_of_the_note(self):
        self.vault.delete("B.md")
        report = self.update()
        self.assertEqual(report.deleted, ["B.md"])
        for table in ("notes", "headings", "tags", "properties", "aliases", "blocks"):
            count = self.ix.conn.execute(f"SELECT COUNT(*) FROM {table} WHERE file_id NOT IN (SELECT id FROM files)").fetchone()[0]
            self.assertEqual(count, 0, table)
        self.assertEqual(self.link("A.md", "[[B]]")["status"], "unresolved")

    def test_attachment_add_and_delete(self):
        self.vault.write("C.md", "# C\n![[img.png]] ![[new.pdf]]")
        self.update()
        self.assertEqual(self.link("C.md", "![[new.pdf]]")["status"], "unresolved")
        self.vault.write("docs/new.pdf", b"%PDF")
        report = self.update()
        self.assertEqual((report.attachments_added, report.parsed), (["docs/new.pdf"], []))
        self.assertEqual(self.link("C.md", "![[new.pdf]]")["target"], "docs/new.pdf")
        self.vault.delete("img.png")
        self.assertEqual(self.update().attachments_deleted, ["img.png"])
        self.assertEqual(self.link("C.md", "![[img.png]]")["status"], "unresolved")


class RippleTests(IncrementalTestCase):
    """Resolution changes in notes that are NOT re-parsed."""

    FILES = {
        "Source.md": "[[GRPO]] [[Index]] [[PPO]] [[Target#Part]] [[Target#^blk]] [[Folder/Moved]]",
        "Index.md": "# Index",
        "Target.md": "# Target\n## Part\ntext ^blk",
        "Folder/Moved.md": "",
    }

    def test_added_note_resolves_links_elsewhere(self):
        self.vault.write("GRPO.md", "# GRPO")
        report = self.update()
        self.assertEqual(report.parsed, ["GRPO.md"])
        self.assertEqual(report.links_changed_elsewhere, 1)
        self.assertEqual(self.link("Source.md", "[[GRPO]]")["target"], "GRPO.md")

    def test_duplicate_basename_makes_links_ambiguous_and_delete_restores(self):
        self.vault.write("Archive/Index.md", "# Old index")
        self.update()
        self.assertEqual(self.link("Source.md", "[[Index]]")["status"], "ambiguous")
        self.assertEqual(json.loads(self.link("Source.md", "[[Index]]")["candidates"]), ["Archive/Index.md", "Index.md"])
        self.vault.delete("Archive/Index.md")
        self.update()
        self.assertEqual(self.link("Source.md", "[[Index]]")["target"], "Index.md")

    def test_alias_added_and_removed(self):
        self.vault.write("Target.md", "---\naliases: [PPO]\n---\n# Target\n## Part\ntext ^blk")
        self.update()
        self.assertEqual(self.link("Source.md", "[[PPO]]")["matched_by"], "alias")
        self.vault.write("Target.md", "# Target\n## Part\ntext ^blk")
        self.update()
        self.assertEqual(self.link("Source.md", "[[PPO]]")["status"], "unresolved")

    def test_heading_and_block_changes_update_subpath_status(self):
        self.vault.write("Target.md", "# Target\n## Renamed\ntext")
        report = self.update()
        self.assertEqual(report.parsed, ["Target.md"])
        self.assertEqual(self.link("Source.md", "[[Target#Part]]")["subpath_status"], "missing")
        self.assertEqual(self.link("Source.md", "[[Target#^blk]]")["subpath_status"], "missing")

    def test_rename_breaks_explicit_path_links(self):
        self.vault.rename("Folder/Moved.md", "Elsewhere/Moved.md")
        report = self.update()
        self.assertEqual((report.added, report.deleted), (["Elsewhere/Moved.md"], ["Folder/Moved.md"]))
        self.assertEqual(self.link("Source.md", "[[Folder/Moved]]")["status"], "unresolved")


class RandomisedEquivalenceTests(unittest.TestCase):
    NOTE_PATHS = ["Alpha.md", "Beta.md", "sub/Alpha.md", "sub/Gamma.md", "Index.md", "other/Index.md", "Delta.md", "sub/deep/Epsilon.md"]
    ATTACHMENTS = ["img.png", "sub/img.png", "doc.pdf", "Board.excalidraw.md"]
    TARGETS = ["Alpha", "beta", "Gamma", "Index", "sub/Alpha", "/Delta", "./Gamma", "../Beta", "Epsilon",
               "Nope", "img.png", "doc.pdf", "Board.excalidraw", "PPO", "Greek", "alpha.md", ""]
    ALIASES = ["PPO", "Greek", "Alpha", "Index"]
    HEADINGS = ["One", "Two", "Three: x"]

    def random_note(self, rng):
        parts = []
        if rng.random() < 0.4:
            parts.append("---\naliases: [" + ", ".join(rng.sample(self.ALIASES, rng.randint(1, 2))) + "]\n---")
        parts += [f"{'#' * rng.randint(1, 3)} {h}" for h in rng.sample(self.HEADINGS, rng.randint(0, 3))]
        if rng.random() < 0.4:
            parts.append(f"para ^{rng.choice(['b1', 'b2'])}")
        for _ in range(rng.randint(0, 5)):
            target = rng.choice(self.TARGETS)
            sub = rng.choice(["", "", f"#{rng.choice(self.HEADINGS)}", f"#{self.HEADINGS[0]}#{self.HEADINGS[1]}", "#^b1", "#^b2"])
            if not target and not sub:
                sub = "#One"
            parts.append(f"{'!' if rng.random() < 0.2 else ''}[[{target}{sub}]] #tag{rng.randint(1, 3)}")
        return "\n\n".join(parts)

    def run_sequence(self, seed, steps=40):
        rng = random.Random(seed)
        vault = TempVault()
        try:
            with vault.index() as ix:
                for step in range(steps):
                    existing = sorted(str(p.relative_to(vault.root)) for p in vault.root.rglob("*") if p.is_file() and ".vaultengine" not in p.parts)
                    op = rng.choice(["write", "write", "write", "delete", "rename", "attach", "touch"])
                    if op == "write" or not existing:
                        vault.write(rng.choice(self.NOTE_PATHS), self.random_note(rng), old_mtime=rng.random() < 0.7)
                    elif op == "delete":
                        vault.delete(rng.choice(existing))
                    elif op == "rename":
                        source = rng.choice(existing)
                        pool = self.NOTE_PATHS if source.endswith(".md") else self.ATTACHMENTS
                        free = [p for p in pool if p not in existing]
                        if free:
                            vault.rename(source, rng.choice(free))
                    elif op == "attach":
                        path = rng.choice(self.ATTACHMENTS)
                        vault.write(path, self.random_note(rng) if path.endswith(".md") else b"\x00")
                    else:
                        vault.touch(rng.choice(existing))
                    ix.update(vault.root)
                    self.assertEqual(snapshot(ix.conn), vault.fresh_snapshot(), f"seed {seed}, step {step}, op {op}")
        finally:
            vault.cleanup()

    def test_incremental_equals_full_rebuild(self):
        for seed in range(6):
            with self.subTest(seed=seed):
                self.run_sequence(seed)


class InvalidationTests(IncrementalTestCase):
    FILES = {"A.md": "[[B]]", "B.md": "# B"}

    def test_parser_version_change_reparses_everything(self):
        with patch.object(index_module, "PARSER_VERSION", 999):
            report = self.ix.update(self.vault.root)
        self.assertEqual(report.forced, ["A.md", "B.md"])
        self.assertIn("parser version", report.full_reason)

    def test_resolver_version_change_reresolves_without_parsing(self):
        with patch.object(index_module, "RESOLVER_VERSION", 999):
            report = self.ix.update(self.vault.root)
        self.assertEqual((report.parsed, report.links_checked), ([], 1))

    def test_alias_setting_change_reresolves(self):
        report = self.ix.update(self.vault.root, resolve_aliases=False)
        self.assertEqual(report.full_reason, "alias resolution setting changed")

    def test_full_flag(self):
        self.assertEqual(self.ix.update(self.vault.root, full=True).forced, ["A.md", "B.md"])

    def test_schema_version_change_rebuilds_database(self):
        self.ix.conn.execute("PRAGMA user_version = 999")
        self.ix.close()
        self.ix = self.vault.index()
        self.assertEqual(self.ix.conn.execute("SELECT COUNT(*) FROM files").fetchone()[0], 0)
        self.assertEqual(self.ix.update(self.vault.root).parsed, ["A.md", "B.md"])

    def test_failed_update_leaves_index_untouched(self):
        before = snapshot(self.ix.conn)
        self.vault.write("A.md", "[[C]]")
        self.vault.write("New.md", "# new")
        with patch.object(index_module, "parse_note", side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError):
                self.ix.update(self.vault.root)
        self.assertEqual(snapshot(self.ix.conn), before)
        self.assertEqual(self.update().parsed, ["A.md", "New.md"])

    def test_staleness(self):
        self.assertFalse(self.ix.staleness(self.vault.root).is_stale)
        self.vault.touch("B.md")
        self.assertFalse(self.ix.staleness(self.vault.root).is_stale)  # same content
        self.vault.write("B.md", "# B changed")
        self.vault.write("C.md", "# C")
        self.vault.delete("A.md")
        stale = self.ix.staleness(self.vault.root)
        self.assertEqual((stale.added, stale.changed, stale.deleted), (["C.md"], ["B.md"], ["A.md"]))


class StoredContentTests(IncrementalTestCase):
    FILES = {
        "N.md": "---\naliases: [Enn]\ntags: [fm/tag]\ncreated: 2026-02-26\nrating: 4\n---\n# Title\n#Inline tag\npara ^blk-1\n",
        "Empty.md": "---\ntags: [x]\n---\n \n\n",
    }

    def test_rows(self):
        conn = self.ix.conn
        props = dict(conn.execute("SELECT key, value_json FROM properties p JOIN files f ON f.id = p.file_id WHERE f.path = 'N.md'").fetchall())
        self.assertEqual(json.loads(props["created"]), "2026-02-26")
        self.assertEqual(json.loads(props["rating"]), 4)
        self.assertEqual(json.loads(props["aliases"]), ["Enn"])
        def rows(sql):
            return sorted(tuple(row) for row in conn.execute(sql))

        self.assertEqual(
            rows("SELECT name, name_key, source, line FROM tags"),
            [("Inline", "inline", "inline", 8), ("fm/tag", "fm/tag", "frontmatter", None), ("x", "x", "frontmatter", None)],
        )
        self.assertEqual(rows("SELECT alias, alias_key FROM aliases"), [("Enn", "enn")])
        self.assertEqual(rows("SELECT level, text, line FROM headings"), [(1, "Title", 7)])
        self.assertEqual(rows("SELECT block_id, line FROM blocks"), [("blk-1", 9)])
        self.assertEqual(
            dict(conn.execute("SELECT f.path, n.body_chars FROM notes n JOIN files f ON f.id = n.file_id").fetchall()),
            {"Empty.md": 0, "N.md": len("#Title#Inlinetagpara^blk-1")},
        )

    def test_index_matches_in_memory_graph_on_real_vault(self):
        with tempfile.TemporaryDirectory() as tmp, VaultIndex(Path(tmp) / "x.sqlite") as ix:
            ix.update(FIXTURES / "vault")
            stored = {
                (r["path"], r["ordinal"]): (r["status"], r["tpath"], r["matched_by"], r["subpath_status"])
                for r in ix.conn.execute(
                    """SELECT s.path, l.ordinal, l.status, t.path AS tpath, l.matched_by, l.subpath_status
                       FROM links l JOIN files s ON s.id = l.source_id LEFT JOIN files t ON t.id = l.target_id"""
                )
            }
        graph = LinkGraph(Vault.load(FIXTURES / "vault"))
        expected = {
            (source, i): (e.status, e.target_path, e.matched_by, e.subpath_status)
            for source in graph.vault.notes
            for i, e in enumerate(graph.outgoing(source))
        }
        self.assertEqual(len(stored), 1787)
        self.assertEqual(stored, expected)


class QueryTests(IncrementalTestCase):
    FILES = {
        "Hub.md": "# Hub\n## Sec\n[[Leaf]] [[Hub]] [[#Sec]]",
        "Leaf.md": "# Leaf\n![[pic.png]] [[Missing]] [[Missing]] [[missing]]",
        "Embedder.md": "![[Hub]] [[Leaf#Nope]] [[Leaf#^gone]]",
        "A/Dup.md": "[[Dup]]",
        "B/Dup.md": "",
        "Blank.md": "---\ntags: [draft]\n---\n\n",
        "pic.png": b"\x89PNG",
    }

    def test_health(self):
        data = health(self.ix.conn)
        self.assertEqual(
            [(g["target"], len(g["links"])) for g in data["unresolved_links"]], [("Missing", 3)]
        )
        self.assertEqual([(a["source"], a["candidates"]) for a in data["ambiguous_links"]], [("A/Dup.md", ["A/Dup.md", "B/Dup.md"])])
        self.assertEqual([(m["kind"], m["subpath"]) for m in data["missing_subpaths"]], [("heading", "Nope"), ("block", "^gone")])
        # Hub is linked only by an embed; Leaf by Hub and Embedder. Self-links don't count.
        self.assertEqual(data["orphans"], ["A/Dup.md", "B/Dup.md", "Blank.md", "Embedder.md"])
        # Leaf links only to an attachment and a missing note; Dup's only link is ambiguous.
        self.assertEqual(data["dead_ends"], ["A/Dup.md", "B/Dup.md", "Blank.md", "Leaf.md"])
        self.assertEqual(data["duplicate_basenames"], [{"name": "Dup", "paths": ["A/Dup.md", "B/Dup.md"]}])
        self.assertEqual(data["empty_notes"], ["B/Dup.md", "Blank.md"])
        self.assertEqual(data["summary"]["unresolved_links"], 3)

    def test_stats(self):
        data = stats(self.ix.conn)
        self.assertEqual((data["notes"], data["attachments"]), (6, 1))
        self.assertEqual(data["links"]["total"], 11)
        self.assertEqual(data["links"]["embeds"], 2)
        self.assertEqual((data["links"]["unresolved"], data["links"]["unresolved_targets"]), (3, 1))
        self.assertEqual(data["most_wanted"], [{"target": "Missing", "links": 3, "linking_notes": 1}])
        self.assertEqual(data["most_linked"][0]["path"], "Leaf.md")
        self.assertEqual(data["note_edges"], 3)  # Hub->Leaf, Embedder->Hub, Embedder->Leaf


class CliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "vault"
        # Never copy an index someone built in the fixture (e.g. by trying the CLI on it).
        shutil.copytree(FIXTURES / "resolver_vault", self.root, ignore=shutil.ignore_patterns(".vaultengine"))

    def tearDown(self):
        self.tmp.cleanup()

    def run_cli(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = cli_main(list(argv))
        return code, out.getvalue(), err.getvalue()

    def test_read_commands_need_an_index(self):
        code, _, err = self.run_cli("health", str(self.root))
        self.assertEqual(code, 2)
        self.assertIn("run `vault index", err)

    def test_missing_vault_folder(self):
        self.assertEqual(self.run_cli("index", str(self.root / "nope"))[0], 2)

    def test_index_then_readable_and_json_reports(self):
        code, out, _ = self.run_cli("index", str(self.root))
        self.assertEqual(code, 0)
        self.assertIn("parsed     6  (6 added, 0 changed, 0 deleted)", out)
        self.assertTrue((self.root / ".vaultengine" / "index.sqlite").exists())

        code, out, _ = self.run_cli("index", str(self.root), "--json")
        self.assertEqual(json.loads(out)["parsed"], [])

        code, out, err = self.run_cli("stats", str(self.root))
        self.assertEqual((code, err), (0, ""))
        self.assertIn("attachments         2", out)

        code, out, _ = self.run_cli("health", str(self.root), "--json")
        data = json.loads(out)
        self.assertEqual(data["summary"]["ambiguous_links"], 1)
        self.assertFalse(data["index"]["stale"])

        code, out, _ = self.run_cli("health", str(self.root))
        self.assertIn("could be Archive/Index.md", out)
        self.assertIn("heading 'Value Function' not in Concepts/Proximal Policy Optimization.md", out)

    def test_stale_index_warns(self):
        self.run_cli("index", str(self.root))
        (self.root / "New.md").write_text("# New")
        code, out, err = self.run_cli("stats", str(self.root), "--json")
        self.assertEqual(code, 0)
        self.assertIn("index is out of date (1 added, 0 changed, 0 deleted)", err)
        self.assertEqual(json.loads(out)["index"]["pending"]["added"], ["New.md"])

    def test_custom_db_path_and_module_entry_point(self):
        db = Path(self.tmp.name) / "elsewhere.sqlite"
        result = subprocess.run(
            [sys.executable, "-m", "vaultengine", "index", str(self.root), "--db", str(db), "--json"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        )
        self.assertEqual(len(json.loads(result.stdout)["parsed"]), 6)
        self.assertTrue(db.exists())
        self.assertFalse((self.root / ".vaultengine").exists())


if __name__ == "__main__":
    unittest.main()
