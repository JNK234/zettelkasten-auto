"""The index-backed vault (vaultengine/indexed.py) gives the same answers as parsing every file.

* every note rebuilt from the index equals a fresh parse of its file;
* a format's answers through the index (issues, links and roles, relations and mirrors,
  metadata, note_links / linked_from, the pipeline, page checks) equal the ones from parsing
  every file, for brain and core on the sample vault;
* the same after random edit sequences made without re-indexing (added, deleted, renamed
  notes, new aliases, renamed headings, new links, a new note that resolves dangling links);
* reading never writes: not the index, not the vault; without a usable index, every file is read.
"""

import hashlib
import json
import random
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from vaultengine import Vault
from vaultengine.formats import TypedVault, load_format
from vaultengine.formats.pages import Pages
from vaultengine.formats.pipeline import Pipeline
from vaultengine.index import VaultIndex
from vaultengine.indexed import IndexedVault, open_typed
from vaultengine.parser import normalize_text, parse_note

TESTS = Path(__file__).parent
REPO_ROOT = TESTS.parent
SAMPLE = TESTS / "fixtures" / "vault"
RESOLVER_VAULT = TESTS / "fixtures" / "resolver_vault"
FORMATS = REPO_ROOT / "formats"


def run_vault(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "vaultengine", *args], capture_output=True, text=True, cwd=REPO_ROOT)


def answers(typed: TypedVault) -> dict:
    """Everything a format works out, in a comparable form."""
    paths = sorted(typed.vault.notes)
    out = {
        "issues": [dict(i.to_dict(), detail=i.detail) for i in typed.issues()],
        "types": {p: typed.type_of(p) for p in paths},
        "metadata": {p: {k: [v.key, v.source, str(v.raw), v.line, v.values] for k, v in typed.metadata(p).items()} for p in paths},
        "relations": {p: [e.to_dict() for e in typed.relations(p)] for p in paths},
        "links": {p: [link.to_dict() for link in typed.links(p)] for p in paths},
        "note_links": {p: typed.note_links(p) for p in paths},
        "linked_from": {p: typed.linked_from(p) for p in paths},
    }
    if typed.spec.pipeline:
        pipeline = Pipeline(typed)
        out["pipeline"] = pipeline.report()
        out["note_facts"] = {p: {k: v for k, v in pipeline.note(p).items() if k != "modified"} for p in paths}
    if typed.spec.pages:
        pages = Pages(typed)
        out["pages"] = [pages.check(p) for p in pages.pages()]
    return json.loads(json.dumps(out, sort_keys=True, default=str))


class Copy:
    def __init__(self, source: Path):
        self.dir = tempfile.TemporaryDirectory()
        self.root = Path(self.dir.name) / "vault"
        shutil.copytree(source, self.root)
        self.db = self.root / ".vaultengine" / "index.sqlite"

    def index(self):
        with VaultIndex(self.db) as index:
            index.update(self.root)

    def close(self):
        self.dir.cleanup()


class RebuiltNotesTests(unittest.TestCase):
    def check(self, source: Path, extra: dict[str, bytes] | None = None):
        copy = Copy(source)
        self.addCleanup(copy.close)
        for path, data in (extra or {}).items():
            (copy.root / path).parent.mkdir(parents=True, exist_ok=True)
            (copy.root / path).write_bytes(data)
        copy.index()
        vault = IndexedVault.open(copy.root, copy.db)
        self.addCleanup(vault.close)
        self.assertFalse(vault.stale.added or vault.stale.changed or vault.stale.deleted)
        self.assertTrue(vault._fresh == {})  # nothing parsed: every note comes from the index
        for path in vault.notes:
            with self.subTest(path=path):
                text = normalize_text((copy.root / path).read_bytes().decode("utf-8", errors="replace"))
                self.assertEqual(vault.notes[path], parse_note(text, path=path))
        full = Vault.load(copy.root)
        self.assertEqual(vault.files, full.files)
        self.assertEqual(sorted(vault.notes), sorted(full.notes))

    def test_sample_vault(self):
        self.check(SAMPLE)

    def test_resolver_vault_and_edge_cases(self):
        self.check(RESOLVER_VAULT, {
            "Edge/frontmatter.md": b"---\naliases: [One, Two]\ntags: [a, b/c]\ncreated: 2026-01-02\nwhen: 2026-01-02T10:00:00\n"
                                   b"nested: {x: [1, 2]}\n---\n# T\nbody #inline [[Concepts/Proximal Policy Optimization#Heading]]\n",
            "Edge/bad yaml.md": b"---\n: : :\n---\nstill a note\n",
            "Edge/crlf.md": b"\xef\xbb\xbfCreated: 01-01-2026 10:00\r\nTags: [[a]] #b\r\n\r\nfrom:: [[x]], [[y]]\r\n"
                            b"See [text](https://example.com) and ![[img.png]] and [md](<Notes/Training log.md>) ^blk\r\n",
            "Edge/fields.md": b"Status: #done\nTags: [[t1]] [[t2|T]] #t3\nTags: [[second]]\n\n- supports:: [[A]]\n> x:: 1\n```\n[[in code]]\n```\n",
            "Edge/empty.md": b"",
        })


class SameAnswersTests(unittest.TestCase):
    def test_sample_vault_fresh_index(self):
        copy = Copy(SAMPLE)
        self.addCleanup(copy.close)
        copy.index()
        for fmt in ("brain", "core"):
            with self.subTest(fmt=fmt):
                spec = load_format(FORMATS / f"{fmt}.yaml")
                via_index = open_typed(spec, copy.root, copy.db)
                self.addCleanup(via_index.vault.close)
                self.assertIsInstance(via_index.vault, IndexedVault)
                self.assertEqual(answers(via_index), answers(TypedVault(spec, Vault.load(copy.root))))

    def test_one_note_does_not_read_the_vault(self):
        copy = Copy(SAMPLE)
        self.addCleanup(copy.close)
        copy.index()
        typed = open_typed(load_format(FORMATS / "core.yaml"), copy.root, copy.db)
        self.addCleanup(typed.vault.close)
        page = "50_MOC/topics/emergent-misalignment-from-reward-hacking.md"
        typed.issues(page)
        Pages(typed).check(page)
        loaded = len(typed.vault.notes._cache)
        self.assertLess(loaded, 60, "a page check should load the page, its zettels and their neighbours, not 180 notes")


class StaleIndexTests(unittest.TestCase):
    """Edits made after the last `vault index`: parsed from disk, affected links resolved again."""

    def edit(self, root: Path, rng: random.Random) -> str:
        notes = sorted(p for p in root.rglob("*.md") if ".vaultengine" not in p.parts)
        kind = rng.choice(["link", "delete", "rename", "add", "alias", "heading", "tags", "resolves-dangling", "move"])
        note = rng.choice(notes)
        if kind == "link":
            other = rng.choice(notes).stem
            note.write_text(note.read_text() + f"\nSee [[{other}]] and [[{other}#What This Is]].\n")
        elif kind == "delete":
            note.unlink()
        elif kind == "rename":
            note.rename(note.with_name(note.stem + " renamed.md"))
        elif kind == "move":
            (root / "50_MOC" / "moved").mkdir(exist_ok=True)
            note.rename(root / "50_MOC" / "moved" / note.name)
        elif kind == "add":
            (root / "20_ZETTLEKASTEN" / f"new-{rng.randrange(10**6)}.md").write_text(
                f"Created: 01-01-2026 10:00\nTags: [[ai-safety]]\n\n# New\n\nFrom [[{rng.choice(notes).stem}]].\n")
        elif kind == "alias":
            note.write_text(f"---\naliases: [{rng.choice(notes).stem} alias, RLHF]\n---\n" + note.read_text())
        elif kind == "heading":
            note.write_text(note.read_text().replace("## ", "## Renamed ", 1))
        elif kind == "tags":
            note.write_text(note.read_text().replace("Tags:", "Tags: [[monitoring]] #ai-safety", 1))
        else:
            (root / "10_SOURCES" / "2507.11473 - Chain of Thought Monitorability.md").write_text(
                "Created: 01-01-2026 10:00\nTags: [[x]]\nSource: y\n\n# S\n")
        return f"{kind} {note.relative_to(root)}"

    def test_random_edit_sequences(self):
        for seed in range(4):
            with self.subTest(seed=seed):
                rng = random.Random(seed)
                copy = Copy(SAMPLE)
                self.addCleanup(copy.close)
                copy.index()
                done = []
                for _ in range(4):
                    done.append(self.edit(copy.root, rng))
                for fmt in ("core", "brain"):
                    spec = load_format(FORMATS / f"{fmt}.yaml")
                    via_index = open_typed(spec, copy.root, copy.db)
                    self.addCleanup(via_index.vault.close)
                    self.assertTrue(via_index.vault.stale.added or via_index.vault.stale.changed or via_index.vault.stale.deleted)
                    self.assertEqual(answers(via_index), answers(TypedVault(spec, Vault.load(copy.root))), f"{fmt}: {done}")


class ReadingWritesNothingTests(unittest.TestCase):
    def snapshot(self, root: Path):
        return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in root.rglob("*") if p.is_file() and not p.name.endswith(("-wal", "-shm"))}

    def test_pipeline_page_check_note(self):
        copy = Copy(SAMPLE)
        self.addCleanup(copy.close)
        copy.index()
        (copy.root / "20_ZETTLEKASTEN/reward.md").write_text("changed since the index\n")  # a stale note too
        before = self.snapshot(copy.root)
        full = {}
        for args in (("pipeline", "--json"), ("page", "check", "--json"), ("note", "monitorability-tax", "--json")):
            with self.subTest(args=args):
                command = [args[0], *([args[1]] if args[0] == "page" else []), str(copy.root),
                           *[a for a in args[1:] if a not in ("check",)], "-f", "core"]
                result = run_vault(*command)
                self.assertIn(result.returncode, (0, 1), result.stderr)
                self.assertNotIn("no usable index", result.stderr)
                full[args[0]] = json.loads(result.stdout)
        self.assertEqual(self.snapshot(copy.root), before)
        # The same answers without the index.
        shutil.rmtree(copy.root / ".vaultengine")
        without = run_vault("pipeline", str(copy.root), "-f", "core", "--json")
        self.assertIn("no usable index", without.stderr)
        self.assertEqual(json.loads(without.stdout), full["pipeline"])

    def test_an_index_of_another_schema_is_left_alone(self):
        copy = Copy(SAMPLE)
        self.addCleanup(copy.close)
        copy.index()
        import sqlite3
        conn = sqlite3.connect(copy.db)
        conn.execute("PRAGMA user_version = 4")
        conn.close()
        before = hashlib.sha256(copy.db.read_bytes()).hexdigest()
        result = run_vault("pipeline", str(copy.root), "-f", "core")
        self.assertEqual(result.returncode, 0)
        self.assertIn("no usable index", result.stderr)
        self.assertEqual(hashlib.sha256(copy.db.read_bytes()).hexdigest(), before)  # not rebuilt by a read


if __name__ == "__main__":
    unittest.main()
