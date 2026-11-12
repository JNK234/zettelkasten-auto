"""`vault move` and `vault merge`, and the write gate that now checks only what a write touches.

On a vault with links of every shape (names, paths, relative and markdown links with %20 and <…>,
headings, blocks, embeds, aliases, frontmatter, a Tags line, self-links, a dangling name):

* every link that resolved before resolves afterwards to the same note, at its new place, and a
  heading or block that was found still is; `vault health` shows no new broken link;
* a dry run writes nothing; a write that would break `vault check` is refused and writes nothing;
* what the engine remembers follows: the baseline, page records, the finding blocks of ideas;
* a crash in the middle is rolled back by the next write, deleted note included;
* the local gate gives the verdict a check of the whole vault would give.
"""

import hashlib
import json
import random
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from vaultengine import Vault
from vaultengine.formats import TypedVault, load_format
from vaultengine.formats.moves import plan_merge, plan_move
from vaultengine.formats.write import FileChange, judge, judge_everything
from vaultengine.graph import LinkGraph
from vaultengine.indexed import open_typed

TESTS = Path(__file__).parent
REPO_ROOT = TESTS.parent
SAMPLE = TESTS / "fixtures" / "vault"
CORE = load_format(REPO_ROOT / "formats" / "core.yaml")


def run_vault(*args: str, env: dict | None = None) -> subprocess.CompletedProcess:
    import os
    return subprocess.run([sys.executable, "-m", "vaultengine", *args], capture_output=True, text=True, cwd=REPO_ROOT,
                          env={**os.environ, **(env or {})})


HEAD = "Created: 01-01-2026 10:00\nTags: [[ml]]\n"
FILES = {
    "10_SOURCES/S1.md": HEAD + "Source: x\n\n# S1\n\n## Part\n\nA claim. ^blk1\n",
    "20_ZETTLEKASTEN/target.md": "---\naliases: [The Target]\n---\n" + HEAD
        + "\n# Target\n\n## What This Is\n\nBody ^tb\n\nFrom [[S1]]. Self [[#What This Is]] and [[target#What This Is]].\n",
    "20_ZETTLEKASTEN/other.md": "Created: 01-01-2026 10:00\nTags: [[ml]] [[target]]\n\n# Other\n\n## Section B\n\n"
        "From [[S1#Part]], see [[target]], [[target|T]], ![[target#^tb]], [[20_ZETTLEKASTEN/target]], [md](target.md), "
        "[rel](../20_ZETTLEKASTEN/target.md#What%20This%20Is), [ang](<target.md>), [[The Target]], [[nowhere]].\n",
    "20_ZETTLEKASTEN/crlf.md": (HEAD + "\nFrom [[S1]].  \nSee [[target#What This Is|the target]]\t\nEnd\n").replace("\n", "\r\n"),
    "50_MOC/p.md": "---\nup: \"[[target]]\"\n---\n" + HEAD + "\n# P\n\n[[target]] [[other]] [[crlf]] [[z3]] [[S1#^blk1]]\n",
    "20_ZETTLEKASTEN/z3.md": HEAD + "\n# z3\n\nFrom [[S1]]. [[renamed-target]] is a name no note has yet.\n",
    "Elsewhere/target-twin.md": "Untyped, elsewhere.\n",
    "Elsewhere/linker.md": "Links [[target-twin]] by name.\n",
}


def edges(root: Path):
    vault = Vault.load(root)
    return list(LinkGraph(vault).edges)


def resolved(root: Path, mapping: dict[str, str], self_links_dropped: bool = False):
    """Resolved links as (source, target, subpath found), with the old paths mapped to the new ones.
    ``self_links_dropped``: leave out the links the mapping turns into links to the note they are in (a merge
    drops them); self-links a note already had stay."""
    found = Counter()
    for e in edges(root):
        if e.status != "resolved":
            continue
        source, target = mapping.get(e.source, e.source), mapping.get(e.target_path, e.target_path)
        if self_links_dropped and source == target and e.source != e.target_path:
            continue
        found[(source, target, e.subpath_status)] += 1
    return found


def edges_between(root: Path, path: str) -> Counter:
    """The note's resolved links to itself, by whether their heading or block is found."""
    return Counter(e.subpath_status or "whole note" for e in edges(root)
                   if e.status == "resolved" and e.source == path and e.target_path == path)


def broken(root: Path) -> Counter:
    return Counter((e.status, e.subpath_status) for e in edges(root) if e.status != "resolved" or e.subpath_status == "missing")


class Disk:
    def __init__(self, files=FILES):
        self.dir = tempfile.TemporaryDirectory()
        self.root = Path(self.dir.name) / "vault"
        for path, text in files.items():
            target = self.root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(text.encode("utf-8"))

    def snapshot(self):
        return {str(p.relative_to(self.root)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in self.root.rglob("*") if p.is_file() and not p.name.endswith(("-wal", "-shm"))}

    def close(self):
        self.dir.cleanup()


class RelocationCase(unittest.TestCase):
    def setUp(self):
        self.disk = Disk()
        self.root = self.disk.root
        self.assertEqual(run_vault("index", str(self.root)).returncode, 0)

    def tearDown(self):
        self.disk.close()

    def vault(self, *args):
        return run_vault(args[0], str(self.root), *args[1:], "-f", "core")

    def health(self) -> dict[str, str]:
        self.assertEqual(run_vault("index", str(self.root)).returncode, 0)
        text = run_vault("health", str(self.root)).stdout
        return {key: re.search(rf"{key}\s+(\d+)", text)[1] for key in ("unresolved links", "ambiguous links",
                                                                           "missing heading/block targets")}

    def keeps_every_link(self, args: list[str], mapping: dict[str, str], gained: int = 0, self_links_dropped: bool = False):
        before = resolved(self.root, mapping, self_links_dropped)
        broken_before = broken(self.root)
        health_before = self.health()
        snapshot = self.disk.snapshot()
        dry = self.vault(*args)
        self.assertEqual(dry.returncode, 0, dry.stdout + dry.stderr)
        self.assertIn("Not written: add --write to write it.", dry.stdout)
        self.assertEqual(self.disk.snapshot(), snapshot)  # a dry run writes nothing
        written = self.vault(*args, "--write")
        self.assertEqual(written.returncode, 0, written.stdout + written.stderr)
        after = resolved(self.root, {})
        self.assertEqual(before - after, Counter(), "links lost their target")
        self.assertEqual(sum((after - before).values()), gained, after - before)
        self.assertFalse(broken(self.root) - broken_before, "new broken links")
        health_after = self.health()
        for key in health_before:
            self.assertLessEqual(int(health_after[key]), int(health_before[key]), key)
        return written


class MoveTests(RelocationCase):
    def test_into_a_folder(self):
        result = self.keeps_every_link(["move", "target", "Elsewhere/"], {"20_ZETTLEKASTEN/target.md": "Elsewhere/target.md"})
        other = (self.root / "20_ZETTLEKASTEN/other.md").read_text()
        # Only what no longer resolved changed: the path link and the relative one. Names and the alias still work.
        self.assertIn("[[Elsewhere/target]]", other)
        self.assertIn("[rel](../Elsewhere/target.md#What%20This%20Is)", other)
        self.assertIn("see [[target]], [[target|T]], ![[target#^tb]]", other)
        self.assertIn("[[The Target]]", other)
        self.assertFalse((self.root / "20_ZETTLEKASTEN/target.md").exists())
        self.assertIn("rename 20_ZETTLEKASTEN/target.md -> Elsewhere/target.md", result.stdout)

    def test_rename(self):
        result = self.keeps_every_link(["move", "target", "renamed-target"],
                                       {"20_ZETTLEKASTEN/target.md": "20_ZETTLEKASTEN/renamed-target.md"}, gained=1)
        other = (self.root / "20_ZETTLEKASTEN/other.md").read_text()
        for written in ("Tags: [[ml]] [[renamed-target]]", "see [[renamed-target]], [[renamed-target|T]], ![[renamed-target#^tb]]",
                        "[[20_ZETTLEKASTEN/renamed-target]]", "[md](renamed-target.md)", "[ang](<renamed-target.md>)",
                        "[[renamed-target|The Target]]" if False else "[[The Target]]", "[[nowhere]]"):
            self.assertIn(written, other)
        # Frontmatter links, self-links by name, and CRLF files.
        self.assertIn('up: "[[renamed-target]]"', (self.root / "50_MOC/p.md").read_text())
        self.assertIn("and [[renamed-target#What This Is]].", (self.root / "20_ZETTLEKASTEN/renamed-target.md").read_text())
        crlf = (self.root / "20_ZETTLEKASTEN/crlf.md").read_bytes()
        self.assertEqual(crlf, FILES["20_ZETTLEKASTEN/crlf.md"].replace("[[target#", "[[renamed-target#").encode())
        # The dangling [[renamed-target]] in z3 reaches the note now, and the output says so.
        self.assertIn("matched no note before, reaches it now: 20_ZETTLEKASTEN/z3.md", result.stdout)

    def test_a_new_name_another_note_has(self):
        # Renaming target to target-twin: Elsewhere/linker's [[target-twin]] would become ambiguous; it is rewritten
        # to the path so it keeps reaching Elsewhere/target-twin.md.
        self.keeps_every_link(["move", "target", "target-twin"], {"20_ZETTLEKASTEN/target.md": "20_ZETTLEKASTEN/target-twin.md"})
        self.assertIn("[[Elsewhere/target-twin]]", (self.root / "Elsewhere/linker.md").read_text())
        self.assertIn("see [[20_ZETTLEKASTEN/target-twin]]", (self.root / "20_ZETTLEKASTEN/other.md").read_text())

    def test_bytes_outside_links_stay(self):
        before = {p: (self.root / p).read_bytes() for p in FILES}
        self.vault("move", "target", "renamed-target", "--write")
        for path, old in before.items():
            new_path = self.root / path
            if not new_path.exists():
                continue
            new = new_path.read_bytes()
            # Every line that changed contains a link that was rewritten; the rest is byte for byte the same.
            for a, b in zip(old.splitlines(keepends=True), new.splitlines(keepends=True)):
                if a != b:
                    self.assertIn(b"renamed-target", b, path)
            self.assertEqual(len(old.splitlines()), len(new.splitlines()), path)

    def test_refused_when_check_would_get_worse(self):
        snapshot = self.disk.snapshot()
        result = self.vault("move", "target", "20_ZETTLEKASTEN/drafts/", "--write")  # no type matches drafts/
        self.assertEqual(result.returncode, 1)
        self.assertIn("untyped-note", result.stdout)
        self.assertEqual(self.disk.snapshot(), snapshot)
        for args, error in ((["move", "target", "other"], "already exists"), (["move", "nope", "x"], "no note 'nope'"),
                            (["move", "target", "../out/"], "not a place in the vault")):
            with self.subTest(args=args):
                result = self.vault(*args)
                self.assertEqual(result.returncode, 2)
                self.assertIn(error, result.stderr)

    def test_a_crash_is_rolled_back_by_the_next_write(self):
        snapshot = self.disk.snapshot()
        crashed = run_vault("move", str(self.root), "target", "renamed-target", "-f", "core", "--write",
                            env={"VAULTENGINE_TEST_CRASH_AFTER_RENAMES": "1"})
        self.assertEqual(crashed.returncode, 70)
        self.assertNotEqual(self.disk.snapshot(), snapshot)
        result = self.vault("move", "target", "renamed-target")  # any write command recovers first
        self.assertIn("recovered an interrupted write", result.stdout + result.stderr)
        after = {k: v for k, v in self.disk.snapshot().items() if not k.startswith(".vaultengine/")}
        self.assertEqual(after, {k: v for k, v in snapshot.items() if not k.startswith(".vaultengine/")})


    def test_a_crash_after_the_deletion_is_rolled_back(self):
        snapshot = {k: v for k, v in self.disk.snapshot().items() if not k.startswith(".vaultengine/")}
        crashed = run_vault("move", str(self.root), "target", "renamed-target", "-f", "core", "--write",
                            env={"VAULTENGINE_TEST_CRASH_AFTER_DELETES": "1"})
        self.assertEqual(crashed.returncode, 70)
        self.assertFalse((self.root / "20_ZETTLEKASTEN/target.md").exists())  # deleted, the write not finished
        self.vault("move", "target", "renamed-target")
        after = {k: v for k, v in self.disk.snapshot().items() if not k.startswith(".vaultengine/")}
        self.assertEqual(after, snapshot)  # the deleted note is back, the new one gone, the links as they were


class MergeTests(RelocationCase):
    def test_two_notes(self):
        survivor_before = (self.root / "20_ZETTLEKASTEN/other.md").read_text()
        target_text = (self.root / "20_ZETTLEKASTEN/target.md").read_text()
        # Every link keeps its target, except the ones that would point at the note they are in: those are
        # dropped; the two with a heading or block become same-note links (gained: they reach other itself).
        self.keeps_every_link(["merge", "other", "target"], {"20_ZETTLEKASTEN/target.md": "20_ZETTLEKASTEN/other.md"},
                              gained=2, self_links_dropped=True)
        self.assertFalse((self.root / "20_ZETTLEKASTEN/target.md").exists())
        survivor = (self.root / "20_ZETTLEKASTEN/other.md").read_text()
        head, _, merged = survivor.partition("\n<!-- vault:merged -->\n")
        self.assertEqual(merged.split("\n")[0], "## Merged from target")
        self.assertTrue(merged.endswith("<!-- /vault:merged -->\n"))
        self.assertIn("```yaml\naliases: [The Target]\n```", merged)  # its frontmatter, kept but inert
        self.assertIn("Body ^tb", merged)
        # The survivor keeps its own text; its links to target would reach itself now, so they are dropped: the
        # Tags link goes, the others become their visible text (an alias included), a heading or block link
        # becomes a same-note link.
        expected = (survivor_before.replace("Tags: [[ml]] [[target]]", "Tags: [[ml]]")
                    .replace("see [[target]], [[target|T]], ![[target#^tb]], [[20_ZETTLEKASTEN/target]], [md](target.md), "
                             "[rel](../20_ZETTLEKASTEN/target.md#What%20This%20Is), [ang](<target.md>), [[The Target]],",
                             "see target, T, ![[#^tb]], 20_ZETTLEKASTEN/target, md, [rel](#What%20This%20Is), ang, The Target,"))
        self.assertEqual(head, expected)
        body = target_text.split("---\n", 2)[2]  # target's text after its frontmatter is in the merged section
        for line in body.splitlines():
            if "[[" not in line:
                self.assertIn(line, merged)
        # target's own self-link by name was a self-link already: it is re-spelled, not dropped.
        self.assertIn("Self [[#What This Is]] and [[other#What This Is]].", merged)
        self.assertEqual(edges_between(self.root, "20_ZETTLEKASTEN/other.md"),
                         Counter({"found": 4}))  # [[#What This Is]] twice, [[#^tb]], [rel](#What%20This%20Is)

    def test_self_links_are_dropped_and_reported(self):
        files = {
            "10_SOURCES/S1.md": HEAD + "Source: x\n\n# S1\n",
            "20_ZETTLEKASTEN/a.md": "---\nrelated:\n  - \"[[b]]\"\n  - \"[[z]]\"\nup: \"[[b]]\"\n---\n" + HEAD
                + "\n# A\n\nFrom [[S1]].\n\n- [[b]]\n- [ ] [[b|task]]\n> [[b]]\n\nSee [[b]] here and [[b#Part|the part]].\n"
                  "supports:: [[b]]\nrelated:: [[z]], [[b]]\n",
            "20_ZETTLEKASTEN/b.md": HEAD + "\n# B\n\n## Part\n\nFrom [[S1]].\n\n[[a]]\n\nBack to [[a|home]], ![[a]] and [[z]].\n",
            "20_ZETTLEKASTEN/z.md": HEAD + "\n# Z\n\nFrom [[S1]]. [[b]] and [[a]].\n",
        }
        self.disk.close()
        self.disk = Disk(files)
        self.root = self.disk.root
        self.assertEqual(run_vault("index", str(self.root)).returncode, 0)
        dry = self.vault("merge", "a", "b", "--json")
        self.assertEqual(dry.returncode, 0, dry.stderr)
        report = json.loads(dry.stdout)
        actions = Counter((link["link"], link["action"], link["result"]) for link in report["self_links"])
        self.assertEqual(actions, Counter({
            ("[[b]]", "line removed", ""): 4,  # frontmatter list item, list item, quote, inline field alone
            ("[[a]]", "line removed", ""): 1,  # b's line holding only [[a]]
            ("[[b|task]]", "line removed", ""): 1,  # a task holding only the link
            ("[[b]]", "unlinked", "b"): 2,  # up: "[[b]]" in frontmatter, and in prose
            ("[[b#Part|the part]]", "same-note link", "[[#Part|the part]]"): 1,
            ("[[b]]", "removed", ""): 1,  # from related:: [[z]], [[b]]
            ("[[a|home]]", "unlinked", "home"): 1,
            ("![[a]]", "removed", ""): 1,
        }))
        self.keeps_every_link(["merge", "a", "b"], {"20_ZETTLEKASTEN/b.md": "20_ZETTLEKASTEN/a.md"}, gained=1,
                              self_links_dropped=True)
        text = (self.root / "20_ZETTLEKASTEN/a.md").read_text()
        head, _, merged = text.partition("<!-- vault:merged -->")
        self.assertEqual(head.split("---\n")[1], 'related:\n  - "[[z]]"\nup: "b"\n')
        self.assertIn("\nFrom [[S1]].\n\n\nSee b here and [[#Part|the part]].\nrelated:: [[z]]\n", head)
        self.assertIn("Back to home and [[z]].", merged)  # the embed went with its comma
        self.assertNotIn("\n[[a]]\n", merged)
        self.assertEqual(edges_between(self.root, "20_ZETTLEKASTEN/a.md"), Counter({"found": 1}))
        self.assertIn("[[a]] and [[a]].", (self.root / "20_ZETTLEKASTEN/z.md").read_text())  # other notes: re-spelled

    def test_a_missing_name_into_a_note(self):
        # z3 links [[renamed-target]], which no note has: merged into target, the link is rewritten to it.
        result = self.keeps_every_link(["merge", "target", "renamed-target"], {}, gained=1)
        self.assertIn("[[target]] is a name no note has yet", (self.root / "20_ZETTLEKASTEN/z3.md").read_text())
        self.assertIn("links to [[renamed-target]] (no note) -> 20_ZETTLEKASTEN/target.md", result.stdout)

    def test_two_missing_names(self):
        self.keeps_every_link(["merge", "nowhere", "renamed-target"], {})
        self.assertIn("[[nowhere]] is a name no note has yet", (self.root / "20_ZETTLEKASTEN/z3.md").read_text())

    def test_refusals(self):
        for args, error in ((["merge", "target", "target"], "into itself"),
                            (["merge", "no-such-note", "target"], "use `vault move"),
                            (["merge", "target", "never-linked"], "no note and no link named")):
            with self.subTest(args=args):
                result = self.vault(*args)
                self.assertEqual(result.returncode, 2)
                self.assertIn(error, result.stderr)


class RecordsFollowTests(RelocationCase):
    def test_baseline_page_record_and_idea(self):
        # A baseline, a page that lists target, an idea from a link finding about target.
        self.assertEqual(self.vault("check", "--update-baseline").returncode, 0)
        page = run_vault("page", "new", str(self.root), "About Target", "-f", "core",
                         "--from", "[[target]] [[other]] [[crlf]] [[z3]]", "--write")
        self.assertEqual(page.returncode, 0, page.stdout + page.stderr)
        idea_block = ("<!-- vault:finding -->\n## Finding\n\n- id: link-0000000000\n- method: link (x)\n"
                      "- notes: [[target]], [[z3]]\n- found: 2026-01-01 10:00\n<!-- /vault:finding -->\n")
        from vaultengine.findings import finding_id
        old_id = finding_id("link", ["20_ZETTLEKASTEN/target.md", "20_ZETTLEKASTEN/z3.md"])
        (self.root / "50_MOC/ideas").mkdir(parents=True)
        (self.root / "50_MOC/ideas/Idea.md").write_text("Created: 01-01-2026 10:00\nTags: [[ml]]\nStatus: #open\n\n# Idea\n\n"
                                                         + idea_block.replace("link-0000000000", old_id))
        self.assertEqual(run_vault("index", str(self.root)).returncode, 0)
        self.assertEqual(self.vault("check", "--update-baseline").returncode, 0)

        result = self.vault("move", "target", "renamed-target", "--write")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("state    .vaultengine/baseline-core.json", result.stdout)
        self.assertIn("state    .vaultengine/pages.json", result.stdout)
        # The baseline follows: nothing new, and no rename had to be guessed.
        check = self.vault("check", "--baseline")
        self.assertEqual(check.returncode, 0, check.stdout)
        self.assertNotIn("followed a renamed note", check.stdout)
        baseline = (self.root / ".vaultengine/baseline-core.json").read_text()
        self.assertNotIn("20_ZETTLEKASTEN/target.md", baseline)
        # The page record follows: its zettels are as recorded.
        data = json.loads(run_vault("page", "check", str(self.root), "about-target", "-f", "core", "--json").stdout)["pages"][0]
        self.assertEqual(data["changed"], [])
        self.assertIn("20_ZETTLEKASTEN/renamed-target.md", json.loads((self.root / ".vaultengine/pages.json").read_text())["pages"]
                      ["50_MOC/about-target.md"]["zettels"])
        # The idea's finding block: its link still reaches target, its id is the one the moved notes give.
        idea = (self.root / "50_MOC/ideas/Idea.md").read_text()
        new_id = finding_id("link", ["20_ZETTLEKASTEN/renamed-target.md", "20_ZETTLEKASTEN/z3.md"])
        self.assertIn(f"- id: {new_id}", idea)
        self.assertNotIn(old_id, idea)

    def test_merge_records(self):
        page = run_vault("page", "new", str(self.root), "About Target", "-f", "core",
                         "--from", "[[target]] [[other]] [[crlf]] [[z3]]", "--write")
        self.assertEqual(page.returncode, 0, page.stdout + page.stderr)
        self.assertEqual(self.vault("merge", "other", "target", "--write").returncode, 0)
        record = json.loads((self.root / ".vaultengine/pages.json").read_text())["pages"]["50_MOC/about-target.md"]
        self.assertNotIn("20_ZETTLEKASTEN/target.md", record["zettels"])
        self.assertIn("20_ZETTLEKASTEN/other.md", record["zettels"])
        # The survivor changed (it holds the merged text): the page says so, and a refresh clears it.
        data = json.loads(run_vault("page", "check", str(self.root), "about-target", "-f", "core", "--json").stdout)["pages"][0]
        self.assertEqual(data["changed"], [{"path": "20_ZETTLEKASTEN/other.md", "status": "changed"}])


class LocalGateTests(unittest.TestCase):
    """The write gate checks only the notes a write can affect; its verdict is the whole vault's."""

    def verdicts(self, typed, files, renamed=None):
        local = judge(typed, files, renamed)
        everything = judge_everything(typed, files, renamed)
        key = lambda issues: sorted((i.path, i.code, i.message) for i in issues)  # noqa: E731
        return (key(local.new_violations), key(local.tolerated)), (key(everything.new_violations), key(everything.tolerated))

    def sample(self) -> Path:
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        root = Path(scratch.name) / "vault"
        shutil.copytree(SAMPLE, root)
        self.assertEqual(run_vault("index", str(root)).returncode, 0)
        return root

    def test_writes_whose_violations_land_elsewhere(self):
        # Each changes violations in notes it does not write: the gate must look there, and agree with the whole vault.
        root = self.sample()
        evd = "50_MOC/discourse/rl-environment-design/EVD - AppWorld grades only the database so any action language scores the same.md"
        clm = "CLM - Correctness must gate all shaping credit"  # not yet supported by this evidence
        source = "10_SOURCES/2503.11926 - Monitoring Reasoning Models (CoT Obfuscation).md"
        cases = {
            # a source zettels cite is deleted: they cite no source any more (core: too-few-links, on the zettels)
            "delete a cited source": ("core", [FileChange(source, (root / source).read_text(), None)], 4),
            # a relation to a claim: the claim owes a mirror (brain: missing-mirror, on the claim)
            "new relation": ("brain", [FileChange(evd, (root / evd).read_text(), (root / evd).read_text() + f"\nsupports:: [[{clm}]]\n")], 1),
            # a note under a name a relation field points at: the link resolves now (brain: its unresolved-target goes)
            "name a missing target": ("brain", [FileChange("10_SOURCES/2601.05808 - EnvScaler.md", None, "Created: 01-01-2026 10:00\n")], 0),
        }
        for label, (fmt, files, at_least) in cases.items():
            with self.subTest(label):
                spec = load_format(REPO_ROOT / "formats" / f"{fmt}.yaml")
                for typed in (TypedVault(spec, Vault.load(root)), open_typed(spec, root, root / ".vaultengine/index.sqlite")):
                    local, everything = self.verdicts(typed, files)
                    self.assertEqual(local, everything)
                    elsewhere = [v for v in everything[0] if v[0] not in {f.path for f in files}]
                    self.assertGreaterEqual(len(elsewhere), at_least)
                    if hasattr(typed.vault, "close"):
                        typed.vault.close()

    def test_random_writes_on_the_sample(self):
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        root = Path(scratch.name) / "vault"
        shutil.copytree(SAMPLE, root)
        self.assertEqual(run_vault("index", str(root)).returncode, 0)
        rng = random.Random(7)
        notes = sorted(p.relative_to(root).as_posix() for p in root.rglob("*.md"))
        for fmt in ("core", "brain"):
            spec = load_format(REPO_ROOT / "formats" / f"{fmt}.yaml")
            for step in range(12):
                typed = open_typed(spec, root, root / ".vaultengine/index.sqlite") if step % 2 else TypedVault(spec, Vault.load(root))
                path = rng.choice(notes)
                text = (root / path).read_text()
                kind = rng.choice(["drop-link", "add-link", "delete", "new-name", "tags", "move", "merge"])
                renamed = None
                if kind == "drop-link":
                    files = [FileChange(path, text, re.sub(r"\[\[[^\]]+\]\]", "", text, count=3))]
                elif kind == "add-link":
                    files = [FileChange(path, text, text + f"\nsupports:: [[{Path(rng.choice(notes)).stem}]]\n")]
                elif kind == "delete":
                    files = [FileChange(path, text, None)]
                elif kind == "new-name":
                    files = [FileChange(f"20_ZETTLEKASTEN/{Path(rng.choice(notes)).stem}.md", None, "Tags: #x\n\nnew\n")]
                    files = [f for f in files if not (root / f.path).exists()] or [FileChange(path, text, text + "\n")]
                elif kind == "tags":
                    files = [FileChange(path, text, text.replace("Tags:", "Tags: [[x]]", 1))]
                elif kind == "move":
                    plan = plan_move(typed, path, f"50_MOC/moved-{step}")
                    files, renamed = plan.files, plan.renamed
                else:
                    other = rng.choice([n for n in notes if n != path])
                    plan = plan_merge(typed, path, other)
                    files, renamed = plan.files, plan.renamed
                with self.subTest(fmt=fmt, step=step, kind=kind, path=path):
                    local, everything = self.verdicts(typed, files, renamed)
                    self.assertEqual(local, everything)
                if hasattr(typed.vault, "close"):
                    typed.vault.close()


if __name__ == "__main__":
    unittest.main()
