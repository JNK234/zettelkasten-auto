"""`vault page new / refresh / check` under formats/core.yaml, on small vaults on disk.

What is proved here: nothing is written without --write; a written page passes `vault check`;
editing a listed zettel makes the page stale and a refresh clears it; every byte outside the
markers stays the same; the "behind" list is exactly the zettels that share the page's tags or
links; the index part and the log line; the engine state lives in .vaultengine, not in the note.
"""

import datetime as dt
import json
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from vaultengine import Vault
from vaultengine.formats import FormatError, TypedVault, load_format, parse_format
from vaultengine.formats.pages import PageError, Pages, excerpt, find_blocks, replace_block

TESTS = Path(__file__).parent
REPO_ROOT = TESTS.parent
CORE = REPO_ROOT / "formats" / "core.yaml"
NOW = dt.datetime(2026, 11, 10, 9, 30)


def run_vault(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "vaultengine", *args], capture_output=True, text=True, cwd=REPO_ROOT)


def source() -> str:
    return "Created: 01-01-2026 10:00\nTags: [[ml]]\nSource: https://example.com\n\n# S\n"


def zettel(tags: str, body: str) -> str:
    return f"Created: 01-01-2026 10:00\nTags: {tags}\n\n# Title\n\n{body}\n"


# The page is made from z1, z2, z3. Every other zettel is there to be in or out of "behind" for one reason.
FILES = {
    "10_SOURCES/S1.md": source(),
    "10_SOURCES/S2.md": source(),
    "10_SOURCES/S3.md": source(),
    "20_ZETTLEKASTEN/z1.md": zettel("[[rl]] [[grading]]", "Graders must fail the empty policy. From [[S1]]."),
    "20_ZETTLEKASTEN/z2.md": zettel("#rl #safety", "## What This Is\n\nEnd-state grading beats transcripts, see [[z1]].\n\nFrom [[S1]] and [[S2]]."),
    "20_ZETTLEKASTEN/z3.md": zettel("[[grading]]", "Partial credit must be gated on correctness.\n\nSee [[z9-target]]. From [[S2]]."),
    # Behind needs two things in common: the page's tags (rl, grading) it carries, the page's links (z1, z2, z3,
    # S1, S2) it links to.
    # behind: four in common
    "20_ZETTLEKASTEN/b-both.md": zettel("[[grading]] [[rl]]", "See [[z2]], from [[S1]]."),
    # behind: two tags (#rl is the same tag as [[rl]])
    "20_ZETTLEKASTEN/b-two-tags.md": zettel("#rl [[grading]]", "Rubrics drift. From [[S3]]."),
    # behind: a tag and a link
    "20_ZETTLEKASTEN/b-tag-link.md": zettel("#grading", "Extends [[z3]]. From [[S3]]."),
    # behind: two links (a zettel and a source on the page)
    "20_ZETTLEKASTEN/b-two-links.md": zettel("[[other]]", "Builds on [[z1]], another take on [[S2]]."),
    # out: one thing in common each (a tag; a zettel on the page; a source on the page)
    "20_ZETTLEKASTEN/b-tag.md": zettel("#rl", "Off-policy data is stale. From [[S3]]."),
    "20_ZETTLEKASTEN/b-link.md": zettel("[[other]]", "Builds on [[z1]]. From [[S3]]."),
    "20_ZETTLEKASTEN/b-source.md": zettel("[[other]]", "Another take on [[S2]]."),
    # out: z3 links to it, but it shares nothing with the page
    "20_ZETTLEKASTEN/z9-target.md": zettel("[[other]]", "Linked from z3 only. From [[S3]]."),
    # out: the page tag is in its text, not in its Tags
    "20_ZETTLEKASTEN/o-bodytag.md": zettel("[[other]]", "About #rl in passing. From [[S3]]."),
    # out: has S1 only in Tags (a tag link does not count as a link)
    "20_ZETTLEKASTEN/o-tagslink.md": zettel("[[S1]]", "Nothing else. From [[S3]]."),
    # out: unrelated
    "20_ZETTLEKASTEN/o-unrelated.md": zettel("[[other]]", "Unrelated. From [[S3]]."),
    # out: not zettels, though they share tags and links
    "50_MOC/other page.md": zettel("[[rl]]", "[[z1]] [[z2]] [[z3]]"),
    "50_MOC/discourse/ignored.md": "Tags: [[rl]]\n\n[[z1]] [[S1]]\n",
}
ON_PAGE = ["20_ZETTLEKASTEN/z1.md", "20_ZETTLEKASTEN/z2.md", "20_ZETTLEKASTEN/z3.md"]
BEHIND = {
    "20_ZETTLEKASTEN/b-both.md": {"tags": ["rl", "grading"], "links": ["S1", "z2"]},
    "20_ZETTLEKASTEN/b-two-tags.md": {"tags": ["rl", "grading"], "links": []},
    "20_ZETTLEKASTEN/b-tag-link.md": {"tags": ["grading"], "links": ["z3"]},
    "20_ZETTLEKASTEN/b-two-links.md": {"tags": [], "links": ["z1", "S2"]},
}


class Disk:
    def __init__(self, files=FILES):
        self.dir = tempfile.TemporaryDirectory()
        self.root = Path(self.dir.name)
        for path, text in files.items():
            self.write(path, text)

    def write(self, path, text, newline="\n"):
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(text.replace("\n", newline).encode("utf-8"))

    def read(self, path) -> bytes:
        return (self.root / path).read_bytes()

    def snapshot(self):
        return sorted((str(p.relative_to(self.root)), p.read_bytes()) for p in self.root.rglob("*") if p.is_file())

    def vault(self, *args):
        return run_vault(*args[:1], str(self.root), *args[1:], "-f", "core")

    def pages(self) -> Pages:
        return Pages(TypedVault(load_format(CORE), Vault.load(self.root)))

    def close(self):
        self.dir.cleanup()


class DiskTest(unittest.TestCase):
    files = FILES

    def setUp(self):
        self.disk = Disk(self.files)

    def tearDown(self):
        self.disk.close()

    def new_page(self, *extra):
        return run_vault("page", "new", str(self.disk.root), "Grading Without Transcripts", "-f", "core",
                         "--from", "[[z1]] [[z2]], [[z3]]", *extra)

    def page(self, *args):
        return run_vault("page", args[0], str(self.disk.root), *args[1:], "-f", "core")


# ----- scaffold ---------------------------------------------------------------------------------------


class NewTests(DiskTest):
    def test_prints_by_default_and_writes_nothing(self):
        before = self.disk.snapshot()
        result = self.new_page()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("# Grading Without Transcripts\n\n<!-- vault:zettels -->", result.stdout)
        self.assertIn("Would write 3 notes (page new):\n  + 50_MOC/grading-without-transcripts.md", result.stdout)
        self.assertIn("Not written: add --write to write it.", result.stdout)
        self.assertEqual(self.disk.snapshot(), before)
        self.assertFalse((self.disk.root / ".vaultengine").exists())
        data = json.loads(self.new_page("--json").stdout)
        self.assertEqual((data["status"], data["written"]), ("dry-run", False))
        self.assertEqual(self.disk.snapshot(), before)

    def test_the_whole_page(self):
        self.assertEqual(self.new_page("--write").returncode, 0)
        text = self.disk.read("50_MOC/grading-without-transcripts.md").decode()
        created = text.splitlines()[0]
        self.assertRegex(created, r"^Created: \d\d-\d\d-\d{4} \d\d:\d\d$")
        self.assertEqual(text, created + "\n" + textwrap.dedent("""\
            Tags: [[rl]] [[grading]]

            # Grading Without Transcripts

            <!-- vault:zettels -->
            ## Zettels

            ### rl
            - [[z1]]: Graders must fail the empty policy. From S1.
            - [[z2]]: End-state grading beats transcripts, see z1.

            ### grading
            - [[z3]]: Partial credit must be gated on correctness.
            <!-- /vault:zettels -->

            <!-- vault:sources -->
            ## Sources

            - [[S1]] (z1, z2)
            - [[S2]] (z2, z3)
            <!-- /vault:sources -->

            """))

    def test_written_page_passes_vault_check(self):
        self.assertEqual(self.disk.vault("check").returncode, 0)  # the vault is clean to start with
        self.assertEqual(self.new_page("--write").returncode, 0)
        check = self.disk.vault("check")
        self.assertEqual(check.returncode, 0, check.stdout)
        self.assertIn("✓ no violations in 22 notes checked", check.stdout)  # 19 + page, index, log

    def test_index_log_and_state(self):
        self.new_page("--write")
        self.assertEqual(self.disk.read("50_MOC/index.md").decode(), textwrap.dedent("""\
            # Index

            <!-- vault:pages -->
            - [[grading-without-transcripts]]: Grading Without Transcripts (3 zettels, 2 sources)
            - [[other page]]: Title (3 zettels, 2 sources)
            <!-- /vault:pages -->
            """))
        log = self.disk.read("50_MOC/log.md").decode()
        self.assertRegex(log, r"^# Log\n\n- \d{4}-\d\d-\d\d \d\d:\d\d page new \[\[grading-without-transcripts\]\] "
                              r"\(3 zettels, 2 sources\)\n$")
        state = json.loads(self.disk.read(".vaultengine/pages.json"))
        self.assertEqual(sorted(state["pages"]["50_MOC/grading-without-transcripts.md"]["zettels"]), ON_PAGE)
        self.assertNotIn("sha256", self.disk.read("50_MOC/grading-without-transcripts.md").decode())  # not in the note

    def test_refused_when_the_page_would_break_a_rule(self):
        before = self.disk.snapshot()
        result = run_vault("page", "new", str(self.disk.root), "Too Thin", "-f", "core", "--from", "[[z1]] [[z2]]", "--write")
        self.assertEqual(result.returncode, 1)
        self.assertIn("✗ refused: the page new would add 1 violation; nothing was written", result.stdout)
        self.assertIn("too-few-links    links to 2 zettel notes; a page needs at least 3", result.stdout)
        self.assertEqual(self.disk.snapshot(), before)

    def test_bad_input(self):
        cases = {
            ("--from", "[[S1]] [[z1]] [[z2]]"): "10_SOURCES/S1.md is a source, not a zettel",
            ("--from", "[[nope]]"): "no note 'nope'",
            ("--from", "reward"): "no index at",
        }
        for extra, error in cases.items():
            with self.subTest(extra=extra):
                result = run_vault("page", "new", str(self.disk.root), "X", "-f", "core", *extra)
                self.assertEqual(result.returncode, 2)
                self.assertIn(error, result.stderr)

    def test_from_a_query_takes_its_zettels_only(self):
        self.assertEqual(run_vault("index", str(self.disk.root)).returncode, 0)
        result = run_vault("page", "new", str(self.disk.root), "Grading", "-f", "core", "--from", "grading OR graders", "--json")
        data = json.loads(result.stdout)
        self.assertEqual(data["status"], "dry-run")
        content = data["page"]["content"]
        for name in ("z1", "z2", "z3"):
            self.assertIn(f"- [[{name}]]", content)
        self.assertNotIn("- [[other page]]", content)  # a page matched too: not a zettel

    def test_tags_option(self):
        self.new_page("--write", "--tags", "[[evaluation]]")
        self.assertIn("Tags: [[evaluation]]\n", self.disk.read("50_MOC/grading-without-transcripts.md").decode())


# ----- staleness and refresh ------------------------------------------------------------------------


class RefreshTests(DiskTest):
    PAGE = "50_MOC/grading-without-transcripts.md"

    def setUp(self):
        super().setUp()
        self.assertEqual(self.new_page("--write").returncode, 0)

    def check(self):
        return json.loads(self.page("check", self.PAGE, "--json").stdout)["pages"][0]

    def test_a_fresh_page_has_nothing_to_refresh(self):
        before = self.disk.snapshot()
        result = self.page("refresh", self.PAGE, "--write")
        self.assertEqual(result.returncode, 0)
        self.assertIn("Nothing to write", result.stdout)
        self.assertEqual(self.disk.snapshot(), before)

    def test_editing_a_listed_zettel_makes_it_stale_and_refresh_clears_it(self):
        before = self.check()
        self.assertEqual((before["changed"], [b["status"] for b in before["blocks"]]), ([], ["current", "current"]))
        self.disk.write("20_ZETTLEKASTEN/z1.md", zettel("[[rl]] [[grading]]", "A do-nothing agent must score zero. From [[S1]]."))
        stale = self.check()
        self.assertEqual(stale["changed"], [{"path": "20_ZETTLEKASTEN/z1.md", "status": "changed"}])
        self.assertEqual([b["status"] for b in stale["blocks"]], ["out-of-date", "current"])
        self.assertEqual(stale["blocks"][0]["removed"], ["- [[z1]]: Graders must fail the empty policy. From S1."])
        self.assertEqual(stale["blocks"][0]["added"], ["- [[z1]]: A do-nothing agent must score zero. From S1."])
        text = self.page("check", self.PAGE).stdout
        self.assertIn("  ✗ zettels block out of date\n      - - [[z1]]: Graders must fail", text)
        self.assertIn("  ✗ 1 zettel changed since the last refresh\n      changed       20_ZETTLEKASTEN/z1.md", text)
        self.assertIn("→ vault page refresh", text)

        shown = self.page("refresh", self.PAGE)  # shows, writes nothing
        self.assertIn("+- [[z1]]: A do-nothing agent must score zero. From S1.", shown.stdout)
        self.assertEqual(self.check()["changed"], stale["changed"])
        self.assertEqual(self.page("refresh", self.PAGE, "--write").returncode, 0)
        after = self.check()
        self.assertEqual((after["changed"], [b["status"] for b in after["blocks"]]), ([], ["current", "current"]))
        self.assertIn("page refresh [[grading-without-transcripts]]", self.disk.read("50_MOC/log.md").decode())

    def test_a_change_below_the_first_line_is_tracked_in_the_engine_only(self):
        self.disk.write("20_ZETTLEKASTEN/z3.md", zettel("[[grading]]", "Partial credit must be gated on correctness.\n\nMore. See [[z9-target]]. From [[S2]]."))
        stale = self.check()
        self.assertEqual(stale["changed"], [{"path": "20_ZETTLEKASTEN/z3.md", "status": "changed"}])
        self.assertEqual([b["status"] for b in stale["blocks"]], ["current", "current"])
        page_before = self.disk.read(self.PAGE)
        log_before = self.disk.read("50_MOC/log.md")
        result = self.page("refresh", self.PAGE, "--write")
        self.assertIn("state    .vaultengine/pages.json", result.stdout)
        self.assertEqual((self.disk.read(self.PAGE), self.disk.read("50_MOC/log.md")), (page_before, log_before))
        self.assertEqual(self.check()["changed"], [])

    def test_bytes_outside_the_markers_stay_the_same(self):
        # The page as the user left it: CRLF endings, trailing spaces, their own text above, between and below the
        # blocks, a tab, no final newline.
        original = self.disk.read(self.PAGE).decode()
        mine = original.replace("# Grading Without Transcripts\n", "# Grading Without Transcripts\n\nMy intro.  \n\t- a tab\n")
        mine = mine.replace("<!-- /vault:zettels -->\n\n", "<!-- /vault:zettels -->\n\nBetween the blocks, mine.\n\n")
        mine = (mine.rstrip("\n") + "\n\n## My part\n\nWhat I think, with a link to [[z2]]. No final newline").replace("\n", "\r\n")
        (self.disk.root / self.PAGE).write_bytes(mine.encode())
        self.disk.write("20_ZETTLEKASTEN/z1.md", zettel("[[rl]] [[grading]]", "Changed first line. From [[S1]]."))
        self.disk.write("20_ZETTLEKASTEN/z4.md", zettel("[[rl]]", "A new one. From [[S3]]."))
        (self.disk.root / self.PAGE).write_bytes(mine.replace("My intro.", "My intro, see [[z4]].").encode())
        before = self.disk.read(self.PAGE)
        self.assertEqual(self.page("refresh", self.PAGE, "--write").returncode, 0)
        after = self.disk.read(self.PAGE)
        self.assertNotEqual(before, after)

        def outside(data: bytes) -> list[bytes]:
            text = data.decode()
            parts = []
            rest = text
            for name in ("zettels", "sources"):
                head, _, tail = rest.partition(f"<!-- vault:{name} -->")
                _, _, rest = tail.partition(f"<!-- /vault:{name} -->")
                parts.append(head.encode())
            parts.append(rest.encode())
            return parts

        self.assertEqual(outside(after), outside(before))
        text = after.decode()
        self.assertNotIn("\n", text.replace("\r\n", ""))  # the new lines are CRLF too
        self.assertIn("- [[z4]]: A new one. From S3.", text)  # linked in my text: now on the page
        self.assertIn("- [[z1]]: Changed first line. From S1.", text)
        self.assertIn("- [[S3]] (z4)", text)

    def test_dropping_a_line_drops_the_zettel(self):
        text = self.disk.read(self.PAGE).decode()
        self.disk.write(self.PAGE, text.replace("- [[z3]]: Partial credit must be gated on correctness.\n", ""))
        self.disk.write(self.PAGE, self.disk.read(self.PAGE).decode().replace("### grading\n", ""))
        self.assertEqual(self.page("refresh", self.PAGE, "--write").returncode, 0)  # 2 zettels: already a violation
        after = self.disk.read(self.PAGE).decode()
        self.assertNotIn("[[z3]]", after)
        self.assertIn("- [[S2]] (z2)", after)
        self.assertIn("(2 zettels, 2 sources)", self.disk.read("50_MOC/index.md").decode())

    def test_broken_markers(self):
        text = self.disk.read(self.PAGE).decode()
        self.disk.write(self.PAGE, text.replace("<!-- /vault:sources -->", ""))
        result = self.page("refresh", self.PAGE, "--write")
        self.assertEqual(result.returncode, 2)
        self.assertIn("the sources block's markers are not one", result.stderr)
        self.assertIn("fix them by hand", self.page("check", self.PAGE).stdout)


# ----- check: behind, ready ----------------------------------------------------------------------------


class BehindTests(DiskTest):
    PAGE = "50_MOC/grading-without-transcripts.md"

    def test_exactly_the_zettels_that_share_tags_or_links(self):
        self.assertEqual(self.new_page("--write").returncode, 0)
        rows = self.disk.pages().behind(self.PAGE)
        self.assertEqual({row["path"]: {"tags": sorted(row["tags"]), "links": sorted(row["links"])} for row in rows},
                         {p: {"tags": sorted(v["tags"]), "links": sorted(v["links"])} for p, v in BEHIND.items()})
        self.assertEqual(rows[0]["path"], "20_ZETTLEKASTEN/b-both.md")  # most in common first
        text = self.page("check", self.PAGE).stdout
        self.assertIn("✗ behind: 4 zettels with at least two of its tags or links in common, not on it", text)

    def test_a_clean_page_is_ready_to_write(self):
        # Exclude two behind zettels, put the others on the page; what they bring (S3) can make the
        # page behind on more, so repeat until nothing is left. Then it is ready.
        self.assertEqual(self.new_page("--write").returncode, 0)
        excluded = self.page("exclude", self.PAGE, "b-two-links", "b-two-tags", "--write")
        self.assertEqual(excluded.returncode, 0, excluded.stdout + excluded.stderr)
        for _ in range(5):
            rows = json.loads(self.page("check", self.PAGE, "--json").stdout)["pages"][0]["behind"]
            if not rows:
                break
            self.assertFalse({"20_ZETTLEKASTEN/b-two-links.md", "20_ZETTLEKASTEN/b-two-tags.md"} & {r["path"] for r in rows})
            links = " ".join(f"[[{Path(row['path']).stem}]]" for row in rows)
            self.disk.write(self.PAGE, self.disk.read(self.PAGE).decode() + f"Also {links}.\n")
            self.assertEqual(self.page("refresh", self.PAGE, "--write").returncode, 0)
        check = self.page("check", self.PAGE)
        self.assertEqual(check.returncode, 0, check.stdout)
        self.assertIn("  ✓ ready to write", check.stdout)
        self.assertIn("  = 2 zettels excluded from it", check.stdout)
        self.assertIn("✓ 1 of 1 managed page ready to write", check.stdout)

    def test_all_pages_and_unmanaged_ones(self):
        self.assertEqual(self.new_page("--write").returncode, 0)
        result = self.page("check")
        self.assertEqual(result.returncode, 1)
        self.assertIn("1 page without vault blocks (not made by `vault page new`): 50_MOC/other page.md", result.stdout)
        self.assertIn("✗ 0 of 1 managed page ready to write", result.stdout)

    def test_no_record_and_index_line(self):
        self.assertEqual(self.new_page("--write").returncode, 0)
        (self.disk.root / ".vaultengine/pages.json").unlink()
        index = self.disk.read("50_MOC/index.md").decode()
        self.disk.write("50_MOC/index.md", index.replace("Grading Without Transcripts (3 zettels", "Old title (3 zettels"))
        data = json.loads(self.page("check", self.PAGE, "--json").stdout)["pages"][0]
        self.assertEqual((data["changed"], data["index"]), (None, "out-of-date"))
        text = self.page("check", self.PAGE).stdout
        self.assertIn("no record of its zettels", text)
        self.assertIn("index line out of date", text)
        self.page("refresh", self.PAGE, "--write")
        data = json.loads(self.page("check", self.PAGE, "--json").stdout)["pages"][0]
        self.assertEqual((data["changed"], data["index"]), ([], "current"))


class ExcludeTests(DiskTest):
    PAGE = "50_MOC/grading-without-transcripts.md"

    def setUp(self):
        super().setUp()
        self.assertEqual(self.new_page("--write").returncode, 0)

    def behind(self):
        return [row["path"] for row in json.loads(self.page("check", self.PAGE, "--json").stdout)["pages"][0]["behind"]]

    def test_exclude_and_undo(self):
        self.assertIn("20_ZETTLEKASTEN/b-two-tags.md", self.behind())
        before = self.disk.snapshot()
        shown = self.page("exclude", self.PAGE, "b-two-tags")
        self.assertEqual(shown.returncode, 0)
        self.assertIn("Would record in the engine state (page exclude); no note changes:", shown.stdout)
        self.assertIn("1 excluded", shown.stdout)
        self.assertIn("Not written: add --write", shown.stdout)
        self.assertEqual(self.disk.snapshot(), before)  # nothing without --write
        page_bytes = self.disk.read(self.PAGE)
        self.assertEqual(self.page("exclude", self.PAGE, "b-two-tags", "--write").returncode, 0)
        self.assertEqual(self.disk.read(self.PAGE), page_bytes)  # kept in the engine, not in the note
        state = json.loads(self.disk.read(".vaultengine/pages.json"))
        self.assertEqual(state["pages"][self.PAGE]["excluded"], ["20_ZETTLEKASTEN/b-two-tags.md"])
        self.assertNotIn("20_ZETTLEKASTEN/b-two-tags.md", self.behind())
        again = self.page("exclude", self.PAGE, "b-two-tags", "--write")
        self.assertIn("already excluded", again.stdout)
        self.assertEqual(self.page("exclude", self.PAGE, "b-two-tags", "--undo", "--write").returncode, 0)
        self.assertIn("20_ZETTLEKASTEN/b-two-tags.md", self.behind())

    def test_refused(self):
        cases = {
            ("z1",): "is on the page; to take it off, delete its line",
            ("S1",): "10_SOURCES/S1.md is a source, not a zettel",
            ("nope",): "no note 'nope'",
        }
        for zettels, error in cases.items():
            with self.subTest(zettels=zettels):
                result = self.page("exclude", self.PAGE, *zettels, "--write")
                self.assertEqual(result.returncode, 2)
                self.assertIn(error, result.stderr)
        (self.disk.root / ".vaultengine/pages.json").unlink()
        result = self.page("exclude", self.PAGE, "b-two-tags", "--write")
        self.assertEqual(result.returncode, 2)
        self.assertIn("has no record yet", result.stderr)


class RenamedPageTests(DiskTest):
    PAGE = "50_MOC/grading-without-transcripts.md"

    def setUp(self):
        super().setUp()
        self.assertEqual(self.new_page("--write").returncode, 0)
        self.assertEqual(self.page("exclude", self.PAGE, "b-two-tags", "--write").returncode, 0)

    def check(self, page):
        return json.loads(self.page("check", page, "--json").stdout)["pages"][0]

    def test_the_record_follows_a_renamed_page(self):
        new = "50_MOC/topics/grading-end-states.md"
        (self.disk.root / "50_MOC/topics").mkdir()
        (self.disk.root / self.PAGE).rename(self.disk.root / new)
        data = self.check(new)
        self.assertEqual(data["renamed_from"], self.PAGE)
        self.assertEqual(data["changed"], [])  # the zettels are compared with the old record, not "no record"
        self.assertEqual(data["excluded"], ["20_ZETTLEKASTEN/b-two-tags.md"])
        self.assertIn("= renamed from 50_MOC/grading-without-transcripts.md: its record followed it",
                      self.page("check", new).stdout)
        # A refresh writes the record under the new path.
        result = self.page("refresh", new, "--write")
        self.assertIn("record moved from 50_MOC/grading-without-transcripts.md (renamed)", result.stdout)
        state = json.loads(self.disk.read(".vaultengine/pages.json"))["pages"]
        self.assertEqual(sorted(state), [new])
        self.assertEqual(state[new]["excluded"], ["20_ZETTLEKASTEN/b-two-tags.md"])
        self.assertIsNone(self.check(new)["renamed_from"])

    def test_renamed_and_edited_follows_a_different_page_does_not(self):
        text = self.disk.read(self.PAGE).decode()
        (self.disk.root / self.PAGE).unlink()
        self.disk.write("50_MOC/renamed.md", text + "\nMy first paragraph about grading end states.\n")
        self.assertEqual(self.check("50_MOC/renamed.md")["renamed_from"], self.PAGE)
        # A page with blocks and no record, but other text, is not taken for the renamed one.
        (self.disk.root / "50_MOC/renamed.md").unlink()
        self.disk.write("50_MOC/hand-made.md", "Created: 01-01-2026 10:00\nTags: [[other]]\n\n# Hand made\n\n"
                        "<!-- vault:zettels -->\n- [[b-link]] [[b-source]] [[o-unrelated]]\n<!-- /vault:zettels -->\n\n"
                        "<!-- vault:sources -->\n<!-- /vault:sources -->\n\nA different page about rubric drift, "
                        "off-policy data and nothing that the first page said at all, written by hand from scratch.\n")
        data = self.check("50_MOC/hand-made.md")
        self.assertEqual((data["managed"], data["renamed_from"], data["changed"]), (True, None, None))


# ----- pieces ------------------------------------------------------------------------------------------


class PieceTests(unittest.TestCase):
    def test_excerpt(self):
        cases = {
            "# T\n\n## What This Is\n\nThe first line.\nSecond.": "The first line.",
            "```\ncode first\n```\n> A quote with [[a|alias]] and [[b#h]] and ![[img.png]].": "A quote with alias and b and .",
            "from:: [[x]]\n---\n| a | b |\n- [text](https://x.y) item": "text item",
            "<!-- c -->\nIt says --> and <!-- inside.": "It says - -> and <!- - inside.",
            "": "",
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(excerpt(text, 1), expected)
        long = "word " * 50
        self.assertEqual(excerpt(long, 1), ("word " * 32).strip() + " …")  # the cut falls on a space: 32 whole words
        self.assertTrue(excerpt("x" * 150 + " `code that is cut " + "y" * 20, 1).endswith("x" * 150 + " …"))

    def test_find_and_replace_blocks(self):
        text = "a\n<!-- vault:zettels -->\nold\n<!-- /vault:zettels -->\nb"
        self.assertEqual(replace_block(text, "zettels", ["new", "lines"]),
                         "a\n<!-- vault:zettels -->\nnew\nlines\n<!-- /vault:zettels -->\nb")
        self.assertEqual(replace_block("x", "pages", ["l"], append_if_missing=True),
                         "x\n\n<!-- vault:pages -->\nl\n<!-- /vault:pages -->\n")
        for bad in ("<!-- /vault:zettels -->\n<!-- vault:zettels -->",
                    "<!-- vault:zettels -->\n<!-- vault:zettels -->\n<!-- /vault:zettels -->",
                    "<!-- vault:zettels -->\n<!-- vault:sources -->\n<!-- /vault:zettels -->\n<!-- /vault:sources -->"):
            with self.subTest(bad=bad):
                with self.assertRaises(PageError):
                    find_blocks(bad.splitlines(), ("zettels", "sources"))

    def test_grouping_by_the_most_shared_tag(self):
        spec = load_format(CORE)
        files = {
            "10_SOURCES/S.md": source(),
            "20_ZETTLEKASTEN/a.md": zettel("[[x]] [[y]]", "A [[S]]"),
            "20_ZETTLEKASTEN/b.md": zettel("[[y]]", "B [[S]]"),
            "20_ZETTLEKASTEN/c.md": zettel("[[x]]", "C [[S]]"),
            "20_ZETTLEKASTEN/d.md": zettel("[[z]] #y", "D [[S]]"),
            "20_ZETTLEKASTEN/e.md": "Created: 01-01-2026 10:00\n\nE [[S]]\n",
        }
        pages = Pages(TypedVault(spec, Vault.from_files(files)))
        lines = pages.render_zettels([f"20_ZETTLEKASTEN/{n}.md" for n in "abcde"])
        # y is on a, b, d (3); x on a, c (2): a goes under y; c under x; d under y (its own z is on 1 only).
        self.assertEqual(lines, ["## Zettels", "", "### y", "- [[a]]", "- [[b]]", "- [[d]]", "", "### x", "- [[c]]",
                                 "", "### (no tags)", "- [[e]]"])
        self.assertEqual(pages.default_tags([f"20_ZETTLEKASTEN/{n}.md" for n in "abcd"]), "[[y]] [[x]]")

    def test_format_validation(self):
        base = textwrap.dedent("""\
            format: 1
            pipeline: {sources: [s], extract: [z], compile: [p]}
            types:
              i: {match: {folder: P, name: index}}
              s: {match: {folder: S}}
              z: {match: {folder: Z}, metadata: {Tags: {kind: [links, tags]}}}
              p: {match: {folder: P}, metadata: {Tags: {kind: [links, tags]}}}
            """)
        ok = parse_format(base + "pages: {index: P/index.md, log: log.md, tags: Tags}\n")
        self.assertEqual(ok.pages.index, "P/index.md")
        cases = {
            "pages: {index: P/other.md, log: log.md, tags: Tags}": "pages.index: P/other.md would be a p; give it a type of its own",
            "pages: {index: P/index.md, log: log.txt, tags: Tags}": "pages.log: expected a note path ending in .md",
            "pages: {index: P/index.md, log: P/index.md, tags: Tags}": "pages: index and log must be different notes",
            "pages: {index: P/index.md, log: log.md, tags: Topics}": "pages.tags: 'Topics' is not a tags or links metadata key of z",
        }
        for line, error in cases.items():
            with self.subTest(line=line):
                with self.assertRaises(FormatError) as raised:
                    parse_format(base + line + "\n")
                self.assertIn(error, str(raised.exception))
        with self.assertRaises(FormatError) as raised:
            parse_format("format: 1\npages: {index: i.md, log: l.md, tags: T}\ntypes: {t: {match: {folder: A}}}\n")
        self.assertIn("pages: needs a `pipeline:` block", str(raised.exception))


class BenchVaultTests(unittest.TestCase):
    """scripts/bench_pages.py makes the vaults the timings in docs/commands.md come from: they must follow core."""

    def test_generated_vault_is_clean_and_its_pages_are_current(self):
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        root = Path(scratch.name) / "bench"
        made = subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "bench_pages.py"), "generate", "200", str(root)],
                              capture_output=True, text=True)
        self.assertEqual(made.returncode, 0, made.stderr)
        self.assertIn("40 sources, 149 zettels, 10 pages, index", made.stdout)
        check = run_vault("check", str(root), "-f", "core")
        self.assertEqual(check.returncode, 0, check.stdout[-2000:])
        data = json.loads(run_vault("page", "check", str(root), "-f", "core", "--json").stdout)
        self.assertEqual(len(data["pages"]), 10)
        for page in data["pages"]:
            self.assertEqual(([b["status"] for b in page["blocks"]], page["changed"], page["index"]),
                             (["current", "current"], [], "current"), page["path"])


if __name__ == "__main__":
    unittest.main()
