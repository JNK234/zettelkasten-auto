"""formats/core.yaml and the engine rules it needs: one home per kind (``subfolders: false``),
untyped notes in governed folders as violations (``typed_folders``), folders outside the format
(``ignored_folders``), minimum links to notes of another kind (``min_links``), ``vault new``
filling Created, and ``vault check --baseline`` / ``--update-baseline`` (renames included).
The pipeline and ``vault note`` are in test_vaultengine_pipeline.py.

Each rule is proved on a small synthetic vault built in memory or in a temp folder;
the sample vault is used only to pin what core reports on it.
"""

import json
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from collections import Counter
from pathlib import Path

from vaultengine import Vault
from vaultengine.formats import FormatError, TypedVault, load_format, parse_format
from vaultengine.formats.check import baseline_path, compare_to_baseline, load_baseline, record_baseline
from vaultengine.formats.spec import Match
from vaultengine.formats.typed import violation_keys
from vaultengine.formats.write import FileChange, judge, plan_new

TESTS = Path(__file__).parent
REPO_ROOT = TESTS.parent
VAULT = TESTS / "fixtures" / "vault"
CORE = REPO_ROOT / "formats" / "core.yaml"


def run_vault(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "vaultengine", *args], capture_output=True, text=True, cwd=REPO_ROOT)


def core() -> TypedVault:
    return load_format(CORE)


def typed(files: dict[str, str], spec=None) -> TypedVault:
    spec = spec if spec is not None else core()
    return TypedVault(spec, Vault.from_files({p: textwrap.dedent(t) for p, t in files.items()}))


def issues_of(vault: TypedVault, path: str | None = None) -> list[tuple[str, str]]:
    return [(i.path, i.code) for i in vault.issues(path)]


# Notes that follow core.yaml. Bodies link as given.
def source(body: str = "", tags: str = "[[ml]]", extra: str = "") -> str:
    return f"Created: 01-01-2026 10:00\nTags: {tags}\nSource: https://arxiv.org/abs/2401.00001\n{extra}\n# A source\n{body}\n"


def zettel(body: str = "", tags: str = "[[ml]]") -> str:
    return f"Created: 01-01-2026 10:00\nTags: {tags}\n\n# A zettel\n{body}\n"


def page(body: str = "", tags: str = "[[ml]]") -> str:
    return f"Created: 01-01-2026 10:00\nTags: {tags}\n\n# A page\n{body}\n"


FINDING = "<!-- vault:finding -->\n- id: link-0123456789\n<!-- /vault:finding -->\n"  # an idea requires its finding block


def idea(status: str = "#open", body: str = "") -> str:
    return f"Created: 01-01-2026 10:00\nTags: [[ml]]\nStatus: {status}\n\n# An idea\n{FINDING}{body}\n"


CLEAN = {
    "10_SOURCES/S1.md": source(),
    "10_SOURCES/S2.md": source(),
    "20_ZETTLEKASTEN/z1.md": zettel("From [[S1]]."),
    "20_ZETTLEKASTEN/z2.md": zettel("From [[S2]], see [[z1]]."),
    "20_ZETTLEKASTEN/z3.md": zettel("From [[S1]] and [[S2]]."),
    "50_MOC/overview.md": page("[[z1]], [[z2]] and [[z3]]."),
    "50_MOC/topics/deep/nested page.md": page("[[z1]] [[z2]] [[z3]]"),
    "50_MOC/ideas/try it.md": idea(),
    "Inbox/loose note.md": "anything\n",
}


# ----- the format file itself -----------------------------------------------------------------------


class CoreFormatTests(unittest.TestCase):
    def test_loads_with_four_types_and_the_governed_folders(self):
        spec = core()
        self.assertEqual(spec.name, "core")
        self.assertEqual([t.name for t in spec.types], ["index", "log", "source", "zettel", "idea", "page"])  # index, log: kept by the engine
        self.assertEqual(spec.typed_folders, ("10_SOURCES", "20_ZETTLEKASTEN", "50_MOC"))
        self.assertEqual(spec.ignored_folders, ("50_MOC/discourse",))
        self.assertEqual((spec.pipeline.sources, spec.pipeline.extract, spec.pipeline.compile), (("source",), ("zettel",), ("page",)))
        self.assertEqual(spec.relations, {})  # the discourse relations are not declared

    def test_only_the_decided_fields(self):
        # Created and Tags everywhere, Source on sources, Status on ideas; nothing computable.
        fields = {t.name: sorted(k.name for k in t.metadata) for t in core().types}
        self.assertEqual(fields, {
            "index": [],
            "log": [],
            "source": ["Created", "Source", "Tags"],
            "zettel": ["Created", "Tags"],
            "idea": ["Created", "Status", "Tags"],
            "page": ["Created", "Tags"],
        })
        self.assertTrue(all(k.required for t in core().types for k in t.metadata))
        self.assertTrue(all(not t.headings and not t.relations_required for t in core().types))

    def test_link_minimums(self):
        minimums = {t.name: [(m.type, m.count) for m in t.min_links] for t in core().types}
        self.assertEqual(minimums, {"index": [], "log": [], "source": [], "zettel": [("source", 1)], "idea": [], "page": [("zettel", 3)]})

    def test_one_home_per_kind(self):
        spec = core()
        cases = {
            "10_SOURCES/a.md": "source",
            "10_sources/A.md": "source",  # folders compare case-insensitively
            "10_SOURCES/old/a.md": None,
            "20_ZETTLEKASTEN/a.md": "zettel",
            "20_ZETTLEKASTEN/drafts/a.md": None,
            "50_MOC/ideas/a.md": "idea",
            "50_MOC/ideas/later/a.md": "page",  # not directly in ideas: the rest of 50_MOC is pages
            "50_MOC/a.md": "page",
            "50_MOC/index.md": "index",
            "50_MOC/log.md": "log",
            "50_MOC/topics/index.md": "page",  # only the one directly in 50_MOC
            "50_MOC/topics/x/y/a.md": "page",
            "50_MOC/discourse/a.md": None,  # ignored
            "50_MOC/discourse/rl/EVD - x.md": None,
            "50_MOC/discourse-notes/a.md": "page",  # only that folder, not a prefix of the name
            "Inbox/a.md": None,
            "a.md": None,
        }
        for path, expected in cases.items():
            with self.subTest(path=path):
                found = spec.type_for(path)
                self.assertEqual(found.name if found else None, expected)

    def test_a_vault_that_follows_it_is_clean(self):
        vault = typed(CLEAN)
        self.assertEqual(vault.issues(), [])
        self.assertEqual(vault.untyped, ["Inbox/loose note.md"])  # outside the governed folders: allowed

    def test_each_field_is_enforced(self):
        cases = {
            "10_SOURCES/S1.md": ("Created: 01-01-2026 10:00\nTags: [[ml]]\n\n# S\n", "missing-key", "missing metadata 'Source'"),
            "50_MOC/ideas/try it.md": ("Created: 01-01-2026 10:00\nTags: [[ml]]\n\n# I\n" + FINDING, "missing-key", "missing metadata 'Status'"),
            "20_ZETTLEKASTEN/z1.md": ("Created: 2026-01-01\nTags: [[ml]]\n\nFrom [[S1]].\n", "bad-value",
                                      "Created: '2026-01-01' is not a datetime like '%d-%m-%Y %H:%M'"),
            "50_MOC/overview.md": ("Created: 01-01-2026 10:00\nTags:\n\n[[z1]] [[z2]] [[z3]]\n", "missing-value", "'Tags' is empty"),
        }
        for path, (text, code, message) in cases.items():
            with self.subTest(path=path):
                vault = typed({**CLEAN, path: text})
                self.assertEqual([(i.path, i.code, i.message) for i in vault.issues()], [(path, code, message)])

    def test_idea_status_is_one_of_open_dropped_done(self):
        for status in ("#open", "#dropped", "#done", "#Done"):
            with self.subTest(status=status):
                self.assertEqual(typed({**CLEAN, "50_MOC/ideas/try it.md": idea(status)}).issues(), [])
        for status, message in (("#parked", "Status: #parked is not an allowed value (open, dropped, done)"),
                                ("#open #done", "Status: expected one value, got 2"),
                                ("open", "Status: 'open' is not a #tag")):
            with self.subTest(status=status):
                found = typed({**CLEAN, "50_MOC/ideas/try it.md": idea(status)}).issues()
                self.assertEqual([(i.code, i.message) for i in found], [("bad-value", message)])

    def test_a_url_is_not_a_source(self):
        # Every source is a note in 10_SOURCES: a URL, or a link to a source note that is not there, is not one.
        for body in ("[Paper](https://arxiv.org/abs/2401.00001)", "https://arxiv.org/abs/2401.00001",
                     "# References\n1. [[2401.00001 - Not Ingested Yet]]\n2. <https://example.com/post>"):
            with self.subTest(body=body):
                found = typed({**CLEAN, "20_ZETTLEKASTEN/z1.md": zettel(body)}).issues()
                self.assertEqual([i.code for i in found], ["too-few-links"])

    def test_tags_as_links_or_hashtags(self):
        vault = typed({**CLEAN, "20_ZETTLEKASTEN/z1.md": zettel("From [[S1]].", tags="#ml #rl")})
        self.assertEqual(vault.issues(), [])


# ----- engine: subfolders: false ------------------------------------------------------------------


class IgnoredFoldersTests(unittest.TestCase):
    DISCOURSE = {
        "50_MOC/discourse/rl/EVD - a.md": "Created: x\n\nsupports:: [[CLM - b]]\nfrom:: [[S1]]\n[[z1]] [[z2]] [[z3]]\n",
        "50_MOC/discourse/rl/CLM - b.md": "no metadata at all\nsupported-by:: [[EVD - a]]\n",
        "50_MOC/discourse/sub/deeper/x.md": "x\n",
    }

    def test_no_type_no_violations_not_even_fields(self):
        vault = typed({**CLEAN, **self.DISCOURSE})
        self.assertEqual(vault.issues(), [])
        self.assertEqual(vault.ignored, sorted(self.DISCOURSE))
        self.assertEqual(vault.untyped, ["Inbox/loose note.md"])
        self.assertTrue(all(vault.type_of(p) is None for p in self.DISCOURSE))

    def test_links_from_ignored_notes_count_for_nothing(self):
        # The only note linking z9's source is ignored; the only "page" linking z9 is ignored.
        vault = typed({**CLEAN, **self.DISCOURSE, "10_SOURCES/S9.md": source(), "20_ZETTLEKASTEN/z9.md": zettel("[[S9]]"),
                       "50_MOC/discourse/p.md": "[[z9]] [[S9]]\n"})
        from vaultengine.formats.coverage import coverage
        self.assertNotIn("10_SOURCES/S9.md", coverage(vault)["sources"]["unlinked"])  # z9 links it
        vault = typed({**CLEAN, **self.DISCOURSE, "10_SOURCES/S9.md": source(), "50_MOC/discourse/p.md": "[[S9]]\n"})
        self.assertIn("10_SOURCES/S9.md", coverage(vault)["sources"]["unlinked"])  # only an ignored note links it

    def test_vault_new_refuses_an_ignored_folder(self):
        from vaultengine.formats.write import WriteError

        with self.assertRaises(WriteError) as raised:
            plan_new(typed(CLEAN), "page", "X", folder="50_MOC/discourse")
        self.assertIn("would be a note of no type, not a page", str(raised.exception))

    def test_validation_and_a_format_without_it(self):
        with self.assertRaises(FormatError) as raised:
            parse_format("format: 1\nignored_folders: [A, A/]\ntypes: {t: {match: {folder: A}}}\n")
        self.assertIn("ignored_folders[1]: 'A' is listed twice", str(raised.exception))
        spec = parse_format("format: 1\ntypes: {t: {match: {folder: A}}}\n")
        self.assertFalse(spec.is_ignored("A/x.md"))
        self.assertEqual(spec.type_for("A/x.md").name, "t")


class SubfoldersTests(unittest.TestCase):
    def test_match(self):
        flat = Match("S", None, subfolders=False)
        self.assertTrue(flat.matches("S/a.md"))
        self.assertTrue(flat.matches("s/A.md"))
        self.assertFalse(flat.matches("S/sub/a.md"))
        self.assertFalse(flat.matches("SX/a.md"))
        self.assertFalse(flat.matches("a.md"))
        deep = Match("S", None)  # the default, as before: any depth
        self.assertTrue(deep.matches("S/sub/a.md"))
        self.assertEqual(flat.describe(), "directly in S/")
        self.assertEqual(deep.describe(), "in S/")

    def test_nested_folder_and_name(self):
        match = Match("A/B", "x *", subfolders=False)
        self.assertTrue(match.matches("A/B/x 1.md"))
        self.assertFalse(match.matches("A/B/C/x 1.md"))
        self.assertFalse(match.matches("A/B/y.md"))

    def test_validation(self):
        bad = {
            "types: {t: {match: {folder: S, subfolders: no}}}": None,  # YAML `no` is false: valid
            "types: {t: {match: {folder: S, subfolders: 'no'}}}": "types.t.match.subfolders: expected true or false, got a string",
            "types: {t: {match: {name: 'x*', subfolders: false}}}":
                "types.t.match.subfolders: only a match with a folder can leave out its subfolders",
            "types: {t: {match: {subfolders: true}}}": "types.t.match: give a folder, a name pattern, or both",
        }
        for body, error in bad.items():
            with self.subTest(body=body):
                if error is None:
                    parse_format("format: 1\n" + body)
                    continue
                with self.assertRaises(FormatError) as raised:
                    parse_format("format: 1\n" + body)
                self.assertIn(error, str(raised.exception))


# ----- engine: typed_folders -> untyped-note ----------------------------------------------------------


class UntypedNoteTests(unittest.TestCase):
    def test_untyped_inside_a_governed_folder_is_a_violation(self):
        vault = typed({**CLEAN, "10_SOURCES/old/S9.md": source(), "20_ZETTLEKASTEN/drafts/z9.md": zettel("[[S1]]")})
        found = [(i.path, i.code, i.line) for i in vault.issues()]
        self.assertEqual(found, [("10_SOURCES/old/S9.md", "untyped-note", 1), ("20_ZETTLEKASTEN/drafts/z9.md", "untyped-note", 1)])
        message = vault.issues("10_SOURCES/old/S9.md")[0].message
        self.assertEqual(message, "no type matches this note, and every note in 10_SOURCES/ needs one; "
                                  "types in 10_SOURCES/: source (directly in 10_SOURCES/)")

    def test_outside_the_governed_folders_is_only_listed(self):
        vault = typed({**CLEAN, "Inbox/x.md": "x\n", "10_SOURCES_old/x.md": "x\n", "templates/t.md": "x\n"})
        self.assertEqual(vault.issues(), [])
        self.assertEqual(vault.untyped, ["10_SOURCES_old/x.md", "Inbox/loose note.md", "Inbox/x.md", "templates/t.md"])

    def test_untyped_notes_still_count_nowhere_else(self):
        # An untyped note is not a source: a zettel citing only it still lacks a source.
        vault = typed({**CLEAN, "10_SOURCES/old/S9.md": source(), "20_ZETTLEKASTEN/z1.md": zettel("From [[S9]].")})
        self.assertEqual(sorted(issues_of(vault)), [("10_SOURCES/old/S9.md", "untyped-note"), ("20_ZETTLEKASTEN/z1.md", "too-few-links")])

    def test_without_typed_folders_nothing_changes(self):
        spec = parse_format("format: 1\ntypes: {source: {match: {folder: S, subfolders: false}}}\n")
        vault = typed({"S/a.md": "x\n", "S/sub/b.md": "x\n"}, spec)
        self.assertEqual(vault.issues(), [])
        self.assertEqual(vault.untyped, ["S/sub/b.md"])

    def test_validation(self):
        cases = {
            "typed_folders: [A, a/]": "typed_folders[1]: 'a' is listed twice",
            "typed_folders: ['/']": "typed_folders[0]: expected a folder, got '/'",
            "typed_folders: {A: 1}": "typed_folders: expected a list, got a mapping",
        }
        for line, error in cases.items():
            with self.subTest(line=line):
                with self.assertRaises(FormatError) as raised:
                    parse_format(f"format: 1\n{line}\ntypes: {{t: {{match: {{folder: A}}}}}}\n")
                self.assertIn(error, str(raised.exception))
        self.assertEqual(parse_format("format: 1\ntyped_folders: A\ntypes: {t: {match: {folder: A}}}\n").typed_folders, ("A",))


# ----- engine: min_links -> too-few-links ---------------------------------------------------------


class MinLinksTests(unittest.TestCase):
    def zettel_issues(self, body: str, extra: dict[str, str] | None = None, tags: str = "[[ml]]"):
        vault = typed({**CLEAN, **(extra or {}), "20_ZETTLEKASTEN/z1.md": zettel(body, tags=tags)})
        return [(i.code, i.message, i.line, i.detail["found"]) for i in vault.issues("20_ZETTLEKASTEN/z1.md")]

    def test_zettel_needs_a_source(self):
        self.assertEqual(self.zettel_issues("No citation."),
                         [("too-few-links", "links to 0 source notes; a zettel needs at least 1", 1, 0)])
        for body in ("From [[S1]].", "![[S1]]", "[[S1#Section|see]]", "- [[10_SOURCES/S1]]", "[text](<../10_SOURCES/S1.md>)"):
            with self.subTest(body=body):
                self.assertEqual(self.zettel_issues(body), [])

    def test_what_does_not_count(self):
        cases = {
            "only other kinds": "See [[z2]] and [[overview]].",
            "a link that matches no note": "From [[S404]].",
            "only in Tags": "",  # tags= below links the source
            "inside code": "`[[S1]]` and\n```\n[[S2]]\n```",
        }
        for label, body in cases.items():
            with self.subTest(label):
                tags = "[[S1]]" if label == "only in Tags" else "[[ml]]"
                found = self.zettel_issues(body, tags=tags)
                self.assertEqual([code for code, *_ in found], ["too-few-links"])
        message = self.zettel_issues("From [[S404]] and [[S405]] and [[S406]].")[0][1]
        self.assertEqual(message, "links to 0 source notes; a zettel needs at least 1 "
                                  "(3 links match no note: [[S404]], [[S405]] +1)")

    def test_ambiguous_link_does_not_count(self):
        # [[S1]] matches the source and a note elsewhere: Obsidian cannot tell which, so neither counts.
        found = self.zettel_issues("From [[S1]], [[S404]].", extra={"Inbox/S1.md": "x\n"})
        self.assertEqual(found, [("too-few-links", "links to 0 source notes; a zettel needs at least 1 "
                                  "(1 link matches no note: [[S404]]; 1 link matches several notes: [[S1]])", 1, 0)])
        self.assertEqual(self.zettel_issues("From [[10_SOURCES/S1]].", extra={"Inbox/S1.md": "x\n"}), [])  # a path is exact

    def test_page_needs_distinct_zettels(self):
        def page_issues(body):
            vault = typed({**CLEAN, "50_MOC/overview.md": page(body)})
            return [(i.message, i.detail["found"]) for i in vault.issues("50_MOC/overview.md")]

        self.assertEqual(page_issues("[[z1]] [[z2]] [[z3]]"), [])
        self.assertEqual(page_issues("[[z1]] [[z1|again]] ![[z1]] [[z2]]"), [("links to 2 zettel notes; a page needs at least 3", 2)])
        self.assertEqual(page_issues("[[z1]] [[S1]] [[S2]] [[try it]]"), [("links to 1 zettel note; a page needs at least 3", 1)])

    def test_relation_fields_count_metadata_does_not(self):
        spec = parse_format(textwrap.dedent("""
            format: 1
            relations: {from: {targets: [source]}}
            types:
              source: {match: {folder: S}}
              evidence: {match: {folder: E}, min_links: {source: 1}, metadata: {Tags: {kind: links}}}
        """))
        vault = typed({"S/paper.md": "x\n", "E/a.md": "from:: [[paper]]\n", "E/b.md": "Tags: [[paper]]\n\nbody\n"}, spec)
        self.assertEqual(issues_of(vault), [("E/b.md", "too-few-links")])

    def test_several_minimums_and_self_links(self):
        spec = parse_format(textwrap.dedent("""
            format: 1
            types:
              source: {match: {folder: S}}
              note: {match: {folder: N}, min_links: {source: 1, note: 2}}
        """))
        vault = typed({"S/s.md": "x\n", "N/a.md": "[[a]] [[b]]\n", "N/b.md": "[[s]] [[a]] [[c]]\n", "N/c.md": "x\n"}, spec)
        found = [(i.path, i.detail["type"], i.detail["found"]) for i in vault.issues()]
        # a: its self-link does not count; c links nothing
        self.assertEqual(found, [("N/a.md", "source", 0), ("N/a.md", "note", 1), ("N/c.md", "source", 0), ("N/c.md", "note", 0)])

    def test_validation(self):
        cases = {
            "{t: {match: {folder: A}, min_links: {nope: 1}}}": "types.t.min_links: unknown type 'nope'",
            "{t: {match: {folder: A}, min_links: {t: 0}}}": "types.t.min_links.t: expected a whole number of at least 1, got 0",
            "{t: {match: {folder: A}, min_links: {t: true}}}": "types.t.min_links.t: expected a whole number of at least 1, got True",
            "{t: {match: {folder: A}, min_links: {t: '2'}}}": "types.t.min_links.t: expected a whole number of at least 1, got '2'",
            "{t: {match: {folder: A}, min_links: [t]}}": "types.t.min_links: expected a mapping, got a list",
        }
        for types, error in cases.items():
            with self.subTest(types=types):
                with self.assertRaises(FormatError) as raised:
                    parse_format(f"format: 1\ntypes: {types}\n")
                self.assertIn(error, str(raised.exception))

    def test_key_does_not_change_as_the_note_improves(self):
        # A page going from 1 to 2 zettels is the same violation (for the baseline and the write gate).
        one = typed({**CLEAN, "50_MOC/overview.md": page("[[z1]]")}).issues("50_MOC/overview.md")[0]
        two = typed({**CLEAN, "50_MOC/overview.md": page("[[z1]] [[z2]]")}).issues("50_MOC/overview.md")[0]
        self.assertNotEqual(one.message, two.message)
        self.assertEqual(violation_keys(one), violation_keys(two))


# ----- the write gate under core ---------------------------------------------------------------------


class VaultNewTests(unittest.TestCase):
    NOW = __import__("datetime").datetime(2026, 11, 10, 9, 5)

    def test_created_is_filled_in(self):
        vault = typed(CLEAN)
        cases = {
            "source": ("Paper Title: Subtitle", "10_SOURCES/Paper Title Subtitle.md", ["Created: 10-11-2026 09:05", "Tags: ", "Source: "]),
            "zettel": ("KL Penalty in RLHF", "20_ZETTLEKASTEN/kl-penalty-in-rlhf.md", ["Created: 10-11-2026 09:05", "Tags: "]),
            "idea": ("Try It Twice", "50_MOC/ideas/Try It Twice.md", ["Created: 10-11-2026 09:05", "Tags: ", "Status: #open"]),
            "page": ("Grader Sanity Checks", "50_MOC/grader-sanity-checks.md", ["Created: 10-11-2026 09:05", "Tags: "]),
        }
        for type_name, (title, path, header) in cases.items():
            with self.subTest(type_name):
                change = plan_new(vault, type_name, title, now=self.NOW).files[0]
                self.assertEqual(change.path, path)
                self.assertEqual([line.rstrip() for line in change.new.splitlines()[:len(header)]], [h.rstrip() for h in header])

    def test_body_carries_the_links_core_requires(self):
        vault = typed(CLEAN)
        values = {"Tags": "[[ml]]"}
        bare = plan_new(vault, "zettel", "Lonely", values=values, now=self.NOW)
        self.assertEqual([i.code for i in judge(vault, bare.files).new_violations], ["too-few-links"])
        cited = plan_new(vault, "zettel", "Cited", values=values, now=self.NOW, body_text="From [[S1]].\n\nMore.\n")
        self.assertEqual(cited.files[0].new, "Created: 10-11-2026 09:05\nTags: [[ml]]\n\n# Cited\n\nFrom [[S1]].\n\nMore.\n")
        self.assertTrue(judge(vault, cited.files).ok)

    def test_cli(self):
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        root = Path(scratch.name)
        for path, text in CLEAN.items():
            (root / path).parent.mkdir(parents=True, exist_ok=True)
            (root / path).write_text(text)
        refused = run_vault("new", str(root), "-f", "core", "--type", "zettel", "--title", "No Source", "--set", "Tags=[[ml]]")
        self.assertEqual(refused.returncode, 1)
        self.assertIn("too-few-links        links to 0 source notes; a zettel needs at least 1", refused.stdout)
        written = run_vault("new", str(root), "-f", "core", "--type", "zettel", "--title", "With Source",
                            "--set", "Tags=[[ml]]", "--body", "Rests on [[S2]].")
        self.assertEqual(written.returncode, 0, written.stdout + written.stderr)
        text = (root / "20_ZETTLEKASTEN/with-source.md").read_text()
        self.assertRegex(text, r"^Created: \d\d-\d\d-\d{4} \d\d:\d\d\nTags: \[\[ml\]\]\n\n# With Source\n\nRests on \[\[S2\]\]\.\n$")
        self.assertEqual(run_vault("check", str(root), "-f", "core").returncode, 0)


class WriteGateTests(unittest.TestCase):
    def test_new_zettel_must_cite_a_source(self):
        vault = typed(CLEAN)
        values = {"Created": "01-01-2026 10:00", "Tags": "[[ml]]"}
        bare = plan_new(vault, "zettel", "Lonely", values=values)
        verdict = judge(vault, bare.files)
        self.assertEqual([(i.code, i.message) for i in verdict.new_violations],
                         [("too-few-links", "links to 0 source notes; a zettel needs at least 1")])
        cited = plan_new(vault, "zettel", "Cited", values=values)
        cited.files[0].new += "From [[S1]].\n"
        self.assertTrue(judge(vault, cited.files).ok)

    def test_new_note_in_a_subfolder_is_refused(self):
        from vaultengine.formats.write import WriteError

        with self.assertRaises(WriteError) as raised:
            plan_new(typed(CLEAN), "source", "X", folder="10_SOURCES/old")
        self.assertIn("10_SOURCES/old/X.md would be a note of no type, not a source", str(raised.exception))

    def test_improving_a_page_that_is_still_short_is_not_refused(self):
        vault = typed({**CLEAN, "50_MOC/overview.md": page("[[z1]]")})
        text = page("[[z1]]")
        verdict = judge(vault, [FileChange("50_MOC/overview.md", text, text.replace("[[z1]]", "[[z1]] [[z2]]"))])
        self.assertTrue(verdict.ok)
        self.assertEqual([i.code for i in verdict.tolerated], ["too-few-links"])


# ----- baseline ----------------------------------------------------------------------------------------


class BaselineApiTests(unittest.TestCase):
    def test_record_compare_and_counts(self):
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        file = Path(scratch.name) / ".vaultengine" / "baseline-core.json"
        messy = {**CLEAN, "20_ZETTLEKASTEN/z1.md": "Tags: [[ml]]\n\nno source\n", "10_SOURCES/old/a.md": "x\n"}
        before = typed(messy)
        record_baseline(before, file)
        baseline = load_baseline(file, "core")
        self.assertEqual(Counter(item["code"] for item in baseline.violations),
                         Counter({"missing-key": 1, "too-few-links": 1, "untyped-note": 1}))
        self.assertTrue(all(item["line"] >= 1 and item["keys"] for item in baseline.violations))

        same = compare_to_baseline(before, before.issues(), baseline)
        self.assertEqual((len(same.new), len(same.known), same.fixed), (0, 3, []))

        # Lines move, one violation is fixed, one is added: only the added one is new.
        after = typed({**messy, "20_ZETTLEKASTEN/z1.md": "\n\nTags: [[ml]]\n\nnow from [[S1]]\n",
                       "50_MOC/overview.md": page("[[z1]]")})
        result = compare_to_baseline(after, after.issues(), baseline)
        self.assertEqual([(i.path, i.code) for i in result.new], [("50_MOC/overview.md", "too-few-links")])
        self.assertEqual([(i.path, i.code) for i in result.known],
                         [("10_SOURCES/old/a.md", "untyped-note"), ("20_ZETTLEKASTEN/z1.md", "missing-key")])
        self.assertEqual([(f["path"], f["code"]) for f in result.fixed], [("20_ZETTLEKASTEN/z1.md", "too-few-links")])

    def test_a_baselined_key_tolerates_only_as_many_as_were_recorded(self):
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        file = Path(scratch.name) / "b.json"
        spec = parse_format("format: 1\nname: core\ntypes: {t: {match: {folder: A}}}\n")
        one = typed({"A/x.md": "foo:: 1\n"}, spec)  # unknown-field
        record_baseline(one, file)
        two = typed({"A/x.md": "foo:: 1\nfoo:: 2\n"}, spec)  # the same message twice
        result = compare_to_baseline(two, two.issues(), load_baseline(file, "core"))
        self.assertEqual((len(result.known), len(result.new)), (1, 1))

    def test_bad_files(self):
        from vaultengine.formats.check import BaselineError

        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        file = Path(scratch.name) / "b.json"
        empty = '"violations": [], "notes": [], "paths": []'
        for content, error in (("{", "not a baseline file (invalid JSON"), ('{"x": 1}', "not a baseline file of version 2"),
                               ('{"baseline": 1, "format": {"name": "core"}, "violations": []}', "not a baseline file of version 2"),
                               ('{"baseline": 2, "format": {"name": "brain"}, ' + empty + "}", "recorded for format 'brain', not 'core'"),
                               ('{"baseline": 2, "format": {"name": "core"}, "violations": [{"keys": "x"}], "notes": [], "paths": []}',
                                "violations[0] has no valid keys"),
                               ('{"baseline": 2, "format": {"name": "core"}, "violations": [], "notes": [{"path": "a"}], "paths": []}',
                                "notes[0] is not a note fingerprint")):
            with self.subTest(content=content):
                file.write_text(content)
                with self.assertRaises(BaselineError) as raised:
                    load_baseline(file, "core")
                self.assertIn(error, str(raised.exception))
        with self.assertRaises(BaselineError):
            load_baseline(Path(scratch.name) / "missing.json", "core")

    def test_file_layout_one_violation_per_line(self):
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        file = Path(scratch.name) / "b.json"
        record_baseline(typed(CLEAN), file)  # nothing to record is still a baseline
        self.assertEqual(load_baseline(file, "core").violations, [])
        record_baseline(typed({**CLEAN, "10_SOURCES/a/b.md": "x\n", "10_SOURCES/a/c.md": "x\n"}), file)
        lines = file.read_text().splitlines()
        rows = [line for line in lines if line.startswith('  {"path": ')]
        self.assertEqual(len(rows), 2)
        self.assertEqual(json.loads(file.read_text())["summary"], {"violations": 2, "notes": 2})
        self.assertFalse(any(p.name.endswith(".tmp") for p in file.parent.iterdir()))

    def test_path(self):
        self.assertEqual(baseline_path("/v", "core"), Path("/v/.vaultengine/baseline-core.json"))
        self.assertEqual(baseline_path("/v", "my format/x"), Path("/v/.vaultengine/baseline-my-format-x.json"))


class BaselineCliTests(unittest.TestCase):
    """`vault check --baseline` on a small vault on disk."""

    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory()
        self.root = Path(self.scratch.name) / "vault"
        files = {**CLEAN, "20_ZETTLEKASTEN/old.md": "Tags: #ml\n\nwritten before the rules\n", "10_SOURCES/2019/s.md": "x\n"}
        for path, text in files.items():
            (self.root / path).parent.mkdir(parents=True, exist_ok=True)
            (self.root / path).write_text(text, encoding="utf-8")
        self.file = self.root / ".vaultengine" / "baseline-core.json"

    def tearDown(self):
        self.scratch.cleanup()

    def check(self, *extra):
        return run_vault("check", str(self.root), "-f", "core", *extra)

    def write(self, path, text):
        (self.root / path).parent.mkdir(parents=True, exist_ok=True)
        (self.root / path).write_text(text, encoding="utf-8")

    def snapshot(self):
        return sorted((str(p.relative_to(self.root)), p.read_bytes()) for p in self.root.rglob("*") if p.is_file())

    def test_check_never_writes_and_baseline_needs_a_recorded_one(self):
        before = self.snapshot()
        missing = self.check("--baseline")
        self.assertEqual(missing.returncode, 2)
        self.assertEqual(missing.stdout, "")
        self.assertIn(f"error: no baseline for format 'core' at {self.file}; record one with "
                      f"`vault check {self.root} -f core --update-baseline`", missing.stderr)
        self.assertEqual(self.check().returncode, 1)
        self.assertEqual(self.check("--json").returncode, 1)
        self.assertEqual(self.snapshot(), before)
        self.assertFalse((self.root / ".vaultengine").exists())
        both = self.check("--baseline", "--update-baseline")
        self.assertEqual(both.returncode, 2)
        self.assertIn("give one", both.stderr)
        self.check("--update-baseline")
        recorded = self.snapshot()
        self.assertEqual(self.check("--baseline").returncode, 0)
        self.assertEqual(self.check("--baseline", "--json").returncode, 0)
        self.assertEqual(self.snapshot(), recorded)  # comparing writes nothing either

    def test_record_then_fail_only_on_new(self):
        plain = self.check()
        self.assertEqual(plain.returncode, 1)
        self.assertIn("✗ 3 violations in 2 notes\n       1  missing-key\n       1  too-few-links\n       1  untyped-note", plain.stdout)

        first = self.check("--update-baseline")
        self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
        self.assertIn(f"Recorded a baseline of 3 violations: {self.file}", first.stdout)
        self.assertTrue(self.file.exists())
        self.assertNotIn("missing-key", first.stdout)  # the wall is not printed

        again = self.check("--baseline")
        self.assertEqual(again.returncode, 0)
        self.assertIn("✓ no new violations in 11 notes checked (3 known violations in the baseline not shown)", again.stdout)

        # The baseline file is not a note, and a check without --baseline is unchanged.
        self.assertEqual(self.check().stdout, plain.stdout)

        self.write("20_ZETTLEKASTEN/new.md", zettel("cites nothing"))
        self.write("50_MOC/thin.md", page("[[z1]]"))
        planted = self.check("--baseline")
        self.assertEqual(planted.returncode, 1)
        self.assertIn("20_ZETTLEKASTEN/new.md  (zettel)\n      1  too-few-links        links to 0 source notes; a zettel needs at least 1",
                      planted.stdout)
        self.assertIn("50_MOC/thin.md  (page)\n      1  too-few-links        links to 1 zettel note; a page needs at least 3",
                      planted.stdout)
        self.assertIn("✗ 2 new violations in 2 notes (3 known violations in the baseline not shown)\n       2  too-few-links",
                      planted.stdout)
        self.assertNotIn("old.md", planted.stdout)

        data = json.loads(self.check("--baseline", "--json").stdout)
        self.assertFalse(data["ok"])
        self.assertEqual(data["summary"]["violations"], 2)
        self.assertEqual({k: data["baseline"][k] for k in ("violations", "new", "known", "fixed", "recorded_now")},
                         {"violations": 3, "new": 2, "known": 3, "fixed": 0, "recorded_now": False})

    def test_fixed_and_update(self):
        self.check("--update-baseline")
        self.write("20_ZETTLEKASTEN/old.md", "Created: 01-01-2026 10:00\nTags: #ml\n\nwritten before the rules\n")
        fixed = self.check("--baseline")
        self.assertEqual(fixed.returncode, 0)
        self.assertIn("1 baselined violation fixed since; `--update-baseline` records the smaller baseline", fixed.stdout)
        update = self.check("--update-baseline")
        self.assertEqual(update.returncode, 0)
        self.assertIn("Recorded a baseline of 2 violations (it had 3)", update.stdout)
        # The fix is now locked in: breaking it again is new.
        self.write("20_ZETTLEKASTEN/old.md", "Tags: #ml\n\nwritten before the rules\n")
        self.assertEqual(self.check("--baseline").returncode, 1)

    def test_with_changed(self):
        self.assertEqual(run_vault("index", str(self.root)).returncode, 0)
        self.check("--update-baseline")
        time.sleep(0.01)
        old = self.root / "20_ZETTLEKASTEN/old.md"
        old.write_text(old.read_text() + "more text, still no source\n")  # changed, nothing new
        quiet = self.check("--changed", "--baseline")
        self.assertEqual(quiet.returncode, 0, quiet.stdout)
        self.assertIn("checked 1 note changed since the last index", quiet.stdout)
        self.assertIn("(2 known violations in the baseline not shown)", quiet.stdout)
        self.write("50_MOC/thin.md", page(""))
        loud = self.check("--changed", "--baseline")
        self.assertEqual(loud.returncode, 1)
        self.assertIn("✗ 1 new violation in 1 note (2 known violations in the baseline not shown)", loud.stdout)

    def test_broken_baseline_is_an_error(self):
        self.file.parent.mkdir(parents=True, exist_ok=True)
        self.file.write_text("not json")
        result = self.check("--baseline")
        self.assertEqual(result.returncode, 2)
        self.assertIn("not a baseline file", result.stderr)
        self.assertEqual(self.file.read_text(), "not json")  # left as it is
        self.assertEqual(self.check("--update-baseline").returncode, 0)  # re-recording replaces it
        self.assertEqual(self.check("--baseline").returncode, 0)


class BaselineRenameTests(unittest.TestCase):
    """A renamed note's baselined violations follow it."""

    PROSE = ("The grader has to fail a policy that does nothing at all, because otherwise the training signal pays for "
             "inaction and the reward climbs while the agent learns to sit still and wait for the episode to end.\n")

    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory()
        self.root = Path(self.scratch.name)
        # old.md: written before the rules, no Created and no source (2 violations).
        files = {**CLEAN, "20_ZETTLEKASTEN/old.md": f"Tags: #ml\n\n# Old\n\n{self.PROSE}",
                 "20_ZETTLEKASTEN/other.md": zettel("From [[S1]]. " + self.PROSE.replace("grader", "judge"))}
        for path, text in files.items():
            self.write(path, text)
        self.assertEqual(run_vault("check", str(self.root), "-f", "core", "--update-baseline").returncode, 0)

    def tearDown(self):
        self.scratch.cleanup()

    def write(self, path, text):
        (self.root / path).parent.mkdir(parents=True, exist_ok=True)
        (self.root / path).write_text(text, encoding="utf-8")

    def check(self):
        return run_vault("check", str(self.root), "-f", "core", "--baseline")

    def test_same_text_new_name(self):
        (self.root / "20_ZETTLEKASTEN/old.md").rename(self.root / "20_ZETTLEKASTEN/graders-must-fail-inaction.md")
        result = self.check()
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("followed a renamed note (same text): 20_ZETTLEKASTEN/old.md → 20_ZETTLEKASTEN/graders-must-fail-inaction.md",
                      result.stdout)
        self.assertIn("(2 known violations in the baseline not shown)", result.stdout)
        self.assertNotIn("fixed since", result.stdout)

    def test_renamed_and_edited(self):
        (self.root / "20_ZETTLEKASTEN/old.md").unlink()
        self.write("20_ZETTLEKASTEN/renamed.md", f"Tags: #ml #rl\n\n# Renamed\n\n{self.PROSE}One more sentence.\n")
        result = self.check()
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("followed a renamed note (similar text): 20_ZETTLEKASTEN/old.md → 20_ZETTLEKASTEN/renamed.md", result.stdout)
        data = json.loads(run_vault("check", str(self.root), "-f", "core", "--baseline", "--json").stdout)
        self.assertEqual(data["baseline"]["renames"][0]["how"], "similar text")
        self.assertGreaterEqual(data["baseline"]["renames"][0]["similarity"], 0.7)

    def test_renamed_and_newly_broken_still_fails_on_the_new_part(self):
        (self.root / "20_ZETTLEKASTEN/old.md").unlink()
        self.write("20_ZETTLEKASTEN/renamed.md", f"Tags: #ml\nCreated: yesterday\n\n# Old\n\n{self.PROSE}")
        result = self.check()
        self.assertEqual(result.returncode, 1)
        # missing Created became a bad Created: that is new; the missing source followed the note.
        self.assertIn("✗ 1 new violation in 1 note (1 known violation in the baseline not shown)\n       1  bad-value", result.stdout)
        self.assertIn("1 baselined violation fixed since", result.stdout)

    def test_a_different_note_does_not_inherit(self):
        (self.root / "20_ZETTLEKASTEN/old.md").unlink()
        self.write("20_ZETTLEKASTEN/unrelated.md", "Tags: #ml\n\n# Unrelated\n\nSomething else entirely, no source, no date.\n")
        result = self.check()
        self.assertEqual(result.returncode, 1)
        self.assertIn("✗ 2 new violations in 1 note", result.stdout)
        self.assertIn("2 baselined violations fixed since", result.stdout)
        self.assertNotIn("followed", result.stdout)

    def test_only_notes_that_did_not_exist_are_candidates(self):
        # old.md is gone and other.md, which was there all along, now looks like it: not a rename.
        (self.root / "20_ZETTLEKASTEN/old.md").unlink()
        self.write("20_ZETTLEKASTEN/other.md", f"Tags: #ml\n\n# Old\n\n{self.PROSE}")
        result = self.check()
        self.assertEqual(result.returncode, 1)
        self.assertIn("20_ZETTLEKASTEN/other.md  (zettel)", result.stdout)
        self.assertNotIn("followed", result.stdout)

    def test_two_equal_candidates_are_not_guessed(self):
        text = (self.root / "20_ZETTLEKASTEN/old.md").read_text()
        (self.root / "20_ZETTLEKASTEN/old.md").unlink()
        self.write("20_ZETTLEKASTEN/copy-a.md", text)
        self.write("20_ZETTLEKASTEN/copy-b.md", text)
        result = self.check()
        self.assertEqual(result.returncode, 1)
        self.assertIn("✗ 4 new violations in 2 notes", result.stdout)
        self.assertNotIn("followed", result.stdout)

    def test_moved_out_of_its_home(self):
        # Into a subfolder: the note is no zettel any more, so its zettel violations are gone and untyped-note is new.
        (self.root / "20_ZETTLEKASTEN/drafts").mkdir()
        (self.root / "20_ZETTLEKASTEN/old.md").rename(self.root / "20_ZETTLEKASTEN/drafts/old.md")
        result = self.check()
        self.assertEqual(result.returncode, 1)
        self.assertIn("20_ZETTLEKASTEN/drafts/old.md  (no type)\n      1  untyped-note", result.stdout)
        self.assertIn("followed a renamed note (same text)", result.stdout)
        self.assertIn("2 baselined violations fixed since", result.stdout)

    def test_fingerprints(self):
        from vaultengine.formats.check import fingerprint, similarity

        a = fingerprint(self.PROSE * 3)
        self.assertEqual(similarity(a, fingerprint(self.PROSE * 3)), 1.0)
        self.assertEqual(len(a["sketch"]), 32)
        self.assertGreater(similarity(a, fingerprint(self.PROSE * 3 + "and a last line")), 0.7)
        self.assertLess(similarity(a, fingerprint("Something else entirely, " * 20)), 0.2)
        self.assertEqual(similarity(fingerprint("short one"), fingerprint("short two")), 0.0)  # too short: only identical texts


# ----- core on the sample vault --------------------------------------------------------------------------


class SampleVaultTests(unittest.TestCase):
    def test_what_core_reports(self):
        vault = TypedVault(core(), Vault.load(VAULT))
        counts = Counter((vault.type_of(i.path), i.code) for i in vault.issues())
        self.assertEqual(dict(counts), {
            ("source", "missing-key"): 28,    # no source has a Source: line yet
            ("source", "missing-value"): 18,  # empty Tags:
            ("zettel", "missing-key"): 15,    # Created
            ("zettel", "too-few-links"): 81,  # cite no source note that is in the vault
            ("page", "too-few-links"): 2,
        })
        self.assertEqual(Counter(vault.type_of(p) for p in vault.vault.notes),
                         Counter({"zettel": 109, None: 40, "source": 28, "page": 3}))
        self.assertEqual(len(vault.ignored), 40)  # 50_MOC/discourse
        self.assertEqual(vault.untyped, [])

    def test_brain_is_unchanged(self):
        vault = TypedVault(load_format(REPO_ROOT / "formats" / "brain.yaml"), Vault.load(VAULT))
        self.assertEqual(len(vault.issues()), 102)
        self.assertFalse({"too-few-links", "untyped-note"} & {i.code for i in vault.issues()})


if __name__ == "__main__":
    unittest.main()
