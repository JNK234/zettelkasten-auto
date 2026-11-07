import json
import subprocess
import sys
import textwrap
import unittest
from collections import Counter
from pathlib import Path

from vaultengine import Vault, parse_note
from vaultengine.formats import FormatError, TypedVault, load_format, parse_format, resolve_format_path
from vaultengine.formats.spec import Match, SectionPattern

TESTS = Path(__file__).parent
VAULT = TESTS / "fixtures" / "vault"
REPO_ROOT = TESTS.parent
BRAIN = REPO_ROOT / "formats" / "brain.yaml"


def spec_of(text: str):
    return parse_format(textwrap.dedent(text))


def typed(format_text: str, files: dict[str, str]) -> TypedVault:
    return TypedVault(spec_of(format_text), Vault.from_files({p: None if t is None else textwrap.dedent(t) for p, t in files.items()}))


def codes(vault: TypedVault, path: str | None = None) -> list[str]:
    return [issue.code for issue in vault.issues(path)]


# ----- parser: generic fields ----------------------------------------------------------


class HeaderFieldTests(unittest.TestCase):
    def test_block_at_top_with_blank_lines(self):
        note = parse_note("\nCreated: 12-05-2026 21:55\n\nStatus: #baby \n\nTags: [[a]] [[b|B]]\n\n# Title\nLater: not a header\n")
        self.assertEqual([(f.key, f.value, f.line) for f in note.header_fields], [
            ("Created", "12-05-2026 21:55", 2), ("Status", "#baby", 4), ("Tags", "[[a]] [[b|B]]", 6),
        ])
        self.assertEqual(note.header_fields[1].tags, ("baby",))
        self.assertEqual([link.target for link in note.header_fields[2].links], ["a", "b"])

    def test_block_ends_at_first_other_line(self):
        note = parse_note("Status: #baby\nSome prose here.\nTags: [[x]]\n")
        self.assertEqual([f.key for f in note.header_fields], ["Status"])

    def test_after_frontmatter(self):
        note = parse_note("---\ntitle: T\n---\nStatus: done\n")
        self.assertEqual([(f.key, f.line) for f in note.header_fields], [("Status", 4)])

    def test_empty_value_and_keys_with_spaces(self):
        note = parse_note("Tags:\nLast reviewed: 2026\n")
        self.assertEqual([(f.key, f.value) for f in note.header_fields], [("Tags", ""), ("Last reviewed", "2026")])

    def test_not_fields(self):
        for text in ("https://example.com\n", "12:30 meeting\n", "key::value\n", "- Status: x\n", "Status:done\n"):
            with self.subTest(text=text):
                self.assertEqual(parse_note(text).header_fields, [])

    def test_inline_field_lines_may_sit_in_the_block(self):
        note = parse_note("Status: #x\nfrom:: [[S]]\nTags: [[t]]\n")
        self.assertEqual([f.key for f in note.header_fields], ["Status", "Tags"])
        self.assertEqual([f.key for f in note.inline_fields], ["from"])

    def test_code_at_the_top_ends_the_block(self):
        note = parse_note("```\nStatus: x\n```\nTags: y\n")
        self.assertEqual(note.header_fields, [])


class InlineFieldTests(unittest.TestCase):
    def test_relation_lines(self):
        note = parse_note("# C\n## Relations\ninforms:: [[QUE 03 - Q]], [[QUE 19]]\nsupported-by:: [[EVD - a]]\n")
        self.assertEqual([(f.key, f.line) for f in note.inline_fields], [("informs", 3), ("supported-by", 4)])
        self.assertEqual([link.target for link in note.inline_fields[0].links], ["QUE 03 - Q", "QUE 19"])
        self.assertEqual(note.inline_fields[0].value, "[[QUE 03 - Q]], [[QUE 19]]")

    def test_list_items_blockquotes_and_spaces_in_keys(self):
        note = parse_note("- supports:: [[A]]\n> from:: [[B]]\n1. different context:: [[C]]\nrating::\n")
        self.assertEqual([f.key for f in note.inline_fields], ["supports", "from", "different context", "rating"])
        self.assertEqual(note.inline_fields[3].value, "")

    def test_not_fields(self):
        # Like Dataview, everything before '::' is the key, so it must look like one.
        text = "std::vector<int> v;\nWell, a sentence:: with a comma.\n`code:: [[x]]`\n```\nfrom:: [[y]]\n```\n%% from:: [[z]] %%\n"
        self.assertEqual([f.key for f in parse_note(text).inline_fields], [])

    def test_frontmatter_is_not_scanned(self):
        self.assertEqual(parse_note("---\nfrom:: x\n---\n").inline_fields, [])

    def test_links_of_a_field_are_the_note_links(self):
        note = parse_note("supports:: [[A]], ![[img.png]]\n")
        field = note.inline_fields[0]
        self.assertEqual(field.links, (note.links[0], note.embeds[0]))

    def test_fixture_vault_counts(self):
        vault = Vault.load(VAULT)
        header = Counter(f.key for note in vault.notes.values() for f in note.header_fields)
        inline = Counter(f.key for note in vault.notes.values() for f in note.inline_fields)
        self.assertEqual(header, Counter({"Tags": 180, "Status": 177, "Created": 165, "Strength": 31,
                                          "Updated": 3, "Type": 3, "Sources": 2, "Zettels": 2}))
        self.assertEqual(inline, Counter({"from": 31, "supports": 31, "informs": 8, "supported-by": 8,
                                          "different-context": 2, "refines": 2, "opposed-by": 1}))


# ----- spec ---------------------------------------------------------------------------

MINIMAL = """
format: 1
types:
  note:
    match: {folder: Notes}
"""


class SpecTests(unittest.TestCase):
    def test_brain_loads(self):
        spec = load_format(BRAIN)
        self.assertEqual(spec.name, "brain")
        self.assertEqual([t.name for t in spec.types], ["question", "claim", "evidence", "zettel", "source", "topic", "comparison"])
        self.assertEqual(spec.relations["supports"].inverse, "supported-by")
        self.assertEqual(spec.relations["supported-by"].inverse, "supports")
        self.assertEqual(spec.relations["opposed-by"].inverse, "opposes")
        for one_way in ("from", "informs", "refines"):
            self.assertIsNone(spec.relations[one_way].inverse)

    def test_inverse_is_declared_automatically_and_keys_are_normalised(self):
        spec = spec_of(MINIMAL + "relations:\n  supports: {inverse: supported-by}\n")
        self.assertEqual(spec.relations["supported-by"].inverse, "supports")
        self.assertIs(spec.relation("Supported By"), spec.relations["supported-by"])
        self.assertIsNone(spec.relation("unknown"))

    def test_self_inverse(self):
        spec = spec_of(MINIMAL + "relations:\n  related: {inverse: related}\n")
        self.assertEqual(spec.relations["related"].inverse, "related")

    def test_errors_name_the_place(self):
        cases = {
            "format: 2\ntypes: {n: {match: {folder: X}}}": "format:",
            "format: 1\ntypes: {}": "types:",
            "format: 1\ntype: {}": "unknown key 'type'",
            "format: 1\ntypes: {n: {}}": "types.n: missing required key 'match'",
            "format: 1\ntypes: {n: {match: {}}}": "types.n.match: give a folder",
            "format: 1\ntypes: {n: {match: {path: X}}}": "types.n.match: unknown key 'path'",
            "format: 1\ntypes: {n: {match: {folder: X}, metadata: {S: {kind: tag}}}}": "types.n.metadata.S.kind: unknown kind 'tag'",
            "format: 1\ntypes: {n: {match: {folder: X}, metadata: {S: {kind: [text, tags]}}}}": "only 'tags' and 'links' can be combined",
            "format: 1\ntypes: {n: {match: {folder: X}, metadata: {S: {format: '%Y'}}}}": "only datetime and date keys take a format",
            "format: 1\ntypes: {n: {match: {folder: X}, metadata: {S: {kind: integer, values: [1]}}}}": "not supported for integer",
            "format: 1\ntypes: {n: {match: {folder: X}, metadata: {S: {required: yes please}}}}": "expected true or false",
            "format: 1\ntypes: {n: {match: {folder: X}, metadata: {S: {}, s: {}}}}": "declared twice",
            "format: 1\ntypes: {n: {match: {folder: X}, relations: {required: [supports]}}}": "unknown relation 'supports'",
            "format: 1\nrelations: {a: {targets: [nope]}}\ntypes: {n: {match: {folder: X}}}": "relations.a.targets: unknown type 'nope'",
            "format: 1\nrelations: {a: {inverse: b}, b: {inverse: c}}\ntypes: {n: {match: {folder: X}}}": "'a' names 'b' as its inverse, but 'b' names 'c'",
            "format: 1\nrelations: {a: {inverse: x}, b: {inverse: x}}\ntypes: {n: {match: {folder: X}}}": "already the inverse of 'a'",
            "format: 1\nlink_roles: {r: ['/(/']}\ntypes: {n: {match: {folder: X}}}": "link_roles.r[0]: invalid regular expression",
            "format: 1\ntypes: {n: {match: {folder: X}}, n: {match: {folder: Y}}}": "duplicate key 'n'",
            "format: 1\ntypes: {n: {match: {folder: X}}}\nmetadata_source: yaml": "metadata_source: 'yaml' is not one of",
            "format: [": "invalid YAML",
            "": "empty",
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                with self.assertRaises(FormatError) as caught:
                    parse_format(text)
                self.assertIn(expected, str(caught.exception))

    def test_load_format_prefixes_the_file(self):
        path = Path(self.id().replace(".", "_") + ".yaml")
        try:
            path.write_text("format: 1\ntypes: {n: {match: {folder: X}, oops: 1}}\n")
            with self.assertRaises(FormatError) as caught:
                load_format(path)
            self.assertTrue(str(caught.exception).startswith(f"{path}: types.n: unknown key 'oops'"))
        finally:
            path.unlink()

    def test_resolve_format_path(self):
        self.assertEqual(resolve_format_path("brain", [REPO_ROOT / "formats"]), REPO_ROOT / "formats" / "brain.yaml")
        self.assertEqual(resolve_format_path("some/file.yaml", []), Path("some/file.yaml"))
        with self.assertRaises(FormatError):
            resolve_format_path("nope", [REPO_ROOT / "formats"])

    def test_match(self):
        match = Match(folder="50_MOC/discourse", name="CLM - *")
        self.assertTrue(match.matches("50_MOC/discourse/rl/CLM - A claim.md"))
        self.assertTrue(match.matches("50_moc/Discourse/clm - lower.md"))
        self.assertFalse(match.matches("50_MOC/discourse/rl/EVD - A.md"))
        self.assertFalse(match.matches("50_MOC/CLM - outside.md"))
        self.assertFalse(Match(folder="20_Z").matches("20_ZETTLEKASTEN/x.md"))
        self.assertTrue(Match(name="QUE *").matches("anywhere/QUE 03 - Q.md"))

    def test_section_patterns(self):
        self.assertTrue(SectionPattern.parse("See Also", "x").matches("see  also"))
        self.assertTrue(SectionPattern.parse("Connections", "x").matches("Connections:"))
        regex = SectionPattern.parse("/connections?$/", "x")
        self.assertTrue(regex.matches("13. Connections"))
        self.assertFalse(regex.matches("Connections to the vault"))


# ----- applying a format -------------------------------------------------------------------

FORMAT = """
format: 1
metadata_source: inline
relations:
  supports: {inverse: supported-by, targets: [claim]}
  supported-by: {targets: [evidence]}
  from: {}
link_roles:
  tension: [Tensions]
  reference: [References]
types:
  claim:
    match: {name: "CLM - *"}
    headings: [Relations, Scope]
    ordered_headings: true
    metadata:
      Status: {required: true, kind: tags, multiple: false, values: [supported, contested]}
    relations: {required: [supported-by]}
  evidence:
    match: {name: "EVD - *"}
    metadata:
      Strength: {required: true, values: [A, B, C, D]}
    relations: {required: [supports], optional: [from]}
  zettel:
    match: {folder: Z}
    headings: [What This Is, Tensions]
    link_roles:
      key-detail: [Key Details]
    metadata:
      Created: {required: true, kind: datetime, format: "%d-%m-%Y %H:%M"}
      Tags: {required: true, kind: [links, tags], link_role: tag}
      Count: {kind: integer}
      Sources: {required: true, allow_empty: true, kind: links}
"""

CLAIM = """\
Status: #supported

# A claim
## Relations
supported-by:: [[EVD - e1]]
## Scope
"""

EVIDENCE = """\
Strength: B

# E
## Relations
supports:: [[CLM - c]]
"""


class TypingTests(unittest.TestCase):
    def test_first_matching_type_wins_and_untyped(self):
        spec_text = MINIMAL + "  any:\n    match: {folder: Notes}\n  other:\n    match: {name: '*'}\n"
        vault = typed(spec_text, {"Notes/a.md": "", "Elsewhere/b.md": "", "img.png": None})
        self.assertEqual(vault.type_of("Notes/a.md"), "note")
        self.assertEqual(vault.type_of("Elsewhere/b.md"), "other")
        self.assertEqual(vault.untyped, [])
        vault = typed(MINIMAL, {"Notes/a.md": "", "b.md": ""})
        self.assertEqual(vault.untyped, ["b.md"])
        self.assertEqual(vault.issues("b.md"), [])

    def test_clean_pair(self):
        vault = typed(FORMAT, {"CLM - c.md": CLAIM, "EVD - e1.md": EVIDENCE})
        self.assertEqual(vault.issues(), [])
        self.assertEqual([(e.relation, e.target_path, e.origin) for e in vault.relations("CLM - c.md")],
                         [("supported-by", "EVD - e1.md", "declared")])


class MetadataTests(unittest.TestCase):
    def zettel(self, header: str, body: str = "## What This Is\n## Tensions\n"):
        vault = typed(FORMAT, {"Z/n.md": header + "\n# T\n" + body})
        return vault, [issue.message for issue in vault.issues("Z/n.md")]

    def test_values(self):
        vault, messages = self.zettel("Created: 24-09-2026 16:29\nTags: [[a]] #b\nCount: 3\nSources:\n")
        self.assertEqual(messages, [])
        meta = vault.metadata("Z/n.md")
        self.assertEqual(meta["Tags"].values, ["a", "b"])
        self.assertEqual([item.kind for item in meta["Tags"].items], ["link", "tag"])
        self.assertEqual(meta["Created"].line, 1)
        self.assertTrue(meta["Sources"].is_empty)

    def test_missing_and_empty(self):
        _, messages = self.zettel("Tags:\n")
        self.assertEqual(messages, ["missing metadata 'Created'", "'Tags' is empty", "missing metadata 'Sources'"])

    def test_bad_values(self):
        _, messages = self.zettel("Created: 2026-09-24\nTags: [[a]] plain\nCount: many\nSources: #x\n")
        self.assertEqual(messages, [
            "Created: '2026-09-24' is not a datetime like '%d-%m-%Y %H:%M'",
            "Tags: 'plain' is not a [[link]] or a #tag",
            "Count: 'many' is not a whole number",
            "Sources: #x is a tag; expected a [[link]]",
        ])

    def test_allowed_values_and_multiple(self):
        vault = typed(FORMAT, {
            "CLM - a.md": CLAIM.replace("#supported", "#Supported"),
            "CLM - b.md": CLAIM.replace("#supported", "#settled"),
            "CLM - c.md": CLAIM.replace("#supported", "#supported #contested"),
            "CLM - d.md": CLAIM.replace("#supported", "supported"),
            "EVD - e1.md": EVIDENCE.replace("[[CLM - c]]", "[[CLM - a]]"),
        })
        messages = {path: [i.message for i in vault.issues(path) if i.code == "bad-value"] for path in vault.vault.notes}
        self.assertEqual(messages["CLM - a.md"], [])
        self.assertEqual(messages["CLM - b.md"], ["Status: #settled is not an allowed value (supported, contested)"])
        self.assertEqual(messages["CLM - c.md"], ["Status: expected one value, got 2"])
        self.assertEqual(messages["CLM - d.md"], ["Status: 'supported' is not a #tag"])
        self.assertEqual(vault.issues("CLM - b.md")[0].line, 1)

    def test_frontmatter_source(self):
        spec_text = FORMAT.replace("metadata_source: inline", "metadata_source: either")
        note = (
            "---\ncreated: 24-09-2026 16:29\ntags: [\"[[a]]\", b]\ncount: 4\nsources: []\n---\n"
            "Created: not used\n# T\n## What This Is\n## Tensions\n"
        )
        vault = typed(spec_text, {"Z/n.md": note})
        self.assertEqual(vault.issues(), [])
        meta = vault.metadata("Z/n.md")
        self.assertEqual(meta["Created"].source, "frontmatter")
        self.assertEqual(meta["Tags"].values, ["a", "b"])

        only_fm = typed(FORMAT.replace("metadata_source: inline", "metadata_source: frontmatter"),
                        {"Z/n.md": "Created: 24-09-2026 16:29\nTags: [[a]]\nSources:\n# T\n## What This Is\n## Tensions\n"})
        self.assertEqual(codes(only_fm), ["missing-key", "missing-key", "missing-key"])

    def test_yaml_dates(self):
        spec_text = MINIMAL.replace("match: {folder: Notes}", "match: {folder: Notes}\n    metadata_source: frontmatter\n"
                                    "    metadata: {d: {kind: date}, t: {kind: datetime}}")
        vault = typed(spec_text, {"Notes/a.md": "---\nd: 2026-09-24\nt: 2026-09-24T10:00:00\n---\n",
                                  "Notes/b.md": "---\nd: 24.09.2026\nt: 2026-09-24\n---\n"})
        self.assertEqual(vault.issues("Notes/a.md"), [])
        self.assertEqual(codes(vault, "Notes/b.md"), ["bad-value", "bad-value"])


class HeadingTests(unittest.TestCase):
    def test_missing_and_order(self):
        vault = typed(FORMAT, {
            "CLM - a.md": "Status: #supported\nsupported-by:: [[EVD - x]]\n# T\n## Scope\n## Relations\n",
            "CLM - b.md": "Status: #supported\nsupported-by:: [[EVD - x]]\n# T\n",
            "CLM - c.md": "Status: #supported\nsupported-by:: [[EVD - x]]\n# T\n# relations\n### SCOPE\n",
            "EVD - x.md": "Strength: A\nsupports:: [[CLM - a]], [[CLM - b]], [[CLM - c]]\n",
        })
        self.assertEqual([(i.code, i.message, i.line) for i in vault.issues("CLM - a.md")], [
            ("heading-order", "heading 'Scope' (line 4) comes before 'Relations' (line 5); expected order: Relations, Scope", 4),
        ])
        self.assertEqual([(i.message, i.line) for i in vault.issues("CLM - b.md")],
                         [("missing headings 'Relations', 'Scope' (expected at the end of the note)", 3)])
        self.assertEqual(vault.issues("CLM - c.md"), [])  # levels and case are not checked


class RelationTests(unittest.TestCase):
    def test_mirror_is_implied_and_reported_on_the_note_to_edit(self):
        claim = CLAIM.replace("supported-by:: [[EVD - e1]]", "supported-by:: [[EVD - e1]]")
        vault = typed(FORMAT, {"CLM - c.md": claim, "EVD - e1.md": EVIDENCE, "EVD - e2.md": EVIDENCE})
        self.assertEqual([(i.path, i.code) for i in vault.issues()], [("CLM - c.md", "missing-mirror")])
        self.assertEqual(
            vault.issues()[0].message,
            "missing `supported-by:: [[EVD - e2]]` (mirror of `supports::` in EVD - e2.md:5)",
        )
        edges = [(e.relation, e.target_path, e.origin, e.mirror_of) for e in vault.relations("CLM - c.md")]
        self.assertEqual(edges, [
            ("supported-by", "EVD - e1.md", "declared", None),
            ("supported-by", "EVD - e2.md", "mirror", "EVD - e2.md"),
        ])
        self.assertEqual(len(vault.relations("CLM - c.md", mirrors=False)), 1)

    def test_mirror_in_the_other_direction(self):
        vault = typed(FORMAT, {"CLM - c.md": CLAIM, "CLM - other.md": CLAIM.replace("[[EVD - e1]]", "[[EVD - e1]]"),
                               "EVD - e1.md": EVIDENCE.replace("[[CLM - c]]", "[[CLM - other]]")})
        issues = vault.issues("EVD - e1.md")
        self.assertEqual([(i.code, i.line, i.related) for i in issues], [("missing-mirror", 5, "CLM - c.md")])
        self.assertIn("missing `supports:: [[CLM - c]]`", issues[0].message)

    def test_one_way_and_unresolved_relations_have_no_mirror(self):
        evidence = EVIDENCE + "from:: [[Paper]]\n"
        vault = typed(FORMAT, {"CLM - c.md": CLAIM.replace("[[EVD - e1]]", "[[EVD - e1]], [[EVD - missing]]"),
                               "EVD - e1.md": evidence, "Paper.md": ""})
        self.assertEqual([(i.path, i.code, i.line) for i in vault.issues()], [("CLM - c.md", "unresolved-target", 5)])
        self.assertEqual([e.origin for e in vault.relations("Paper.md")], [])

    def test_symmetric_relation(self):
        spec_text = FORMAT.replace("  from: {}", "  from: {}\n  related: {inverse: related}")
        vault = typed(spec_text, {"a.md": "related:: [[b]]\n", "b.md": "", "c.md": "related:: [[d]]\n", "d.md": "related:: [[c]]\n"})
        self.assertEqual([(i.path, i.message) for i in vault.issues()],
                         [("b.md", "missing `related:: [[a]]` (mirror of `related::` in a.md:1)")])

    def test_field_problems(self):
        evidence = "Strength: B\nsupports:: [[CLM - c]] and more\nfrom :: [[P]]\nrating:: 5\nsupported-by:: [[CLM - c]]\n"
        vault = typed(FORMAT, {"CLM - c.md": CLAIM.replace("[[EVD - e1]]", "[[EVD - e]]"), "EVD - e.md": evidence})
        issues = [(i.code, i.line) for i in vault.issues("EVD - e.md")]
        self.assertEqual(issues, [
            ("bad-relation", 2),
            ("unresolved-target", 3),
            ("unknown-field", 4),
            ("unexpected-relation", 5),
            ("relation-target-type", 5),
        ])
        messages = [i.message for i in vault.issues("EVD - e.md")]
        self.assertEqual(messages[0], "`supports::` has text that is not a link: 'and more'")
        self.assertEqual(messages[1], "`from::` [[P]] matches no note")
        self.assertIn("`rating::` is not a declared relation", messages[2])
        self.assertEqual(messages[4], "`supported-by::` [[CLM - c]] is a claim; expected evidence")

    def test_wrong_target_type_is_not_mirrored(self):
        vault = typed(FORMAT, {"EVD - e.md": "Strength: B\nsupports:: [[z]]\n", "Z/z.md": ""})
        self.assertEqual([i.code for i in vault.issues("EVD - e.md")], ["relation-target-type"])
        self.assertNotIn("missing-mirror", codes(vault))
        self.assertEqual(vault.relations("Z/z.md"), [])

    def test_missing_required_relation(self):
        vault = typed(FORMAT, {"EVD - e.md": "Strength: B\nfrom:: [[p]]\n", "p.md": ""})
        self.assertEqual([(i.message, i.line) for i in vault.issues()],
                         [("missing relation field `supports::` (required for evidence)", 2)])


class LinkRoleTests(unittest.TestCase):
    NOTE = """\
    Created: 24-09-2026 16:29
    Tags: [[t1]] #t2
    Sources:
    supports:: [[CLM - c]]

    # Title [[in-title]]
    ## What This Is
    [[plain]]
    ## Tensions
    [[tension-1]]
    ### Deeper
    [[tension-2]]
    ## Key Details
    [[detail]]
    # References
    1. [[ref]]
    """

    def test_roles(self):
        vault = typed(FORMAT, {"Z/n.md": self.NOTE})
        roles = [(link.edge.link.target, link.role, link.section, link.via) for link in vault.links("Z/n.md")]
        self.assertEqual(roles, [
            ("t1", "tag", None, "metadata"),
            ("CLM - c", "supports", None, "field"),
            ("in-title", None, "Title [[in-title]]", None),
            ("plain", None, "What This Is", None),
            ("tension-1", "tension", "Tensions", "section"),
            ("tension-2", "tension", "Deeper", "section"),
            ("detail", "key-detail", "Key Details", "section"),
            ("ref", "reference", "References", "section"),
        ])

    def test_untyped_notes_get_global_roles(self):
        vault = typed(FORMAT, {"n.md": "## Tensions\n[[a]]\nkey detail:: [[b]]\n## Key Details\n[[c]]\n"})
        self.assertEqual([link.role for link in vault.links("n.md")], ["tension", "key-detail", None])


# ----- brain.yaml on the fixture vault -----------------------------------------------------------


class BrainFormatTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.typed = TypedVault(load_format(BRAIN), Vault.load(VAULT))
        cls.report = cls.typed.report()

    def test_every_note_is_typed(self):
        self.assertEqual((self.report["notes"], self.report["typed"], self.report["untyped"]), (180, 180, []))
        rows = {row["name"]: (row["notes"], row["clean"]) for row in self.report["types"]}
        self.assertEqual(rows, {
            "question": (1, 1), "claim": (8, 1), "evidence": (31, 4), "zettel": (109, 67),
            "source": (28, 27), "topic": (2, 2), "comparison": (1, 1),
        })

    def test_issues_are_the_known_drift(self):
        by_code = {row["code"]: (row["count"], row["notes"]) for row in self.report["issues"]["by_code"]}
        # Most relation targets are notes the fixture does not include.
        self.assertEqual(by_code, {"unresolved-target": (59, 34), "missing-heading": (28, 28), "missing-key": (15, 15)})
        self.assertTrue(all(i.line >= 1 for i in self.typed.issues()))
        off_template = [i.path for i in self.typed.issues() if i.code == "missing-heading" and i.path.startswith("20_")]
        self.assertEqual(len(off_template), 27)  # 23 on the older template, 4 stubs
        self.assertEqual([i.path for i in self.typed.issues() if i.path.startswith("10_")], ["10_SOURCES/RLHF.md"])

    def test_relations_and_mirrors(self):
        rows = {row["name"]: row for row in self.report["relations"]}
        self.assertEqual((rows["supports"]["declared"], rows["supports"]["resolved"]), (37, 28))
        self.assertEqual((rows["supported-by"]["declared"], rows["supported-by"]["resolved"]), (37, 28))
        self.assertEqual(sum(row["implied_by_mirror"] for row in rows.values()), 0)  # all pairs written both ways
        claim = "50_MOC/discourse/rl-environment-design/CLM - State graders should check what must not change as well as what must.md"
        self.assertIn(
            ("refines", "50_MOC/discourse/rl-environment-design/CLM - Grade the observed end state rather than the trajectory.md"),
            [(e.relation, e.target_path) for e in self.typed.relations(claim)],
        )

    def test_link_roles(self):
        roles = {row["role"]: row["links"] for row in self.report["link_roles"]}
        self.assertEqual(roles["tension"], 481)
        self.assertEqual(roles["tag"], 505)
        self.assertEqual(roles["from"], 31)
        self.assertEqual(roles["position"], 12)
        self.assertEqual(roles["connection"], 97)
        self.assertEqual(sum(roles.values()) + self.report["links_without_role"], 1787)

    def test_metadata_of_a_source_with_two_status_tags(self):
        meta = self.typed.metadata("10_SOURCES/DeepSeek R1 - Paper Notes.md")
        self.assertEqual(meta["Status"].values, ["baby", "processed"])
        self.assertEqual(meta["Created"].raw, "25-01-2025 10:03")


# ----- CLI -------------------------------------------------------------------------


def run_vault(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "vaultengine", *args], capture_output=True, text=True, cwd=REPO_ROOT)


class FormatsCliTests(unittest.TestCase):
    def test_summary(self):
        result = run_vault("formats", str(VAULT), "--format", "brain")
        self.assertEqual(result.returncode, 0, result.stderr)
        for line in (
            "notes      180   180 typed, 0 untyped",
            "issues     102   in 77 notes",
            "  zettel           109     67      42   in 20_ZETTLEKASTEN/",
            "  missing-heading          28 / 28",
            "  supports                   37       28          9        0   supported-by",
            "     481  tension",
        ):
            self.assertIn(line, result.stdout)

    def test_note_json_and_strict(self):
        result = run_vault("formats", str(VAULT), "-f", "brain", "--note", "EVD - tau2 reward is a product so one zero kills the task", "--json")
        data = json.loads(result.stdout)
        self.assertEqual(data["type"], "evidence")
        self.assertEqual({e["relation"] for e in data["relations"]}, {"from", "supports"})
        self.assertEqual(run_vault("formats", str(VAULT), "-f", "brain", "--strict").returncode, 1)
        self.assertEqual(run_vault("formats", str(VAULT), "-f", "brain", "--type", "comparison", "--strict").returncode, 0)

    def test_errors(self):
        result = run_vault("formats", str(VAULT), "-f", "no-such-format")
        self.assertEqual(result.returncode, 2)
        self.assertIn("no format named 'no-such-format'", result.stderr)
        result = run_vault("formats", str(VAULT), "-f", "brain", "--type", "nope")
        self.assertEqual(result.returncode, 2)
        self.assertIn("has no type 'nope'", result.stderr)
        result = run_vault("formats", str(VAULT), "-f", "brain", "--note", "does-not-exist")
        self.assertEqual(result.returncode, 2)


if __name__ == "__main__":
    unittest.main()
