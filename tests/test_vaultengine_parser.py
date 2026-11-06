import subprocess
import sys
import unittest
from pathlib import Path

from vaultengine import parse_file, parse_note

VAULT = Path(__file__).parent / "fixtures" / "vault"
CASES = Path(__file__).parent / "fixtures" / "parser_cases"
REPO_ROOT = Path(__file__).parent.parent


def targets(links):
    return [link.target for link in links]


class FrontmatterTests(unittest.TestCase):
    def test_list_tags_and_aliases(self):
        note = parse_note("---\naliases: [PPO, Proximal Policy Optimisation]\ntags: [rl, ml/optim]\n---\nBody\n")
        self.assertTrue(note.has_frontmatter)
        self.assertEqual(note.aliases, ["PPO", "Proximal Policy Optimisation"])
        self.assertEqual(note.tag_names, ["rl", "ml/optim"])
        self.assertTrue(all(tag.source == "frontmatter" for tag in note.tags))
        self.assertEqual(note.body_start_line, 5)

    def test_block_list_form_and_hash_prefix(self):
        note = parse_note('---\ntags:\n  - "#alpha"\n  - beta/gamma\naliases:\n  - One\n---\n')
        self.assertEqual(note.tag_names, ["alpha", "beta/gamma"])
        self.assertEqual(note.aliases, ["One"])

    def test_string_values_are_split(self):
        note = parse_note("---\ntags: alpha, beta gamma\naliases: First, Second name\n---\n")
        self.assertEqual(note.tag_names, ["alpha", "beta", "gamma"])
        self.assertEqual(note.aliases, ["First", "Second name"])

    def test_property_names_are_case_insensitive(self):
        note = parse_note("---\nTags: [x1]\nAliases: [Y]\n---\n")
        self.assertEqual(note.tag_names, ["x1"])
        self.assertEqual(note.aliases, ["Y"])

    def test_numeric_frontmatter_tag_is_rejected(self):
        note = parse_note("---\ntags: [2024, y2024]\n---\n")
        self.assertEqual(note.tag_names, ["y2024"])

    def test_invalid_yaml_is_reported_and_hidden_from_body(self):
        note = parse_note("---\ntags: [unclosed\n---\n#real-tag\n")
        self.assertTrue(note.has_frontmatter)
        self.assertIsNotNone(note.frontmatter_error)
        self.assertEqual(note.frontmatter, {})
        self.assertEqual(note.tag_names, ["real-tag"])

    def test_non_mapping_yaml_is_an_error(self):
        note = parse_note("---\n- a\n- b\n---\n")
        self.assertIn("not a mapping", note.frontmatter_error)

    def test_frontmatter_must_start_on_first_line(self):
        note = parse_note("\n---\ntags: [x]\n---\n")
        self.assertFalse(note.has_frontmatter)
        self.assertEqual(note.tag_names, [])

    def test_unclosed_frontmatter_is_body(self):
        note = parse_note("---\ntitle: x\n#tag-in-body\n")
        self.assertFalse(note.has_frontmatter)
        self.assertEqual(note.tag_names, ["tag-in-body"])

    def test_frontmatter_content_is_not_scanned_as_body(self):
        note = parse_note("---\nsource: \"[[Not A Body Link]] #nottag\"\n---\n# Title\n")
        self.assertEqual(note.links, [])
        self.assertEqual(note.tag_names, [])
        self.assertEqual([(h.text, h.line) for h in note.headings], [("Title", 4)])

    def test_crlf_and_bom(self):
        note = parse_note("\ufeff---\r\ntags: [a]\r\n---\r\n# Head\r\n[[Link]] #b\r\n")
        self.assertTrue(note.has_frontmatter)
        self.assertEqual(note.tag_names, ["a", "b"])
        self.assertEqual([(h.text, h.line) for h in note.headings], [("Head", 4)])
        self.assertEqual([(link.target, link.line) for link in note.links], [("Link", 5)])


class TagTests(unittest.TestCase):
    def tags(self, text):
        return [tag.name for tag in parse_note(text).tags]

    def test_inline_and_nested(self):
        self.assertEqual(self.tags("Status: #baby #ai-safety and #area/sub/leaf"), ["baby", "ai-safety", "area/sub/leaf"])

    def test_purely_numeric_is_not_a_tag(self):
        self.assertEqual(self.tags("issue #1984 and #4512. but #y1984"), ["y1984"])

    def test_punctuation_ends_a_tag(self):
        self.assertEqual(self.tags("see #alpha), #beta. #gamma! #delta's"), ["alpha", "beta", "gamma", "delta"])

    def test_hash_inside_word_or_run_is_not_a_tag(self):
        self.assertEqual(self.tags("C#sharp emit ###STOP### here"), [])

    def test_heading_marker_is_not_a_tag_but_tag_in_heading_is(self):
        self.assertEqual(self.tags("# Heading #intro\n#start-of-line"), ["intro", "start-of-line"])

    def test_unicode_letters(self):
        self.assertEqual(self.tags("#café #日本語"), ["café", "日本語"])

    def test_hash_in_link_targets_and_urls_is_not_a_tag(self):
        text = "[[Note #weird]] [x](<https://a.com/ #frag>) [y](<Note.md #sub>)"
        self.assertEqual(self.tags(text), [])

    def test_tag_names_are_case_insensitively_distinct(self):
        note = parse_note("#Alpha #alpha #beta")
        self.assertEqual(note.tag_names, ["Alpha", "beta"])
        self.assertEqual(len(note.tags), 3)


class WikilinkTests(unittest.TestCase):
    def one(self, text):
        note = parse_note(text)
        self.assertEqual(len(note.links) + len(note.embeds), 1, note)
        return (note.links or note.embeds)[0]

    def test_plain(self):
        link = self.one("See [[Some Note]].")
        self.assertEqual((link.target, link.display, link.heading_path, link.block_id), ("Some Note", None, (), None))
        self.assertEqual(link.syntax, "wikilink")

    def test_alias(self):
        link = self.one("[[Some Note|the note]]")
        self.assertEqual((link.target, link.display), ("Some Note", "the note"))

    def test_heading(self):
        link = self.one("[[Some Note#Section Two|alias]]")
        self.assertEqual((link.target, link.heading_path, link.display), ("Some Note", ("Section Two",), "alias"))

    def test_nested_heading(self):
        self.assertEqual(self.one("[[Note#Part A#Sub B]]").heading_path, ("Part A", "Sub B"))

    def test_block(self):
        link = self.one("[[Note#^abc-123]]")
        self.assertEqual((link.target, link.block_id, link.heading_path), ("Note", "abc-123", ()))

    def test_same_note_heading_and_block(self):
        heading = self.one("[[#Local Heading]]")
        self.assertTrue(heading.is_same_note)
        self.assertEqual(heading.heading_path, ("Local Heading",))
        block = self.one("[[#^blk]]")
        self.assertTrue(block.is_same_note)
        self.assertEqual(block.block_id, "blk")

    def test_escaped_pipe_in_table(self):
        link = self.one("| col | [[Note\\|Alias]] |")
        self.assertEqual((link.target, link.display), ("Note", "Alias"))

    def test_path_and_extension_are_kept_as_written(self):
        self.assertEqual(self.one("[[folder/Note.md]]").target, "folder/Note.md")

    def test_empty_is_not_a_link(self):
        note = parse_note("[[]] [[ ]] [[#]] [[|x]]")
        self.assertEqual(note.links, [])

    def test_links_do_not_span_lines(self):
        self.assertEqual(parse_note("[[broken\nlink]]").links, [])

    def test_line_numbers(self):
        note = parse_note("a\nb [[One]]\n\nc [[Two]]")
        self.assertEqual([(link.target, link.line) for link in note.links], [("One", 2), ("Two", 4)])


class EmbedTests(unittest.TestCase):
    def test_wikilink_embeds(self):
        note = parse_note("![[diagram.png|300]] ![[Note#Part]] ![[Note#^blk]]")
        self.assertEqual(note.links, [])
        self.assertEqual(targets(note.embeds), ["diagram.png", "Note", "Note"])
        self.assertEqual(note.embeds[0].display, "300")
        self.assertEqual(note.embeds[1].heading_path, ("Part",))
        self.assertEqual(note.embeds[2].block_id, "blk")

    def test_markdown_image_embed(self):
        note = parse_note("![alt text](images/pic%201.png)")
        self.assertEqual(targets(note.embeds), ["images/pic 1.png"])
        self.assertEqual(note.embeds[0].syntax, "markdown")


class MarkdownLinkTests(unittest.TestCase):
    def test_internal_with_encoding_and_subpath(self):
        note = parse_note("[text](Some%20Note.md#Heading%20Two)")
        link = note.links[0]
        self.assertEqual((link.target, link.heading_path, link.display), ("Some Note.md", ("Heading Two",), "text"))
        self.assertEqual(link.syntax, "markdown")

    def test_angle_brackets_and_title(self):
        note = parse_note('[a](<My Note.md#^blk> "title")')
        self.assertEqual((note.links[0].target, note.links[0].block_id), ("My Note.md", "blk"))

    def test_same_note_anchor(self):
        link = parse_note("[jump](#Local%20Heading)").links[0]
        self.assertTrue(link.is_same_note)
        self.assertEqual(link.heading_path, ("Local Heading",))

    def test_external_links_are_separate(self):
        note = parse_note("[site](https://example.com/a_(b)#frag) [mail](mailto:x@y.z) ![img](https://x/y.png)")
        self.assertEqual(note.links, [])
        self.assertEqual(note.embeds, [])
        self.assertEqual([e.url for e in note.external_links], ["https://example.com/a_(b)#frag", "mailto:x@y.z", "https://x/y.png"])
        self.assertTrue(note.external_links[2].is_embed)

    def test_bracketed_link_text(self):
        note = parse_note("[[2305.18290] Paper title](https://arxiv.org/abs/2305.18290)")
        self.assertEqual(note.links, [])
        self.assertEqual(note.external_links[0].display, "[2305.18290] Paper title")

    def test_escaped_bracket_is_not_a_link(self):
        self.assertEqual(parse_note("\\[not](link.md)").links, [])


class HeadingTests(unittest.TestCase):
    def headings(self, text):
        return [(h.level, h.text, h.line) for h in parse_note(text).headings]

    def test_levels_and_closing_hashes(self):
        text = "# One\n## Two ##\n###### Six\n####### Seven\n#NoSpace\n    # indented code-ish"
        self.assertEqual(self.headings(text), [(1, "One", 1), (2, "Two", 2), (6, "Six", 3)])

    def test_inline_code_kept_in_heading_text(self):
        self.assertEqual(self.headings("## Using `foo()` here"), [(2, "Using `foo()` here", 1)])

    def test_empty_heading_is_ignored(self):
        self.assertEqual(self.headings("#\n# \n## ##"), [])


class BlockIdTests(unittest.TestCase):
    def ids(self, text):
        return [(b.id, b.line) for b in parse_note(text).block_ids]

    def test_end_of_line_and_own_line(self):
        self.assertEqual(self.ids("A paragraph. ^para-1\n\n| t |\n^table-id\n"), [("para-1", 1), ("table-id", 4)])

    def test_not_block_ids(self):
        self.assertEqual(self.ids("x^2 here\nfootnote[^1]\n^ spaced\nmid ^id text\nbad ^under_score"), [])


class CodeIsIgnoredTests(unittest.TestCase):
    DECOYS = "[[Hidden]] ![[HiddenEmbed]] [md](hidden.md) #hidden ^hiddenblock"

    def assertNothingFound(self, text):
        note = parse_note(text)
        self.assertEqual((note.links, note.embeds, note.tags, note.block_ids, note.headings), ([], [], [], [], []), note)

    def test_backtick_fence(self):
        self.assertNothingFound(f"```python\n# comment not heading\n{self.DECOYS}\n```\n")

    def test_tilde_fence(self):
        self.assertNothingFound(f"~~~\n{self.DECOYS}\n~~~\n")

    def test_longer_fence_is_not_closed_by_shorter(self):
        self.assertNothingFound(f"````\n```\n{self.DECOYS}\n```\n````\n")

    def test_indented_fence_in_list_item(self):
        note = parse_note(f"- item [[Visible]]\n  ```\n  {self.DECOYS}\n  ```\n- after #visible\n")
        self.assertEqual(targets(note.links), ["Visible"])
        self.assertEqual(note.tag_names, ["visible"])

    def test_fence_in_blockquote(self):
        self.assertNothingFound(f"> ```\n> {self.DECOYS}\n> ```\n")

    def test_unclosed_fence_runs_to_end(self):
        self.assertNothingFound(f"```\n{self.DECOYS}\n")

    def test_inline_code(self):
        self.assertNothingFound(f"Text `{self.DECOYS}` and ``a ` {self.DECOYS}``")

    def test_unmatched_backtick_is_literal(self):
        note = parse_note("a ` stray [[Visible]] #seen")
        self.assertEqual(targets(note.links), ["Visible"])
        self.assertEqual(note.tag_names, ["seen"])

    def test_inline_code_does_not_cross_a_blank_line(self):
        note = parse_note("open ` here\n\n[[Visible]] ` close")
        self.assertEqual(targets(note.links), ["Visible"])

    def test_obsidian_comments(self):
        self.assertNothingFound(f"%% {self.DECOYS} %%\n%%\nmulti\n{self.DECOYS}\n%%")

    def test_four_space_indent_in_list_is_not_code(self):
        note = parse_note("- item\n\n    continued [[Visible]] #seen\n")
        self.assertEqual(targets(note.links), ["Visible"])
        self.assertEqual(note.tag_names, ["seen"])


class FixtureVaultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notes = {path.relative_to(VAULT).as_posix(): parse_file(path, VAULT) for path in sorted(VAULT.rglob("*.md"))}

    def note(self, fragment):
        matches = [note for name, note in self.notes.items() if fragment in name]
        self.assertEqual(len(matches), 1, fragment)
        return matches[0]

    def test_whole_vault_parses(self):
        self.assertEqual(len(self.notes), 180)
        self.assertTrue(all(note.frontmatter_error is None for note in self.notes.values()))
        self.assertEqual(sum(len(n.links) for n in self.notes.values()), 1787)
        self.assertEqual(sum(len(n.embeds) for n in self.notes.values()), 0)

    def test_no_numeric_tags_anywhere(self):
        for note in self.notes.values():
            self.assertFalse(any(tag.name.isdigit() for tag in note.tags), note.path)

    def test_list_continuation_link_is_kept(self):
        urls = [link.url for link in self.note("10_SOURCES/RLHF.md").external_links]
        self.assertTrue(any("rlhf-without-rl" in url for url in urls))


class ParserCaseTests(unittest.TestCase):
    """Edge cases on a small synthetic vault (tests/fixtures/parser_cases)."""

    @classmethod
    def setUpClass(cls):
        cls.notes = {path.stem: parse_file(path, CASES) for path in sorted(CASES.rglob("*.md"))}

    def test_frontmatter_note(self):
        note = self.notes["Frontmatter note"]
        self.assertTrue(note.has_frontmatter)
        self.assertEqual(note.frontmatter["status"], "complete")
        self.assertEqual([t.name for t in note.tags if t.source == "frontmatter"], ["parsing", "fixtures", "frontmatter"])
        self.assertEqual(note.body_start_line, 6)
        self.assertEqual((note.headings[0].text, note.headings[0].line), ("Frontmatter note", 7))
        # The later '---' is a horizontal rule, not a second frontmatter block.
        self.assertIn("Zettel with tags line", targets(note.links))

    def test_hash_comment_in_fenced_code_is_not_a_tag(self):
        self.assertNotIn("Unzip", self.notes["Fenced code"].tag_names)

    def test_wikilink_shaped_type_hint_in_inline_code_is_not_a_link(self):
        self.assertNotIn("list, list", targets(self.notes["Fenced code"].links))

    def test_python_comments_in_fences_are_not_headings(self):
        texts = [h.text for h in self.notes["Fenced code"].headings]
        self.assertNotIn("configs/training/run.yaml", texts)
        self.assertIn("6. Implementation Plan", texts)

    def test_issue_numbers_are_not_tags(self):
        self.assertEqual(self.notes["Issue numbers"].tag_names, ["baby", "triage"])

    def test_excalidraw_embeds(self):
        self.assertEqual(targets(self.notes["Board embed"].embeds), ["Board 2026-01-01 10.00.00.excalidraw"])

    def test_zettel_tags_line_is_wikilinks_and_status_is_a_tag(self):
        note = self.notes["Zettel with tags line"]
        self.assertEqual(note.tag_names, ["baby"])
        self.assertEqual(targets(note.links)[:2], ["reinforcement-learning", "policy-gradient-methods"])


class IsolationTests(unittest.TestCase):
    def test_vaultengine_imports_no_llm_or_db_code(self):
        # Diff sys.modules around the import: some site .pth hooks preload
        # namespace stubs (e.g. 'google') before any project code runs.
        code = (
            "import sys\n"
            "before = set(sys.modules)\n"
            "import vaultengine, vaultengine.parser, vaultengine.index, vaultengine.queries, vaultengine.cli\n"
            "import vaultengine.service, vaultengine.rpc, vaultengine.rpc.schema\n"
            "added = set(sys.modules) - before\n"
            "bad = [m for m in added if m.split('.')[0] in "
            "{'src', 'openai', 'anthropic', 'google', 'chromadb', 'sentence_transformers', 'torch', 'requests', 'httpx', 'dotenv'}]\n"
            "print(','.join(sorted(bad)))\n"
        )
        result = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
        self.assertEqual(result.stdout.strip(), "")


if __name__ == "__main__":
    unittest.main()
