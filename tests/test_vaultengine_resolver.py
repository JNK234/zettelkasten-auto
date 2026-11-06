import unicodedata
import unittest
from pathlib import Path

from vaultengine import LinkGraph, Resolver, Vault, parse_note

FIXTURES = Path(__file__).parent / "fixtures"


def resolve_one(files, source, link_text, **options):
    """Resolve the single link in ``link_text`` as if it were written in ``source``."""
    vault = Vault.from_files({**files, source: link_text})
    note = vault.notes[source]
    links = note.links + note.embeds
    assert len(links) == 1, links
    return Resolver(vault, **options).resolve(source, links[0])


class ExplicitPathTests(unittest.TestCase):
    FILES = {
        "A/Note.md": "# Note",
        "B/Note.md": "# Note",
        "B/Sub/Deep.md": "# Deep",
        "Other.md": "# Other",
    }

    def check(self, link, expected_path, source="B/Source.md"):
        result = resolve_one(self.FILES, source, link)
        self.assertEqual((result.status, result.target_path), ("resolved", expected_path), result)
        self.assertEqual(result.matched_by, "path")

    def test_vault_path_with_or_without_extension(self):
        self.check("[[A/Note]]", "A/Note.md")
        self.check("[[A/Note.md]]", "A/Note.md")

    def test_path_is_case_insensitive(self):
        self.check("[[a/note]]", "A/Note.md")

    def test_leading_slash_is_vault_root(self):
        self.check("[[/A/Note]]", "A/Note.md")

    def test_dot_relative_paths_use_source_folder(self):
        self.check("[[./Sub/Deep]]", "B/Sub/Deep.md")
        self.check("[[../A/Note]]", "A/Note.md")
        self.check("[x](../Other.md)", "Other.md")

    def test_plain_path_falls_back_to_source_folder(self):
        self.check("[[Sub/Deep]]", "B/Sub/Deep.md")

    def test_explicit_path_disambiguates_duplicate_basenames(self):
        self.check("[[B/Note]]", "B/Note.md", source="Other2.md")

    def test_wrong_path_is_unresolved_without_basename_fallback(self):
        result = resolve_one(self.FILES, "Source.md", "[[Wrong/Other]]")
        self.assertEqual(result.status, "unresolved")

    def test_path_escaping_the_vault_is_unresolved(self):
        self.assertEqual(resolve_one(self.FILES, "Source.md", "[[../Other]]").status, "unresolved")


class BasenameTests(unittest.TestCase):
    def test_case_insensitive(self):
        result = resolve_one({"Folder/My Note.md": ""}, "Source.md", "[[my NOTE]]")
        self.assertEqual((result.status, result.target_path, result.matched_by), ("resolved", "Folder/My Note.md", "name"))

    def test_extension_may_be_written(self):
        self.assertEqual(resolve_one({"x/Note.md": ""}, "S.md", "[[Note.md]]").target_path, "x/Note.md")

    def test_unicode_normalisation(self):
        nfd_name = unicodedata.normalize("NFD", "Café.md")
        vault = Vault.from_files({nfd_name: "", "S.md": "[[Café]]"})
        edge = LinkGraph(vault).outgoing("S.md")[0]
        self.assertEqual(edge.status, "resolved")

    def test_duplicate_basename_is_ambiguous(self):
        result = resolve_one({"A/Index.md": "", "B/Index.md": ""}, "S.md", "[[Index]]")
        self.assertEqual(result.status, "ambiguous")
        self.assertIsNone(result.target_path)
        self.assertEqual(result.candidates, ("A/Index.md", "B/Index.md"))

    def test_same_name_differing_only_in_case_is_ambiguous(self):
        result = resolve_one({"A/note.md": "", "B/NOTE.md": ""}, "S.md", "[[Note]]")
        self.assertEqual(result.status, "ambiguous")

    def test_missing_note_is_kept_unresolved(self):
        result = resolve_one({"A.md": ""}, "S.md", "[[Nowhere|shown]]")
        self.assertEqual((result.status, result.target_path, result.candidates), ("unresolved", None, ()))
        self.assertEqual(result.link.target, "Nowhere")
        self.assertEqual(result.link.display, "shown")


class AliasTests(unittest.TestCase):
    PPO = {"Concepts/Proximal Policy Optimization.md": "---\naliases: [PPO, Proximal Policy Optimisation]\n---\n"}

    def test_alias_resolves(self):
        result = resolve_one(self.PPO, "S.md", "[[PPO]]")
        self.assertEqual((result.status, result.target_path, result.matched_by), ("resolved", "Concepts/Proximal Policy Optimization.md", "alias"))

    def test_alias_is_case_insensitive_and_keeps_display(self):
        result = resolve_one(self.PPO, "S.md", "[[ppo|the algorithm]]")
        self.assertEqual(result.matched_by, "alias")
        self.assertEqual(result.link.display, "the algorithm")

    def test_name_and_alias_of_same_file_is_not_ambiguous(self):
        files = {"PPO.md": "---\naliases: [PPO]\n---\n"}
        result = resolve_one(files, "S.md", "[[PPO]]")
        self.assertEqual((result.status, result.matched_by), ("resolved", "name"))

    def test_name_of_one_file_and_alias_of_another_is_ambiguous(self):
        files = {**self.PPO, "PPO.md": "# a different note"}
        result = resolve_one(files, "S.md", "[[PPO]]")
        self.assertEqual(result.status, "ambiguous")
        self.assertEqual(result.candidates, ("Concepts/Proximal Policy Optimization.md", "PPO.md"))

    def test_shared_alias_is_ambiguous(self):
        files = {"A.md": "---\naliases: [Shared]\n---\n", "B.md": "---\naliases: Shared\n---\n"}
        self.assertEqual(resolve_one(files, "S.md", "[[shared]]").status, "ambiguous")

    def test_alias_resolution_can_be_disabled(self):
        self.assertEqual(resolve_one(self.PPO, "S.md", "[[PPO]]", resolve_aliases=False).status, "unresolved")

    def test_alias_does_not_apply_to_explicit_paths(self):
        self.assertEqual(resolve_one(self.PPO, "S.md", "[[Concepts/PPO]]").status, "unresolved")


class SubpathTests(unittest.TestCase):
    TARGET = {
        "Target.md": (
            "# Title\n"
            "## Setup: the basics\n"
            "text ^Block-1\n"
            "### Details\n"
            "## Results\n"
            "```\n## Heading in code\n```\n"
            "A line `^not-a-block`\n"
        ),
        "pic.png": None,
    }

    def status(self, link_text, source="S.md"):
        result = resolve_one(self.TARGET, source, link_text)
        return result.status, result.subpath_status

    def test_no_subpath_has_no_subpath_status(self):
        self.assertEqual(self.status("[[Target]]"), ("resolved", None))

    def test_heading_found(self):
        self.assertEqual(self.status("[[Target#Results]]"), ("resolved", "found"))

    def test_heading_case_and_special_characters(self):
        self.assertEqual(self.status("[[Target#setup the basics]]"), ("resolved", "found"))
        self.assertEqual(self.status("[[Target#Setup: the basics]]"), ("resolved", "found"))

    def test_heading_missing_still_resolves_the_note(self):
        self.assertEqual(self.status("[[Target#Nope]]"), ("resolved", "missing"))

    def test_heading_inside_code_does_not_count(self):
        self.assertEqual(self.status("[[Target#Heading in code]]"), ("resolved", "missing"))

    def test_nested_heading_path(self):
        self.assertEqual(self.status("[[Target#Setup the basics#Details]]"), ("resolved", "found"))
        self.assertEqual(self.status("[[Target#Title#Details]]"), ("resolved", "found"))
        self.assertEqual(self.status("[[Target#Results#Details]]"), ("resolved", "missing"))
        # "Results" comes after "Setup" but is a sibling, not inside its section.
        self.assertEqual(self.status("[[Target#Setup the basics#Results]]"), ("resolved", "missing"))
        self.assertEqual(self.status("[[Target#Details#Setup the basics]]"), ("resolved", "missing"))

    def test_block(self):
        self.assertEqual(self.status("[[Target#^block-1]]"), ("resolved", "found"))
        self.assertEqual(self.status("[[Target#^nope]]"), ("resolved", "missing"))
        self.assertEqual(self.status("[[Target#^not-a-block]]"), ("resolved", "missing"))

    def test_markdown_link_subpath(self):
        self.assertEqual(self.status("[r](Target.md#Results)"), ("resolved", "found"))

    def test_same_note_heading_and_block(self):
        vault = Vault.from_files({"S.md": "# Here\npara ^mine\n[[#Here]] [[#There]] [[#^mine]] [[#^gone]]"})
        edges = LinkGraph(vault).outgoing("S.md")
        self.assertEqual(
            [(e.status, e.target_path, e.matched_by, e.subpath_status) for e in edges],
            [("resolved", "S.md", "self", "found"), ("resolved", "S.md", "self", "missing"),
             ("resolved", "S.md", "self", "found"), ("resolved", "S.md", "self", "missing")],
        )

    def test_subpath_not_checked_when_note_not_resolved_or_not_markdown(self):
        self.assertEqual(self.status("[[Missing#Heading]]"), ("unresolved", "not_checked"))
        self.assertEqual(self.status("![[pic.png#anything]]"), ("resolved", "not_checked"))
        result = resolve_one({"A/X.md": "# H", "B/X.md": "# H"}, "S.md", "[[X#H]]")
        self.assertEqual((result.status, result.subpath_status), ("ambiguous", "not_checked"))


class AttachmentTests(unittest.TestCase):
    FILES = {
        "attachments/diagram.png": None,
        "attachments/paper.pdf": None,
        "Drawings/Board 2025.excalidraw.md": "",
        "Notes/figs/local.jpg": None,
    }

    def check(self, link_text, expected, source="Notes/S.md"):
        result = resolve_one(self.FILES, source, link_text)
        self.assertEqual((result.status, result.target_path), ("resolved", expected), result)
        return result

    def test_image_embed_by_name(self):
        self.assertTrue(self.check("![[diagram.png]]", "attachments/diagram.png").link.is_embed)

    def test_embed_with_size(self):
        self.check("![[DIAGRAM.PNG|300]]", "attachments/diagram.png")

    def test_pdf_link(self):
        self.check("[[paper.pdf#page=3]]", "attachments/paper.pdf")

    def test_excalidraw_drawing_stored_as_markdown(self):
        self.check("![[Board 2025.excalidraw]]", "Drawings/Board 2025.excalidraw.md")

    def test_markdown_image_relative_path(self):
        self.check("![fig](figs/local.jpg)", "Notes/figs/local.jpg")

    def test_non_markdown_needs_its_extension(self):
        self.assertEqual(resolve_one(self.FILES, "S.md", "![[diagram]]").status, "unresolved")

    def test_missing_attachment_is_unresolved(self):
        self.assertEqual(resolve_one(self.FILES, "S.md", "![[nope.png]]").status, "unresolved")


class GraphTests(unittest.TestCase):
    def setUp(self):
        self.vault = Vault.from_files({
            "Hub.md": "# Hub\n[[#Hub]] [[Hub]]",
            "A.md": "Tags: [[Hub]] [[ghost-tag]]\n\n[[Hub]] ![[img.png]] [[Missing]] [[Dup]]",
            "B.md": "[[hub#Hub]]",
            "x/Dup.md": "",
            "y/Dup.md": "",
            "img.png": None,
        })
        self.graph = LinkGraph(self.vault)

    def test_every_link_and_embed_is_an_edge(self):
        self.assertEqual(len(self.graph.edges), 2 + 6 + 1)

    def test_backlinks_exclude_self_links(self):
        self.assertEqual(self.graph.backlink_sources("Hub.md"), ["A.md", "B.md"])
        self.assertEqual(len(self.graph.backlinks("Hub.md")), 3)

    def test_embeds_create_backlinks_to_attachments(self):
        self.assertEqual(self.graph.backlink_sources("img.png"), ["A.md"])

    def test_ambiguous_and_unresolved_create_no_backlinks(self):
        self.assertEqual(self.graph.backlinks("x/Dup.md"), [])
        self.assertEqual([e.link.target for e in self.graph.ambiguous()], ["Dup"])

    def test_obsidian_shaped_summaries(self):
        self.assertEqual(self.graph.resolved_links["A.md"], {"Hub.md": 2, "img.png": 1})
        self.assertEqual(self.graph.resolved_links["Hub.md"], {"Hub.md": 1})  # [[Hub]] counts, [[#Hub]] does not
        self.assertEqual(self.graph.unresolved_links, {"A.md": {"ghost-tag": 1, "Missing": 1}})

    def test_tags_line_wikilinks_are_edges(self):
        first_line = [e for e in self.graph.outgoing("A.md") if e.link.line == 1]
        self.assertEqual([(e.link.target, e.status) for e in first_line], [("Hub", "resolved"), ("ghost-tag", "unresolved")])

    def test_missing_subpaths(self):
        graph = LinkGraph(Vault.from_files({"T.md": "# Real", "S.md": "[[T#Real]] [[T#Fake]]"}))
        self.assertEqual([e.link.raw for e in graph.missing_subpaths()], ["[[T#Fake]]"])


class SampleResolverVaultTests(unittest.TestCase):
    """The hand-made vault used for the alias / heading / missing-note samples."""

    @classmethod
    def setUpClass(cls):
        cls.vault = Vault.load(FIXTURES / "resolver_vault")
        cls.graph = LinkGraph(cls.vault)

    def test_hidden_folders_are_skipped_and_attachments_indexed(self):
        self.assertNotIn(".obsidian/app.json", self.vault.files)
        self.assertIn("attachments/diagram.png", self.vault.files)

    def test_alias_note(self):
        edges = self.graph.outgoing("Notes/Training log.md")
        self.assertEqual([e.matched_by for e in edges], ["alias", "alias", "alias", "name"])
        self.assertEqual({e.target_path for e in edges}, {"Concepts/Proximal Policy Optimization.md"})

    def test_heading_note(self):
        edges = self.graph.outgoing("Notes/Reading notes.md")
        self.assertTrue(all(e.status == "resolved" for e in edges))
        self.assertEqual(
            [e.subpath_status for e in edges],
            ["found", "found", "found", "missing", "found", "found", "found", None, "not_checked"],
        )

    def test_missing_note(self):
        edges = self.graph.outgoing("Notes/Open questions.md")
        self.assertEqual(
            [(e.link.target, e.status) for e in edges],
            [("reinforcement-learning", "unresolved"), ("PPO", "resolved"), ("GRPO", "unresolved"),
             ("Concepts/Trust Region Methods", "unresolved"), ("Index", "ambiguous"), ("missing-figure.png", "unresolved")],
        )

    def test_backlinks(self):
        self.assertEqual(
            self.graph.backlink_sources("Concepts/Proximal Policy Optimization.md"),
            ["Notes/Open questions.md", "Notes/Reading notes.md", "Notes/Training log.md"],
        )


class RealVaultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vault = Vault.load(FIXTURES / "vault")
        cls.graph = LinkGraph(cls.vault)

    def test_every_parsed_link_becomes_an_edge(self):
        parsed = sum(len(n.links) + len(n.embeds) for n in self.vault.notes.values())
        self.assertEqual(len(self.graph.edges), parsed)
        self.assertEqual(parsed, 1787)

    def test_no_ambiguous_links_or_missing_subpaths(self):
        self.assertEqual(self.graph.ambiguous(), [])
        self.assertEqual(self.graph.missing_subpaths(), [])

    def test_kl_penalty_links(self):
        edges = self.graph.outgoing("20_ZETTLEKASTEN/kl-penalty-in-rlhf.md")
        resolved = [e.target_path for e in edges if e.status == "resolved"]
        self.assertIn("20_ZETTLEKASTEN/reinforcement-learning-from-human-feedback-rlhf.md", resolved)  # from the Tags: line
        self.assertIn("20_ZETTLEKASTEN/direct-preference-optimization-dpo.md", resolved)
        self.assertIn("policy-gradient-methods", self.graph.unresolved_links["20_ZETTLEKASTEN/kl-penalty-in-rlhf.md"])
        self.assertEqual(
            self.graph.backlink_sources("20_ZETTLEKASTEN/kl-penalty-in-rlhf.md"),
            ["20_ZETTLEKASTEN/the-environment-not-flops-bottlenecks-agentic-rl.md"],
        )

    def test_explicit_path_to_missing_file_is_unresolved(self):
        edges = [e for e in self.graph.outgoing("10_SOURCES/RLHF.md") if e.link.target == "20_ZETTLEKASTEN/Reinforcement Learning"]
        self.assertEqual([e.status for e in edges], ["unresolved"])

    def test_excalidraw_embeds_without_drawing_files_stay_unresolved(self):
        graph = LinkGraph(Vault.load(FIXTURES / "parser_cases"))
        edges = graph.outgoing("Notes/Board embed.md")
        self.assertEqual([(e.link.is_embed, e.status) for e in edges if e.link.is_embed], [(True, "unresolved")])


if __name__ == "__main__":
    unittest.main()
