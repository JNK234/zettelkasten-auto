import io
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from src import main
from src.db import get_client, get_collection, get_collection_name, index_zettel


TEMPLATE_CONTENT = """Created: {{date}} {{time}}

Tags:

# {{Title}}

# See Also

# References
"""


class FakeEmbeddingFunction:
    def __call__(self, input):
        embeddings = []
        for text in input:
            lowered = text.lower()
            embeddings.append(
                [
                    float(lowered.count("alpha")),
                    float(lowered.count("beta")),
                    float(lowered.count("gamma")),
                    float(len(text)) / 1000.0,
                ]
            )
        return embeddings


class SequenceProvider:
    def __init__(self, responses):
        self.responses = list(responses)

    def extract(self, prompt, system_prompt):
        response = self.responses.pop(0)
        if callable(response):
            return response(prompt, system_prompt)
        return response


def make_concept(title, content, tags):
    return {
        "title": title,
        "content": content,
        "suggested_tags": tags,
        "concept_type": "mechanism",
        "abstraction_level": "general-principle",
    }


class IntegrationTests(unittest.TestCase):
    def make_workspace(self):
        tempdir = tempfile.TemporaryDirectory()
        root = Path(tempdir.name)
        sources_dir = root / "sources"
        zettel_dir = root / "zettels"
        bundles_dir = root / "bundles"
        template_path = root / "template.md"
        db_path = root / "chromadb"

        sources_dir.mkdir()
        zettel_dir.mkdir()
        bundles_dir.mkdir()
        template_path.write_text(TEMPLATE_CONTENT, encoding="utf-8")

        config = {
            "vault": {
                "root": str(root),
                "sources_dir": "sources",
                "zettel_dir": "zettels",
                "bundles_dir": "bundles",
                "template_path": "template.md",
            },
            "llm": {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "base_url": "http://localhost:11434",
            },
            "embeddings": {
                "db_path": "chromadb",
                "provider": "sentence-transformer",
                "model": "all-MiniLM-L6-v2",
                "top_k": 5,
                "max_distance": 0.5,
            },
            "bundle": {
                "default_depth": 1,
                "default_top_k": 5,
            },
            "processing": {
                "min_source_chars": 20,
                "duplicate_max_distance": 0.15,
                "chunk_target_chars": 120,
                "chunk_max_chars": 160,
            },
        }
        return tempdir, config, sources_dir, zettel_dir, bundles_dir, template_path, db_path

    def run_main(self, argv, config, provider, extra_patchers=None):
        stdout = io.StringIO()
        with ExitStack() as stack:
            stack.enter_context(patch("src.main.load_config", return_value=config))
            stack.enter_context(patch("src.main.get_embedding_function", return_value=FakeEmbeddingFunction()))
            stack.enter_context(patch("src.db.client.get_embedding_function", return_value=FakeEmbeddingFunction()))
            stack.enter_context(patch("src.llm.extraction.get_llm_provider", return_value=provider))
            for patcher in extra_patchers or []:
                stack.enter_context(patcher)
            stack.enter_context(patch("sys.argv", ["main.py"] + argv))
            with redirect_stdout(stdout):
                try:
                    main.main()
                except SystemExit as exc:
                    return stdout.getvalue(), exc.code
        return stdout.getvalue(), 0

    def test_process_dry_run_is_read_only(self):
        tempdir, config, sources_dir, zettel_dir, _, _, db_path = self.make_workspace()
        try:
            source_path = sources_dir / "source-a.md"
            source_path.write_text("Alpha systems explain alpha feedback loops in detail.", encoding="utf-8")
            provider = SequenceProvider([
                {"concepts": [make_concept("Alpha Concept", "Alpha concept content alpha.", ["Alpha Tag", "Core Idea"])]}
            ])

            output, exit_code = self.run_main(["process", "--dry-run"], config, provider)
            self.assertEqual(exit_code, 0)
            self.assertIn("created: 1", output.lower())
            self.assertEqual(list(zettel_dir.glob("*.md")), [])
            self.assertNotIn("#processed", source_path.read_text(encoding="utf-8"))

            client = get_client(str(db_path))
            collection = get_collection(
                client,
                config["embeddings"]["provider"],
                config["embeddings"]["model"],
                create=False,
            )
            self.assertIsNone(collection)
        finally:
            tempdir.cleanup()

    def test_embedding_provider_and_model_use_distinct_collections(self):
        tempdir, config, _, _, _, _, db_path = self.make_workspace()
        try:
            with patch("src.db.client.get_embedding_function", return_value=FakeEmbeddingFunction()):
                client = get_client(str(db_path))
                index_zettel(client, "note-one", "alpha alpha", "sentence-transformer", "model-a", force=True)
                index_zettel(client, "note-two", "beta beta", "openai", "model-b", force=True)

                sentence_collection = get_collection(client, "sentence-transformer", "model-a", create=False)
                openai_collection = get_collection(client, "openai", "model-b", create=False)

                self.assertIsNotNone(sentence_collection)
                self.assertIsNotNone(openai_collection)
                self.assertNotEqual(
                    get_collection_name("sentence-transformer", "model-a"),
                    get_collection_name("openai", "model-b"),
                )
                self.assertEqual(sentence_collection.get(include=[])["ids"], ["note-one"])
                self.assertEqual(openai_collection.get(include=[])["ids"], ["note-two"])
        finally:
            tempdir.cleanup()

    def test_bundle_dry_run_requires_fresh_index(self):
        tempdir, config, sources_dir, zettel_dir, _, _, _ = self.make_workspace()
        try:
            (sources_dir / "source-a.md").write_text("Alpha source content for bundling.", encoding="utf-8")
            (zettel_dir / "existing-note.md").write_text(
                "Tags: [[alpha-tag]]\n\n# Existing Note\n\nAlpha body.\n\n# References\n\n1. [[source-a]]\n",
                encoding="utf-8",
            )
            provider = SequenceProvider([])

            output, exit_code = self.run_main(["bundle", "--source", "source-a", "--dry-run"], config, provider)
            self.assertEqual(exit_code, 1)
            self.assertIn("active index is stale", output.lower())
        finally:
            tempdir.cleanup()

    def test_process_chunks_and_dedupes_long_sources(self):
        tempdir, config, sources_dir, zettel_dir, _, _, _ = self.make_workspace()
        try:
            config["processing"]["chunk_target_chars"] = 60
            config["processing"]["chunk_max_chars"] = 80
            source_path = sources_dir / "source-a.md"
            source_path.write_text(
                "# Section One\n\nAlpha systems explain alpha loops in detail with reusable mechanism notes.\n\n"
                "# Section Two\n\nAlpha systems explain alpha loops in another way with additional reusable mechanism notes.\n",
                encoding="utf-8",
            )
            provider = SequenceProvider(
                [{"concepts": [make_concept("Alpha Concept", "Alpha content alpha.", ["Alpha Tag", "Systems Thinking"]) ]}]
                + [{"concepts": [make_concept("Alpha Concept", "Alpha content alpha repeated.", ["Alpha Tag", "Systems Thinking"])]}] * 9
            )

            output, exit_code = self.run_main(["process"], config, provider)
            self.assertEqual(exit_code, 0)
            self.assertEqual(len(list(zettel_dir.glob("*.md"))), 1)
            self.assertIn("after dedupe", output)
            self.assertIn("Status: #processed", source_path.read_text(encoding="utf-8"))
        finally:
            tempdir.cleanup()

    def test_invalid_provider_output_is_reported_and_not_marked_processed(self):
        tempdir, config, sources_dir, zettel_dir, _, _, _ = self.make_workspace()
        try:
            source_path = sources_dir / "source-a.md"
            source_path.write_text("Alpha content with enough detail to process.", encoding="utf-8")
            provider = SequenceProvider([{"not_concepts": []}])

            output, exit_code = self.run_main(["process"], config, provider)
            self.assertEqual(exit_code, 0)
            self.assertIn("invalid_output: 1", output)
            self.assertEqual(list(zettel_dir.glob("*.md")), [])
            self.assertNotIn("#processed", source_path.read_text(encoding="utf-8"))
        finally:
            tempdir.cleanup()

    def test_new_notes_separate_tags_and_see_also_and_bundle_follows_tags_only(self):
        tempdir, config, sources_dir, zettel_dir, _, _, db_path = self.make_workspace()
        try:
            config["processing"]["duplicate_max_distance"] = 0.01
            related_note = zettel_dir / "related-idea.md"
            related_note.write_text(
                "Tags: [[related-tag]]\n\n# Related Idea\n\nAlpha reference beta.\n\n# See Also\n\n- [[next-sim]]\n\n# References\n\n1. [[seed-source]]\n",
                encoding="utf-8",
            )
            alpha_tag = zettel_dir / "alpha-tag.md"
            alpha_tag.write_text(
                "Tags: [[deep-tag]]\n\n# Alpha Tag\n\nBridge note body.\n\n# References\n\n1. [[seed-source]]\n",
                encoding="utf-8",
            )
            deep_tag = zettel_dir / "deep-tag.md"
            deep_tag.write_text(
                "Tags:\n\n# Deep Tag\n\nDeeper body.\n\n# References\n\n1. [[seed-source]]\n",
                encoding="utf-8",
            )
            next_sim = zettel_dir / "next-sim.md"
            next_sim.write_text(
                "Tags:\n\n# Next Sim\n\nUnrelated note body.\n\n# References\n\n1. [[seed-source]]\n",
                encoding="utf-8",
            )

            with patch("src.db.client.get_embedding_function", return_value=FakeEmbeddingFunction()):
                client = get_client(str(db_path))
                index_zettel(client, "related-idea", related_note.read_text(encoding="utf-8"), "sentence-transformer", "all-MiniLM-L6-v2", force=True)
                index_zettel(client, "alpha-tag", alpha_tag.read_text(encoding="utf-8"), "sentence-transformer", "all-MiniLM-L6-v2", force=True)
                index_zettel(client, "deep-tag", deep_tag.read_text(encoding="utf-8"), "sentence-transformer", "all-MiniLM-L6-v2", force=True)
                index_zettel(client, "next-sim", next_sim.read_text(encoding="utf-8"), "sentence-transformer", "all-MiniLM-L6-v2", force=True)

            source_path = sources_dir / "source-a.md"
            source_path.write_text("Alpha concept source with strong alpha and beta overlap.", encoding="utf-8")
            provider = SequenceProvider([
                {"concepts": [make_concept("Core Idea", "Alpha concept source with strong alpha and beta overlap.", ["Alpha Tag", "Topic Label"])]}
            ])

            output, exit_code = self.run_main(
                ["process"],
                config,
                provider,
                extra_patchers=[
                    patch("src.main.find_similar", return_value=["related-idea"]),
                    patch("src.main.zettel_exists", return_value=(False, None)),
                ],
            )
            self.assertEqual(exit_code, 0)

            created_path = zettel_dir / "core-idea.md"
            created_note = created_path.read_text(encoding="utf-8")
            self.assertIn("Tags: [[alpha-tag]] [[topic-label]]", created_note)
            self.assertIn("- [[related-idea]]", created_note)
            self.assertNotIn("[[related-idea]] [[topic-label]]", created_note)

            bundle_output, bundle_exit = self.run_main(["bundle", "--note", "Core Idea", "--dry-run"], config, SequenceProvider([]))
            self.assertEqual(bundle_exit, 0)
            self.assertIn("#### deep-tag", bundle_output)
            self.assertNotIn("#### next-sim", bundle_output)
        finally:
            tempdir.cleanup()

    def test_legacy_notes_without_see_also_still_bundle(self):
        tempdir, config, _, zettel_dir, _, _, _ = self.make_workspace()
        try:
            legacy_note = zettel_dir / "legacy-note.md"
            legacy_note.write_text(
                "Tags: [[legacy-tag]]\n\n# Legacy Note\n\nLegacy body.\n\n# References\n\n1. [[seed-source]]\n",
                encoding="utf-8",
            )
            legacy_tag = zettel_dir / "legacy-tag.md"
            legacy_tag.write_text(
                "Tags:\n\n# Legacy Tag\n\nTag body.\n\n# References\n\n1. [[seed-source]]\n",
                encoding="utf-8",
            )

            output, exit_code = self.run_main(["bundle", "--note", "Legacy Note", "--dry-run"], config, SequenceProvider([]))
            self.assertEqual(exit_code, 0)
            self.assertIn("#### legacy-tag", output)
        finally:
            tempdir.cleanup()


if __name__ == "__main__":
    unittest.main()
