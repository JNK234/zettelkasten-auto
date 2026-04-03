# Zettelkasten Automation

Convert substantive source notes into atomic Zettels, keep the knowledge graph local, and assemble exportable bundles around a source or note.

## What The Tool Does

- Processes markdown source notes from a configured source folder.
- Extracts reusable concepts with a structured LLM response.
- Writes new Zettels from a template.
- Stores semantic embeddings in a local ChromaDB collection scoped to the active embedding provider and model.
- Builds context bundles around a source or note.

The tool is intentionally CLI-first and local-first. Source files are never deleted. Successful processing marks them as processed through a `Status: #processed` marker.

## Installation

Using `uv`:

```bash
uv sync
```

Or with `pip`:

```bash
pip install -e .
```

## Configuration

1. Copy `.env.example` to `.env`.
2. Add the API key(s) for the providers you plan to use.
3. Edit `config/config.yaml`.

Portable example:

```yaml
vault:
  root: "/path/to/your/obsidian/vault"
  sources_dir: "10_SOURCES"
  zettel_dir: "20_ZETTLEKASTEN"
  bundles_dir: "110_BUNDLES"
  template_path: "60_TEMPLATES/Note Template.md"

llm:
  provider: "anthropic"  # "openai", "ollama", "gemini", or "anthropic"
  model: "claude-sonnet-4-5"  # Model name (provider-specific)
  base_url: "http://localhost:11434"  # Only for Ollama
  # OpenAI models: gpt-4o, gpt-4o-mini
  # Ollama models: llama3.1:8b, mistral:7b, qwen2.5:7b
  # Gemini models: gemini-1.5-flash, gemini-1.5-pro
  # Anthropic models: claude-sonnet-4-5, claude-haiku-4-5, claude-opus-4-6

embeddings:
  db_path: "chromastores/zettledb"
  provider: "sentence-transformer"  # "openai" or "sentence-transformer"
  model: "all-MiniLM-L6-v2"
  top_k: 5
  max_distance: 0.5

bundle:
  default_depth: 1
  default_top_k: 10

processing:
  min_source_chars: 100
  duplicate_max_distance: 0.25
  chunk_target_chars: 12000
  chunk_max_chars: 16000
```

`template_path` should point to a markdown template containing these anchors:

- `Tags:`
- `# {{Title}}`
- `# References`

Including `# See Also` in the template is recommended but optional. The writer inserts it if it is missing.

## CLI

Show help:

```bash
zettel --help
```

Main commands:

```bash
zettel process [--dry-run] [--limit N]
zettel rebuild-index [--dry-run]
zettel bundle --source SOURCE [--depth N] [--top-k N] [--dry-run]
zettel bundle --note "Note Title" [--depth N] [--dry-run]
```

Legacy maintenance commands:

```bash
zettel normalize-links [--dry-run]
zettel rename-files [--dry-run]
zettel sync-backlinks [--dry-run]
```

## Recommended Workflow

1. Put substantive, reusable source notes into `sources_dir`.
2. Preview extraction:

```bash
zettel process --dry-run --limit 5
```

3. Run the real ingestion:

```bash
zettel process --limit 5
```

4. Rebuild the active embedding collection whenever notes are renamed, merged, or heavily edited by hand:

```bash
zettel rebuild-index
```

5. Generate a context bundle:

```bash
zettel bundle --source source-file-name
zettel bundle --note "core-idea"
```

`bundle --dry-run` is read-only and, for source bundles, requires a fresh index. If the index is stale it exits with a clear error instead of silently using inaccurate semantic results.

## How Processing Works

For each source note, the tool:

1. Skips notes already marked as processed through `Status: #processed` or a legacy terminal `#processed` line.
2. Skips notes shorter than `processing.min_source_chars`.
3. Splits long sources by markdown headings and then paragraphs.
4. Runs extraction per chunk and validates the provider output.
5. Deduplicates candidate concepts by normalized title and semantic similarity.
6. Writes new Zettels and indexes them in the active embedding collection.
7. Marks the source as processed when the note was successfully handled or intentionally skipped as non-extractable.

Per-source outcomes are summarized with categories such as `processed`, `too_short`, `non_extractable`, `provider_error`, `invalid_output`, and `no_valid_concepts`.

## Note Format

Newly generated notes separate semantic relationships from aboutness:

```markdown
Created: DD-MM-YYYY HH:MM

Tags: [[alpha-tag]] [[topic-label]]

# Concept Title

Concept content...

# See Also

- [[related-idea]]
- [[another-note]]

# References

1. [[Source File Name]]
```

- `Tags:` contains only the extracted aboutness links.
- `# See Also` contains semantically similar notes.
- Bundle traversal follows `Tags` beyond degree `0`; it does not recursively follow `See Also`.

## Project Structure

```text
zettelkasten-auto/
├── config/
│   └── config.yaml          # All user configuration
├── src/
│   ├── main.py              # CLI entry point, orchestration, file operations
│   ├── prompts.py           # LLM prompt templates
│   ├── bundle.py            # Context bundle assembly and export
│   ├── db/
│   │   ├── client.py        # ChromaDB collection management, indexing, search
│   │   └── embeddings.py    # Embedding function factory (OpenAI, SentenceTransformer)
│   └── llm/
│       ├── extraction.py    # Concept extraction with validation
│       └── providers.py     # LLM providers (OpenAI, Anthropic, Ollama, Gemini)
└── tests/
    └── test_integration.py
```

## Tests

Run the integration suite with:

```bash
python -m unittest discover -s tests
```
