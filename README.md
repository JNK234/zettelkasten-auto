# Zettelkasten Automation

Converts source notes into atomic Zettelkasten notes using LLM extraction and embedding-based semantic linking.

## Features

- Extracts atomic concepts from source notes using OpenAI structured outputs
- Finds semantically similar existing notes using ChromaDB embeddings
- Creates linked Zettel files with proper formatting
- Incremental processing (skips already processed files)

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

1. Copy `.env.example` to `.env` and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_key_here
   ```

2. Edit `config/config.yaml` to set your vault paths:
   ```yaml
   vault:
     root: "/path/to/your/obsidian/vault"
     sources_dir: "10_SOURCES"
     zettel_dir: "20_ZETTLEKASTEN"
   ```

## Usage

```bash
python -m src.main
```

The script will:
1. Index existing Zettels in the output directory
2. Scan source files (skipping those marked with both `#processed` and `#adult`)
3. Extract atomic concepts via LLM
4. Find similar existing notes for linking
5. Create new Zettel files
6. Mark sources as processed

## Project Structure

```
zettelkasten-auto/
├── config/
│   └── config.yaml    # Vault paths and settings
├── src/
│   ├── db.py          # ChromaDB operations
│   ├── llm.py         # LLM concept extraction
│   └── main.py        # Orchestration
├── .env               # API keys (not tracked)
└── requirements.txt
```

## How It Works

1. Source notes are read from the configured sources directory
2. Each source is sent to OpenAI to extract atomic concepts (title, content, tags)
3. For each concept, ChromaDB finds semantically similar existing Zettels
4. A new Zettel file is created with proper linking to similar notes
5. The source file is marked with `#processed` to avoid reprocessing

## Output Format

Each generated Zettel follows this format:

```markdown
Created: DD-MM-YYYY HH:MM

Status: #baby

Tags: [[Tag1]] [[Tag2]]

# Concept Title

Content explaining the concept...

# See Also

- [[Similar Note 1]]
- [[Similar Note 2]]

# References

1. [[Source File Name]]
```
