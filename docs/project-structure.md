# Project structure

[← Documentation index](../README.md#documentation)

```text
zettelkasten-auto/
├── README.md                # overview and the documentation index
├── docs/                    # the documentation (linked from README.md)
├── vaultengine/             # Vault engine (no LLM)
│   ├── parser.py            # Obsidian-flavoured markdown parsing (also header and key:: fields)
│   ├── resolver.py          # Link resolution rules
│   ├── graph.py             # Link graph and backlinks (in memory)
│   ├── index.py             # Incremental SQLite index (also partial updates for writes)
│   ├── queries.py           # stats / health / note queries on the index
│   ├── service.py           # Operations used by the JSON-RPC server
│   ├── cli.py               # `vault` command
│   ├── formats/             # the formats layer
│   │   ├── spec.py          # loading and validating a format file
│   │   ├── typed.py         # a vault seen through a format: types, metadata, relations, roles, violations
│   │   ├── check.py         # vault check (and --changed)
│   │   ├── fix.py           # vault fix
│   │   ├── edit.py          # byte-preserving line edits
│   │   ├── write.py         # the safe write path: new, link, set; the gate, the atomic commit, index sync
│   │   ├── ingest.py        # vault ingest: conversion and the duplicate gate
│   │   ├── similarity.py    # title similarity and content overlap
│   │   └── coverage.py      # vault coverage
│   ├── experimental/        # vault suggest: lexical / embedding ranking, and its evaluation data
│   └── rpc/                 # JSON-RPC 2.0 server, Pydantic models, schema.json
├── formats/brain.yaml       # the format of the sample vault
├── tui/                     # `vault-tui` terminal app (TypeScript + Ink), see tui/README.md
├── scripts/
│   ├── verify.sh            # every check of the engine, CLI, API and app
│   ├── check_docs.py        # every link in the docs resolves
│   └── semantic_eval.py     # the embedding experiment
├── config/
│   └── config.yaml          # zettel tool configuration
├── src/                     # zettel tool (LLM extraction), see docs/zettel.md
│   ├── main.py              # CLI entry point, orchestration, file operations
│   ├── prompts.py           # LLM prompt templates
│   ├── bundle.py            # Context bundle assembly and export
│   ├── db/
│   │   ├── client.py        # ChromaDB collection management, indexing, search
│   │   └── embeddings.py    # Embedding function factory (OpenAI, SentenceTransformer)
│   └── llm/
│       ├── extraction.py    # Concept extraction with validation
│       └── providers.py     # LLM providers (OpenAI, Anthropic, Ollama, Gemini)
└── tests/                   # Python tests; fixtures/vault is the sample vault, fixtures/rpc the recorded API sessions
```
