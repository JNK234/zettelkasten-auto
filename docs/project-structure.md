# Project structure

[← Documentation index](../README.md#documentation)

```text
zettelkasten-auto/
├── README.md                # overview and the documentation index
├── docs/                    # the documentation (linked from README.md)
│   └── experiments/         # vault experiment reports, one per run that set a default (<date>.md)
├── vaultengine/             # Vault engine (no LLM)
│   ├── parser.py            # Obsidian-flavoured markdown parsing (also header and key:: fields)
│   ├── resolver.py          # Link resolution rules
│   ├── graph.py             # Link graph and backlinks (in memory)
│   ├── index.py             # Incremental SQLite index (also partial updates for writes)
│   ├── queries.py           # stats / health / note queries on the index
│   ├── service.py           # Operations used by the JSON-RPC server
│   ├── cli.py               # `vault` command
│   ├── settings.py          # engine settings: flag > env (VAULTENGINE_*) > config file (engine:) > default
│   ├── graphrank.py         # the link graph (missing notes included, link directions kept): PageRank (numpy), co-citation, coupling
│   ├── related.py           # vault related: graph, embedding (text) and tag signals; TextSimilarity
│   ├── relations.py         # vault explain: how two notes relate, signal by signal, with the evidence
│   ├── tagrule.py           # which links are tags (Tags: [[topic]]): the format's link_role tag, else fields named tags
│   ├── shape.py             # vault map: communities (Louvain), topic pages, meaning edges, betweenness, constraint, HITS
│   ├── lenses.py            # vault deepen: neighbourhood, cited together, same foundations, backbone, cluster
│   ├── discover.py          # vault discover: hidden link, same role, bridge, gap, surprise, unwritten, wander
│   ├── connect.py           # vault suggest (links to add) and vault unlinked
│   ├── suggest.py           # vault suggest (notes to write, drafts)
│   ├── experiment.py        # vault experiment: hidden-link recovery, text variants, communities vs topics; seeded
│   ├── explore.py           # vault explore: the lens registry, sections, equal space, near-duplicates, where to start
│   ├── memory.py            # what explore remembers (memory.sqlite): runs, seen, marks, links made, map snapshots
│   ├── linkit.py            # vault link-it / unlink: a See Also line in both notes, through the safe write path
│   ├── history.py           # vault map --record / --since: snapshots and what changed (gaps, communities, missing notes)
│   ├── stats.py             # the vault-wide numbers explore keeps between runs (stats.sqlite): exact, reused, fresh
│   ├── operations.py        # the explore commands as operations: what --json prints and the JSON-RPC methods return
│   ├── embedding/           # embeddings
│   │   ├── provider.py      # the EmbeddingProvider interface, registry, load_provider
│   │   ├── model2vec.py     # static Model2Vec models (tokenizer.json + model.safetensors), numpy + tokenizers
│   │   ├── text.py          # what a note is embedded as, and its hash
│   │   ├── store.py         # vectors cached in the index by text hash and model id; vault embed
│   │   └── explain.py       # the words behind a similarity (static models)
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
│   └── rpc/                 # JSON-RPC 2.0 server, Pydantic models, schema.json
├── formats/brain.yaml       # the format of the sample vault
├── tui/                     # `vault-tui` terminal app (TypeScript + Ink), see tui/README.md
├── scripts/
│   ├── verify.sh            # every check of the engine, CLI, API and app
│   ├── check_docs.py        # every link in the docs resolves
│   ├── setup.sh             # uv sync + npm ci with absolute paths (source it in the sandbox)
│   ├── bench_shape.py       # times vault map and vault discover on generated vaults (thousands to 50,000 notes)
│   ├── bench_commands.py    # times every vault command on a generated vault
├── config/
│   └── config.yaml          # zettel tool configuration, and `engine:`: every vault engine setting
├── models/potion-base-8M/   # the embedding model (Model2Vec; shipped next to the repository, not in git)
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
