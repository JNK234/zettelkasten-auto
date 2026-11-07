# Getting started: the sample vault

[← Documentation index](../README.md#documentation)

Work on a copy, so no index is written into the test fixtures (`-p` keeps file times, which keeps the outputs below exact):

```bash
cp -Rp tests/fixtures/vault /tmp/sample-vault
```

In the outputs below, timings, timestamps and the index date differ on your machine; everything else should match.

## 1. Build the index

```bash
vault index /tmp/sample-vault
```
```
Indexed /tmp/sample-vault in 0.41s
  index      /tmp/sample-vault/.vaultengine/index.sqlite
  files      180 notes, 0 attachments
  parsed     180  (180 added, 0 changed, 0 deleted)
  read       180 notes hashed, 0 skipped by size/mtime
  links      1787 resolved this run, 0 changed status in notes that were not re-parsed
    + 10_SOURCES/2312.06942 - AI Control - Improving Safety Despite Intentional Subversion.md
    …
  … 170 more (--limit 0 shows all)
```

Run it again: nothing is read or parsed.

```bash
vault index /tmp/sample-vault
```
```
Indexed /tmp/sample-vault in 0.01s
  index      /tmp/sample-vault/.vaultengine/index.sqlite
  files      180 notes, 0 attachments
  parsed     0  (index already up to date)
  read       0 notes hashed, 180 skipped by size/mtime
  links      0 resolved this run, 0 changed status in notes that were not re-parsed
```

Edit one note and index again: exactly that note is parsed.

```bash
echo "Edited." >> "/tmp/sample-vault/20_ZETTLEKASTEN/kl-penalty-in-rlhf.md"
vault index /tmp/sample-vault
```
```
  parsed     1  (0 added, 1 changed, 0 deleted)
  read       1 note hashed, 179 skipped by size/mtime
  links      11 resolved this run, 0 changed status in notes that were not re-parsed
    ~ 20_ZETTLEKASTEN/kl-penalty-in-rlhf.md
```

## 2. Stats and health

```bash
vault stats /tmp/sample-vault --top 3
```
```
Vault stats: /tmp/sample-vault
  index 2026-11-06T20:29:16+00:00 (up to date)

  notes             180   (0 with frontmatter, 0 frontmatter errors)
  attachments         0
  links            1787   wikilinks 1787 · markdown 0 · embeds 0
    resolved        472   unresolved 1315 (612 distinct targets) · ambiguous 0
  note edges        439   distinct note→note connections
  tags              241   uses of 28 distinct tags
  properties          0   distinct keys
  aliases             0
  headings         1391
  blocks              0

Notes per top-level folder
    109  20_ZETTLEKASTEN
     43  50_MOC
     28  10_SOURCES

Top tags (notes)
    133  #baby
     31  #evidence
     15  #ai-safety

Most linked (linking notes / links)
   26 / 27  20_ZETTLEKASTEN/reinforcement-learning-in-language-models.md
   20 / 22  20_ZETTLEKASTEN/reward-hacking-as-specification-gaming.md
   16 / 16  20_ZETTLEKASTEN/reward.md

Most wanted missing notes (linking notes / links)
   52 / 52  rl-environment-design
   40 / 40  discourse-graph
   31 / 31  evaluation-methodology
```

```bash
vault health /tmp/sample-vault --limit 3
```
```
Vault health: /tmp/sample-vault
  index 2026-11-06T20:29:16+00:00 (up to date)

  unresolved links                      1315   across 612 missing targets
  ambiguous links                          0
  missing heading/block targets            0
  orphans (no backlinks)                  41
  dead ends (no outgoing note links)      21
  duplicate basenames                      0
  empty notes                              0

Unresolved link targets (612)
  rl-environment-design  — 52 links from 52 notes, e.g. 20_ZETTLEKASTEN/agentic-tasks-are-artifact-bundles-not-dataset-rows.md:5 (+51 more notes)
  discourse-graph  — 40 links from 40 notes, e.g. 50_MOC/discourse/rl-environment-design/CLM - A fixed typed contract should separate environment from trainer.md:3 (+39 more notes)
  evaluation-methodology  — 31 links from 31 notes, e.g. 20_ZETTLEKASTEN/a-bounty-list-is-a-map-of-what-cannot-be-verified.md:3 (+30 more notes)
  … 609 more (--limit 0 shows all)
  …
```

Add `--json` to `index`, `stats` or `health` for machine-readable output, e.g.
`vault health /tmp/sample-vault --json | python3 -c "import sys, json; print(json.load(sys.stdin)['summary'])"` prints
`{'unresolved_links': 1315, 'unresolved_targets': 612, 'ambiguous_links': 0, 'missing_subpaths': 0, 'orphans': 41, 'dead_ends': 21, 'duplicate_basenames': 0, 'empty_notes': 0}`.

If files change after indexing, `stats` and `health` still answer from the index and print a warning on stderr:
``warning: index is out of date (1 added, 0 changed, 0 deleted); run `vault index` to refresh``.

## 3. The engine as a Python library

```bash
python - <<'EOF'
from vaultengine import LinkGraph, Vault

vault = Vault.load("/tmp/sample-vault")
graph = LinkGraph(vault)
print(len(vault.notes), "notes,", len(graph.edges), "links,", sum(e.status == "resolved" for e in graph.edges), "resolved")
note = "20_ZETTLEKASTEN/kl-penalty-in-rlhf.md"
print("tags:", vault.notes[note].tag_names)
print("backlinks:", graph.backlink_sources(note))
EOF
```
```
180 notes, 1787 links, 472 resolved
tags: ['baby']
backlinks: ['20_ZETTLEKASTEN/the-environment-not-flops-bottlenecks-agentic-rl.md']
```

## Next

- The JSON-RPC server: [The JSON-RPC API](api.md) (`vault serve`, the same session continued).
- Your own conventions: [The format file](formats.md), then every format command in [Commands](commands.md).
- The terminal app: [The terminal app](tui.md).
