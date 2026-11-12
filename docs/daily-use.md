# Daily use: your own vault

[← Documentation index](../README.md#documentation) · [Install](install.md) · [The reading workflow](workflow.md) · [Exploring](exploring.md)

How to run `vault` on your own vault, every day. Every command below was run on a copy of the sample vault; the
outputs are real, shortened.

## Once: install, model, a personal config

In the repository: `uv sync` (the `vault` command, in `.venv/bin`: put it on your PATH or
`source .venv/bin/activate`); for the app, `cd tui && npm ci` ([Install](install.md)). The embedding model is not in
git: copy the `potion-base-8M` folder (`model.safetensors`, `tokenizer.json`, `config.json`) to `models/potion-base-8M/`.

Keep your settings in a file **outside the repository**, e.g. `~/vault-config.yaml`. It replaces `config/config.yaml`
(settings it leaves out take their defaults), so give the model path. Keep the index and the explore memory outside
the vault, so a sync service never sees them:

```yaml
engine:
  vault: ~/path/to/your/vault
  index: ~/.local/state/vault/index.sqlite     # default: <vault>/.vaultengine/index.sqlite
  memory: ~/.local/state/vault/memory.sqlite   # what explore remembers; default: <vault>/.vaultengine/
  format: core                                 # the format with the pipeline blocks (pipeline, pages, ideas)
  embedding:
    model_path: ~/path/to/zettelkasten-auto/models/potion-base-8M
```

Point the engine at it, in each shell or once in your shell profile; `--config FILE` on any command does the same:

```bash
export VAULTENGINE_CONFIG=~/vault-config.yaml
V=~/path/to/your/vault                       # `vault page` and `vault idea` take the vault path even with a config
vault config show                            # every value and where it came from: check vault, index, format, model
```

`~` is expanded; relative paths in the file are relative to the file's folder. Any setting can also be given as
`--setting KEY=VALUE` or `VAULTENGINE_<KEY>` (e.g. `VAULTENGINE_FORMAT=brain`): flag > env > file > default.
`brain` has no pipeline blocks: `next`, `pipeline`, `page` and `idea` need `core` (or [blocks added](formats.md#pipeline-for-vault-pipeline-and-vault-note)).

## First run on a vault

```bash
vault index                      # 180 notes, 1787 links; later runs parse only what changed
vault embed                      # vectors    180 embedded, 0 cached (180 notes)
vault check --update-baseline    # Recorded a baseline of 144 violations: <vault>/.vaultengine/baseline-core.json
vault check --baseline           # ✓ no new violations in 180 notes checked (144 known violations in the baseline not shown)
```
The baseline lets you start with an imperfect vault: from now on `--baseline` fails only on what is new.

## The daily loop

```bash
vault next --limit 5             # one queue, most trusted first: each item with its reason and the command that does it
vault pipeline                   # where each source stands: ingested, extracted, compiled; what no page uses yet
```
```
1. write    write [[transcript-spotting-heuristics-for-reward-hacking]]   (trust 97)
   why  6 notes link to it and it does not exist
   run  vault new /tmp/daily/vault -f core --type zettel --name transcript-spotting-heuristics-for-reward-hacking …
```

Explore: from nowhere (where to start), or from a note. Results have refs (`1.2`) valid until the next explore run.

```bash
vault explore                    # no start: gaps, surprises, unwritten notes, communities without a topic page
vault explore kl-penalty-in-rlhf # ten lenses on one note, a section each, with evidence
vault mark keep 1.1              # remembered in the memory, never in notes; keeps reorder the sections
vault mark dismiss 2.1           # gone from later runs (`--fresh` also hides what you have seen)
vault kept                       # what you kept, with lens and evidence
vault link-it 1.2                # dry run: the diff of both notes (a `See Also` line each) and the check
vault link-it 1.2 --write        # writes both notes or neither; the index follows (`vault unlink A B --write` undoes it)
```

Pages from zettels, then what the engine found, then an idea from a finding:

```bash
vault page new "$V" "RL fine-tuning" --from '[[kl-penalty-in-rlhf]] [[direct-preference-optimization-dpo]] [[covariance-targeted-entropy-intervention]]'
vault page new "$V" "RL fine-tuning" --from '[[kl-penalty-in-rlhf]] [[direct-preference-optimization-dpo]] [[covariance-targeted-entropy-intervention]]' --write
vault page check "$V"            # what each page needs: stale blocks, changed zettels, zettels it is behind on
vault page refresh "$V" rl-fine-tuning --write   # rebuilds the generated blocks; your text around them stays
vault findings --limit 2         # missing notes, two names for one idea, names of existing notes, links worth adding
vault idea new "$V" --from link-6b21be7696       # dry run of the idea note; add --write to write it
vault check --baseline           # end of the day: exit 0 means nothing new is wrong
```
```
50_MOC/rl-fine-tuning.md  (RL fine-tuning; refreshed 2026-11-12T15:55:52)
  ✗ behind: 1 zettel with at least two of its tags or links in common, not on it
      20_ZETTLEKASTEN/reinforcement-learning-from-human-feedback-rlhf.md   tags policy-gradient-methods · links RLHF
  → link the behind zettels that belong in the text and refresh; `vault page exclude` the ones that do not
```

The same loop runs in the app (`/next`, `/explore`, `/page`, `/findings`, …): [The terminal app](tui.md).

## The weekly loop

```bash
vault map --record               # the shape of the vault, and a snapshot of it in the memory
vault map --since 2026-11-12     # against the snapshot of that date: gaps closed, communities formed, missing notes written
vault experiment --output /tmp/vault-experiment.md   # when the vault has grown: test the engine's options on your notes
```
`vault experiment` changes nothing; it says which defaults it would change (`related.unlinked_weights`, community
settings) and why. Change them in your config only when it says an option wins clearly ([Experiments](experiment.md)).

## What writes, and where

Everything else only reads (index, stats, health, search, related, suggest, unlinked, explain, map without
`--record`, deepen, discover, kept, pipeline, note, page check, findings, next, check without `--update-baseline`,
experiment apart from its `--output` file).

| Command | Writes |
|---|---|
| `index`, `embed` | the index file (`engine.index`); `embed` caches vectors in it |
| `explore` (unless `--no-record`), `mark`, `map --record` | the memory (`engine.memory`), and `stats.sqlite` next to it |
| `link-it --write`, `unlink --write` | the two notes, and the memory |
| `page new` / `refresh` / `exclude` with `--write` | the page, the format's index and log notes, `<vault>/.vaultengine/pages.json` |
| `idea new` / `set` with `--write` | the idea note |
| `move` / `merge` with `--write` | the notes and every link to them |
| `new`, `link`, `set`, `fix`, `ingest` | notes, **by default**: pass `--dry-run` to see first |
| `check --update-baseline` | `<vault>/.vaultengine/baseline-<format>.json` |

Every note write goes through one gate: refused if it adds a violation, all files or none (a journal in
`<vault>/.vaultengine/journal` while it runs), the index updated after.

## With an assistant

An assistant uses the same commands with `--json` and never edits a note file itself: dry run, show you, then the
same command with `--write`. The rules and the fields: [For assistants](agents.md) (explore, mark, link-it, map) and
[The reading workflow](workflow.md#driving-it-from-an-assistant) (check, pipeline, pages, findings, ideas, next).

## Troubleshooting

- **No model** (moved, or `model_path` wrong): nothing fails; the text signal is off and the rest rescale.
  `vault config show` marks the path `✗ not found`. For example:
  ```bash
  vault related kl-penalty-in-rlhf --setting embedding.model_path=/no/such/model
  ```
  ```
  score = 0.71 × graph + 0.29 × tags · 183 notes compared
  text signal unavailable: embedding model folder not found: /no/such/model
  ```
- **Stale index** (you edited notes in Obsidian): reads still answer, from the old index, and say so:
  ``warning: index is out of date (0 added, 1 changed, 0 deleted); run `vault index` to refresh``. Run `vault index`
  (only changed notes are parsed). Writes keep the index current themselves.
- **macOS, Apple silicon, after `npm ci`**: the app's tests stop on `Cannot find native binding` (rolldown's
  platform package was not installed), or a terminal cannot start (`posix_spawnp failed.`: node-pty's helper lost
  its execute bit). In `tui/`:
  ```bash
  npm i --no-save @rolldown/binding-darwin-arm64@$(node -p "require('rolldown/package.json').version")
  chmod +x node_modules/node-pty/prebuilds/darwin-*/spawn-helper
  ```
