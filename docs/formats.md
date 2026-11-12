# The format file

[← Documentation index](../README.md#documentation)

The engine reads Obsidian syntax and knows nothing about how *you* write notes. A format file says that, as data:
[`formats/brain.yaml`](../formats/brain.yaml) is the one for the sample vault, and its comments give the evidence for each rule (counts from reading
all 180 notes). [`formats/core.yaml`](../formats/core.yaml) is the lean one: four kinds of note (source, zettel, page, idea), each in
one folder, only `Created`, `Tags`, a `Source` line on sources and a `Status` (`#open`, `#dropped`, `#done`) on ideas. The
structure is enforced by the engine ([structure rules](#structure-rules-typed_folders-ignored_folders-subfolders-min_links)),
and where each source stands is computed ([pipeline](#pipeline-for-vault-pipeline-and-vault-note)). Commands take `--format <name or file>`: a bare name is looked up as `formats/<name>.yaml` (in the current
folder, then in this repository). The file is validated strictly: unknown keys, wrong value types, duplicate keys and references
to undeclared types or relations are errors that name the place, and every command exits with 2:

```
error: /tmp/bad.yaml: types.zettel.metadata.Status: unknown key 'require' (allowed: required, allow_empty, kind, values, multiple, format, link_role, description)
```

What the parser provides, for any vault, is generic: the **header fields** (the `Key: value` lines at the very top of a note;
blank lines may sit between them, the block ends at the first other line) and the **inline fields** (any Dataview-style
`key:: value` line, also in a list item or blockquote; `::` must be followed by a space or the end of the line, so
`std::vector` is not a field). A format gives them meaning.

## Top level

| Key | Value | Default |
|---|---|---|
| `format` | `1` (the version of this file layout) | required |
| `name` | the format's name | the file name |
| `description` | text | none |
| `metadata_source` | where metadata is read: `inline` (header fields), `frontmatter` (YAML properties), `either` (frontmatter wins) | `either` |
| `relations` | the typed relation fields, see [relations](#relations) | none |
| `link_roles` | `role: [section headings]`: a link gets the role of the innermost enclosing section that has one | none |
| `typed_folders` | folders in which every note (at any depth) must have a type; one that matches none is an `untyped-note` violation, see [structure rules](#structure-rules-typed_folders-ignored_folders-subfolders-min_links) | none (untyped notes are only listed by `vault formats`) |
| `ignored_folders` | folders outside the format: their notes have no type, no violations of any kind, and their links count for nothing (`min_links`, coverage, pipeline) | none |
| `pipeline` | which types are sources and which extract and compile them, see [pipeline](#pipeline-for-vault-pipeline-and-vault-note) | none (then `vault pipeline` and `vault note` exit 2) |
| `pages` | what `vault page` keeps in step: `index` (a note with one line per page), `log` (a note it appends to), `tags` (the metadata key zettels are grouped by), see [pages](#pages-for-vault-page) | none (then `vault page` exits 2) |
| `ideas` | what `vault idea` writes: `type`, its `status` key, the `block` the finding goes in, the `tags` key, see [ideas](#ideas-for-vault-idea) | none (then `vault idea` exits 2) |
| `coverage` | what `vault coverage` reports on, see [coverage](#coverage-for-vault-coverage) | none (then `vault coverage` exits 2) |
| `types` | the note types, see [types](#typesname); the first type whose `match` fits a note is its type | required, at least one |

Headings in `link_roles`, `headings`, `relations_heading` and `title_section` match the way heading links do (case, spacing
and `# ^ | : % [ ] \` are ignored; the level is not checked), or as a `/regex/` searched case-insensitively:
`connection: [Connections, "/connections?$/"]` also catches `13. Connections`.

## `relations`

```yaml
relations:
  supports: {inverse: supported-by, targets: [claim]}   # a mirror pair: A supports B means B supported-by A
  supported-by: {targets: [evidence]}                   # its targets; the inverse of an inverse is implied
  from: {targets: [source]}                              # no inverse: one-way
  related: {inverse: related}                            # its own inverse: symmetric
```

| Key | Value |
|---|---|
| `inverse` | the mirror relation. Declaring it declares the inverse too; if both sides name an inverse, they must agree |
| `targets` | note types allowed at the other end (`relation-target-type` otherwise); none means any |
| `description` | text |

Relation names are compared case-insensitively, with spaces and `_` as `-`. A `key::` field whose key is not declared is an
`unknown-field` violation.

## `types.<name>`

| Key | Value | Default |
|---|---|---|
| `match` | `{folder: F, name: "glob"}` or a list of them (any one fits). `folder` matches at any depth (`subfolders: false`: only notes directly in it), `name` is a glob on the file name without `.md`, both case-insensitive | required |
| `description` | text | none |
| `headings` | required headings, by name or `/regex/` | none |
| `ordered_headings` | `true`: the required headings must come in this order | `false` |
| `metadata` | `Key: {options}` for each metadata key, see [metadata keys](#metadata-keys) | none |
| `metadata_source` | as at the top level, for this type | the top level's |
| `relations` | `{required: [...], optional: [...]}`. With this block the type allows only these relations; without it, any declared relation | none (anything) |
| `relations_heading` | the heading relation fields go under when a write adds the first one | none |
| `link_roles` | roles for this type, tried before the top-level ones | none |
| `title_section` | the first link text under this heading is also a title of the note (for the duplicate gate) | none |
| `new` | how `vault new` writes one, see [new](#new-for-vault-new) | see [new](#new-for-vault-new) |
| `ingest` | how `vault ingest` fills one, see [ingest](#ingest-for-vault-ingest); without it the type cannot be ingested | none |
| `min_links` | `{type: N}`: a note of this type must link to at least N distinct notes of that type (`too-few-links` otherwise), see [structure rules](#structure-rules-typed_folders-ignored_folders-subfolders-min_links) | none |
| `required_blocks` | block names: the note must have each `<!-- vault:NAME -->` … `<!-- /vault:NAME -->` block, once, closed and not empty (`missing-block` otherwise). Markers inside code do not count | none |

## `metadata` keys

```yaml
metadata:
  Created: {required: true, kind: datetime, format: "%d-%m-%Y %H:%M"}
  Status: {required: true, kind: tags, multiple: false, values: [supported, contested, weak]}
  Tags: {required: true, kind: [links, tags], link_role: tag}
  Sources: {required: true, kind: integer}
```

| Option | Value | Default |
|---|---|---|
| `kind` | `text`, `tags` (written `#tag`; in frontmatter, plain names count), `links` (`[[note]]`), `[links, tags]` (either), `datetime` / `date` (a `strptime` `format`; YAML dates count in frontmatter), `integer` | `text` |
| `required` | the key must be there and not empty | `false` |
| `allow_empty` | with `required`: the line must be there but may be blank | `false` |
| `values` | allowed values, compared case-insensitively (for tags and links: each item) | any |
| `multiple` | more than one item allowed (tags, links) | `true` for tags/links, `false` otherwise |
| `format` | for `datetime` / `date` | `%Y-%m-%dT%H:%M` / `%Y-%m-%d` |
| `link_role` | the role of links in this key's value, e.g. `tag` | none |
| `description` | text | none |

## `new` (for `vault new`)

| Option | Value | Default |
|---|---|---|
| `name` | the file name without `.md`: `{title}` (the title without `/ \ : * ? " < > \| # ^ [ ]`) or `{slug}` (`lower-case-with-dashes`) | the `match` name glob with `*` replaced by `{title}`, else `{title}` |
| `max_name_length` | cut longer names (the sample cuts discourse names at 90) | none |
| `title_heading` | write `# {title}` after the metadata | `true` |
| `spaced_metadata` | a blank line between metadata lines | `false` |
| `defaults` | `Key: value` used when no value is given; `{now}` is the current time in the key's date format | none |

Required headings are written in order, each at the level it has in most notes of the type (so `# References` is an H1 in a
new zettel, as in most zettels), with relation fields under `relations_heading`.

## `ingest` (for `vault ingest`)

```yaml
ingest:
  body_heading: Notes                 # the converted document goes under this required heading
  references_heading: References      # "1. [title](url)" goes here
  provenance: {file: Original, hash: Content hash, date: Ingested}   # metadata keys of the type
  url: Source                         # a metadata key the document's URL fills (unless --set gives it)
```

Every key is optional. Without `body_heading`, the document goes right after the note's title and its headings start at
level 2: that is `formats/core.yaml`'s source, `ingest: {url: Source}`, which records nothing beyond the type's own
fields (Created, Tags, Source).

## `coverage` (for `vault coverage`)

```yaml
coverage:
  sources: [source]                               # types that count as sources
  citing: [zettel, evidence, topic, comparison]   # types expected to cite one
  citation_roles: [reference, from, source]       # links in these roles are citations, also when the target is missing
```

## Structure rules: `typed_folders`, `ignored_folders`, `subfolders`, `min_links`

Rules about where notes live and how the kinds connect, checked by the engine on every note so they hold however big the
vault gets. From [`formats/core.yaml`](../formats/core.yaml):

```yaml
typed_folders: [10_SOURCES, 20_ZETTLEKASTEN, 50_MOC]   # every note in these, at any depth, needs a type
ignored_folders: [50_MOC/discourse]                    # except these: outside the format altogether
types:
  source:
    match: {folder: 10_SOURCES, subfolders: false}    # directly in 10_SOURCES; 10_SOURCES/old/x.md is not a source
  zettel:
    match: {folder: 20_ZETTLEKASTEN, subfolders: false}
    min_links: {source: 1}                            # cites at least one source
  idea:
    match: {folder: 50_MOC/ideas, subfolders: false}  # before page: the first matching type wins
  page:
    match: {folder: 50_MOC}                           # the rest of 50_MOC, at any depth, but not 50_MOC/discourse
    min_links: {zettel: 3}
```

- **`untyped-note`**: a note inside a `typed_folders` folder that no type matches, reported at line 1 with the types that
  live there (`no type matches this note, and every note in 10_SOURCES/ needs one; types in 10_SOURCES/: source (directly in
  10_SOURCES/)`). Notes outside those folders may stay untyped. `vault new` already refuses to create a note of no type.
- **`ignored_folders`**: take precedence over everything. A note there has no type and no violation of any kind, not even an
  `unknown-field` for a `key::` line. Its links make nothing linked, extracted or compiled.
- **`too-few-links`**: counted over the note's links and embeds, relation fields included: each distinct note of the named type
  that a link resolves to counts once. A link that matches no note or several notes counts for nothing, and the message names
  it (`links to 0 source notes; a zettel needs at least 1 (1 link matches no note: [[2601.05808 - EnvScaler]])`). Links in
  metadata values (`Tags: [[x]]`) do not count: they file a note, they do not cite. Self-links and links inside code do not count
  either, and neither do URLs: a source is a note. Reported at the metadata block. The same "links to"
  (`TypedVault.note_links`) is used by `vault pipeline` and `vault note`.

Both count in the write gate like any other violation, so under core `vault new --type zettel` is refused unless the new
note links to a source. A note that improves without reaching the minimum (a page going from 1 zettel to 2) keeps the same
violation: the gate and [`vault check --baseline`](commands.md#a-baseline---baseline---update-baseline) compare by what a
violation is about (note, code, target type), not by its message.

## `pipeline` (for `vault pipeline` and `vault note`)

```yaml
pipeline:
  sources: [source]    # a note of these types is ingested
  extract: [zettel]    # ... extracted once a note of these types links to it
  compile: [page]      # ... compiled once a note of these types links to one of those
```

Each is a type or a list of types; a type can play one role only. Stages are computed from the links on every run, with the
`note_links` definition above. For the zettels that rest on no source, the reason comes from `coverage:`'s `citation_roles`
(core: links under `References`): `missing` (they cite notes that are not in the vault), `not-a-source`, `url-only` or `none`.
See [Commands: vault pipeline](commands.md#vault-pipeline-vault--f-f---limit-n---json).

## `pages` (for `vault page`)

```yaml
pages:
  index: 50_MOC/index.md   # one line per page between <!-- vault:pages --> markers; give it a type that is not a page
  log: 50_MOC/log.md       # a line appended on every page write
  tags: Tags               # the tags (or links) metadata key: groups a page's zettels, finds the zettels it is behind on
```

Needs a `pipeline` block: pages are its `compile` types, zettels its `extract` types, sources its `sources` types. In core,
`index` and `log` are types of their own (`match: {folder: 50_MOC, name: index, subfolders: false}`, no rules) so neither
note is a page. See [Commands: vault page](commands.md#vault-page-newrefreshcheckexclude).

## `ideas` (for `vault idea`)

```yaml
ideas:
  type: idea        # ideas are notes of this type
  status: Status    # a one-#tag key with values (core: open, dropped, done), set by `vault idea set`
  block: finding    # the finding is written between <!-- vault:finding --> markers; the type must require it
  tags: Tags        # filled with the tags the finding's notes share
```

In core, `idea` has `required_blocks: [finding]`. So an idea without its finding block fails `vault check`, and the write
gate refuses `vault new --type idea` without one. See [Commands: vault findings and vault idea](commands.md#vault-findings-vault--f-f---method-m---id-id---limit-n---json).

## Violations

Every violation has a file and a line. For something missing, the line is where it belongs: the heading a missing heading
should precede (or the last line), the relation field a missing mirror should join, the header block for a missing key.
Frontmatter values are reported at line 1.

| Code | Means | Example (sample vault) | `vault fix` |
|---|---|---|---|
| `missing-heading` | a required heading is not there | `missing headings 'What This Is', 'Why It Matters', 'Key Details', 'Tensions' (expected before 'References')` | adds it, with an empty section |
| `heading-order` | required headings out of order | `heading 'Scope' (line 4) comes before 'Relations' (line 5); expected order: …` | no |
| `missing-key` | a required metadata key is not there | `missing metadata 'Created'` | no |
| `missing-value` | it is there but empty | `'Tags' is empty` | no |
| `bad-value` | wrong kind, not an allowed value, several where one is allowed | `Status: #suported is not an allowed value (supported, contested, weak)` | no |
| `missing-relation` | a required relation field is not there | ``missing relation field `supported-by::` (required for claim)`` | no |
| `unexpected-relation` | a relation the type does not allow | ``…`supported-by::` is not a relation of evidence (allowed: from, supports, opposes)`` | no |
| `unknown-field` | a `key::` field that is not a declared relation | ``…`endorses::` is not a declared relation (declared: …)`` | no |
| `bad-relation` | text in a relation value that is not a link | ``…`supports::` has text that is not a link: 'and more'`` | no |
| `unresolved-target` | a relation link that matches no note | ``…`from::` [[2601.05808 - EnvScaler]] matches no note`` | no |
| `ambiguous-target` | a relation link that matches several files | ``…`supports::` [[Index]] matches 2 files: …`` | no |
| `relation-target-type` | a relation pointing at a note of the wrong type | ``…`supports::` [[kl-penalty-in-rlhf]] is a zettel; expected claim`` | no |
| `missing-mirror` | A says `supports:: [[B]]`, B lacks `supported-by:: [[A]]` (reported on B, the note to edit) | ``missing `supported-by:: [[EVD - …]]` (mirror of `supports::` in …EVD - ….md:13)`` | adds it |
| `too-few-links` | fewer links to notes of a type than the type's `min_links` | `links to 1 zettel note; a page needs at least 3` (core) | no |
| `untyped-note` | a note in a `typed_folders` folder that no type matches | `no type matches this note, and every note in 10_SOURCES/ needs one; …` (core) | no |
| `missing-block` | a `required_blocks` block that is absent, not one opening followed by one closing marker, or empty | `no finding block (<!-- vault:finding --> … <!-- /vault:finding -->)` (core ideas) | no |

## From Python

```python
from vaultengine import Vault
from vaultengine.formats import TypedVault, load_format

typed = TypedVault(load_format("formats/brain.yaml"), Vault.load("/tmp/sample-vault"))
# with core: typed.note_links(path), typed.linked_from(path); Pipeline(typed).report() and .note(path)
# (from vaultengine.formats.pipeline import Pipeline)
path = "50_MOC/discourse/rl-environment-design/CLM - Grade the observed end state rather than the trajectory.md"
typed.type_of(path)                          # 'claim'
typed.metadata(path)["Status"].values        # ['supported']
[(e.relation, e.origin) for e in typed.relations(path)][:2]   # [('informs', 'declared'), ('informs', 'declared')]
[(l.role, l.edge.link.target) for l in typed.links(path)][-1]  # ('related-zettel', 'replay-and-hash-end-state-grading')
len(typed.issues())                          # 102 on a fresh copy
```

See [Commands](commands.md) for the commands that use a format, and [Verify it yourself](verify.md) for the tests.
