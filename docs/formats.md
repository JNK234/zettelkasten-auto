# The format file

[← Documentation index](../README.md#documentation)

The engine reads Obsidian syntax and knows nothing about how *you* write notes. A format file says that, as data:
[`formats/brain.yaml`](../formats/brain.yaml) is the one for the sample vault, and its comments give the evidence for each rule (counts from reading
all 180 notes). Commands take `--format <name or file>`: a bare name is looked up as `formats/<name>.yaml` (in the current
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
| `match` | `{folder: F, name: "glob"}` or a list of them (any one fits). `folder` matches at any depth, `name` is a glob on the file name without `.md`, both case-insensitive | required |
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
```

## `coverage` (for `vault coverage`)

```yaml
coverage:
  sources: [source]                               # types that count as sources
  citing: [zettel, evidence, topic, comparison]   # types expected to cite one
  citation_roles: [reference, from, source]       # links in these roles are citations, also when the target is missing
```

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

## From Python

```python
from vaultengine import Vault
from vaultengine.formats import TypedVault, load_format

typed = TypedVault(load_format("formats/brain.yaml"), Vault.load("/tmp/sample-vault"))
path = "50_MOC/discourse/rl-environment-design/CLM - Grade the observed end state rather than the trajectory.md"
typed.type_of(path)                          # 'claim'
typed.metadata(path)["Status"].values        # ['supported']
[(e.relation, e.origin) for e in typed.relations(path)][:2]   # [('informs', 'declared'), ('informs', 'declared')]
[(l.role, l.edge.link.target) for l in typed.links(path)][-1]  # ('related-zettel', 'replay-and-hash-end-state-grading')
len(typed.issues())                          # 102 on a fresh copy
```

See [Commands](commands.md) for the commands that use a format, and [Verify it yourself](verify.md) for the tests.
