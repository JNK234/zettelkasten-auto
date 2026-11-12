"""A vault read from the SQLite index instead of from every file: the backend that lets
`vault pipeline`, `vault page check` and `vault note` answer in seconds on a large vault.

``IndexedVault.open(root, db)`` gives a ``Vault`` whose ``notes`` load on first use from
the index, and an ``IndexGraph`` giving each note's resolved links and backlinks from the
index's ``links`` table. Nothing is written: not the index, not the vault.

Freshness. Opening compares the vault with the index the way ``vault index`` does (size
and mtime, then SHA-256). Notes added or changed since the last ``vault index`` are parsed
from disk, in memory only; deleted ones are dropped. Stored link resolutions are reused
except the ones an edit can affect, which are resolved again against the vault as it is
now. The rule is the index's own (``VaultIndex.update``): a link can only be affected if
its target's basename or full text equals a name or alias key of a file that was added,
deleted or changed; links in changed notes come from their fresh parse. So the answers are
the same as from parsing every file, which the tests check on random edit sequences.

When the index is missing or needs a rebuild (another schema, parser or resolver version),
``open`` returns None and callers read every file instead.
"""

from __future__ import annotations

import json
import posixpath
import sqlite3
from collections import defaultdict
from collections.abc import Iterator, Mapping
from pathlib import Path

from vaultengine.frontmatter import split_frontmatter
from vaultengine.index import SCHEMA_VERSION, Staleness, VaultIndex, _name_keys
from vaultengine.model import (BlockId, ExternalLink, HeaderField, Heading, InlineField, Link, Marker, ParsedNote, Tag,
                               field_kinds)
from vaultengine.parser import normalize_text, parse_note
from vaultengine.resolver import ResolvedLink, Resolver
from vaultengine.vault import Vault, is_markdown, link_name, lookup_key

_CHUNK = 500  # ids per IN (...) query


def _chunks(items: list, size: int = _CHUNK) -> Iterator[list]:
    for start in range(0, len(items), size):
        yield items[start:start + size]


class LazyNotes(Mapping):
    """{path: ParsedNote}: the keys are known up front, a note is built from the index (or was
    parsed fresh) the first time it is looked up. ``prefetch`` loads many with a few queries."""

    def __init__(self, vault: "IndexedVault", paths: set[str]) -> None:
        self._vault = vault
        self._paths = paths
        self._cache: dict[str, ParsedNote] = dict(vault._fresh)

    def __contains__(self, path: object) -> bool:
        return path in self._paths

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(self._paths))

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, path: str) -> ParsedNote:
        if path not in self._paths:
            raise KeyError(path)
        if path not in self._cache:
            self.prefetch([path])
        return self._cache[path]

    def prefetch(self, paths) -> None:
        wanted = [p for p in paths if p in self._paths and p not in self._cache]
        if wanted:
            self._cache.update(self._vault._load_notes(wanted))


class IndexedVault(Vault):
    def __init__(self, root: Path, index: VaultIndex, stale: Staleness, texts: dict[str, str] | None = None,
                 owner: bool = True) -> None:
        """``texts``: contents to use instead of the file on disk (a write's planned changes, see ``overlay``)."""
        self.root = root
        self.index = index
        self.owner = owner  # an overlay shares its base's index connection and does not close it
        self.conn: sqlite3.Connection = index.conn
        self.stale = stale
        self._texts = dict(texts or {})
        rows = self.conn.execute("SELECT id, path, kind FROM files").fetchall()
        self.ids: dict[str, int] = {row["path"]: row["id"] for row in rows}
        self.paths_by_id: dict[int, str] = {row["id"]: row["path"] for row in rows}
        deleted = set(stale.deleted)
        changed = set(stale.changed)
        added = set(stale.added)
        indexed_notes = {row["path"] for row in rows if row["kind"] == "note"}
        self.files = sorted((set(self.ids) - deleted) | added)
        self._fresh: dict[str, ParsedNote] = {}
        for path in sorted((changed | added)):
            if is_markdown(path):
                if path in self._texts:
                    text = normalize_text(self._texts[path])
                else:
                    text = normalize_text((root / path).read_bytes().decode("utf-8", errors="replace"))
                self._fresh[path] = parse_note(text, path=path)
        self.notes = LazyNotes(self, (indexed_notes - deleted) | set(self._fresh))
        self._by_path: dict[str, list[str]] = defaultdict(list)
        self._by_name: dict[str, list[str]] = defaultdict(list)
        self._by_alias: dict[str, list[str]] = defaultdict(list)
        for path in self.files:
            keys = {lookup_key(path)}
            if is_markdown(path):
                keys.add(lookup_key(path[:-3]))
            for key in keys:
                self._by_path[key].append(path)
            for key in {lookup_key(posixpath.basename(path)), lookup_key(link_name(path))}:
                self._by_name[key].append(path)
        aliases: dict[str, list[str]] = defaultdict(list)
        for row in self.conn.execute("SELECT f.path, a.alias FROM aliases a JOIN files f ON f.id = a.file_id ORDER BY a.rowid"):
            aliases[row["path"]].append(row["alias"])
        self.indexed_aliases = aliases
        for path in sorted(self.notes._paths):
            names = self._fresh[path].aliases if path in self._fresh else aliases.get(path, [])
            for alias in {lookup_key(alias) for alias in names}:
                if alias:
                    self._by_alias[alias].append(path)

    @classmethod
    def open(cls, root: str | Path, db_path: str | Path) -> "IndexedVault | None":
        """The vault from its index, or None when there is no usable index (then read every file)."""
        db_path = Path(db_path)
        if not db_path.exists():
            return None
        probe = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            version = probe.execute("PRAGMA user_version").fetchone()[0]
        except sqlite3.DatabaseError:
            return None
        finally:
            probe.close()
        if version != SCHEMA_VERSION:
            return None  # opening it as an index would rebuild its schema: leave that to `vault index`
        index = VaultIndex.open_existing(db_path)
        root = Path(root).resolve()
        stale = index.staleness(root)
        if stale.reason:
            index.close()
            return None
        return cls(root, index, stale)

    def close(self) -> None:
        if self.owner:
            self.index.close()

    def overlay(self, changes: dict[str, str | None]) -> "IndexedVault":
        """This vault with ``changes`` applied in memory ({path: new text, or None to delete}): what a
        write would leave. Nothing on disk is touched."""
        indexed = set(self.ids)
        added, changed, deleted = set(self.stale.added), set(self.stale.changed), set(self.stale.deleted)
        texts = dict(self._texts)
        for path, text in changes.items():
            if text is None:
                added.discard(path)
                changed.discard(path)
                texts.pop(path, None)
                if path in indexed:
                    deleted.add(path)
            else:
                texts[path] = text
                if path in indexed:
                    changed.add(path)
                    deleted.discard(path)
                else:
                    added.add(path)
        stale = Staleness(sorted(added), sorted(changed), sorted(deleted))
        return IndexedVault(self.root, self.index, stale, texts, owner=False)

    def sources_linking(self, keys: set[str]) -> set[str]:
        """Notes with a link whose target (whole, or its last part) is one of ``keys``."""
        found = set()
        stale = self.stale_paths
        for chunk in _chunks(sorted(keys)):
            marks = ",".join("?" * len(chunk))
            for row in self.conn.execute(
                    f"""SELECT DISTINCT f.path FROM links l JOIN files f ON f.id = l.source_id
                        WHERE l.target_key IN ({marks}) OR l.target_name_key IN ({marks})""", chunk + chunk):
                if row["path"] not in stale:
                    found.add(row["path"])
        for path, note in self._fresh.items():
            if any(lookup_key(l.target.strip()) in keys or lookup_key(posixpath.basename(l.target.strip())) in keys
                   for l in note.links + note.embeds):
                found.add(path)
        return found

    @property
    def stale_paths(self) -> set[str]:
        return set(self.stale.added) | set(self.stale.changed) | set(self.stale.deleted)

    # ----- rebuilding ParsedNotes from the index -------------------------------------------------

    def _load_notes(self, paths: list[str]) -> dict[str, ParsedNote]:
        conn = self.conn
        by_id = {self.ids[p]: p for p in paths}
        notes: dict[int, ParsedNote] = {}
        for ids in _chunks(list(by_id)):
            marks = ",".join("?" * len(ids))
            for row in conn.execute(f"SELECT * FROM notes WHERE file_id IN ({marks})", ids):
                block = split_frontmatter(row["frontmatter_block"]) if row["frontmatter_block"] is not None else None
                notes[row["file_id"]] = ParsedNote(
                    path=by_id[row["file_id"]],
                    frontmatter=block.data if block else {},
                    frontmatter_error=row["frontmatter_error"],
                    has_frontmatter=bool(row["has_frontmatter"]),
                    body_start_line=row["body_start_line"],
                    last_line=row["last_line"],
                )
            for row in conn.execute(f"SELECT file_id, alias FROM aliases WHERE file_id IN ({marks}) ORDER BY rowid", ids):
                notes[row["file_id"]].aliases.append(row["alias"])
            for row in conn.execute(f"SELECT file_id, name, source, line FROM tags WHERE file_id IN ({marks}) ORDER BY rowid", ids):
                notes[row["file_id"]].tags.append(Tag(row["name"], row["source"], row["line"]))
            for row in conn.execute(
                    f"SELECT file_id, level, text, line FROM headings WHERE file_id IN ({marks}) ORDER BY file_id, ordinal", ids):
                notes[row["file_id"]].headings.append(Heading(row["level"], row["text"], row["line"]))
            for row in conn.execute(f"SELECT file_id, block_id, line FROM blocks WHERE file_id IN ({marks}) ORDER BY rowid", ids):
                notes[row["file_id"]].block_ids.append(BlockId(row["block_id"], row["line"]))
            for row in conn.execute(
                    f"SELECT file_id, name, closing, line, content FROM markers WHERE file_id IN ({marks}) ORDER BY file_id, ordinal", ids):
                notes[row["file_id"]].markers.append(Marker(row["name"], bool(row["closing"]), row["line"], row["content"]))
            for row in conn.execute(
                    f"SELECT * FROM external_links WHERE file_id IN ({marks}) ORDER BY file_id, ordinal", ids):
                notes[row["file_id"]].external_links.append(
                    ExternalLink(row["raw"], row["url"], row["display"], bool(row["is_embed"]), row["line"]))
            for row in conn.execute(
                    f"""SELECT source_id, syntax, is_embed, raw, target, heading_path, block_id, display, line, start_offset,
                               end_offset FROM links WHERE source_id IN ({marks}) ORDER BY source_id, ordinal""", ids):
                link = _link(row)
                note = notes[row["source_id"]]
                (note.embeds if link.is_embed else note.links).append(link)
            fields: dict[int, list[sqlite3.Row]] = defaultdict(list)
            for row in conn.execute(f"SELECT * FROM fields WHERE file_id IN ({marks}) ORDER BY file_id, ordinal", ids):
                fields[row["file_id"]].append(row)
            for file_id, rows in fields.items():
                note = notes[file_id]
                every = sorted(note.links + note.embeds, key=lambda item: item.start)
                for row in rows:
                    start, end, line = row["value_start"], row["value_end"], row["line"]
                    links = tuple(link for link in every if link.start >= start and link.end <= end)
                    tags = tuple(tag.name for tag in note.tags if tag.source == "inline" and tag.line == line)
                    kind = HeaderField if row["kind"] == "header" else InlineField
                    field = kind(row["key"], row["value"], line, start, end, links, tags)
                    (note.header_fields if row["kind"] == "header" else note.inline_fields).append(field)
        return {by_id[file_id]: note for file_id, note in notes.items()}

    # ----- queries the formats layer uses ------------------------------------------------------

    def notes_with_header_item(self, key: str, value: str) -> list[str]:
        """Notes whose first ``Key: value`` header field named ``key`` holds [[value]] or #value."""
        rows = self.conn.execute(
            "SELECT DISTINCT f.path FROM header_items h JOIN files f ON f.id = h.file_id WHERE h.key_key = ? AND h.value_key = ?",
            (lookup_key(key), lookup_key(value)),
        )
        stale = self.stale_paths
        found = {row["path"] for row in rows if row["path"] not in stale}
        key_key, value_key = lookup_key(key), lookup_key(value)
        for path, note in self._fresh.items():
            f = next((f for f in note.header_fields if lookup_key(f.key) == key_key), None)  # the first one counts
            if f is not None and (any(lookup_key(l.target) == value_key for l in f.links)
                                  or any(lookup_key(t) == value_key for t in f.tags)):
                found.add(path)
        return sorted(found)


def _link(row: sqlite3.Row) -> Link:
    return Link(
        raw=row["raw"], syntax=row["syntax"], target=row["target"], heading_path=tuple(json.loads(row["heading_path"])),
        block_id=row["block_id"], display=row["display"], is_embed=bool(row["is_embed"]), line=row["line"],
        start=row["start_offset"], end=row["end_offset"],
    )


class IndexGraph:
    """Each note's resolved links and backlinks, from the index's ``links`` table (see the module doc)."""

    def __init__(self, vault: IndexedVault, resolve_aliases: bool = True) -> None:
        self.vault = vault
        self.conn = vault.conn
        self.resolver = Resolver(vault, resolve_aliases=resolve_aliases)
        stale = vault.stale
        affected: set[str] = set()
        for path in stale.deleted:
            affected |= _name_keys(path) | {lookup_key(a) for a in vault.indexed_aliases.get(path, [])}
        for path in stale.added:
            affected |= _name_keys(path)
        for path in stale.changed:
            affected |= _name_keys(path) | {lookup_key(a) for a in vault.indexed_aliases.get(path, [])}
        for note in vault._fresh.values():
            affected |= {lookup_key(a) for a in note.aliases}
        self._stale_sources = {vault.ids[p] for p in vault.stale_paths if p in vault.ids}
        # Stored links an edit may resolve differently now: resolved again.
        self._overrides: dict[tuple[str, int], ResolvedLink] = {}
        self._overrides_to: dict[str, list[tuple[str, int]]] = defaultdict(list)
        if affected:
            keys = sorted(affected)
            for chunk in _chunks(keys):
                marks = ",".join("?" * len(chunk))
                rows = self.conn.execute(
                    f"""SELECT l.*, f.path AS source_path FROM links l JOIN files f ON f.id = l.source_id
                        WHERE l.target_name_key IN ({marks}) OR l.target_key IN ({marks})""", chunk + chunk).fetchall()
                for row in rows:
                    if row["source_id"] in self._stale_sources or (row["source_path"], row["ordinal"]) in self._overrides:
                        continue
                    edge = self.resolver.resolve(row["source_path"], _link(row))
                    self._overrides[(row["source_path"], row["ordinal"])] = edge
        for (source, ordinal), edge in self._overrides.items():
            if edge.status == "resolved" and edge.target_path != source:
                self._overrides_to[edge.target_path].append((source, ordinal))
        # Notes parsed fresh: every link resolved now.
        self._fresh_edges: dict[str, list[ResolvedLink]] = {}
        self._fresh_to: dict[str, list[tuple[str, str | None]]] = defaultdict(list)
        for path, note in vault._fresh.items():
            edges = [self.resolver.resolve(path, link) for link in sorted(note.links + note.embeds, key=lambda l: l.start)]
            self._fresh_edges[path] = edges
            kinds = field_kinds(note)
            for edge in edges:
                if edge.status == "resolved" and edge.target_path != path:
                    self._fresh_to[edge.target_path].append((path, kinds.get(edge.link.start)))
        self._out_rows: dict[str, list[sqlite3.Row]] = {}
        self._prefetched = False

    # ----- stored rows ------------------------------------------------------------------------

    def prefetch_rows(self, paths) -> None:
        """The stored link rows of many notes, in a few queries (for questions about most of the vault)."""
        by_id = {self.vault.ids[p]: p for p in paths if p not in self._out_rows and p in self.vault.ids}
        for path in by_id.values():
            self._out_rows[path] = []
        for ids in _chunks(list(by_id)):
            marks = ",".join("?" * len(ids))
            for row in self.conn.execute(f"SELECT * FROM links WHERE source_id IN ({marks}) ORDER BY source_id, ordinal", ids):
                self._out_rows[by_id[row["source_id"]]].append(row)

    def _rows_of(self, path: str) -> list[sqlite3.Row]:
        if path not in self._out_rows:
            file_id = self.vault.ids.get(path)
            self._out_rows[path] = [] if file_id is None else self.conn.execute(
                "SELECT * FROM links WHERE source_id = ? ORDER BY ordinal", (file_id,)).fetchall()
        return self._out_rows[path]

    def prefetch(self) -> None:
        """Load the columns of every stored link row that ``outgoing_fields`` and ``incoming_fields``
        need, once, as plain tuples (for questions about most of the vault, like the pipeline)."""
        if self._prefetched:
            return
        paths = self.vault.paths_by_id
        cursor = self.conn.cursor()
        cursor.row_factory = None  # tuples: sqlite3.Row objects for every link cost more than the query
        fields: dict[str, list[tuple[int, str | None, str, str | None]]] = defaultdict(list)
        incoming: dict[str, list[tuple[str, int, str | None]]] = defaultdict(list)
        for source_id, ordinal, target_id, status, field_kind in cursor.execute(
                "SELECT source_id, ordinal, target_id, status, field_kind FROM links ORDER BY source_id, ordinal"):
            source = paths[source_id]
            target = paths.get(target_id) if target_id is not None else None
            fields[source].append((ordinal, target, status, field_kind))
            if status == "resolved" and target is not None:
                incoming[target].append((source, ordinal, field_kind))
        self._field_rows = fields
        self._incoming_rows = incoming
        self._prefetched = True

    def _stored_edge(self, source: str, row: sqlite3.Row, link: Link) -> ResolvedLink:
        override = self._overrides.get((source, row["ordinal"]))
        if override is not None:
            return ResolvedLink(source, link, override.status, override.target_path, override.candidates,
                                override.matched_by, override.subpath_status)
        status = row["status"]
        target = self.vault.paths_by_id.get(row["target_id"]) if row["target_id"] is not None else None
        candidates = tuple(json.loads(row["candidates"])) if status == "ambiguous" else ((target,) if target else ())
        return ResolvedLink(source, link, status, target, candidates, row["matched_by"], row["subpath_status"])

    # ----- the graph interface TypedVault uses --------------------------------------------------

    def outgoing(self, path: str) -> list[ResolvedLink]:
        if path in self._fresh_edges:
            return list(self._fresh_edges[path])
        if path not in self.vault.notes:
            return []
        note = self.vault.notes[path]
        links = sorted(note.links + note.embeds, key=lambda l: l.start)
        return [self._stored_edge(path, row, link) for row, link in zip(self._rows_of(path), links)]

    def outgoing_fields(self, path: str) -> list[tuple[str | None, str, str | None]]:
        if path in self._fresh_edges:
            kinds = field_kinds(self.vault.notes[path])
            return [(e.target_path, e.status, kinds.get(e.link.start)) for e in self._fresh_edges[path]]
        if path not in self.vault.notes:
            return []
        if self._prefetched:
            rows = self._field_rows.get(path, [])
        else:
            rows = [(row["ordinal"], self.vault.paths_by_id.get(row["target_id"]) if row["target_id"] is not None else None,
                     row["status"], row["field_kind"]) for row in self._rows_of(path)]
        result = []
        for ordinal, target, status, kind in rows:
            override = self._overrides.get((path, ordinal)) if self._overrides else None
            if override is not None:
                result.append((override.target_path, override.status, kind))
            else:
                result.append((target, status, kind))
        return result

    def incoming_fields(self, path: str) -> list[tuple[str, str | None]]:
        """(source, field kind) of each resolved link to ``path`` from another note."""
        found: list[tuple[str, str | None]] = []
        file_id = self.vault.ids.get(path)
        if file_id is not None and path not in self.vault.stale.deleted:
            if self._prefetched:
                rows = self._incoming_rows.get(path, [])
            else:
                rows = [(self.vault.paths_by_id[r["source_id"]], r["ordinal"], r["field_kind"]) for r in self.conn.execute(
                    "SELECT source_id, ordinal, field_kind FROM links WHERE target_id = ? AND status = 'resolved'", (file_id,))]
            for source, ordinal, kind in rows:
                if source == path or self.vault.ids.get(source) in self._stale_sources or (source, ordinal) in self._overrides:
                    continue
                found.append((source, kind))
        for source, ordinal in self._overrides_to.get(path, []):
            kind = next(r["field_kind"] for r in self._rows_of(source) if r["ordinal"] == ordinal)
            found.append((source, kind))
        found.extend(self._fresh_to.get(path, []))
        return found

    def backlink_sources(self, path: str) -> list[str]:
        return sorted({source for source, _ in self.incoming_fields(path)})


def open_typed(spec, root: str | Path, db_path: str | Path):
    """A TypedVault read from the index (None when there is none usable)."""
    from vaultengine.formats.typed import TypedVault

    vault = IndexedVault.open(root, db_path)
    if vault is None:
        return None
    return TypedVault(spec, vault, IndexGraph(vault))
