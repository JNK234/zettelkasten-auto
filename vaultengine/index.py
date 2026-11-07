"""Incremental SQLite index of a vault.

The index is a cache: everything in it can be rebuilt from the vault files.

Change detection, per file:
1. size and mtime match the stored values  -> skipped without reading;
2. otherwise the file is read and hashed (SHA-256) -> parsed only if the
   hash differs from the stored one.
A file modified within RACY_WINDOW_NS of an index run is stored with a sentinel
mtime so the next run re-hashes it (an edit in the same clock tick as the
run could otherwise go unnoticed; git handles "racily clean" files the same way).

Resolution is incremental too, but it cannot be limited to changed files:
adding "GRPO.md" resolves [[GRPO]] in untouched notes, a second "Index.md"
makes existing [[Index]] links ambiguous, and renaming a heading changes
subpath checks elsewhere. A link can only be affected if its target's
basename or full text equals a name/alias key of a file that was added,
deleted or changed. Those links are re-resolved from stored rows, with no
other files read or parsed. The invariant "incremental == full rebuild" is
tested with randomised edit sequences.

Stored versions (schema, parser, resolver, alias setting, vault root) force a
rebuild, full re-parse or full re-resolution when they change.
"""

from __future__ import annotations

import hashlib
import json
import posixpath
import sqlite3
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from vaultengine.model import BlockId, Heading, Link, ParsedNote
from vaultengine.parser import PARSER_VERSION, normalize_text, parse_note
from vaultengine.resolver import RESOLVER_VERSION, Resolver
from vaultengine.vault import Vault, is_markdown, iter_vault_files, link_name, lookup_key

SCHEMA_VERSION = 1
DEFAULT_DB_RELATIVE = Path(".vaultengine") / "index.sqlite"
RACY_WINDOW_NS = 2_000_000_000
RACY_MTIME = -1

_SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);

-- Every vault file. kind = 'note' (markdown) or 'attachment' (anything else).
CREATE TABLE files (
    id           INTEGER PRIMARY KEY,
    path         TEXT NOT NULL UNIQUE,      -- vault-relative, '/'-separated, NFC
    name_key     TEXT NOT NULL,             -- lookup key of the name links use
    kind         TEXT NOT NULL CHECK (kind IN ('note', 'attachment')),
    size         INTEGER NOT NULL,
    mtime_ns     INTEGER NOT NULL,          -- -1: re-hash on next run
    content_hash TEXT                       -- SHA-256 of the bytes (notes only)
);
CREATE INDEX files_name_key ON files (name_key);

CREATE TABLE notes (
    file_id           INTEGER PRIMARY KEY REFERENCES files (id) ON DELETE CASCADE,
    has_frontmatter   INTEGER NOT NULL,
    frontmatter_error TEXT,
    body_start_line   INTEGER NOT NULL,
    body_chars        INTEGER NOT NULL,     -- non-whitespace characters after frontmatter
    line_count        INTEGER NOT NULL
);

CREATE TABLE properties (
    file_id    INTEGER NOT NULL REFERENCES files (id) ON DELETE CASCADE,
    key        TEXT NOT NULL,
    value_json TEXT NOT NULL
);
CREATE INDEX properties_file ON properties (file_id);
CREATE INDEX properties_key ON properties (key);

CREATE TABLE aliases (
    file_id   INTEGER NOT NULL REFERENCES files (id) ON DELETE CASCADE,
    alias     TEXT NOT NULL,
    alias_key TEXT NOT NULL
);
CREATE INDEX aliases_file ON aliases (file_id);
CREATE INDEX aliases_key ON aliases (alias_key);

CREATE TABLE tags (
    file_id  INTEGER NOT NULL REFERENCES files (id) ON DELETE CASCADE,
    name     TEXT NOT NULL,                 -- as written, without '#'
    name_key TEXT NOT NULL,                 -- case-folded; nested: 'a/b' LIKE 'a/%'
    source   TEXT NOT NULL CHECK (source IN ('frontmatter', 'inline')),
    line     INTEGER
);
CREATE INDEX tags_file ON tags (file_id);
CREATE INDEX tags_key ON tags (name_key);

CREATE TABLE headings (
    file_id INTEGER NOT NULL REFERENCES files (id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL,
    level   INTEGER NOT NULL,
    text    TEXT NOT NULL,
    line    INTEGER NOT NULL,
    PRIMARY KEY (file_id, ordinal)
);

CREATE TABLE blocks (
    file_id  INTEGER NOT NULL REFERENCES files (id) ON DELETE CASCADE,
    block_id TEXT NOT NULL,
    line     INTEGER NOT NULL
);
CREATE INDEX blocks_file ON blocks (file_id);

-- One row per link or embed. Parse columns come from the source note;
-- resolution columns are maintained by the resolver.
CREATE TABLE links (
    id              INTEGER PRIMARY KEY,
    source_id       INTEGER NOT NULL REFERENCES files (id) ON DELETE CASCADE,
    ordinal         INTEGER NOT NULL,
    syntax          TEXT NOT NULL,          -- 'wikilink' | 'markdown'
    is_embed        INTEGER NOT NULL,
    raw             TEXT NOT NULL,
    target          TEXT NOT NULL,          -- as written; '' for same-note links
    heading_path    TEXT NOT NULL,          -- JSON array
    block_id        TEXT,
    display         TEXT,
    line            INTEGER NOT NULL,
    start_offset    INTEGER NOT NULL,
    end_offset      INTEGER NOT NULL,
    target_key      TEXT NOT NULL,          -- lookup key of the whole target
    target_name_key TEXT NOT NULL,          -- lookup key of the target's last path part
    status          TEXT NOT NULL,          -- 'resolved' | 'ambiguous' | 'unresolved'
    target_id       INTEGER REFERENCES files (id) ON DELETE SET NULL,
    matched_by      TEXT,                   -- 'self' | 'path' | 'name' | 'alias'
    candidates      TEXT NOT NULL,          -- JSON array of paths when ambiguous
    subpath_status  TEXT,                   -- NULL | 'found' | 'missing' | 'not_checked'
    UNIQUE (source_id, ordinal)
);
CREATE INDEX links_target ON links (target_id);
CREATE INDEX links_target_key ON links (target_key);
CREATE INDEX links_target_name_key ON links (target_name_key);

-- Resolved note-to-note edges with multiplicity (self-links excluded):
-- the input for graph algorithms.
CREATE VIEW note_edges AS
SELECT s.path AS source, t.path AS target, COUNT(*) AS weight
FROM links l
JOIN files s ON s.id = l.source_id
JOIN files t ON t.id = l.target_id
WHERE l.status = 'resolved' AND t.kind = 'note' AND l.target_id != l.source_id
GROUP BY l.source_id, l.target_id;
"""

class IndexNeedsRebuild(RuntimeError):
    """A partial update was asked for, but the index must first be rebuilt with `vault index`."""


_NOTE_CHILD_TABLES = ("notes", "properties", "aliases", "tags", "headings", "blocks")


def default_db_path(vault_root: str | Path) -> Path:
    return Path(vault_root) / DEFAULT_DB_RELATIVE


def _name_keys(path: str) -> set[str]:
    return {lookup_key(posixpath.basename(path)), lookup_key(link_name(path))}


def _target_name_key(target: str) -> str:
    return lookup_key(posixpath.basename(target))


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class IndexReport:
    vault: str
    db: str
    notes: int = 0
    attachments: int = 0
    added: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)
    forced: list[str] = field(default_factory=list)  # unchanged but re-parsed (full run)
    deleted: list[str] = field(default_factory=list)
    attachments_added: list[str] = field(default_factory=list)
    attachments_deleted: list[str] = field(default_factory=list)
    hashed: int = 0  # note files read and hashed
    full_reason: str | None = None
    links_checked: int = 0  # links re-resolved this run
    links_changed_elsewhere: int = 0  # of those, links in notes NOT re-parsed whose result changed
    seconds: float = 0.0

    @property
    def parsed(self) -> list[str]:
        return sorted(self.added + self.changed + self.forced)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["parsed"] = self.parsed
        return data


@dataclass
class Staleness:
    added: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    reason: str | None = None

    @property
    def is_stale(self) -> bool:
        return bool(self.added or self.changed or self.deleted or self.reason)

    def summary(self) -> str:
        if self.reason:
            return self.reason
        return f"{len(self.added)} added, {len(self.changed)} changed, {len(self.deleted)} deleted"


class VaultIndex:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")
        # The index is a rebuildable cache: in WAL mode NORMAL can lose the last
        # commit on power loss but cannot corrupt the database.
        self.conn.execute("PRAGMA synchronous = NORMAL")
        self._ensure_schema()

    @classmethod
    def open_existing(cls, db_path: str | Path) -> "VaultIndex":
        if not Path(db_path).exists():
            raise FileNotFoundError(db_path)
        return cls(db_path)

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "VaultIndex":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ----- schema & meta ------------------------------------------------

    def _ensure_schema(self) -> None:
        version = self.conn.execute("PRAGMA user_version").fetchone()[0]
        if version == SCHEMA_VERSION:
            return
        objects = self.conn.execute(
            "SELECT type, name FROM sqlite_master WHERE name NOT LIKE 'sqlite_%' AND type IN ('table', 'view')"
        ).fetchall()
        self.conn.execute("PRAGMA foreign_keys = OFF")
        for row in objects:
            self.conn.execute(f'DROP {row["type"].upper()} IF EXISTS "{row["name"]}"')
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(f"BEGIN;\n{_SCHEMA}\nPRAGMA user_version = {SCHEMA_VERSION};\nCOMMIT;")

    def meta(self) -> dict[str, str]:
        return {row["key"]: row["value"] for row in self.conn.execute("SELECT key, value FROM meta")}

    def _set_meta(self, **values: object) -> None:
        self.conn.executemany(
            "INSERT INTO meta (key, value) VALUES (?, ?) ON CONFLICT (key) DO UPDATE SET value = excluded.value",
            [(key, str(value)) for key, value in values.items()],
        )

    def _invalidation(self, root: Path, resolve_aliases: bool) -> tuple[str | None, str | None]:
        """(reason to re-parse everything, reason to re-resolve everything)."""
        meta = self.meta()
        if "parser_version" not in meta:
            return None, None  # new index: everything is added anyway
        reparse = None
        if meta["parser_version"] != str(PARSER_VERSION):
            reparse = f"parser version {meta['parser_version']} -> {PARSER_VERSION}"
        elif meta.get("vault_root") != str(root):
            reparse = f"vault root changed from {meta.get('vault_root')}"
        reresolve = reparse
        if reresolve is None and meta.get("resolver_version") != str(RESOLVER_VERSION):
            reresolve = f"resolver version {meta.get('resolver_version')} -> {RESOLVER_VERSION}"
        if reresolve is None and meta.get("resolve_aliases") != str(resolve_aliases):
            reresolve = "alias resolution setting changed"
        return reparse, reresolve

    # ----- update -------------------------------------------------------

    def update(
        self, vault_root: str | Path, *, full: bool = False, resolve_aliases: bool = True, only: Iterable[str] | None = None,
    ) -> IndexReport:
        """Bring the index up to date with the vault.

        ``only``: vault-relative paths to update, and nothing else (used by the write
        commands, so that other pending edits stay "changed since the last index").
        A path that no longer exists is removed. Refused with ``IndexNeedsRebuild`` when
        the index needs a full re-parse or re-resolution, which a partial update cannot do.
        """
        started = time.perf_counter()
        scan_started_ns = time.time_ns()
        root = Path(vault_root).resolve()
        report = IndexReport(vault=str(root), db=str(self.db_path))
        reparse_reason, reresolve_reason = self._invalidation(root, resolve_aliases)
        if full:
            reparse_reason = reresolve_reason = "full re-index requested"
        selected = None if only is None else {unicodedata.normalize("NFC", path) for path in only}
        if selected is not None and ("parser_version" not in self.meta() or reparse_reason or reresolve_reason):
            raise IndexNeedsRebuild(reparse_reason or reresolve_reason or "index has never been built")
        report.full_reason = reresolve_reason

        def kept_mtime(mtime_ns: int) -> int:
            return RACY_MTIME if mtime_ns >= scan_started_ns - RACY_WINDOW_NS else mtime_ns

        disk = {}
        if selected is None:
            for rel, full_path in iter_vault_files(root):
                stat = full_path.stat()
                disk[rel] = (full_path, stat.st_size, stat.st_mtime_ns)
        else:
            for rel in selected:
                full_path = root / rel
                if full_path.is_file():
                    stat = full_path.stat()
                    disk[rel] = (full_path, stat.st_size, stat.st_mtime_ns)

        conn = self.conn
        conn.execute("BEGIN IMMEDIATE")
        try:
            rows = {row["path"]: row for row in conn.execute("SELECT id, path, kind, size, mtime_ns, content_hash FROM files")}
            affected_keys: set[str] = set()
            reparsed_ids: list[int] = []

            gone = rows.keys() - disk.keys() if selected is None else (rows.keys() & selected) - disk.keys()
            for path in sorted(gone):
                row = rows[path]
                affected_keys |= _name_keys(path) | self._alias_keys(row["id"])
                conn.execute("DELETE FROM files WHERE id = ?", (row["id"],))
                (report.deleted if row["kind"] == "note" else report.attachments_deleted).append(path)

            for path in sorted(disk):
                full_path, size, mtime_ns = disk[path]
                row = rows.get(path)
                if not is_markdown(path):
                    report.attachments += 1
                    if row is None:
                        conn.execute(
                            "INSERT INTO files (path, name_key, kind, size, mtime_ns) VALUES (?, ?, 'attachment', ?, ?)",
                            (path, lookup_key(link_name(path)), size, kept_mtime(mtime_ns)),
                        )
                        affected_keys |= _name_keys(path)
                        report.attachments_added.append(path)
                    elif (row["size"], row["mtime_ns"]) != (size, mtime_ns):
                        conn.execute("UPDATE files SET size = ?, mtime_ns = ? WHERE id = ?", (size, kept_mtime(mtime_ns), row["id"]))
                    continue

                report.notes += 1
                if row is not None and reparse_reason is None and (row["size"], row["mtime_ns"]) == (size, mtime_ns):
                    continue
                data = full_path.read_bytes()
                report.hashed += 1
                digest = hashlib.sha256(data).hexdigest()
                unchanged = row is not None and row["content_hash"] == digest
                if unchanged and reparse_reason is None:
                    conn.execute("UPDATE files SET size = ?, mtime_ns = ? WHERE id = ?", (size, kept_mtime(mtime_ns), row["id"]))
                    continue

                text = normalize_text(data.decode("utf-8", errors="replace"))
                note = parse_note(text, path=path)
                if row is None:
                    file_id = conn.execute(
                        "INSERT INTO files (path, name_key, kind, size, mtime_ns, content_hash) VALUES (?, ?, 'note', ?, ?, ?)",
                        (path, lookup_key(link_name(path)), size, kept_mtime(mtime_ns), digest),
                    ).lastrowid
                    report.added.append(path)
                else:
                    file_id = row["id"]
                    affected_keys |= self._alias_keys(file_id)
                    self._delete_note_rows(file_id)
                    conn.execute(
                        "UPDATE files SET size = ?, mtime_ns = ?, content_hash = ? WHERE id = ?",
                        (size, kept_mtime(mtime_ns), digest, file_id),
                    )
                    (report.forced if unchanged else report.changed).append(path)
                self._write_note(file_id, note, text)
                affected_keys |= _name_keys(path) | {lookup_key(alias) for alias in note.aliases}
                reparsed_ids.append(file_id)

            if reresolve_reason or affected_keys or reparsed_ids:
                report.links_checked, report.links_changed_elsewhere = self._resolve(
                    everything=reresolve_reason is not None,
                    keys=affected_keys,
                    source_ids=reparsed_ids,
                    resolve_aliases=resolve_aliases,
                )
            if selected is not None:  # totals of the whole index, not of the selection
                report.notes, report.attachments = conn.execute(
                    "SELECT SUM(kind = 'note'), SUM(kind = 'attachment') FROM files"
                ).fetchone()
                report.notes, report.attachments = report.notes or 0, report.attachments or 0
            changed = bool(
                reparsed_ids or report.deleted or report.attachments_added or report.attachments_deleted or reresolve_reason
            )
            generation = int(self.meta().get("generation", "0")) + (1 if changed else 0)
            self._set_meta(
                generation=generation,  # bumped only when the index content changed; a cache key for readers
                schema_version=SCHEMA_VERSION,
                parser_version=PARSER_VERSION,
                resolver_version=RESOLVER_VERSION,
                resolve_aliases=resolve_aliases,
                vault_root=root,
                last_indexed_at=_now_iso(),
            )
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        # Keep planner statistics current for ad-hoc and graph queries (cheap no-op when nothing changed much).
        conn.execute("PRAGMA analysis_limit = 1000")
        conn.execute("PRAGMA optimize")
        report.seconds = round(time.perf_counter() - started, 3)
        return report

    def _alias_keys(self, file_id: int) -> set[str]:
        return {row[0] for row in self.conn.execute("SELECT alias_key FROM aliases WHERE file_id = ?", (file_id,))}

    def _delete_note_rows(self, file_id: int) -> None:
        for table in _NOTE_CHILD_TABLES:
            self.conn.execute(f"DELETE FROM {table} WHERE file_id = ?", (file_id,))
        self.conn.execute("DELETE FROM links WHERE source_id = ?", (file_id,))

    def _write_note(self, file_id: int, note: ParsedNote, text: str) -> None:
        conn = self.conn
        body = "\n".join(text.split("\n")[note.body_start_line - 1 :])
        conn.execute(
            "INSERT INTO notes (file_id, has_frontmatter, frontmatter_error, body_start_line, body_chars, line_count) VALUES (?, ?, ?, ?, ?, ?)",
            (file_id, note.has_frontmatter, note.frontmatter_error, note.body_start_line,
             sum(1 for char in body if not char.isspace()), text.count("\n") + 1 if text else 0),
        )
        conn.executemany(
            "INSERT INTO properties (file_id, key, value_json) VALUES (?, ?, ?)",
            [(file_id, key, json.dumps(value, default=str, ensure_ascii=False)) for key, value in note.frontmatter.items()],
        )
        conn.executemany(
            "INSERT INTO aliases (file_id, alias, alias_key) VALUES (?, ?, ?)",
            [(file_id, alias, lookup_key(alias)) for alias in note.aliases],
        )
        conn.executemany(
            "INSERT INTO tags (file_id, name, name_key, source, line) VALUES (?, ?, ?, ?, ?)",
            [(file_id, tag.name, lookup_key(tag.name), tag.source, tag.line) for tag in note.tags],
        )
        conn.executemany(
            "INSERT INTO headings (file_id, ordinal, level, text, line) VALUES (?, ?, ?, ?, ?)",
            [(file_id, index, h.level, h.text, h.line) for index, h in enumerate(note.headings)],
        )
        conn.executemany(
            "INSERT INTO blocks (file_id, block_id, line) VALUES (?, ?, ?)",
            [(file_id, block.id, block.line) for block in note.block_ids],
        )
        links = sorted(note.links + note.embeds, key=lambda link: link.start)
        conn.executemany(
            """INSERT INTO links (source_id, ordinal, syntax, is_embed, raw, target, heading_path, block_id, display,
                                  line, start_offset, end_offset, target_key, target_name_key, status, candidates)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'unresolved', '[]')""",
            [
                (file_id, index, link.syntax, link.is_embed, link.raw, link.target,
                 json.dumps(list(link.heading_path), ensure_ascii=False), link.block_id, link.display,
                 link.line, link.start, link.end, lookup_key(link.target), _target_name_key(link.target))
                for index, link in enumerate(links)
            ],
        )

    # ----- resolution ---------------------------------------------------

    def _resolution_vault(self) -> tuple[Vault, dict[str, int]]:
        """An in-memory Vault holding just what the resolver needs (no file reads)."""
        ids = {row["path"]: row["id"] for row in self.conn.execute("SELECT id, path FROM files")}
        notes = {
            row["path"]: ParsedNote(path=row["path"])
            for row in self.conn.execute("SELECT path FROM files WHERE kind = 'note'")
        }
        for row in self.conn.execute("SELECT f.path, a.alias FROM aliases a JOIN files f ON f.id = a.file_id ORDER BY a.rowid"):
            notes[row["path"]].aliases.append(row["alias"])
        for row in self.conn.execute(
            "SELECT f.path, h.level, h.text, h.line FROM headings h JOIN files f ON f.id = h.file_id ORDER BY h.file_id, h.ordinal"
        ):
            notes[row["path"]].headings.append(Heading(row["level"], row["text"], row["line"]))
        for row in self.conn.execute("SELECT f.path, b.block_id, b.line FROM blocks b JOIN files f ON f.id = b.file_id ORDER BY b.rowid"):
            notes[row["path"]].block_ids.append(BlockId(row["block_id"], row["line"]))
        return Vault(ids.keys(), notes), ids

    def _resolve(self, *, everything: bool, keys: set[str], source_ids: list[int], resolve_aliases: bool) -> tuple[int, int]:
        conn = self.conn
        vault, ids = self._resolution_vault()
        resolver = Resolver(vault, resolve_aliases=resolve_aliases)
        select = "SELECT l.*, f.path AS source_path FROM links l JOIN files f ON f.id = l.source_id"
        if everything:
            rows = conn.execute(select).fetchall()
        else:
            conn.execute("CREATE TEMP TABLE IF NOT EXISTS affected_keys (key TEXT PRIMARY KEY)")
            conn.execute("CREATE TEMP TABLE IF NOT EXISTS affected_sources (id INTEGER PRIMARY KEY)")
            conn.execute("DELETE FROM affected_keys")
            conn.execute("DELETE FROM affected_sources")
            conn.executemany("INSERT OR IGNORE INTO affected_keys VALUES (?)", [(key,) for key in keys])
            conn.executemany("INSERT OR IGNORE INTO affected_sources VALUES (?)", [(i,) for i in source_ids])
            rows = conn.execute(
                select
                + """ WHERE l.source_id IN (SELECT id FROM affected_sources)
                      OR l.target_name_key IN (SELECT key FROM affected_keys)
                      OR l.target_key IN (SELECT key FROM affected_keys)"""
            ).fetchall()

        reparsed = set(source_ids)
        updates = []
        changed_elsewhere = 0
        for row in rows:
            link = Link(
                raw=row["raw"], syntax=row["syntax"], target=row["target"],
                heading_path=tuple(json.loads(row["heading_path"])), block_id=row["block_id"],
                display=row["display"], is_embed=bool(row["is_embed"]), line=row["line"],
            )
            result = resolver.resolve(row["source_path"], link)
            new = (
                result.status,
                ids.get(result.target_path) if result.target_path else None,
                result.matched_by,
                json.dumps(list(result.candidates) if result.status == "ambiguous" else [], ensure_ascii=False),
                result.subpath_status,
            )
            if new != (row["status"], row["target_id"], row["matched_by"], row["candidates"], row["subpath_status"]):
                updates.append((*new, row["id"]))
                changed_elsewhere += row["source_id"] not in reparsed
        conn.executemany(
            "UPDATE links SET status = ?, target_id = ?, matched_by = ?, candidates = ?, subpath_status = ? WHERE id = ?",
            updates,
        )
        return len(rows), changed_elsewhere

    # ----- freshness ----------------------------------------------------

    def staleness(self, vault_root: str | Path, resolve_aliases: bool = True) -> Staleness:
        """What ``update`` would change, without writing anything."""
        root = Path(vault_root).resolve()
        meta = self.meta()
        if "parser_version" not in meta:
            return Staleness(reason="index has never been built")
        reparse, reresolve = self._invalidation(root, resolve_aliases)
        if reparse or reresolve:
            return Staleness(reason=reparse or reresolve)
        rows = {row["path"]: row for row in self.conn.execute("SELECT path, kind, size, mtime_ns, content_hash FROM files")}
        result = Staleness()
        seen = set()
        for rel, full_path in iter_vault_files(root):
            seen.add(rel)
            row = rows.get(rel)
            if row is None:
                result.added.append(rel)
                continue
            if row["kind"] != "note":
                continue
            stat = full_path.stat()
            if (row["size"], row["mtime_ns"]) != (stat.st_size, stat.st_mtime_ns):
                if hashlib.sha256(full_path.read_bytes()).hexdigest() != row["content_hash"]:
                    result.changed.append(rel)
        result.deleted = sorted(rows.keys() - seen)
        return result
