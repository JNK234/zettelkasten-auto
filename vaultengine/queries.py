"""Read-only reports computed from the SQLite index (no vault files are read).

Definitions used by ``health``:
* unresolved link   - no file matches the link target;
* ambiguous link    - more than one file matches;
* missing subpath   - the note resolved but the #heading / #^block is not in it;
* orphan            - a note that no other note links to (resolved links/embeds);
* dead end          - a note with no resolved links/embeds to another note
                      (links to attachments, unresolved and self links don't count);
* duplicate basename- two or more files that a bare [[name]] would both match;
* empty note        - nothing but whitespace after the frontmatter.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from vaultengine.vault import link_name

# Set-based (NOT IN over one subquery) rather than correlated NOT EXISTS:
# evaluated once, so the cost stays linear in the number of links whatever
# plan SQLite picks.
_ORPHANS = """
SELECT path FROM files
WHERE kind = 'note' AND id NOT IN (
    SELECT target_id FROM links
    WHERE status = 'resolved' AND target_id IS NOT NULL AND target_id != source_id)
ORDER BY path
"""

_DEAD_ENDS = """
SELECT path FROM files
WHERE kind = 'note' AND id NOT IN (
    SELECT l.source_id FROM links l JOIN files t ON t.id = l.target_id
    WHERE l.status = 'resolved' AND t.kind = 'note' AND l.target_id != l.source_id)
ORDER BY path
"""


def _scalar(conn: sqlite3.Connection, sql: str, *args: Any) -> int:
    return conn.execute(sql, args).fetchone()[0] or 0


def _link_ref(row: sqlite3.Row) -> dict[str, Any]:
    return {"source": row["source"], "line": row["line"], "raw": row["raw"]}


def stats(conn: sqlite3.Connection, top: int = 10) -> dict[str, Any]:
    links = conn.execute(
        """SELECT COUNT(*) AS total,
                  SUM(syntax = 'wikilink') AS wikilink, SUM(syntax = 'markdown') AS markdown,
                  SUM(is_embed) AS embeds,
                  SUM(status = 'resolved') AS resolved, SUM(status = 'unresolved') AS unresolved,
                  SUM(status = 'ambiguous') AS ambiguous,
                  COUNT(DISTINCT CASE WHEN status = 'unresolved' THEN target_key END) AS unresolved_targets
           FROM links"""
    ).fetchone()
    return {
        "notes": _scalar(conn, "SELECT COUNT(*) FROM files WHERE kind = 'note'"),
        "attachments": _scalar(conn, "SELECT COUNT(*) FROM files WHERE kind = 'attachment'"),
        "notes_with_frontmatter": _scalar(conn, "SELECT COUNT(*) FROM notes WHERE has_frontmatter"),
        "frontmatter_errors": _scalar(conn, "SELECT COUNT(*) FROM notes WHERE frontmatter_error IS NOT NULL"),
        "links": {key: links[key] or 0 for key in links.keys()},
        "note_edges": _scalar(conn, "SELECT COUNT(*) FROM note_edges"),
        "tags": {
            "uses": _scalar(conn, "SELECT COUNT(*) FROM tags"),
            "distinct": _scalar(conn, "SELECT COUNT(DISTINCT name_key) FROM tags"),
            "top": [
                {"tag": row["tag"], "notes": row["notes"]}
                for row in conn.execute(
                    "SELECT MIN(name) AS tag, COUNT(DISTINCT file_id) AS notes FROM tags GROUP BY name_key ORDER BY notes DESC, tag LIMIT ?",
                    (top,),
                )
            ],
        },
        "properties": {
            "distinct_keys": _scalar(conn, "SELECT COUNT(DISTINCT key) FROM properties"),
            "top": [
                {"key": row["key"], "notes": row["notes"]}
                for row in conn.execute(
                    "SELECT key, COUNT(DISTINCT file_id) AS notes FROM properties GROUP BY key ORDER BY notes DESC, key LIMIT ?",
                    (top,),
                )
            ],
        },
        "aliases": _scalar(conn, "SELECT COUNT(*) FROM aliases"),
        "headings": _scalar(conn, "SELECT COUNT(*) FROM headings"),
        "blocks": _scalar(conn, "SELECT COUNT(*) FROM blocks"),
        "folders": [
            {"folder": row["folder"], "notes": row["notes"]}
            for row in conn.execute(
                """SELECT CASE WHEN instr(path, '/') = 0 THEN '/' ELSE substr(path, 1, instr(path, '/') - 1) END AS folder,
                          COUNT(*) AS notes
                   FROM files WHERE kind = 'note' GROUP BY folder ORDER BY notes DESC, folder"""
            )
        ],
        "most_linked": [
            {"path": row["path"], "backlinks": row["backlinks"], "linking_notes": row["linking_notes"]}
            for row in conn.execute(
                """SELECT t.path, COUNT(*) AS backlinks, COUNT(DISTINCT l.source_id) AS linking_notes
                   FROM links l JOIN files t ON t.id = l.target_id
                   WHERE l.status = 'resolved' AND l.source_id != l.target_id
                   GROUP BY l.target_id ORDER BY linking_notes DESC, backlinks DESC, t.path LIMIT ?""",
                (top,),
            )
        ],
        "most_wanted": [
            {"target": row["target"], "links": row["links"], "linking_notes": row["linking_notes"]}
            for row in conn.execute(
                """SELECT MIN(target) AS target, COUNT(*) AS links, COUNT(DISTINCT source_id) AS linking_notes
                   FROM links WHERE status = 'unresolved'
                   GROUP BY target_key ORDER BY linking_notes DESC, links DESC, target LIMIT ?""",
                (top,),
            )
        ],
    }


def health(conn: sqlite3.Connection) -> dict[str, Any]:
    unresolved: dict[str, dict[str, Any]] = {}
    for row in conn.execute(
        """SELECT l.target_key, l.target, s.path AS source, l.line, l.raw
           FROM links l JOIN files s ON s.id = l.source_id
           WHERE l.status = 'unresolved' ORDER BY s.path, l.ordinal"""
    ):
        group = unresolved.setdefault(row["target_key"], {"target": row["target"], "links": []})
        group["links"].append(_link_ref(row))
    unresolved_groups = sorted(
        unresolved.values(),
        key=lambda g: (-len({ref["source"] for ref in g["links"]}), -len(g["links"]), g["target"].lower()),
    )

    ambiguous = [
        {**_link_ref(row), "candidates": json.loads(row["candidates"])}
        for row in conn.execute(
            """SELECT s.path AS source, l.line, l.raw, l.candidates FROM links l JOIN files s ON s.id = l.source_id
               WHERE l.status = 'ambiguous' ORDER BY s.path, l.ordinal"""
        )
    ]
    missing_subpaths = [
        {
            **_link_ref(row),
            "target_path": row["target_path"],
            "kind": "block" if row["block_id"] is not None else "heading",
            "subpath": ("^" + row["block_id"]) if row["block_id"] is not None else "#".join(json.loads(row["heading_path"])),
        }
        for row in conn.execute(
            """SELECT s.path AS source, l.line, l.raw, l.block_id, l.heading_path, t.path AS target_path
               FROM links l JOIN files s ON s.id = l.source_id JOIN files t ON t.id = l.target_id
               WHERE l.subpath_status = 'missing' ORDER BY s.path, l.ordinal"""
        )
    ]
    duplicates: dict[str, list[str]] = {}
    for row in conn.execute(
        """SELECT name_key, path FROM files
           WHERE name_key IN (SELECT name_key FROM files GROUP BY name_key HAVING COUNT(*) > 1)
           ORDER BY name_key, path"""
    ):
        duplicates.setdefault(row["name_key"], []).append(row["path"])

    report = {
        "unresolved_links": unresolved_groups,
        "ambiguous_links": ambiguous,
        "missing_subpaths": missing_subpaths,
        "orphans": [row[0] for row in conn.execute(_ORPHANS)],
        "dead_ends": [row[0] for row in conn.execute(_DEAD_ENDS)],
        "duplicate_basenames": [{"name": link_name(paths[0]), "paths": paths} for paths in duplicates.values()],
        "empty_notes": [
            row[0]
            for row in conn.execute(
                "SELECT f.path FROM notes n JOIN files f ON f.id = n.file_id WHERE n.body_chars = 0 ORDER BY f.path"
            )
        ],
    }
    report["summary"] = {
        "unresolved_links": sum(len(g["links"]) for g in unresolved_groups),
        "unresolved_targets": len(unresolved_groups),
        "ambiguous_links": len(ambiguous),
        "missing_subpaths": len(missing_subpaths),
        "orphans": len(report["orphans"]),
        "dead_ends": len(report["dead_ends"]),
        "duplicate_basenames": len(duplicates),
        "empty_notes": len(report["empty_notes"]),
    }
    return report


def pick_candidates(conn: sqlite3.Connection) -> list[tuple[str, str, str]]:
    """(path, field, text) for every note title and alias, for fuzzy pickers."""
    candidates = [(row[0], "title", link_name(row[0])) for row in conn.execute("SELECT path FROM files WHERE kind = 'note' ORDER BY path")]
    candidates += [
        (row[0], "alias", row[1])
        for row in conn.execute("SELECT f.path, a.alias FROM aliases a JOIN files f ON f.id = a.file_id ORDER BY f.path, a.rowid")
    ]
    return candidates


def note_detail(conn: sqlite3.Connection, path: str) -> dict[str, Any] | None:
    """Everything the index knows about one note, or None if it is not indexed."""
    note = conn.execute(
        """SELECT f.id, f.path, n.has_frontmatter, n.frontmatter_error, n.body_chars, n.line_count
           FROM files f JOIN notes n ON n.file_id = f.id WHERE f.path = ?""",
        (path,),
    ).fetchone()
    if note is None:
        return None
    file_id = note["id"]

    def rows(sql: str) -> list[sqlite3.Row]:
        return conn.execute(sql, (file_id,)).fetchall()

    return {
        "path": note["path"],
        "title": link_name(note["path"]),
        "aliases": [row["alias"] for row in rows("SELECT alias FROM aliases WHERE file_id = ? ORDER BY rowid")],
        "has_frontmatter": bool(note["has_frontmatter"]),
        "frontmatter_error": note["frontmatter_error"],
        "properties": {
            row["key"]: json.loads(row["value_json"])
            for row in rows("SELECT key, value_json FROM properties WHERE file_id = ? ORDER BY rowid")
        },
        "tags": [
            {"name": row["name"], "source": row["source"], "line": row["line"]}
            for row in rows("SELECT name, source, line FROM tags WHERE file_id = ? ORDER BY rowid")
        ],
        "headings": [
            {"level": row["level"], "text": row["text"], "line": row["line"]}
            for row in rows("SELECT level, text, line FROM headings WHERE file_id = ? ORDER BY ordinal")
        ],
        "blocks": [{"id": row["block_id"], "line": row["line"]} for row in rows("SELECT block_id, line FROM blocks WHERE file_id = ? ORDER BY rowid")],
        "body_chars": note["body_chars"],
        "line_count": note["line_count"],
        "links_out": [
            {
                "line": row["line"],
                "raw": row["raw"],
                "syntax": row["syntax"],
                "is_embed": bool(row["is_embed"]),
                "target": row["target"],
                "heading_path": json.loads(row["heading_path"]),
                "block_id": row["block_id"],
                "display": row["display"],
                "status": row["status"],
                "target_path": row["target_path"],
                "matched_by": row["matched_by"],
                "candidates": json.loads(row["candidates"]),
                "subpath_status": row["subpath_status"],
            }
            for row in rows(
                """SELECT l.*, t.path AS target_path FROM links l LEFT JOIN files t ON t.id = l.target_id
                   WHERE l.source_id = ? ORDER BY l.ordinal"""
            )
        ],
        "links_in": [
            {
                "source": row["source"],
                "line": row["line"],
                "raw": row["raw"],
                "is_embed": bool(row["is_embed"]),
                "heading_path": json.loads(row["heading_path"]),
                "block_id": row["block_id"],
                "subpath_status": row["subpath_status"],
            }
            for row in rows(
                """SELECT s.path AS source, l.line, l.raw, l.is_embed, l.heading_path, l.block_id, l.subpath_status
                   FROM links l JOIN files s ON s.id = l.source_id
                   WHERE l.target_id = ? AND l.status = 'resolved' AND l.source_id != l.target_id
                   ORDER BY s.path, l.ordinal"""
            )
        ],
    }
