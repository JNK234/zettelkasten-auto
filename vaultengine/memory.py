"""What `vault explore` remembers across sessions: the engine's own store, never the notes.

One SQLite file (``engine.memory``, default ``<vault>/.vaultengine/memory.sqlite``), apart from the index so a
rebuilt index (a new schema, ``vault index --full``) loses nothing. It holds:

sessions, session_results
    Each `vault explore` run and the results it showed, with their refs (``2.3``: section 2, result 3), so
    ``vault mark`` and ``vault link-it`` can name a result of the last run.
seen
    Every result shown, by (start, lens, target): ``--fresh`` hides what was seen before, with its first and
    last time and how often.
marks
    keep / dismiss per (start, lens, target), with what the result said (title, evidence): ``vault kept`` lists
    the kept ones; the keep rate of a lens (kept / marked) orders explore's sections.
links
    Every link ``vault link-it --write`` made: both notes, the lines it wrote, whether it created the section and
    whether it put a blank line before that section's heading. ``vault unlink`` undoes exactly those.
snapshots
    ``vault map --record``: a small summary of the map (communities by their notes, the gaps, the missing
    notes, the communities without a topic page), zlib-compressed JSON; ``vault map --since`` compares.

A *start* is what explore started from: a note path, ``[[missing note]]``, or "" (no starting point). A
*target* is what a result is about: a note path, ``a ↔ b`` for a pair, ``[[name]]`` for a missing note,
``community: …`` for a community. Times are ISO 8601, local time, to the second; a mark's to the microsecond,
so marks made within one second still list in the order they were made (``kept``, oldest first).
"""

from __future__ import annotations

import datetime as dt
import json
import sqlite3
import zlib
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = 2  # 2: links.blank_a, links.blank_b (a file of version 1 gains them when opened to write)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY, at TEXT NOT NULL, start TEXT NOT NULL, fresh INTEGER NOT NULL, lenses TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS session_results (
    session INTEGER NOT NULL REFERENCES sessions(id), ref TEXT NOT NULL, lens TEXT NOT NULL, kind TEXT NOT NULL,
    target TEXT NOT NULL, title TEXT NOT NULL, evidence TEXT NOT NULL, notes TEXT NOT NULL, link_a TEXT, link_b TEXT,
    PRIMARY KEY (session, ref)
);
CREATE TABLE IF NOT EXISTS seen (
    start TEXT NOT NULL, lens TEXT NOT NULL, target TEXT NOT NULL, first TEXT NOT NULL, last TEXT NOT NULL,
    times INTEGER NOT NULL, PRIMARY KEY (start, lens, target)
);
CREATE TABLE IF NOT EXISTS marks (
    start TEXT NOT NULL, lens TEXT NOT NULL, target TEXT NOT NULL, verdict TEXT NOT NULL CHECK (verdict IN ('keep', 'dismiss')),
    kind TEXT NOT NULL, title TEXT NOT NULL, evidence TEXT NOT NULL, link_a TEXT, link_b TEXT, at TEXT NOT NULL,
    PRIMARY KEY (start, lens, target)
);
CREATE TABLE IF NOT EXISTS links (
    id INTEGER PRIMARY KEY, at TEXT NOT NULL, a TEXT NOT NULL, b TEXT NOT NULL, lens TEXT, section TEXT NOT NULL,
    line_a TEXT NOT NULL, line_b TEXT NOT NULL, created_a INTEGER NOT NULL, created_b INTEGER NOT NULL,
    undone TEXT, blank_a INTEGER, blank_b INTEGER
);
CREATE TABLE IF NOT EXISTS snapshots (id INTEGER PRIMARY KEY, at TEXT NOT NULL, data BLOB NOT NULL);
"""


def now(precise: bool = False) -> str:
    """Local time, ISO 8601: to the second, or to the microsecond when ``precise``."""
    moment = dt.datetime.now()
    return moment.isoformat(timespec="microseconds") if precise else moment.replace(microsecond=0).isoformat()


class MemoryStoreError(ValueError):
    """A ref or date that names nothing in the memory; the message says what there is."""


class Memory:
    def __init__(self, path: str | Path, readonly: bool = False) -> None:
        """``readonly``: open an existing file without writing a byte of it (``vault kept``, ``map --since``)."""
        self.path = Path(path)
        if readonly:
            self.conn = sqlite3.connect(f"file:{self.path.resolve()}?mode=ro", uri=True)
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        version = self.conn.execute("PRAGMA user_version").fetchone()[0]
        if version not in ((1, SCHEMA_VERSION) if readonly else (0, 1, SCHEMA_VERSION)):
            raise MemoryStoreError(f"{self.path}: memory schema {version}, this engine knows {SCHEMA_VERSION}")
        if not readonly and version == 0:
            self.conn.executescript(_SCHEMA + f"PRAGMA user_version = {SCHEMA_VERSION};")
        elif not readonly and version == 1:  # links made then have no blank_a / blank_b: NULL, "not recorded"
            self.conn.executescript("ALTER TABLE links ADD COLUMN blank_a INTEGER; ALTER TABLE links ADD COLUMN blank_b INTEGER;"
                                    f"PRAGMA user_version = {SCHEMA_VERSION};")

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Memory":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ----- sessions ------------------------------------------------------------------------------

    def record_session(self, start: str, fresh: bool, lenses: list[str], results: Iterable[dict[str, Any]]) -> int:
        """Store a run and the results it showed; mark them seen. Returns the session id."""
        at = now()
        with self.conn:
            session = self.conn.execute("INSERT INTO sessions (at, start, fresh, lenses) VALUES (?, ?, ?, ?)",
                                        (at, start, int(fresh), json.dumps(lenses))).lastrowid
            for r in results:
                link = r.get("link") or (None, None)
                self.conn.execute(
                    "INSERT INTO session_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (session, r["ref"], r["lens"], r["kind"], r["target"], r["title"], r["evidence"],
                     json.dumps(r.get("notes", [])), link[0], link[1]))
                self.conn.execute(
                    """INSERT INTO seen VALUES (?, ?, ?, ?, ?, 1) ON CONFLICT (start, lens, target)
                       DO UPDATE SET last = excluded.last, times = times + 1""",
                    (start, r["lens"], r["target"], at, at))
        return session

    def last_session(self) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM sessions ORDER BY id DESC LIMIT 1").fetchone()

    def result(self, ref: str) -> dict[str, Any]:
        """A result of the last explore run by its ref (``2.3``)."""
        session = self.last_session()
        if session is None:
            raise MemoryStoreError("no explore run yet: `vault explore` first, then mark its results by their numbers (2.3)")
        row = self.conn.execute("SELECT * FROM session_results WHERE session = ? AND ref = ?", (session["id"], ref)).fetchone()
        if row is None:
            refs = [r[0] for r in self.conn.execute("SELECT ref FROM session_results WHERE session = ?", (session["id"],))]
            shown = ", ".join(refs[:8]) + (" …" if len(refs) > 8 else "")
            raise MemoryStoreError(f"no result {ref} in the last explore run "
                                   f"({'from ' + session['start'] if session['start'] else 'no starting point'}); "
                                   f"its results: {shown or 'none'}")
        out = dict(row)
        out["start"] = session["start"]
        out["notes"] = json.loads(out["notes"])
        out["link"] = (row["link_a"], row["link_b"]) if row["link_a"] else None
        return out

    # ----- seen and marks ---------------------------------------------------------------------------

    def seen_targets(self, start: str) -> set[str]:
        """Everything shown from this start before, in any lens."""
        return {r[0] for r in self.conn.execute("SELECT target FROM seen WHERE start = ?", (start,))}

    def mark(self, result: dict[str, Any], verdict: str) -> None:
        link = result.get("link") or (None, None)
        with self.conn:
            self.conn.execute(
                """INSERT INTO marks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT (start, lens, target)
                   DO UPDATE SET verdict = excluded.verdict, at = excluded.at, title = excluded.title,
                   evidence = excluded.evidence""",
                (result["start"], result["lens"], result["target"], verdict, result["kind"], result["title"],
                 result["evidence"], link[0], link[1], now(precise=True)))

    def unmark(self, result: dict[str, Any]) -> bool:
        with self.conn:
            cursor = self.conn.execute("DELETE FROM marks WHERE start = ? AND lens = ? AND target = ?",
                                       (result["start"], result["lens"], result["target"]))
        return cursor.rowcount > 0

    def marks_for(self, start: str) -> dict[tuple[str, str], str]:
        """(lens, target) -> verdict, for one start."""
        return {(r["lens"], r["target"]): r["verdict"]
                for r in self.conn.execute("SELECT lens, target, verdict FROM marks WHERE start = ?", (start,))}

    def dismissed_targets(self, start: str) -> set[str]:
        return {r[0] for r in self.conn.execute("SELECT target FROM marks WHERE start = ? AND verdict = 'dismiss'", (start,))}

    def keep_counts(self) -> dict[str, tuple[int, int]]:
        """lens -> (kept, dismissed), over every start."""
        out: dict[str, list[int]] = {}
        for lens, verdict, n in self.conn.execute("SELECT lens, verdict, COUNT(*) FROM marks GROUP BY lens, verdict"):
            out.setdefault(lens, [0, 0])[0 if verdict == "keep" else 1] += n
        return {k: (v[0], v[1]) for k, v in out.items()}

    def kept(self, lens: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM marks WHERE verdict = 'keep'" + (" AND lens = ?" if lens else "") + " ORDER BY at, start, lens, target"
        return [dict(r) for r in self.conn.execute(sql, (lens,) if lens else ())]

    # ----- links ----------------------------------------------------------------------------------

    def record_link(self, a: str, b: str, lens: str | None, section: str, line_a: str, line_b: str,
                    created_a: bool, created_b: bool, blank_a: bool = False, blank_b: bool = False) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT INTO links (at, a, b, lens, section, line_a, line_b, created_a, created_b, blank_a, blank_b)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (now(), a, b, lens, section, line_a, line_b, int(created_a), int(created_b), int(blank_a), int(blank_b)))

    def link_between(self, a: str, b: str) -> dict[str, Any] | None:
        """The last link link-it made between two notes (either order) that is not undone, oriented as (a, b)."""
        row = self.conn.execute(
            "SELECT * FROM links WHERE undone IS NULL AND ((a = ? AND b = ?) OR (a = ? AND b = ?)) ORDER BY id DESC LIMIT 1",
            (a, b, b, a)).fetchone()
        if row is None:
            return None
        out = dict(row)
        if out["a"] != a:  # orient it as asked
            out["a"], out["b"] = out["b"], out["a"]
            out["line_a"], out["line_b"] = out["line_b"], out["line_a"]
            out["created_a"], out["created_b"] = out["created_b"], out["created_a"]
            out["blank_a"], out["blank_b"] = out.get("blank_b"), out.get("blank_a")
        return out

    def undo_link(self, link_id: int) -> None:
        with self.conn:
            self.conn.execute("UPDATE links SET undone = ? WHERE id = ?", (now(), link_id))

    # ----- map snapshots ------------------------------------------------------------------------------

    def save_snapshot(self, data: dict[str, Any]) -> str:
        at = now()
        blob = zlib.compress(json.dumps(data, sort_keys=True).encode("utf-8"), 9)
        with self.conn:
            self.conn.execute("INSERT INTO snapshots (at, data) VALUES (?, ?)", (at, blob))
        return at

    def snapshots(self) -> list[tuple[int, str, int]]:
        """(id, time, bytes) of every snapshot, oldest first."""
        return [(r[0], r[1], r[2]) for r in self.conn.execute("SELECT id, at, length(data) FROM snapshots ORDER BY id")]

    def snapshot(self, snapshot_id: int) -> dict[str, Any]:
        row = self.conn.execute("SELECT data FROM snapshots WHERE id = ?", (snapshot_id,)).fetchone()
        return json.loads(zlib.decompress(row[0]).decode("utf-8"))

    def snapshot_since(self, when: str) -> tuple[int, str, str]:
        """The snapshot that stands for the vault as it was at ``when`` (a date or an ISO time): the first one at or
        after it, else the last one before it. Returns (id, its time, how it was chosen)."""
        try:
            moment = dt.datetime.fromisoformat(when)
        except ValueError:
            raise MemoryStoreError(f"--since {when!r}: give a date (2026-11-10) or a time (2026-11-10T14:30)") from None
        start = moment.replace(microsecond=0).isoformat()
        rows = self.snapshots()
        if not rows:
            raise MemoryStoreError("no map snapshot yet: `vault map --record` saves one")
        after = [r for r in rows if r[1] >= start]
        if after:
            return after[0][0], after[0][1], f"the first snapshot at or after {when}"
        before = rows[-1]
        return before[0], before[1], f"no snapshot at or after {when}: the last one before it"
