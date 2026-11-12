"""The vault-wide numbers the explore lenses compare against, computed once and kept for the next run.

`vault explore` used to compute everything on every run: the distributions its cutoffs are percentiles of
(every unlinked pair's hidden-link score, the role distances, co-citation and coupling values, the
similarity of linked pairs, the near-duplicate cutoff), the vault's key terms, betweenness, constraint,
HITS and the communities. At 5,000 notes that was half a minute, at 50,000 several minutes, for numbers that
do not change between two runs, and barely change when a few notes do.

They are kept in ``stats.sqlite`` next to the memory (``<vault>/.vaultengine/``; derived data, safe to
delete), one set per *fingerprint*: the settings they depend on (seed, resolution, path budget, which links
are tags, topic pages, the embedding model, the near-duplicate percentile) and this module's ``VERSION``.
A set has a *base*: the vault's state (every note's path and content hash, every attachment's path) when
everything was last computed; and, once the vault has moved on a little, an *overlay* for the state now.

On a run, the state now is compared with the base:

exact
    Nothing changed: every stored number is used as it is, so the output is what a full computation gives.
reused
    Some notes changed, at most ``engine.explore.stats_refresh`` of them (2%, at least one note). The
    *distributions* (``tolerant=True``) and the near-duplicate cutoff are used from the base: a percentile over
    thousands of values hardly moves when a few of them change. So are the key terms of the notes that did not
    change; the changed and new ones are computed now. Everything about particular notes (the communities,
    betweenness, constraint, HITS, roles) is computed for the vault as it is now, and kept in the overlay, so
    the next run on the same state is as fast as an exact one. The output says when the base was computed and
    how many notes changed since.
fresh
    Nothing stored, different settings, more notes changed than that, or ``--refresh``: everything is
    computed and becomes the new base.

Items are pickled one per row and read only when a lens asks for them.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import pickle
import sqlite3
import zlib
from pathlib import Path
from typing import Any, Callable

VERSION = 1  # bump when a stored number is computed differently

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sets (fingerprint TEXT PRIMARY KEY, at TEXT NOT NULL, notes INTEGER NOT NULL,
                                 base BLOB NOT NULL, overlay TEXT);
CREATE TABLE IF NOT EXISTS items (fingerprint TEXT NOT NULL, name TEXT NOT NULL, layer TEXT NOT NULL, data BLOB NOT NULL,
                                  PRIMARY KEY (fingerprint, name, layer));
"""


def vault_state(conn: sqlite3.Connection) -> dict[str, str | None]:
    """path -> content hash of every note (None for an attachment): what the numbers were computed from."""
    return {path: (digest if kind == "note" else None)
            for path, kind, digest in conn.execute("SELECT path, kind, content_hash FROM files ORDER BY path")}


def state_hash(state: dict[str, str | None]) -> str:
    return hashlib.sha256(json.dumps(sorted(state.items())).encode()).hexdigest()


def fingerprint(**settings: Any) -> str:
    return hashlib.sha256(json.dumps({"version": VERSION, **settings}, sort_keys=True, default=str).encode()).hexdigest()[:24]


class NoStats:
    """Compute everything, keep nothing (`vault deepen`, `discover`, `map`: they compute as they always did)."""

    mode = "off"
    changed: set[str] = set()
    removed: set[str] = set()

    def get(self, name: str, compute: Callable[[], Any], tolerant: bool = False) -> Any:
        return compute()

    def key_terms(self, compute_all: Callable[[], dict[str, Any]], compute_one: Callable[[str], Any]) -> dict[str, Any]:
        return compute_all()

    def describe(self) -> dict[str, Any]:
        return {"mode": "off"}

    def save(self) -> None:
        pass


def _pack(value: Any) -> bytes:
    return zlib.compress(pickle.dumps(value, protocol=5), 1)


def _unpack(blob: bytes) -> Any:
    return pickle.loads(zlib.decompress(blob))


class Stats(NoStats):
    def __init__(self, path: str | Path, fingerprint_: str, state: dict[str, str | None], refresh_share: float = 0.02,
                 refresh: bool = False) -> None:
        self.path = Path(path)
        self.fingerprint = fingerprint_
        self.state = state
        self.now = state_hash(state)
        self.values: dict[str, Any] = {}  # what this run computed or read
        self.computed: set[str] = set()  # what this run computed (to store)
        self.at: str | None = None
        self.changed: set[str] = set()  # notes added or edited since the base
        self.removed: set[str] = set()
        self.overlay_ok = False  # the overlay was computed for the state now
        notes = sum(1 for digest in state.values() if digest is not None)
        self.threshold = max(1, int(refresh_share * notes))
        row = None
        if self.path.is_file():
            with self._connect() as conn:
                row = conn.execute("SELECT at, base, overlay FROM sets WHERE fingerprint = ?", (self.fingerprint,)).fetchone()
        if row is None or refresh:
            self.mode = "fresh"
            self.why = "--refresh" if row is not None else "no stored numbers for these settings"
            return
        self.at, base, overlay = row[0], _unpack(row[1]), row[2]
        self.changed = {p for p, digest in state.items() if base.get(p, "absent") != digest}
        self.removed = {p for p in base if p not in state}
        moved = len(self.changed) + len(self.removed)
        if moved == 0:
            self.mode, self.why = "exact", "nothing changed since"
        elif moved <= self.threshold:
            self.mode, self.why = "reused", f"{moved} changed since, at most {self.threshold} reuse them"
            self.overlay_ok = overlay == self.now
        else:
            self.mode, self.why = "fresh", f"{moved} notes changed since, more than {self.threshold}"
            self.changed, self.removed = set(), set()

    def _connect(self):
        """A connection for one read or write, closed after it (a server may keep an explorer for long)."""
        from contextlib import closing

        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path)
        conn.executescript(_SCHEMA)
        return closing(conn)

    def _stored(self, name: str, layer: str) -> Any:
        if not self.path.is_file():
            return None
        with self._connect() as conn:
            row = conn.execute("SELECT data FROM items WHERE fingerprint = ? AND name = ? AND layer = ?",
                               (self.fingerprint, name, layer)).fetchone()
        return _unpack(row[0]) if row is not None else None

    def get(self, name: str, compute: Callable[[], Any], tolerant: bool = False) -> Any:
        """A number (or structure) by name: stored, when the mode allows it (``tolerant``: fine from the base of a
        few notes ago), else computed now."""
        if name in self.values:
            return self.values[name]
        value = None
        if self.mode == "exact" or (self.mode == "reused" and tolerant):
            value = self._stored(name, "base")
        if value is None and self.overlay_ok:
            value = self._stored(name, "overlay")
        if value is None:
            value = compute()
            self.computed.add(name)
        self.values[name] = value
        return value

    def key_terms(self, compute_all: Callable[[], dict[str, Any]], compute_one: Callable[[str], Any]) -> dict[str, Any]:
        """path -> key terms: the base's for notes that did not change (``reused``), computed for the rest."""
        if "key_terms" in self.values:
            return self.values["key_terms"]
        terms = self._stored("key_terms", "overlay") if self.overlay_ok else None
        if terms is None:
            stored = self._stored("key_terms", "base") if self.mode in ("exact", "reused") else None
            if stored is None:
                terms = compute_all()
            else:
                terms = {p: t for p, t in stored.items() if p not in self.removed}
                for path in sorted(self.changed):
                    if self.state.get(path) is not None:
                        terms[path] = compute_one(path)
            if stored is None or self.mode == "reused":
                self.computed.add("key_terms")
        self.values["key_terms"] = terms
        return terms

    def describe(self) -> dict[str, Any]:
        return {"mode": self.mode, "why": self.why, "computed_at": self.at if self.mode != "fresh" else None,
                "changed_since": len(self.changed) + len(self.removed), "threshold": self.threshold,
                "file": str(self.path)}

    def save(self) -> None:
        """Store what this run computed: as the new base (``fresh``), added to the base (``exact``), or in the
        overlay for the state now (``reused``: the base stays what the drift is measured from)."""
        if not self.computed:
            return
        now = dt.datetime.now().replace(microsecond=0).isoformat()
        with self._connect() as conn, conn:
            if self.mode == "fresh" and self.at is None:  # the first save of a fresh run replaces the set
                conn.execute("DELETE FROM items WHERE fingerprint = ?", (self.fingerprint,))
                conn.execute("INSERT OR REPLACE INTO sets VALUES (?, ?, ?, ?, NULL)",
                             (self.fingerprint, now, sum(1 for d in self.state.values() if d is not None), _pack(self.state)))
                self.at = now
            if self.mode == "reused":
                if not self.overlay_ok:  # a new overlay for the state now
                    conn.execute("DELETE FROM items WHERE fingerprint = ? AND layer = 'overlay'", (self.fingerprint,))
                    conn.execute("UPDATE sets SET overlay = ? WHERE fingerprint = ?", (self.now, self.fingerprint))
                    self.overlay_ok = True
                layer = "overlay"
            else:
                layer = "base"
            for name in sorted(self.computed):
                conn.execute("INSERT OR REPLACE INTO items VALUES (?, ?, ?, ?)",
                             (self.fingerprint, name, layer, _pack(self.values[name])))
        self.computed.clear()
