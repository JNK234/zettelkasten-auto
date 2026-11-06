"""Transport-independent operations on one vault and its index.

Front-ends (the JSON-RPC server today, others later) call these methods and
map the domain errors below to their own error codes. Results are plain dicts
shaped like the models in ``vaultengine.rpc.models``.
"""

from __future__ import annotations

import unicodedata
from pathlib import Path
from typing import Any

from vaultengine import queries
from vaultengine.fuzzy import Prepared, fuzzy_match, prepare
from vaultengine.index import VaultIndex, default_db_path
from vaultengine.vault import link_name


class ServiceError(Exception):
    def __init__(self, message: str, **data: Any) -> None:
        super().__init__(message)
        self.message = message
        self.data = data


class IndexNotBuilt(ServiceError):
    pass


class NoteNotFound(ServiceError):
    pass


class VaultService:
    def __init__(self, vault_root: str | Path, db_path: str | Path | None = None) -> None:
        self.root = Path(vault_root).resolve()
        self.db_path = Path(db_path).resolve() if db_path else default_db_path(self.root)
        self._index: VaultIndex | None = None
        # Prepared pick candidates, keyed by the index generation they were built from.
        self._pick_cache: tuple[str | None, list[tuple[str, str, Prepared]]] | None = None

    def close(self) -> None:
        if self._index is not None:
            self._index.close()
            self._index = None

    # ----- helpers ------------------------------------------------------

    def _open(self, create: bool) -> VaultIndex | None:
        if self._index is None:
            if not create and not self.db_path.exists():
                return None
            self._index = VaultIndex(self.db_path)
        return self._index

    def _built_index(self) -> VaultIndex:
        index = self._open(create=False)
        if index is None or "parser_version" not in index.meta():
            raise IndexNotBuilt("Index has not been built; call index.build first", db=str(self.db_path))
        return index

    # ----- operations ---------------------------------------------------

    def build(self, full: bool = False) -> dict[str, Any]:
        index = self._open(create=True)
        return index.update(self.root, full=full).to_dict()

    def status(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "vault": str(self.root),
            "db": str(self.db_path),
            "built": False,
            "last_indexed_at": None,
            "schema_version": None,
            "parser_version": None,
            "resolver_version": None,
            "stale": None,
            "pending": None,
            "notes": 0,
            "attachments": 0,
            "links": 0,
        }
        index = self._open(create=False)
        if index is None:
            return result
        meta = index.meta()
        if "parser_version" not in meta:
            return result
        stale = index.staleness(self.root)
        counts = index.conn.execute(
            """SELECT (SELECT COUNT(*) FROM files WHERE kind = 'note'),
                      (SELECT COUNT(*) FROM files WHERE kind = 'attachment'),
                      (SELECT COUNT(*) FROM links)"""
        ).fetchone()
        result.update(
            built=True,
            last_indexed_at=meta.get("last_indexed_at"),
            schema_version=int(meta["schema_version"]),
            parser_version=int(meta["parser_version"]),
            resolver_version=int(meta["resolver_version"]),
            stale=stale.is_stale,
            pending={"added": stale.added, "changed": stale.changed, "deleted": stale.deleted, "reason": stale.reason},
            notes=counts[0],
            attachments=counts[1],
            links=counts[2],
        )
        return result

    def stats(self, top: int = 10) -> dict[str, Any]:
        return queries.stats(self._built_index().conn, top=top)

    def health(self, limit: int | None = None) -> dict[str, Any]:
        report = queries.health(self._built_index().conn)
        if limit is not None:
            for key in ("unresolved_links", "ambiguous_links", "missing_subpaths", "orphans", "dead_ends", "duplicate_basenames", "empty_notes"):
                report[key] = report[key][:limit]
        report["limit"] = limit
        return report

    def _pick_candidates(self) -> list[tuple[str, str, Prepared]]:
        index = self._built_index()
        stamp = index.meta().get("generation")  # also catches re-indexing by another process
        if self._pick_cache is None or self._pick_cache[0] != stamp:
            prepared = [(path, field, prepare(text)) for path, field, text in queries.pick_candidates(index.conn)]
            self._pick_cache = (stamp, prepared)
        return self._pick_cache[1]

    def pick(self, query: str, limit: int = 20) -> dict[str, Any]:
        """Best match per note over its title (file name) and aliases."""
        best: dict[str, dict[str, Any]] = {}
        for path, field, prepared in self._pick_candidates():
            text = prepared.text
            match = fuzzy_match(query, prepared)
            if match is None:
                continue
            candidate = {
                "path": path,
                "title": link_name(path),
                "matched_field": field,
                "matched_text": text,
                "score": match.score,
                "positions": list(match.positions),
            }
            current = best.get(path)
            if current is None or (match.score, -len(text)) > (current["score"], -len(current["matched_text"])):
                best[path] = candidate
        ranked = sorted(best.values(), key=lambda c: (-c["score"], len(c["matched_text"]), c["path"]))
        return {"query": query, "total": len(ranked), "results": ranked[:limit]}

    def note(self, path: str) -> dict[str, Any]:
        detail = queries.note_detail(self._built_index().conn, unicodedata.normalize("NFC", path))
        if detail is None:
            raise NoteNotFound(f"No indexed note at {path!r}", path=path)
        return detail
