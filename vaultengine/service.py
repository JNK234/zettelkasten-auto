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


class FormatNotConfigured(ServiceError):
    """No format was given (serve --format), or it cannot be loaded."""


class WriteRejected(ServiceError):
    """The write cannot be planned: bad arguments, a name already taken, ... Nothing was written."""


class WriteFailedError(ServiceError):
    """The commit failed and every file was restored, or a file changed meanwhile. Nothing was written."""


class VaultService:
    def __init__(self, vault_root: str | Path, db_path: str | Path | None = None, format_path: str | Path | None = None) -> None:
        self.root = Path(vault_root).resolve()
        self.db_path = Path(db_path).resolve() if db_path else default_db_path(self.root)
        self.format_path = Path(format_path).resolve() if format_path else None
        self._format: tuple[int, Any] | None = None  # (mtime_ns, FormatSpec): reloaded when the file changes
        self._parse_cache: dict[str, Any] = {}  # {path: (sha256, ParsedNote)}: only changed notes are re-parsed
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

    # ----- formats: check, coverage and the safe writes -------------------------------

    def _spec(self):
        from vaultengine.formats import FormatError, load_format

        if self.format_path is None:
            raise FormatNotConfigured("No format configured; start the server with --format <name or file>")
        try:
            mtime = self.format_path.stat().st_mtime_ns
        except OSError as exc:
            raise FormatNotConfigured(f"Format file not readable: {exc}", format=str(self.format_path)) from None
        if self._format is None or self._format[0] != mtime:
            try:
                self._format = (mtime, load_format(self.format_path))
            except FormatError as exc:
                raise FormatNotConfigured(f"Invalid format file: {exc}", format=str(self.format_path)) from None
        return self._format[1]

    def _typed(self):
        from vaultengine.formats import TypedVault
        from vaultengine.vault import Vault

        return TypedVault(self._spec(), Vault.load(self.root, cache=self._parse_cache))

    def check(self, changed: bool = False, limit: int | None = None) -> dict[str, Any]:
        from vaultengine.formats.check import check_report, scope_all, scope_changed, select_issues

        spec = self._spec()
        if changed:
            index = self._built_index()
            stale = index.staleness(self.root)
            last = index.meta().get("last_indexed_at")
        typed = self._typed()
        if changed:
            scope = scope_changed(typed, stale.added, stale.changed, stale.deleted, last, stale.reason)
        else:
            scope = scope_all(typed)
        report = check_report(typed, scope, select_issues(typed, scope))
        report["format"] = {"name": spec.name, "path": str(self.format_path)}
        if limit is not None:
            report["notes"] = report["notes"][:limit]
        report["limit"] = limit
        return report

    def coverage(self, limit: int | None = None) -> dict[str, Any]:
        from vaultengine.formats.coverage import coverage

        typed = self._typed()
        if typed.spec.coverage is None:
            raise FormatNotConfigured(f"Format {typed.spec.name!r} has no `coverage:` block", format=str(self.format_path))
        report = coverage(typed)
        report["format"] = {"name": typed.spec.name, "path": str(self.format_path)}
        if limit is not None:
            report["sources"]["unlinked"] = report["sources"]["unlinked"][:limit]
            report["citing"]["notes"] = report["citing"]["notes"][:limit]
        report["limit"] = limit
        return report

    def _write(self, build, dry_run: bool) -> dict[str, Any]:
        from vaultengine.formats.write import NoteMissing, WriteError, WriteFailed, outcome_to_dict, run_write

        spec = self._spec()
        try:
            outcome = run_write(spec, self.root, self.db_path, build, dry_run=dry_run, parse_cache=self._parse_cache)
        except NoteMissing as exc:
            raise NoteNotFound(str(exc)) from None
        except WriteError as exc:
            raise WriteRejected(str(exc)) from None
        except WriteFailed as exc:
            raise WriteFailedError(str(exc)) from None
        return outcome_to_dict(outcome, dry_run)

    def new(self, type: str, title: str, *, folder: str | None = None, name: str | None = None,
            values: dict[str, str] | None = None, relations: list[tuple[str, str]] | None = None,
            sections: dict[str, str] | None = None, dry_run: bool = False) -> dict[str, Any]:
        from vaultengine.formats.write import plan_new

        return self._write(lambda typed: plan_new(
            typed, type, title, folder=folder, name=name, values=values, relations=relations, sections=sections,
        ), dry_run)

    def link(self, source: str, target: str, relation: str, dry_run: bool = False) -> dict[str, Any]:
        from vaultengine.formats.write import plan_link

        return self._write(lambda typed: plan_link(typed, source, target, relation), dry_run)

    def set(self, note: str, key: str, value: str, dry_run: bool = False) -> dict[str, Any]:
        from vaultengine.formats.write import plan_set

        return self._write(lambda typed: plan_set(typed, note, key, value), dry_run)

    def fix(self, dry_run: bool = False, only: list[str] | None = None) -> dict[str, Any]:
        from vaultengine.formats.fix import FIXABLE, plan_fixes
        from vaultengine.formats.write import FileChange, WritePlan

        found: dict[str, Any] = {}

        def build(typed):
            plan = plan_fixes(typed, only=tuple(only) if only else FIXABLE)
            found["plan"] = plan
            return WritePlan("fix", [FileChange(f.path, f.original, f.new, f.changes) for f in plan.files])

        result = self._write(build, dry_run)
        plan = found["plan"]
        result["skipped"] = [{**issue.to_dict(), "reason": reason} for issue, reason in plan.skipped]
        result["remaining"] = [
            {"code": code, "count": count} for code, count in sorted(plan.remaining.items(), key=lambda kv: (-kv[1], kv[0]))
        ]
        return result

    def ingest(self, *, path: str | None = None, content: str | None = None, filename: str | None = None,
               type: str = "source", title: str | None = None, folder: str | None = None,
               values: dict[str, str] | None = None, dry_run: bool = False) -> dict[str, Any]:
        from vaultengine.formats.ingest import (
            DuplicateFound, document_from_bytes, document_to_dict, plan_ingest, read_document,
        )
        from vaultengine.formats.write import WriteError

        try:
            if path is not None:
                document = read_document(Path(path).expanduser(), title=title)
            else:
                document = document_from_bytes(filename or "document.txt", (content or "").encode("utf-8"), title)
        except WriteError as exc:
            raise WriteRejected(str(exc)) from None
        found: dict[str, Any] = {}
        try:
            result = self._write(
                lambda typed: plan_ingest(typed, document, type, folder=folder, values=values, found=found), dry_run,
            )
        except DuplicateFound as dup:
            return {
                "operation": "ingest", "status": "duplicate", "dry_run": dry_run, "written": False, "files": [],
                "already": [], "new_violations": [], "tolerated": [], "index": None, "recovered": [],
                "document": document_to_dict(document),
                "duplicates": [c.to_dict() for c in dup.duplicates],
                "candidates": [c.to_dict() for c in dup.candidates[:5]],
            }
        result["document"] = document_to_dict(document)
        result["duplicates"] = []
        result["candidates"] = [c.to_dict() for c in found.get("candidates", [])[:5]]
        return result
