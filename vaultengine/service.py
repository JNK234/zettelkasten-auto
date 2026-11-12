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


class InvalidQuery(ServiceError):
    """A search query that does not parse; data says where (start/end offsets in the query) and how to fix it."""


class NotAvailable(ServiceError):
    """Nothing to act on: no explore run yet, a ref that is not in it, a result that is not two notes to link, no map
    snapshot, a lens that does not run from this start. The message says what there is instead."""
class NoBaseline(ServiceError):
    """vault.check with baseline=true, and no baseline is recorded (or it cannot be read)."""


class VaultService:
    def __init__(self, vault_root: str | Path, db_path: str | Path | None = None, format_path: str | Path | None = None,
                 settings: Any = None) -> None:
        """``settings``: engine settings (``vaultengine.settings``) for the embedding model; default: the configured ones."""
        self.root = Path(vault_root).resolve()
        self._settings = settings
        self.db_path = Path(db_path).resolve() if db_path else default_db_path(self.root)
        self.format_path = Path(format_path).resolve() if format_path else None
        self._format: tuple[int, Any] | None = None  # (mtime_ns, FormatSpec): reloaded when the file changes
        self._parse_cache: dict[str, Any] = {}  # {path: (sha256, ParsedNote)}: only changed notes are re-parsed
        self._index: VaultIndex | None = None
        # Prepared pick candidates, keyed by the index generation they were built from.
        self._pick_cache: tuple[str | None, list[tuple[str, str, Prepared]]] | None = None
        # Link graph, tag weights and note vectors for related/suggest, keyed by the index generation.
        self._relatedness: tuple[str | None, Any] | None = None

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

    def check(self, changed: bool = False, limit: int | None = None, baseline: bool = False,
              update_baseline: bool = False) -> dict[str, Any]:
        """``baseline``: only violations not in the recorded baseline count (``ok`` is about those);
        ``update_baseline``: record the whole vault's violations as the baseline (the only write of a check)."""
        from vaultengine.formats.check import (
            BaselineError, BaselineResult, baseline_path, check_report, compare_to_baseline, load_baseline,
            record_baseline, scope_all, scope_changed, select_issues,
        )

        spec = self._spec()
        if baseline and update_baseline:
            raise WriteRejected("baseline compares with the baseline, update_baseline records it; give one")
        baseline_file = baseline_path(self.root, spec.name) if baseline or update_baseline else None
        recorded = None
        if baseline:
            if not baseline_file.exists():
                raise NoBaseline(f"No baseline for format {spec.name!r}; record one with vault.check update_baseline=true",
                                 path=str(baseline_file))
            try:
                recorded = load_baseline(baseline_file, spec.name)
            except BaselineError as exc:
                raise NoBaseline(str(exc), path=str(baseline_file)) from None
        if changed:
            index = self._built_index()
            stale = index.staleness(self.root)
            last = index.meta().get("last_indexed_at")
        typed = self._typed()
        if changed:
            scope = scope_changed(typed, stale.added, stale.changed, stale.deleted, last, stale.reason)
        else:
            scope = scope_all(typed)
        issues = select_issues(typed, scope)
        result = None
        if update_baseline:
            previous = None
            if baseline_file.exists():
                try:
                    previous = len(load_baseline(baseline_file, spec.name).violations)
                except BaselineError:
                    previous = None
            written = record_baseline(typed, baseline_file)
            result = BaselineResult(written, [], issues, [], recorded=True, previous=previous)
        elif recorded is not None:
            result = compare_to_baseline(typed, issues, recorded)
        report = check_report(typed, scope, result.new if result else issues)
        report["format"] = {"name": spec.name, "path": str(self.format_path)}
        report["baseline"] = result.to_dict() if result is not None else None
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

    def _write(self, build, dry_run: bool, after=None) -> dict[str, Any]:
        """``after(outcome)``: called once the write is done (or planned); what it returns is added to the result."""
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
        extra = after(outcome) if after is not None else None
        return {**outcome_to_dict(outcome, dry_run), **(extra or {})}

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

    # ----- the reading pipeline: pipeline, note facts, pages, findings, ideas, moves, next -----------

    def _reading(self):
        """The vault through the format for a read: from the index when there is a usable one (notes changed
        since are parsed from disk), else from every file. Close it with ``_done``."""
        from vaultengine.indexed import open_typed

        spec = self._spec()
        typed = open_typed(spec, self.root, self.db_path) if self.db_path.exists() else None
        return typed if typed is not None else self._typed()

    @staticmethod
    def _done(typed) -> None:
        if hasattr(typed.vault, "close"):
            typed.vault.close()

    def _needs(self, block: str):
        spec = self._spec()
        if getattr(spec, block) is None:
            raise FormatNotConfigured(f"Format {spec.name!r} has no `{block}:` block", format=str(self.format_path))
        return spec

    def _find(self, typed, ref: str) -> str:
        from vaultengine.formats.write import NoteMissing, find_note

        try:
            return find_note(typed, ref)
        except NoteMissing as exc:
            raise NoteNotFound(str(exc), note=ref) from None

    def pipeline(self, limit: int | None = None) -> dict[str, Any]:
        from vaultengine.formats.pipeline import Pipeline

        self._needs("pipeline")
        typed = self._reading()
        try:
            report = Pipeline(typed).report()
        finally:
            self._done(typed)
        if limit is not None:
            report["sources"] = report["sources"][:limit]
            report["extract"]["unsourced"]["notes"] = report["extract"]["unsourced"]["notes"][:limit]
            report["gaps"] = {key: items[:limit] for key, items in report["gaps"].items()}
        report["limit"] = limit
        return report

    def note_facts(self, note: str) -> dict[str, Any]:
        from vaultengine.formats.pipeline import Pipeline

        self._needs("pipeline")
        typed = self._reading()
        try:
            pipeline = Pipeline(typed)
            return pipeline.note(self._find(typed, note))
        finally:
            self._done(typed)

    def _state(self, outcome) -> dict[str, Any]:
        return {"state": [{"path": f.path, "changes": list(f.changes)} for f in outcome.plan.side_files]}

    def page_new(self, title: str, zettels: list[str] | None = None, queries: list[str] | None = None, limit: int = 10,
                 tags: str | None = None, folder: str | None = None, dry_run: bool = False) -> dict[str, Any]:
        """A page from zettels: ``zettels`` (refs) and the best-ranked zettels of each query in ``queries``."""
        from vaultengine.formats.pages import PageError, Pages

        self._needs("pages")
        ranked = [(query, [hit.path for hit in self._search(query, limit=None, facets=0).hits]) for query in queries or []]
        found: dict[str, Any] = {"from": []}

        def build(typed):
            pages = Pages(typed)
            chosen = [self._find(typed, ref) for ref in zettels or []]
            for query, paths in ranked:
                hits = [p for p in paths if p in typed.vault.notes and pages.is_zettel(p)]
                found["from"].append({"query": query, "matched": len(paths), "zettels": len(hits),
                                      "taken": hits[:limit]})
                chosen.extend(hits[:limit])
            if not chosen:
                raise PageError("no zettels to make the page from (zettels and queries gave none)")
            plan = pages.plan_new(title, list(dict.fromkeys(chosen)), folder=folder, tags=tags)
            found["page"] = {"path": plan.files[0].path, "content": plan.files[0].new}
            return plan

        return self._write(build, dry_run, after=lambda outcome: {**self._state(outcome), "page": found["page"],
                                                                   "from": found["from"]})

    def page_refresh(self, page: str, dry_run: bool = False) -> dict[str, Any]:
        from vaultengine.formats.pages import Pages

        self._needs("pages")
        return self._write(lambda typed: Pages(typed).plan_refresh(page), dry_run, after=self._state)

    def page_exclude(self, page: str, zettels: list[str], undo: bool = False, dry_run: bool = False) -> dict[str, Any]:
        from vaultengine.formats.pages import Pages

        self._needs("pages")
        return self._write(lambda typed: Pages(typed).plan_exclude(page, zettels, undo=undo), dry_run, after=self._state)

    def page_check(self, page: str | None = None) -> dict[str, Any]:
        """One page, or every page; ``ready``: every managed page (or the one asked about) is ready to write."""
        from vaultengine.formats.pages import PageError, Pages

        spec = self._needs("pages")
        typed = self._reading()
        try:
            pages = Pages(typed)
            targets = [self._find(typed, page)] if page else pages.pages()
            try:
                results = [pages.check(p) for p in targets]
            except PageError as exc:
                raise WriteRejected(str(exc)) from None
        finally:
            self._done(typed)
        attention = [r for r in results if (r["managed"] or page) and not r["ready"]]
        return {"format": {"name": spec.name, "path": str(self.format_path)}, "pages": results, "ready": not attention,
                "managed": sum(1 for r in results if r["managed"]),
                "ready_pages": sum(1 for r in results if r["managed"] and r["ready"])}

    def _findings(self):
        from vaultengine.findings import Findings, counts_for, skip_for

        spec = self._spec()
        return Findings(self._built_index().conn, skip_for(spec), counts_for(spec), self._engine())

    def _index_info(self) -> dict[str, Any]:
        index = self._built_index()
        stale = index.staleness(self.root)
        return {"db": str(self.db_path), "last_indexed_at": index.meta().get("last_indexed_at"), "stale": stale.is_stale,
                "pending": {"added": stale.added, "changed": stale.changed, "deleted": stale.deleted, "reason": stale.reason}}

    def findings(self, methods: list[str] | None = None, id: str | None = None, limit: int | None = None) -> dict[str, Any]:
        """Findings from the index (read only). ``limit``: per method; ``total`` counts them all."""
        findings = self._findings()
        items = findings.all(methods or None)
        if id is not None:
            items = [f for f in items if f.id == id]
            if not items:
                raise NoteNotFound(f"No finding {id!r} now (findings.list gives the ids)", id=id)
        shown = items
        if limit is not None:
            per: dict[str, int] = {}
            shown = []
            for finding in items:
                per[finding.method] = per.get(finding.method, 0) + 1
                if per[finding.method] <= limit:
                    shown.append(finding)
        by_method: dict[str, int] = {}
        for finding in items:
            by_method[finding.method] = by_method.get(finding.method, 0) + 1
        return {"index": self._index_info(), "total": len(items), "by_method": by_method,
                "thresholds": findings.thresholds, "findings": [f.to_dict() for f in shown], "limit": limit}

    def idea_new(self, finding: str, title: str | None = None, tags: str | None = None, dry_run: bool = False) -> dict[str, Any]:
        from vaultengine.formats.ideas import plan_idea_new

        self._needs("ideas")
        found = self._findings().get(finding)
        if found is None:
            raise NoteNotFound(f"No finding {finding!r} now (findings.list gives the ids)", id=finding)
        made: dict[str, Any] = {}

        def build(typed):
            plan = plan_idea_new(typed, found, title=title, tags=tags)
            made["idea"] = {"path": plan.files[0].path, "content": plan.files[0].new}
            return plan

        return self._write(build, dry_run, after=lambda outcome: {**self._state(outcome), "finding": found.to_dict(),
                                                                   "idea": made["idea"]})

    def idea_set(self, idea: str, status: str, dry_run: bool = False) -> dict[str, Any]:
        from vaultengine.formats.ideas import plan_idea_set

        self._needs("ideas")
        return self._write(lambda typed: plan_idea_set(typed, idea, status), dry_run, after=self._state)

    def _relocated(self, outcome) -> dict[str, Any]:
        from dataclasses import asdict

        report = getattr(outcome.plan, "report", None)
        return {**self._state(outcome), "moved": dict(outcome.plan.renamed),
                "rewritten": list(report.rewritten) if report else [],
                "now_resolving": list(report.now_resolving) if report else [],
                "self_links": [asdict(link) for link in report.self_links] if report else []}

    def move(self, note: str, to: str, dry_run: bool = False) -> dict[str, Any]:
        from vaultengine.formats.moves import plan_move

        return self._write(lambda typed: plan_move(typed, note, to), dry_run, after=self._relocated)

    def merge(self, survivor: str, other: str, dry_run: bool = False) -> dict[str, Any]:
        from vaultengine.formats.moves import plan_merge

        return self._write(lambda typed: plan_merge(typed, survivor, other), dry_run, after=self._relocated)

    def next(self, limit: int | None = 10) -> dict[str, Any]:
        """The queue of what needs doing, most trusted first; each item's ``command`` resolves it (CLI)."""
        from vaultengine.formats.todo import Queue

        spec = self._needs("pipeline")
        findings = self._findings() if self.db_path.exists() else None
        typed = self._reading()
        try:
            queue = Queue(typed, str(self.root), str(self.format_path), findings)
            items = queue.items()
        finally:
            self._done(typed)
        shown = items if limit is None else items[:limit]
        return {"format": {"name": spec.name, "path": str(self.format_path)}, "total": len(items), "counts": queue.counts,
                "items": [item.to_dict() for item in shown], "limit": limit}

    # ----- search, related notes, suggestions, unlinked mentions ----------

    def _provider(self):
        """The embedding provider, loaded once for the life of the service.

        Raises EmbeddingError or SettingsError; ``TextSimilarity`` turns that into "text unavailable"."""
        from vaultengine.settings import load_settings

        if self._settings is None:
            self._settings = load_settings()
        return self._settings.provider()

    @staticmethod
    def _text_status(text) -> dict[str, Any]:
        on = text.available()
        return {"available": on, "model_id": text.vectors.model_id if on else None, "unavailable": text.unavailable,
                "uncached": len(text.uncached)}

    def _engine(self, text: bool = True):
        """Relatedness for the current index generation (vectors are computed in memory, never written)."""
        from vaultengine.related import Relatedness

        index = self._built_index()
        stamp = index.meta().get("generation")
        if self._relatedness is None or self._relatedness[0] != stamp:
            from vaultengine.formats import FormatError
            from vaultengine.tagrule import default_rule, rule_for_format

            try:  # which links are tags: the format's tag fields, else fields named tags
                rule = rule_for_format(self.format_path) if self.format_path else default_rule()
            except FormatError as exc:
                raise FormatNotConfigured(f"Invalid format file: {exc}", format=str(self.format_path)) from None
            weights = self._settings.engine.related.weights.model_dump() if self._settings is not None else None
            self._relatedness = (stamp, Relatedness(index.conn, self._provider, self.root, rule, weights))
        return self._relatedness[1]

    def _note_id(self, ref: str, missing_ok: bool = False) -> tuple[int | None, Any]:
        """(file id, seed) for a note reference; a missing link target gives (None, seed) when missing_ok."""
        from vaultengine.querylang import QueryError

        engine = self._engine(text=False)  # resolving a name needs no embeddings
        try:
            seed = engine.seed(ref)
        except QueryError as exc:
            raise NoteNotFound(exc.message, note=ref, **({"hint": exc.hint} if exc.hint else {})) from None
        if seed.path is None and not missing_ok:
            raise NoteNotFound(f'"{ref}" is not a note yet (only links to it exist)', note=ref)
        return (engine.graph.note_ids[seed.node] if seed.path else None), seed

    def _search(self, query: str, **options: Any):
        from vaultengine.querylang import QueryError
        from vaultengine.search import search

        try:
            return search(self._built_index().conn, query, **options)
        except QueryError as exc:
            raise InvalidQuery(exc.message, query=query, start=exc.start, end=exc.end, hint=exc.hint) from None

    def search(self, query: str, limit: int = 10, fuzzy: bool = True, facets: int = 8, snippet_width: int = 100) -> dict[str, Any]:
        from vaultengine.snippets import hit_context

        result = self._search(query, limit=limit, fuzzy=fuzzy, facets=facets)

        def marks(pieces) -> list[dict[str, Any]]:
            found, position = [], 0
            for piece in pieces:
                if piece.mark:
                    found.append({"start": position, "end": position + len(piece.text), "kind": piece.mark})
                position += len(piece.text)
            return found

        hits = []
        for hit in result.hits:
            context = hit_context(self.root, hit, query, width=snippet_width)
            snippet = None
            if context.snippet is not None:
                snippet = {
                    "field": context.snippet.field,
                    "text": context.snippet.text,
                    "marks": marks(context.snippet.pieces),
                    "truncated_start": context.snippet.truncated_start,
                    "truncated_end": context.snippet.truncated_end,
                }
            hits.append({
                "path": hit.path,
                "title": hit.title,
                "score": hit.score,
                "exact": hit.exact,
                "matches": [
                    {"clause": m.clause, "exact": m.exact, "term": m.term, "edits": m.edits,
                     "fields": [{"field": name, "count": f.tf} for name, f in m.fields.items()]}
                    for m in hit.matches.values()
                ],
                "title_marks": marks(context.title),
                "snippet": snippet,
            })
        clauses = [
            {"label": label, "kind": stats.kind, "required": stats.required, "notes": stats.df,
             "variants": [{"term": term, "edits": edits, "notes": stats.variant_df[term]} for term, edits in stats.variants.items()]}
            for label, stats in result.clauses.items()
        ]
        order = "path" if not result.clauses else ("exact_then_score" if len(result.clauses) > 1 or any(c["variants"] for c in clauses) else "score")
        return {
            "query": query,
            "tree": result.tree,
            "notes": result.notes,
            "total": result.total,
            "order": order,
            "warnings": result.warnings,
            "clauses": clauses,
            "hits": hits,
            "facets": {
                "tags": [{"name": name, "notes": n} for name, n in (result.facets.tags if result.facets else [])],
                "folders": [{"name": name, "notes": n} for name, n in (result.facets.folders if result.facets else [])],
            },
        }

    def preview(self, path: str, query: str | None = None) -> dict[str, Any]:
        from vaultengine.snippets import preview

        path = unicodedata.normalize("NFC", path)
        index = self._built_index()
        if index.conn.execute("SELECT 1 FROM files WHERE path = ? AND kind = 'note'", (path,)).fetchone() is None:
            raise NoteNotFound(f"No indexed note at {path!r}", path=path)
        hit = None
        if query and query.strip():
            hit = next((h for h in self._search(query, limit=None, facets=0).hits if h.path == path), None)
        lines = preview(self.root, path, hit, query)
        if lines is None:
            raise NoteNotFound(f"{path!r} is in the index but not on disk; run index.build", path=path)
        marked = [line for line in lines if line.marks]
        return {
            "path": path,
            "title": link_name(path),
            "query": query,
            "matched": hit is not None,
            "marks": sum(len(line.marks) for line in lines),
            "first_match_line": marked[0].number if marked else None,
            "lines": [{"number": line.number, "text": line.text,
                       "marks": [{"start": a, "end": b, "kind": kind} for a, b, kind in line.marks]} for line in lines],
        }

    def related(self, note: str, limit: int = 10, unlinked: bool = False) -> dict[str, Any]:
        _, seed = self._note_id(note, missing_ok=True)
        from vaultengine.related import UNLINKED_WEIGHTS

        unlinked_weights = (self._settings.engine.related.unlinked_weights.model_dump() if self._settings is not None
                            else UNLINKED_WEIGHTS)
        result = self._engine().related(seed, limit=limit, unlinked_only=unlinked,
                                        weights=unlinked_weights if unlinked else None)
        return {
            "seed": {"label": seed.label, "path": seed.path, "linkers": seed.linkers},
            "compared": result.compared,
            "hits": [
                {"path": h.path, "title": h.title, "score": h.score, "graph": h.graph, "text": h.text, "tags": h.tags,
                 "cocitation": h.cocitation, "coupling": h.coupling, "linked": h.linked, "hops": h.hops, "reason": h.reason}
                for h in result.hits
            ],
            "text_signal": self._text_status(self._engine().text),
            "weights": result.weights,
        }

    def suggest_links(self, limit: int = 10, note: str | None = None, path: str | None = None) -> dict[str, Any]:
        from vaultengine.connect import LinkSuggester

        for_note = self._note_id(note)[0] if note else None
        suggester = LinkSuggester(self._built_index().conn, self._engine())
        pairs = suggester.suggestions(limit=limit, for_note=for_note, path=path)
        keys = ("a", "b", "score", "shared", "cited", "tags", "text", "shared_with", "cited_by", "shared_tags", "terms", "reason")
        return {"note": note, "path": path, "links": [{key: getattr(p, key) for key in keys} for p in pairs],
                "text_signal": self._text_status(self._engine().text), "weights": suggester.weights}

    def suggest_notes(self, limit: int = 10, min_links: int = 2) -> dict[str, Any]:
        from vaultengine.suggest import Suggester

        suggester = Suggester(self._built_index().conn, self._engine(text=False))  # missing notes need no embeddings
        result = suggester.suggestions(limit=limit, min_links=min_links)
        return {
            "total_missing": result.total_missing,
            "notes": [
                {"name": m.name, "linking_notes": m.linking_notes, "links": m.links, "mentions": m.mentions,
                 "folders": [{"name": name, "notes": n} for name, n in m.folders], "about": m.about}
                for m in result.notes
            ],
            "same_idea": [[{"name": name, "linking_notes": n} for name, n in group] for group in result.same_idea],
            "existing": [{"name": name, "linking_notes": n, "path": path} for name, n, path in result.existing],
        }

    def unlinked(self, note: str | None = None, min_words: int = 1, limit: int | None = 20, per_note: int = 20) -> dict[str, Any]:
        """Mentions grouped by the note mentioned, most mentioning notes first."""
        from vaultengine.connect import MentionFinder

        finder = MentionFinder(self._built_index().conn, self.root, min_words=min_words)
        if note:
            file_id, _ = self._note_id(note)
            mentions = finder.of(file_id)
            target = finder.paths[file_id]
        else:
            mentions, target = finder.all(), None
        groups: dict[str, list] = {}
        for mention in mentions:
            groups.setdefault(mention.target, []).append(mention)
        order = sorted(groups, key=lambda t: (-len({m.source for m in groups[t]}), t))
        keys = ("source", "line", "start", "end", "before", "text", "after")
        return {
            "note": target,
            "total": len(mentions),
            "notes": len({m.source for m in mentions}),
            "targets": [
                {
                    "path": t,
                    "title": link_name(t),
                    "names": sorted({m.name for m in groups[t]}),
                    "total": len(groups[t]),
                    "notes": len({m.source for m in groups[t]}),
                    "mentions": [{key: getattr(m, key) for key in keys} for m in groups[t][:per_note]],
                }
                for t in (order if limit is None else order[:limit])
            ],
        }

    # ----- explore: explore.*, map.*, note.deepen, vault.discover, notes.explain, embed.status, config.show -------

    def _engine_settings(self):
        from vaultengine.settings import load_settings

        if self._settings is None:
            self._settings = load_settings()
        return self._settings

    def memory_path(self) -> Path:
        """The explore memory: engine.memory, else <vault>/.vaultengine/memory.sqlite (this server's vault)."""
        memory = self._engine_settings().engine.memory
        return Path(memory) if memory is not None else self.root / ".vaultengine" / "memory.sqlite"

    def _workbench(self):
        from vaultengine.formats import FormatError
        from vaultengine.operations import Workbench

        settings = self._engine_settings()
        try:
            return Workbench(self._built_index().conn, self.root, settings,
                             self.format_path or settings.engine.format, self.memory_path())
        except FormatError as exc:
            raise FormatNotConfigured(f"Invalid format file: {exc}", format=str(self.format_path)) from None

    @staticmethod
    def _operation(run):
        from vaultengine.operations import OperationError

        try:
            return run()
        except OperationError as exc:
            data = {**exc.data, **({"hint": exc.hint} if exc.hint else {})}
            if exc.kind == "not_found":
                raise NoteNotFound(exc.message, **data) from None
            raise NotAvailable(exc.message, **data) from None

    def explore(self, note: str | None = None, lenses: list[str] | None = None, per_lens: int | None = None,
                fresh: bool = False, record: bool = True, refresh: bool = False) -> dict[str, Any]:
        from contextlib import nullcontext

        from vaultengine.memory import Memory
        from vaultengine.operations import explore_json

        path = self.memory_path()
        memory = Memory(path) if record or path.is_file() else None
        with memory or nullcontext():
            out, explorer = self._operation(lambda: self._workbench().explore(note, lenses, memory, fresh, per_lens, refresh, record))
            return explore_json(out, path, fresh, text=explorer.engine.text)

    def explore_mark(self, verdict: str, refs: list[str]) -> dict[str, Any]:
        from vaultengine.memory import Memory
        from vaultengine.operations import mark

        path = self.memory_path()
        if not path.is_file():
            raise NotAvailable(f"nothing remembered yet ({path} does not exist)", hint="call explore.run first")
        with Memory(path) as memory:
            return self._operation(lambda: mark(memory, verdict, refs))

    def explore_kept(self, lens: str | None = None) -> dict[str, Any]:
        from vaultengine.operations import kept

        return kept(self.memory_path(), lens)

    def explore_link(self, ref: str | None = None, a: str | None = None, b: str | None = None, reason: str | None = None,
                     dry_run: bool = True) -> dict[str, Any]:
        from vaultengine.operations import LinkIt

        write = self._operation(lambda: LinkIt(self.memory_path(), self._engine_settings().engine.explore.link_section,
                                               ref, a, b, reason))

        def after(outcome):
            write.after(outcome)
            return write.extra(outcome)

        return self._write(write.build, dry_run, after)

    def explore_unlink(self, a: str, b: str, dry_run: bool = True) -> dict[str, Any]:
        from vaultengine.operations import Unlink

        write = Unlink(self.memory_path(), self._engine_settings().engine.explore.link_section, a, b)

        def after(outcome):
            write.after(outcome)
            return write.extra(outcome)

        return self._write(write.build, dry_run, after)

    def map(self, limit: int = 10, meaning: bool | None = None, record: bool = False) -> dict[str, Any]:
        data, _, _ = self._workbench().map(limit, meaning, record)
        return data

    def map_since(self, since: str) -> dict[str, Any]:
        return self._operation(lambda: self._workbench().map_since(since))

    def deepen(self, note: str, lenses: list[str] | None = None, limit: int = 8) -> dict[str, Any]:
        data = self._operation(lambda: self._workbench().deepen(note, lenses, limit))
        data.pop("topic_pages_known", None)
        return data

    def discover(self, note: str | None = None, lenses: list[str] | None = None, limit: int = 3, steps: int = 8,
                 seed: int | None = None) -> dict[str, Any]:
        from vaultengine.operations import text_signal

        data, engine = self._operation(lambda: self._workbench().discover(note, lenses, limit, steps, seed))
        return {**data, "text_signal": text_signal(engine.text)}

    def explain(self, a: str, b: str, words: int = 8) -> dict[str, Any]:
        data, _ = self._operation(lambda: self._workbench().explain(a, b, words))
        return data

    def embed_status(self) -> dict[str, Any]:
        from vaultengine.embedding.store import status

        engine = self._engine_settings().engine
        model_path = engine.embedding.model_path
        data = status(self._built_index().conn, engine.model_id)
        data.update(provider=engine.embedding.provider, model_path=str(model_path) if model_path else None,
                    model_found=bool(model_path and Path(model_path).is_dir()))
        return data

    def config_show(self) -> dict[str, Any]:
        return self._engine_settings().to_dict()
