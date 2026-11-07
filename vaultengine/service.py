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


class InvalidQuery(ServiceError):
    """A search query that does not parse; data says where (start/end offsets in the query) and how to fix it."""


class VaultService:
    def __init__(self, vault_root: str | Path, db_path: str | Path | None = None) -> None:
        self.root = Path(vault_root).resolve()
        self.db_path = Path(db_path).resolve() if db_path else default_db_path(self.root)
        self._index: VaultIndex | None = None
        # Prepared pick candidates, keyed by the index generation they were built from.
        self._pick_cache: tuple[str | None, list[tuple[str, str, Prepared]]] | None = None
        # Link graph, term profiles and tag weights for related/suggest, keyed by the index generation.
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

    # ----- search, related notes, suggestions, unlinked mentions ----------

    def _engine(self):
        from vaultengine.related import Relatedness

        index = self._built_index()
        stamp = index.meta().get("generation")
        if self._relatedness is None or self._relatedness[0] != stamp:
            self._relatedness = (stamp, Relatedness(index.conn))
        return self._relatedness[1]

    def _note_id(self, ref: str, missing_ok: bool = False) -> tuple[int | None, Any]:
        """(file id, seed) for a note reference; a missing link target gives (None, seed) when missing_ok."""
        from vaultengine.querylang import QueryError

        engine = self._engine()
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
        result = self._engine().related(seed, limit=limit, unlinked_only=unlinked)
        return {
            "seed": {"label": seed.label, "path": seed.path, "linkers": seed.linkers},
            "compared": result.compared,
            "hits": [
                {"path": h.path, "title": h.title, "score": h.score, "graph": h.graph, "text": h.text, "tags": h.tags,
                 "linked": h.linked, "hops": h.hops, "reason": h.reason}
                for h in result.hits
            ],
        }

    def suggest_links(self, limit: int = 10, note: str | None = None, path: str | None = None) -> dict[str, Any]:
        from vaultengine.connect import LinkSuggester

        for_note = self._note_id(note)[0] if note else None
        pairs = LinkSuggester(self._built_index().conn, self._engine()).suggestions(limit=limit, for_note=for_note, path=path)
        keys = ("a", "b", "score", "shared", "cited", "text", "shared_with", "cited_by", "terms", "reason")
        return {"note": note, "path": path, "links": [{key: getattr(p, key) for key in keys} for p in pairs]}

    def suggest_notes(self, limit: int = 10, min_links: int = 2) -> dict[str, Any]:
        from vaultengine.suggest import Suggester

        suggester = Suggester(self._built_index().conn)
        suggester.engine = self._engine()
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
