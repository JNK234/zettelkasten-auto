"""The explore commands as operations on one vault, shared by the CLI (`--json`) and the JSON-RPC server.

Each operation returns the plain dict that `vault <command> --json` prints and the matching JSON-RPC method
returns (its model is in ``rpc/models.py``), so the two cannot drift apart. The CLI formats text from the
same dicts (or the objects behind them). Errors are ``OperationError``: a message, an optional hint, and a
kind the front-ends map to an exit code or an error code (``not_found``: no such note; ``unavailable``:
nothing to act on, e.g. no explore run yet, a result that is not two notes, no map snapshot).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

DEEPEN_LENSES = ("neighbourhood", "cited", "foundations", "backbone", "cluster")
DISCOVER_LENSES = ("hidden", "role", "bridge", "gap", "surprise", "unwritten", "wander")
DISCOVER_NOTE_LENSES = {"hidden", "role", "bridge", "wander"}


class OperationError(Exception):
    def __init__(self, message: str, hint: str | None = None, kind: str = "unavailable", **data: Any) -> None:
        super().__init__(message)
        self.message, self.hint, self.kind, self.data = message, hint, kind, data


def _query_error(exc) -> OperationError:
    return OperationError(exc.message, exc.hint, "not_found")


class Workbench:
    """One vault, its index connection, the engine settings and the format: what every explore command needs.

    ``settings``: ``settings.Settings`` (engine settings and the embedding provider); ``format_name``: a format
    name or file, or None (then fields named tags are tag links and there are no topic pages)."""

    def __init__(self, conn, root: str | Path, settings, format_name: str | Path | None = None,
                 memory_path: str | Path | None = None) -> None:
        from vaultengine.tagrule import rule_for_format

        self.conn, self.root, self.settings = conn, str(root), settings
        self.engine_settings = settings.engine
        self.memory_path = Path(memory_path) if memory_path is not None else Path(settings.engine.memory_path)
        self.format_name = format_name
        self.rule = rule_for_format(format_name)  # raises formats.FormatError
        self.is_page, self.pages_from = self._topic_pages()

    # ----- pieces -------------------------------------------------------------------------------------------

    def _topic_pages(self):
        """(path -> is it a topic page, how that was decided) from the format's types; (None, why not) without one."""
        from vaultengine.formats import BUNDLED_FORMATS_DIR, load_format, resolve_format_path

        types = self.engine_settings.explore.topic_types
        if not self.format_name:
            return None, "no format: the format's note types say which notes are topic pages (--format or engine.format)"
        spec = load_format(resolve_format_path(self.format_name, [Path("formats"), BUNDLED_FORMATS_DIR]))
        known = [t for t in types if any(x.name == t for x in spec.types)]
        if not known:
            return None, f"format {spec.name} has no type named {' or '.join(types)} (engine.explore.topic_types)"

        def is_page(path: str) -> bool:
            note_type = spec.type_for(path)
            return note_type is not None and note_type.name in known

        return is_page, f"notes the format {spec.name} types as {' or '.join(known)}"

    def weights(self, unlinked: bool = False) -> dict[str, float]:
        related = self.engine_settings.related
        return (related.unlinked_weights if unlinked else related.weights).model_dump()

    def relatedness(self, unlinked: bool = False):
        from vaultengine.related import Relatedness

        return Relatedness(self.conn, lambda: self.settings.provider(), self.root, self.rule, self.weights(unlinked))

    def shape(self, graph=None, meaning=None):
        from vaultengine.shape import VaultShape

        e = self.engine_settings.explore
        return VaultShape(self.conn, self.rule, e.seed, e.resolution, e.path_budget, self.is_page, graph, meaning,
                          e.meaning_percentile, e.meaning_neighbours)

    def explorer(self, memory=None, per_lens: int | None = None, refresh: bool = False, keep_stats: bool = True):
        """The explorer; ``keep_stats``: with the vault-wide numbers kept between runs (``stats.py``)."""
        from vaultengine.explore import Explorer
        from vaultengine.stats import Stats, fingerprint, vault_state

        settings = self.engine_settings.explore
        if per_lens:
            settings = settings.model_copy(update={"per_lens": per_lens})
        engine = self.relatedness(unlinked=True)
        shape = self.shape(engine.graph)
        stats = None
        if keep_stats:
            key = fingerprint(seed=settings.seed, resolution=settings.resolution, path_budget=settings.path_budget,
                              tags=engine.graph.tag_rule, pages=self.pages_from, topic_types=settings.topic_types,
                              model=engine.text.vectors.model_id if engine.text.available() else None,
                              duplicate_percentile=settings.duplicate_percentile)
            stats = Stats(self.memory_path.with_name("stats.sqlite"), key, vault_state(self.conn),
                          settings.stats_refresh, refresh)
        return Explorer(engine, shape, settings, memory, stats)

    # ----- explore --------------------------------------------------------------------------------------------

    def check_lenses(self, note: str | None, lenses: list[str] | None) -> list[str]:
        from vaultengine.explore import REGISTRY

        settings = self.engine_settings.explore
        chosen = list(lenses or (settings.lenses if note else settings.start_lenses))
        unknown = [n for n in chosen if n not in REGISTRY]
        if unknown:
            raise OperationError(f"unknown lens {', '.join(unknown)}", f"lenses: {', '.join(REGISTRY)}", "usage")
        wrong = [n for n in chosen if not (REGISTRY[n].from_start if note else REGISTRY[n].vault_wide)]
        if wrong:
            need = "without a starting point" if note else "from a note"
            raise OperationError(f"{', '.join(wrong)} only run {need}", None, "usage")
        return chosen

    def explore(self, note: str | None, lenses: list[str] | None = None, memory=None, fresh: bool = False,
                per_lens: int | None = None, refresh: bool = False, record: bool = True):
        """(the run: ``Explorer.run``'s dict, the explorer). Records the run in ``memory`` when given and ``record``."""
        from vaultengine.querylang import QueryError

        chosen = self.check_lenses(note, lenses)
        explorer = self.explorer(memory, per_lens, refresh)
        try:
            out = explorer.run(note, chosen, fresh=fresh)
        except QueryError as exc:
            raise _query_error(exc) from None
        out["lenses"] = chosen
        if memory is not None and record:
            memory.record_session(out["start"], fresh, chosen, [r.to_dict() for s in out["sections"] for r in s.results])
        return out, explorer

    # ----- deepen, discover, explain --------------------------------------------------------------------------

    def deepen(self, note: str, lenses: list[str] | None = None, limit: int = 8) -> dict[str, Any]:
        from vaultengine.lenses import Lenses
        from vaultengine.querylang import QueryError

        e = self.engine_settings.explore
        wanted = list(lenses or DEEPEN_LENSES)
        engine = self.relatedness()
        lens = Lenses(engine, self.shape(engine.graph), e.percentile, e.seed)
        try:
            node = lens.node(note)
        except QueryError as exc:
            raise _query_error(exc) from None
        compute = {"neighbourhood": lambda: lens.neighbourhood(node, limit),
                   "cited": lambda: lens.cited_together(node, limit),
                   "foundations": lambda: lens.foundations(node, limit),
                   "backbone": lambda: lens.backbone(node),
                   "cluster": lambda: lens.cluster(node, 3)}
        results = {name: compute[name]() for name in wanted}
        return {"note": engine.graph.labels[node], "percentile": e.percentile, "lenses": results,
                "topic_pages_known": lens.shape.topic_page is not None}

    def discover(self, note: str | None, lenses: list[str] | None = None, limit: int = 3, steps: int = 8,
                 seed: int | None = None) -> tuple[dict[str, Any], Any]:
        """(the result, the relatedness engine behind it: its text signal says whether embeddings were there)."""
        from vaultengine.discover import Discover
        from vaultengine.querylang import QueryError

        e = self.engine_settings.explore
        seed = seed if seed is not None else e.seed
        wanted = list(lenses or [lens for lens in DISCOVER_LENSES if note or lens not in DISCOVER_NOTE_LENSES])
        if not note and DISCOVER_NOTE_LENSES & set(wanted):
            raise OperationError(f"{', '.join(sorted(DISCOVER_NOTE_LENSES & set(wanted)))} start from a note",
                                 "vault discover <vault> <note>", "usage")
        engine = self.relatedness(unlinked=True)
        shape = self.shape(engine.graph)
        discover = Discover(engine, shape, e.percentile, seed)
        node = None
        if note:
            try:
                seed_note = engine.seed(note)
            except QueryError as exc:
                raise _query_error(exc) from None
            if seed_note.path is None:
                raise OperationError(f'"{note}" is not a note yet (only links to it exist)',
                                     "`vault discover` lists unwritten notes without a note", "not_found")
            node = seed_note.node
        results: dict[str, Any] = {}
        for lens in wanted:
            if lens == "hidden":
                results[lens] = discover.hidden_link(node, limit)
            elif lens == "role":
                results[lens] = discover.same_role(node, limit)
            elif lens == "bridge":
                results[lens] = discover.bridge(node, limit)
            elif lens == "wander":
                results[lens] = discover.wander(node, steps, seed)
            else:
                fn = {"gap": discover.gaps, "surprise": discover.surprises, "unwritten": discover.unwritten}[lens]
                only = node if lens != "gap" else (shape.membership[node] if node is not None else None)
                results[lens] = {"with_embeddings": fn(True, only, limit), "without_embeddings": fn(False, only, limit)}
        return {"note": engine.graph.labels[node] if node is not None else None, "percentile": e.percentile,
                "seed": seed, "lenses": results}, engine

    def explain(self, a: str, b: str, words: int = 8):
        """(the explanation dict, the Relations object's result) for two notes."""
        from vaultengine.querylang import QueryError
        from vaultengine.relations import Relations

        weights = self.engine_settings.explore.weights.model_dump()
        relations = Relations(self.conn, lambda: self.settings.provider(), self.root, weights, tag_rule=self.rule)
        try:
            rel = relations.explain(a, b, words=words)
        except QueryError as exc:
            raise _query_error(exc) from None
        return {"configured_weights": weights, **rel.to_dict()}, rel

    # ----- map ----------------------------------------------------------------------------------------------------

    def map(self, limit: int | None = 10, meaning: bool | None = None, record: bool = False) -> tuple[dict[str, Any], Any, Any]:
        """(the map as `vault map --json` prints it, the shape, the engine or None); ``record`` also saves a snapshot."""
        from vaultengine.shape import compare

        explore = self.engine_settings.explore
        meaning = explore.meaning if meaning is None else meaning
        engine = None
        changes = None
        if meaning:
            engine = self.relatedness()
            plain = self.shape(engine.graph)
            shape = self.shape(engine.graph, engine.text)
            if shape.meaning_edges:
                changes = compare(plain, shape)
        elif record:
            engine = self.relatedness(unlinked=True)
            shape = self.shape(engine.graph)
        else:
            shape = self.shape()
        recorded = self.record_map(engine, shape) if record else None
        data = {"percentile": explore.percentile, "topic_pages": self.pages_from, **shape.to_dict(limit or None)}
        if meaning:
            data["meaning"] = {"edges": [{"a": shape.sg.labels[a], "b": shape.sg.labels[b], "cosine": c}
                                         for a, b, c in shape.meaning_edges],
                               "cutoff": shape.meaning_cutoff, "percentile": explore.meaning_percentile,
                               "unavailable": shape.meaning_unavailable, "changes": changes}
        if recorded:
            data["recorded"] = recorded
        return data, shape, engine

    def record_map(self, engine, shape) -> dict[str, Any]:
        from vaultengine.discover import Discover
        from vaultengine.history import snapshot
        from vaultengine.memory import Memory

        explore = self.engine_settings.explore
        discover = Discover(engine, shape, explore.percentile, explore.seed)
        data = snapshot(shape, discover, engine.text.available())
        with Memory(self.memory_path) as memory:
            at = memory.save_snapshot(data)
            size = memory.snapshots()[-1][2]
        return {"at": at, "bytes": size, "memory": str(memory.path), "communities": len(data["communities"]),
                "gaps": len(data["gaps"]), "basis": data["gap_basis"], "missing": len(data["missing"])}

    def map_since(self, since: str) -> dict[str, Any]:
        """What changed since the snapshot that stands for ``since``; reads the memory, writes nothing."""
        from vaultengine.discover import Discover
        from vaultengine.history import changes
        from vaultengine.memory import Memory, MemoryStoreError

        path = self.memory_path
        if not path.is_file():
            raise OperationError(f"no map snapshot yet ({path} does not exist)", "`vault map --record` saves one")
        with Memory(path, readonly=True) as memory:
            try:
                snap_id, at, how = memory.snapshot_since(since)
            except MemoryStoreError as exc:
                raise OperationError(str(exc)) from None
            old = memory.snapshot(snap_id)
        explore = self.engine_settings.explore
        engine = self.relatedness(unlinked=True)
        shape = self.shape(engine.graph)
        discover = Discover(engine, shape, explore.percentile, explore.seed)
        return {"since": since, "snapshot": at, "chosen": how, **changes(old, shape, discover, engine)}


# ----- explore's JSON, marks and kept ------------------------------------------------------------------------------


def numbers_json(stats: dict, timing: bool = False) -> dict[str, Any]:
    """Where explore's vault-wide numbers came from: "current" (computed now, or stored from this very state: the
    same numbers either way), or "reused" from a few notes ago. ``timing`` adds whether they were stored."""
    out: dict[str, Any] = {"source": "reused" if stats["mode"] == "reused" else "current"}
    if stats["mode"] == "reused":
        out.update(computed_at=stats["computed_at"], changed_since=stats["changed_since"], threshold=stats["threshold"])
    if timing:
        out["stored"] = stats["mode"]
    return out


def text_signal(text) -> dict[str, Any]:
    """The state of the text (embedding) signal of a ``related.TextSimilarity`` that was used."""
    on = text.available()
    return {"available": on, "model_id": text.vectors.model_id if on else None, "unavailable": text.unavailable,
            "uncached": len(text.uncached)}


def explore_json(out: dict[str, Any], memory_path: str | Path, fresh: bool, timing: bool = False,
                 text=None) -> dict[str, Any]:
    """The result of `vault explore --json` and of explore.run (``text``: the explorer's TextSimilarity)."""
    import math

    seed = out["seed"]
    cutoff = out["duplicate_cutoff"]
    return {
        "start": out["start"] or None,
        "start_kind": None if seed is None else ("note" if seed.path is not None else "missing"),
        "fresh": fresh,
        "duplicate_cutoff": None if cutoff is None or math.isinf(cutoff) else cutoff,
        "similarity": "embedding" if out["meaning"] else "key terms",
        "memory": str(memory_path),
        "numbers": numbers_json(out["stats"], timing),
        **({"text_signal": text_signal(text)} if text is not None else {}),
        "sections": [{"lens": s.lens.name, "kind": s.lens.kind, "title": s.lens.title, "what": s.lens.what,
                      "budget": s.budget, "found": s.candidates, "hidden": s.hidden, "duplicates": s.duplicates,
                      "kept": s.kept, "dismissed": s.dismissed, "keep_score": s.keep_score, "applies": s.applies,
                      "note": s.note, **({"seconds": s.seconds} if timing else {}),
                      "results": [r.to_dict() for r in s.results]}
                     for s in out["sections"]],
    }


def mark(memory, verdict: str, refs: list[str]) -> dict[str, Any]:
    """keep / dismiss / clear results of the last explore run, by ref."""
    from vaultengine.memory import MemoryStoreError

    if verdict not in ("keep", "dismiss", "clear"):
        raise OperationError(f"unknown verdict {verdict!r}", "keep, dismiss or clear", "usage")
    try:
        results = [memory.result(ref) for ref in refs]
    except MemoryStoreError as exc:
        raise OperationError(str(exc)) from None
    marked = []
    for r in results:
        if verdict == "clear":
            changed = memory.unmark(r)
        else:
            memory.mark(r, verdict)
            changed = True
        marked.append({"ref": r["ref"], "lens": r["lens"], "target": r["target"], "title": r["title"],
                       "verdict": verdict, "changed": changed})
    counts = memory.keep_counts()
    return {"marked": marked, "keep_rates": [{"lens": lens, "kept": counts.get(lens, (0, 0))[0],
                                               "dismissed": counts.get(lens, (0, 0))[1]}
                                              for lens in sorted({r["lens"] for r in results})]}


def kept(memory_path: str | Path, lens: str | None = None) -> dict[str, Any]:
    from vaultengine.memory import Memory

    rows = []
    if Path(memory_path).is_file():
        with Memory(memory_path, readonly=True) as memory:
            rows = memory.kept(lens)
    return {"memory": str(memory_path), "kept": [
        {"start": r["start"] or None, "lens": r["lens"], "target": r["target"], "kind": r["kind"], "title": r["title"],
         "evidence": r["evidence"], "link": [r["link_a"], r["link_b"]] if r["link_a"] else None, "at": r["at"]}
        for r in rows]}


# ----- link it and unlink ----------------------------------------------------------------------------------------


class LinkIt:
    """One `vault link-it` / explore.link write: ``build`` plans it for ``formats.write.run_write``, ``after`` records
    a written link in the memory (for unlink) and counts it as a keep of the result it came from.

    From a result of the last explore run (``ref``), or from two notes (``a``, ``b``) and a reason."""

    def __init__(self, memory_path: str | Path, section: str, ref: str | None = None, a: str | None = None,
                 b: str | None = None, reason: str | None = None) -> None:
        from vaultengine.explore import REGISTRY
        from vaultengine.linkit import note_reason
        from vaultengine.memory import Memory, MemoryStoreError

        self.memory_path, self.section = Path(memory_path), section
        self.result: dict[str, Any] | None = None
        self.written: list[Any] = []
        if ref is not None:
            if not self.memory_path.is_file():
                raise OperationError(f"nothing remembered yet ({self.memory_path} does not exist)", "run `vault explore` first")
            with Memory(self.memory_path) as memory:
                try:
                    self.result = memory.result(ref)
                except MemoryStoreError as exc:
                    raise OperationError(str(exc)) from None
            r = self.result
            if not r["link"]:
                raise OperationError(f"result {r['ref']} ({r['lens']} · {r['title']}) is not two notes to link ({r['kind']})",
                                     "link-it joins a note result with the start, or the two notes of a gap")
            self.a, self.b = r["link"]
            self.reason = reason or note_reason(REGISTRY[r["lens"]].title, r["evidence"])
        elif a and b:
            self.a, self.b = a, b
            self.reason = reason or "linked by hand (vault link-it)"
        else:
            raise OperationError("give a result of the last explore run, or two notes", None, "usage")

    def build(self, typed):
        from vaultengine.linkit import plan_link_it

        plan, lines = plan_link_it(typed, self.a, self.b, self.reason, self.reason, self.section)
        self.written = lines
        return plan

    def after(self, outcome) -> None:
        if outcome.status != "written":
            return
        from vaultengine.memory import Memory

        by_path = {w.path: w for w in self.written}
        paths = [f.path for f in outcome.plan.files]
        with Memory(self.memory_path) as memory:
            if len(paths) == 2:
                x, y = paths
                memory.record_link(x, y, self.result["lens"] if self.result else None, self.section, by_path[x].line,
                                   by_path[y].line, by_path[x].created, by_path[y].created, by_path[x].blank,
                                   by_path[y].blank)
            if self.result is not None:
                memory.mark(self.result, "keep")

    def extra(self, outcome) -> dict[str, Any]:
        r = self.result
        return {"ref": r["ref"] if r else None, "lens": r["lens"] if r else None, "notes": [self.a, self.b],
                "reason": self.reason, "kept": bool(r) and outcome.status == "written"}


def _flag(value) -> bool | None:
    return None if value is None else bool(value)


class Unlink:
    """One `vault unlink` / explore.unlink write: the list lines linking the other note under the link section, in
    both notes; the section too when link-it created it and nothing else is left in it."""

    def __init__(self, memory_path: str | Path, section: str, a: str, b: str) -> None:
        self.memory_path, self.section, self.a_ref, self.b_ref = Path(memory_path), section, a, b
        self.record: dict[str, Any] = {}

    def build(self, typed):
        from vaultengine.formats.write import find_note
        from vaultengine.linkit import plan_unlink
        from vaultengine.memory import Memory

        a, b = find_note(typed, self.a_ref), find_note(typed, self.b_ref)
        if self.memory_path.is_file():
            with Memory(self.memory_path, readonly=True) as memory:
                self.record = memory.link_between(a, b) or {}
        created = {a: bool(self.record.get("created_a")), b: bool(self.record.get("created_b"))} if self.record else {}
        blank = {a: _flag(self.record.get("blank_a")), b: _flag(self.record.get("blank_b"))} if self.record else {}
        return plan_unlink(typed, a, b, self.section, created, blank)

    def after(self, outcome) -> None:
        if outcome.status == "written" and self.record:
            from vaultengine.memory import Memory

            with Memory(self.memory_path) as memory:
                memory.undo_link(self.record["id"])

    def extra(self, outcome) -> dict[str, Any]:
        return {"recorded": bool(self.record)}
