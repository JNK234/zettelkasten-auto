"""`vault explore`: the deepen and discover lenses through one registry, one section per lens.

From a starting point (a note, or a missing note: a topic the vault links to but has no note for), every lens
in ``engine.explore.lenses`` runs and gives its candidates in its own order; each section then shows
``engine.explore.per_lens`` of them (3 to 5), every result with its lens and a one-line evidence.

Equal space. Deepen (around the note) and discover (what the vault does not say yet) get the same number of
results: with D deepen and S discover sections that apply, each kind's budget is ``per_lens`` x (D + S) / 2,
split evenly over its sections and kept between 3 and 5 per section. With the default lens list (five of each)
that is ``per_lens`` everywhere.

Diversity. A section never shows two versions of the same note: candidates are taken in the lens's order and
one is skipped when it shares a note with a result already shown, or when one of its notes is a near-duplicate
of one of theirs: similarity at or above the ``duplicate_percentile`` (99) of note-pair similarities in the vault
(centered embedding cosine; key-term cosine without the model). A missing note is a duplicate of the same name.

Memory (``memory.py``). Every result shown is recorded as seen, under its start. ``--fresh`` hides what was seen
in earlier runs from the same start, or dismissed there, and fills the section from further down the lens's
list. The sections are ordered by how often you keep what a lens shows: (kept + 1) / (marked + 2) over all your
marks, so a lens you never marked counts as one half; ties keep the configured order. The rate is printed in each
section's header.

No starting point. Explore then proposes where to start, from what the vault-wide lenses find
(``engine.explore.start_lenses``): the widest gaps, the most surprising links, the most wanted missing notes,
the communities without a topic page. Each proposal names where it would start (``explore <that>``).

Relatedness here uses ``engine.related.unlinked_weights``: explore looks for what is not linked yet, the task
`vault experiment` measured those weights on. Notes already linked to the start are left out of the note
lenses (neighbourhood, cited together, same foundations, hidden link, same role); the backbone, the cluster and
surprise are about the links that are there and keep them.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable

POOL = 30  # candidates asked of each lens before --fresh and diversity pick the shown ones
GAP_POOL = 12  # gaps asked for: each one's joiners cost a hidden-link search over a community (34 s at 50,000 notes for 30)


@dataclass(frozen=True)
class LensInfo:
    name: str
    kind: str  # "deepen", "discover" or "map"
    title: str
    what: str
    from_start: bool  # runs from a starting point
    vault_wide: bool  # proposes starting points without one


REGISTRY: dict[str, LensInfo] = {lens.name: lens for lens in (
    LensInfo("neighbourhood", "deepen", "Neighbourhood", "related notes not linked yet, at or above the percentile of this note's scores", True, False),
    LensInfo("cited", "deepen", "Cited together", "notes the same notes link to, not linked with it", True, False),
    LensInfo("foundations", "deepen", "Same foundations", "notes that link to the same notes, not linked with it", True, False),
    LensInfo("backbone", "deepen", "Backbone", "the main chain through its community, nearest first", True, False),
    LensInfo("cluster", "deepen", "Cluster", "its community: topic page, hubs, the communities next to it", True, False),
    LensInfo("hidden", "discover", "Hidden link", "no link either way, but two or more middles in common (Swanson's A-B-C)", True, False),
    LensInfo("role", "discover", "Same role elsewhere", "notes in other communities holding the same position", True, False),
    LensInfo("bridge", "discover", "Bridge", "the communities whose shortest paths run through it", True, False),
    LensInfo("gap", "discover", "Gap", "communities close in meaning with fewer links than chance predicts", True, True),
    LensInfo("surprise", "discover", "Surprise", "links between notes far apart in meaning, across communities", True, True),
    LensInfo("unwritten", "discover", "Unwritten", "missing notes linked from the most communities", True, True),
    LensInfo("wander", "discover", "Wander", "a seeded random walk on links", True, False),
    LensInfo("topicless", "map", "No topic page", "communities without a topic page, largest first", False, True),
)}
DOES_NOT_APPLY = "does not apply to a missing note (it needs the note's own links)"
NOTE_ONLY = {"cited", "foundations", "backbone", "role", "bridge", "wander"}


@dataclass
class Result:
    lens: str
    kind: str  # "note", "pair" (two notes not linked), "link" (a link that is there), "missing", "community"
    target: str  # what it is about, as remembered (memory.py)
    title: str
    evidence: str  # one line
    notes: tuple[str, ...] = ()  # the notes it is about (paths): diversity, and what link-it could join
    link: tuple[str, str] | None = None  # the two notes `vault link-it` would link
    start_at: str | None = None  # proposals: what `vault explore` would start from
    ref: str = ""
    mark: str | None = None  # "keep" / "dismiss" when you marked it before

    def to_dict(self) -> dict[str, Any]:
        return {"ref": self.ref, "lens": self.lens, "kind": self.kind, "target": self.target, "title": self.title,
                "evidence": self.evidence, "notes": list(self.notes), "link": list(self.link) if self.link else None,
                "start_at": self.start_at, "mark": self.mark}


@dataclass
class Section:
    lens: LensInfo
    results: list[Result] = field(default_factory=list)
    candidates: int = 0  # what the lens found
    hidden: int = 0  # left out by --fresh
    duplicates: int = 0  # left out as near-duplicates
    budget: int = 0
    note: str | None = None  # why it found nothing / does not apply
    applies: bool = True  # False: the lens does not apply to this start (it takes no space)
    seconds: float = 0.0  # the lens's time, with whatever shared part it computed first (e.g. betweenness for role)
    kept: int = 0
    dismissed: int = 0

    @property
    def keep_score(self) -> float:
        return (self.kept + 1) / (self.kept + self.dismissed + 2)


def _tie(step: str) -> str:
    """The backbone's step text ("links", "both → x", "both #tag", "both [[x]]") in words."""
    if step == "links":
        return "a direct link"
    what = step.removeprefix("both ")
    if what.startswith("→ "):
        return f"both linking {what[2:]} (no note yet)"
    return f"both tagged {what}" if what.startswith(("#", "[[")) else f"both linked with {what}"


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" + ("" if n == 1 else "s")


def one_line(text: str) -> str:
    return " ".join(str(text).split())


class Explorer:
    """Runs the lenses from a start. ``engine``: a ``related.Relatedness`` with the unlinked weights; ``shape``: the
    map; the lenses and discover objects are built on them."""

    def __init__(self, engine, shape, settings, memory=None, stats=None) -> None:
        """``stats``: a ``stats.Stats`` keeping the vault-wide numbers between runs (None: compute them all)."""
        from vaultengine.discover import Discover
        from vaultengine.lenses import Lenses
        from vaultengine.stats import NoStats

        self.engine, self.shape, self.settings, self.memory = engine, shape, settings, memory
        self.stats = stats if stats is not None else NoStats()
        shape.stats = self.stats
        self.graph = engine.graph
        self.lenses = Lenses(engine, shape, settings.percentile, settings.seed, self.stats)
        self.discover = Discover(engine, shape, settings.percentile, settings.seed, stats=self.stats)
        self.meaning = engine.text.available()
        self._dup_cut: float | None = None
        self._similarity: Callable[[str, str], float | None] | None = None

    # ----- names ----------------------------------------------------------------------------------------

    def name(self, path: str) -> str:
        from vaultengine.vault import link_name

        return link_name(path)

    def is_linked(self, a: str, b: str) -> bool:
        g = self.graph
        fa, fb = g.by_file.get(self._file(a)), g.by_file.get(self._file(b))
        return fa is not None and fb is not None and ((fa, fb) in g.direct or (fb, fa) in g.direct)

    def _file(self, path: str) -> int | None:
        if not hasattr(self, "_file_of"):
            self._file_of = {p: f for f, p in self.engine.paths.items()}
        return self._file_of.get(path)

    # ----- near-duplicates --------------------------------------------------------------------------------

    @property
    def similarity(self) -> Callable[[str, str], float | None]:
        """(path, path) -> the similarity near-duplicates are judged by (centered cosine, else key-term cosine)."""
        if self._similarity is None:
            d = self.discover
            by_node = d.note_similarity(self.meaning) or d.note_similarity(False)
            node = lambda p: self.graph.by_file.get(self._file(p))

            def sim(a: str, b: str) -> float | None:
                x, y = node(a), node(b)
                return by_node(x, y) if x is not None and y is not None else None

            self._similarity = sim
        return self._similarity

    @property
    def duplicate_cutoff(self) -> float:
        """The ``duplicate_percentile`` of note-pair similarity: every pair up to 2,000 notes (700 for key terms, which are
        slower), else 200,000 seeded pairs."""
        if self._dup_cut is None:
            self._dup_cut = self.stats.get("duplicate_cutoff", self._duplicate_cutoff, tolerant=True)
        return self._dup_cut

    def _duplicate_cutoff(self) -> float:
        from vaultengine.shape import percentile

        paths = sorted(self.engine.paths.values())
        n = len(paths)
        values: list[float] = []
        if self.meaning:
            import numpy as np

            text = self.engine.text
            rows = [text.vectors.row_of[f] for f in sorted(self.engine.paths) if f in text.vectors.row_of]
            m = text.centered[rows]
            m = m[np.abs(m).sum(axis=1) > 0]
            if len(m) <= 2000:
                sims = m @ m.T
                values = sims[np.triu_indices(len(m), 1)].tolist()
            else:
                rng = np.random.default_rng(self.settings.seed)
                x, y = rng.integers(0, len(m), 200_000), rng.integers(0, len(m), 200_000)
                keep = x != y
                values = np.einsum("ij,ij->i", m[x[keep]], m[y[keep]]).tolist()
        else:
            rng = random.Random(self.settings.seed)
            pairs = ([(a, b) for i, a in enumerate(paths) for b in paths[i + 1:]] if n <= 700 else
                     [tuple(rng.sample(paths, 2)) for _ in range(200_000)])
            values = [s for a, b in pairs if (s := self.similarity(a, b)) is not None]
        return percentile(values, self.settings.duplicate_percentile) if values else math.inf

    def near_duplicate(self, r: Result, shown: list[Result]) -> bool:
        for other in shown:
            if r.target == other.target:
                return True
            if r.kind == "missing" and other.kind == "missing":
                continue
            if set(r.notes) & set(other.notes):
                return True
            for a in r.notes:
                for b in other.notes:
                    s = self.similarity(a, b)
                    if s is not None and s >= self.duplicate_cutoff:
                        return True
        return False

    # ----- the start ----------------------------------------------------------------------------------------

    def resolve(self, ref: str):
        """(start key, seed) for a note or a missing note; raises querylang.QueryError."""
        seed = self.engine.seed(ref)
        return (seed.path if seed.path is not None else f"[[{seed.label}]]"), seed

    # ----- the lenses ------------------------------------------------------------------------------------------

    def candidates(self, lens: str, seed) -> tuple[list[Result], str | None]:
        """A lens's candidates from ``seed`` (None: vault-wide proposals), best first; or why there are none."""
        if seed is not None and seed.path is None and lens in NOTE_ONLY:
            return [], DOES_NOT_APPLY
        fn = getattr(self, f"_lens_{lens}")
        return fn(seed)

    def _lens_neighbourhood(self, seed):
        from vaultengine.shape import percentile, percentile_rank

        engine = self.engine
        everything = engine.related(seed, limit=None, words=False, reasons=False)  # one ranking, reasons only for the shown
        seed_file = self.graph.note_ids[seed.node]
        scores = {h.path: h.score for h in everything.hits}
        values = [scores.get(p, 0.0) for f, p in engine.paths.items() if f != seed_file]
        cut = percentile(values, self.settings.percentile)
        hits = [h for h in everything.hits if not h.linked][:POOL]
        engine.describe(seed, hits)
        hits = [h for h in hits if h.score >= cut and h.score > 0]
        out = []
        for h in hits:
            out.append(Result("neighbourhood", "note", h.path, self.name(h.path),
                              one_line(f"{h.reason or 'distant links only'}; score {h.score:.2f}"
                                       f" (p{percentile_rank(values, h.score):.0f} of this note's scores)"),
                              (h.path,), (seed.path, h.path) if seed.path else None))
        return out, None if out else "none at or above the cutoff that is not linked already"

    def _pair(self, seed, kind: str):
        r = (self.lenses.cited_together if kind == "cited" else self.lenses.foundations)(seed.node, 0)
        out = []
        for row in r["notes"]:
            path = row["path"]
            if self.is_linked(seed.path, path):
                continue
            ev = row["evidence"]
            names = ", ".join(e["name"] for e in ev[:3]) + (f" +{len(ev) - 3}" if len(ev) > 3 else "")
            what = f"cited together by {names}" if kind == "cited" else f"both link {names}"
            value = "co-citation" if kind == "cited" else "coupling"
            out.append(Result(kind, "note", path, self.name(path),
                              one_line(f"{what}; {value} {row['value']:.2f} (p{row['percentile']:.0f} of the vault's pairs)"),
                              (path,), (seed.path, path)))
        why = None if out else ("none stands out in this vault" if not r["notes"] else "every one is linked with it already")
        return out, why

    def _lens_cited(self, seed):
        return self._pair(seed, "cited")

    def _lens_foundations(self, seed):
        return self._pair(seed, "foundations")

    def _lens_backbone(self, seed):
        r = self.lenses.backbone(seed.node)
        if r["community"] is None:
            return [], "this note is in no community of two or more notes"
        chain, steps = r["chain"], r["steps"]
        anchor = seed.path if seed.path in chain else r["joins_at"]
        if anchor is None:
            return [], "the backbone does not reach this note"
        here = chain.index(anchor)
        order = sorted((i for i in range(len(chain)) if chain[i] != seed.path), key=lambda i: (abs(i - here), i))
        on = seed.path in chain
        whose = f"the backbone of community {r['community']} ({r['length']} of its {r['community_notes']} notes)"
        out = []
        for i in order:
            path = chain[i]
            k = abs(i - here)
            if k == 0:
                text = f"where this note joins {whose}: {_plural(r['steps_off'], 'step')} off it"
            else:
                step = steps[min(i, here)]
                tie = f"; tied by {_tie(step)}" if k == 1 else ""
                text = (f"{_plural(k, 'step')} along {whose}" + ("" if on else f" from {self.name(anchor)}, where this note joins it")
                        + tie)
            linked = self.is_linked(seed.path, path)
            out.append(Result("backbone", "note", path, self.name(path), one_line(text + ("; linked" if linked else "")),
                              (path,), None if linked else (seed.path, path)))
        return out, None

    def _lens_cluster(self, seed):
        r = self.lenses.cluster(seed.node, 5)
        if r["community"] is None:
            return [], "in no community of two or more notes"
        out = []
        c = r["community"]
        for page in r["topic_pages"]:
            out.append(Result("cluster", "note", page["path"], self.name(page["path"]),
                              f"topic page of community {c} ({r['name']}, {r['notes']} notes): links {page['reaches']} of them",
                              (page["path"],), None if seed.path is None or self.is_linked(seed.path, page["path"]) else (seed.path, page["path"])))
        if self.shape.topic_page is not None and not r["topic_pages"]:
            out.append(Result("cluster", "community", f"community: {r['name']}", f"community {c}: {r['name']}",
                              f"{r['notes']} notes and no topic page; key phrases: "
                              + ", ".join(x["phrase"] for x in r["phrases"][:3]), (), None,
                              self.name(r["hubs"][0]) if r["hubs"] else None))
        for hub in r["hubs"]:
            if hub == seed.path:
                continue
            linked = seed.path is not None and self.is_linked(seed.path, hub)
            out.append(Result("cluster", "note", hub, self.name(hub),
                              f"a hub of community {c} ({r['name']}): among its most tied-in notes" + ("; linked" if linked else ""),
                              (hub,), None if linked or seed.path is None else (seed.path, hub)))
        for n in r["neighbours"]:
            through = ", ".join(self.name(t) for t in n["through"][:2])
            out.append(Result("cluster", "community", f"community: {n['name']}", f"next community {n['community']}: {n['name']}",
                              f"weight {n['weight']:g} between the two communities, carried by {through}", (),
                              None, self.name(n["through"][0]) if n["through"] else None))
        return out, None

    def linkers(self, seed) -> set[int]:
        """A missing note's linkers (graph nodes)."""
        g = self.graph
        return {u for u in g.adjacency[seed.node] if not g.is_missing(u) and u != seed.node}

    def _topic_hidden(self, seed):
        """Notes that do not link the missing note but are linked with two or more of the notes that do: each such
        note a middle, weighing 1 / ln(1 + the notes linked with it), as hidden link weighs middles."""
        from vaultengine.shape import percentile, percentile_rank

        d, g = self.discover, self.graph
        _, members, weight = d.middles
        linkers = self.linkers(seed)
        score: dict[int, float] = {}
        via: dict[int, list[int]] = {}
        for linker in sorted(linkers):
            m = ("node", linker)
            if m not in weight:
                continue
            for c in members[m]:
                if c in linkers or g.is_missing(c):
                    continue
                score[c] = score.get(c, 0.0) + weight[m]
                via.setdefault(c, []).append(linker)
        values, _ = d.hidden_distribution()
        cut = percentile(values, self.settings.percentile)
        found = sorted(((s, c) for c, s in score.items() if len(via[c]) >= 2 and s >= cut),
                       key=lambda item: (-item[0], g.labels[item[1]]))
        out = []
        for s, c in found[:POOL]:
            names = ", ".join(self.name(g.labels[v]) for v in sorted(via[c], key=lambda v: g.labels[v])[:3])
            more = f" +{len(via[c]) - 3}" if len(via[c]) > 3 else ""
            out.append(Result("hidden", "note", g.labels[c], self.name(g.labels[c]),
                              one_line(f"does not link it; linked with {len(via[c])} notes that do: {names}{more}; score {s:.2f}"
                                       f" (p{percentile_rank(values, s):.0f} of the vault's unlinked pairs)"),
                              (g.labels[c],), None, self.name(g.labels[c])))
        return out, None if out else "no note is linked with two or more of its linkers strongly enough"

    def _lens_hidden(self, seed):
        if seed.path is None:
            return self._topic_hidden(seed)
        r = self.discover.hidden_link(seed.node, POOL)
        out = [Result("hidden", "note", row["note"], self.name(row["note"]), one_line(row["reason"]), (row["note"],),
                      (seed.path, row["note"])) for row in r["results"]]
        return out, None if out else f"none at or above the cutoff ({r['found']} unlinked notes share two or more middles)"

    def _lens_role(self, seed):
        r = self.discover.same_role(seed.node, POOL)
        out = []
        for row in r["results"]:
            parts = row["reason"].split("; ")
            out.append(Result("role", "note", row["note"], self.name(row["note"]), one_line("; ".join(parts[:3])),
                              (row["note"],), None if self.is_linked(seed.path, row["note"]) else (seed.path, row["note"])))
        return out, None if out else "no note elsewhere holds a role this close"

    def _lens_bridge(self, seed):
        r = self.discover.bridge(seed.node, POOL)
        out = [Result("bridge", "community", f"communities {b['communities'][0]} ↔ {b['communities'][1]}",
                      f"communities {b['communities'][0]} and {b['communities'][1]}",
                      one_line(b["reason"] + f"; its betweenness p{r['betweenness_percentile']:.0f}")) for b in r["brokers"]]
        return out, None if out else "no shortest path between notes runs through it"

    def _lens_gap(self, seed):
        only = None
        if seed is not None:
            number = self.shape.membership[seed.node]
            if number not in {c.number for c in self.shape.communities}:
                return [], "in no community of two or more notes"
            only = number
        r = self.discover.gaps(self.meaning, only, GAP_POOL)
        if "unavailable" in r:
            r = self.discover.gaps(False, only, GAP_POOL)
        basis = "meaning" if self.meaning else "key terms"
        out = []
        for row in r["results"]:
            x, y = row["communities"]
            join = row["join"][0] if row["join"] else None
            notes = (join["a"], join["b"]) if join else ()
            target = f"gap: {join['a']} ↔ {join['b']}" if join else f"gap: {row['names'][0]} | {row['names'][1]}"
            evidence = row["reason"] + (f"; join {self.name(join['a'])} ↔ {self.name(join['b'])}"
                                        f" (via {', '.join(join['middles'][:3])})" if join else "; no pair across shares two middles")
            hub = next((c.hubs[0] for c in self.shape.communities if c.number == x and c.hubs), None)
            start_at = self.name(join["a"]) if join else (self.name(self.shape.sg.labels[hub]) if hub is not None else None)
            out.append(Result("gap", "pair", target, f"communities {x} and {y}: {row['names'][0]} / {row['names'][1]}",
                              one_line(f"{evidence} [{basis}]"), notes, notes if join else None, start_at))
        return out, None if out else "no pair of communities is close in meaning and linked less than chance predicts"

    def _lens_surprise(self, seed):
        only = seed.node if seed is not None and seed.path is not None else None
        among = self.linkers(seed) if seed is not None and seed.path is None else None
        r = self.discover.surprises(self.meaning, only, POOL, among)
        if "unavailable" in r:
            r = self.discover.surprises(False, only, POOL, among)
        out = []
        for row in r["results"]:
            a, b = row["a"], row["b"]
            reason = row["reason"].split("; about:")[0]
            out.append(Result("surprise", "link", f"{a} → {b}", f"{self.name(a)} → {self.name(b)}", one_line(reason),
                              (a, b), None, self.name(a if seed is None or a != seed.path else b)))
        return out, None if out else "no link here joins notes this far apart"

    def _lens_unwritten(self, seed):
        only = seed.node if seed is not None and seed.path is not None else None
        among = self.linkers(seed) if seed is not None and seed.path is None else None
        r = self.discover.unwritten(self.meaning, only, POOL, among)
        if "unavailable" in r:
            r = self.discover.unwritten(False, only, POOL, among)
        out = []
        for row in r["results"]:
            name = row["names"][0]
            if seed is not None and seed.path is None and seed.label in row["names"]:
                continue  # the missing note explored itself
            out.append(Result("unwritten", "missing", f"[[{name}]]", f"[[{name}]]" + (f" +{len(row['names']) - 1}" if len(row["names"]) > 1 else ""),
                              one_line(row["reason"]), (), None, f"[[{name}]]"))
        return out, None if out else "no missing note it links is wanted from this many communities"

    def _lens_wander(self, seed):
        r = self.discover.wander(seed.node, POOL)
        out, seen = [], {seed.path}
        for hop in r["hops"]:
            path = hop["to"]
            if path in seen:
                continue
            seen.add(path)
            linked = self.is_linked(seed.path, path)
            out.append(Result("wander", "note", path, self.name(path), one_line(f"step {len(seen) - 1} of a walk (seed {r['seed']}): {hop['reason']}"),
                              (path,), None if linked else (seed.path, path)))
        return out, None if out else "no links to walk"

    def _lens_topicless(self, seed):
        if self.shape.topic_page is None:
            return [], "no format with topic page types: which notes are topic pages is unknown"
        out = []
        for c in self.shape.communities:
            if c.topic_pages:
                continue
            hub = self.shape.sg.labels[c.hubs[0]] if c.hubs else None
            phrases = ", ".join(p.phrase for p in c.phrases[:3]) or "no shared tag or link"
            out.append(Result("topicless", "community", f"community: {c.name}", f"community {c.number}: {c.name}",
                              f"{len(c.notes)} notes and no topic page; key phrases {phrases}; hub {self.name(hub) if hub else '—'}",
                              (), None, self.name(hub) if hub else None))
        return out, None if out else "every community has a topic page"

    # ----- a run ---------------------------------------------------------------------------------------------

    def run(self, ref: str | None, lenses: list[str], fresh: bool = False) -> dict[str, Any]:
        """Every lens from ``ref`` (None: proposals), diversified and ordered; nothing is recorded here."""
        start, seed = self.resolve(ref) if ref else ("", None)
        names = [n for n in lenses if (REGISTRY[n].from_start if seed is not None else REGISTRY[n].vault_wide)]
        sections = [Section(REGISTRY[n]) for n in names]
        counts = self.memory.keep_counts() if self.memory is not None else {}
        marks = self.memory.marks_for(start) if self.memory is not None else {}
        hide = (self.memory.seen_targets(start) | self.memory.dismissed_targets(start)) if fresh and self.memory else set()
        found: dict[str, list[Result]] = {}
        for s in sections:
            s.kept, s.dismissed = counts.get(s.lens.name, (0, 0))
            started = time.perf_counter()
            found[s.lens.name], s.note = self.candidates(s.lens.name, seed)
            s.seconds = time.perf_counter() - started
            s.candidates = len(found[s.lens.name])
            s.applies = s.note != DOES_NOT_APPLY
        self.budget(sections, [s for s in sections if s.applies])
        for s in sections:
            shown: list[Result] = []
            for r in found[s.lens.name]:
                if len(shown) >= s.budget:
                    break
                if r.target in hide:
                    s.hidden += 1
                    continue
                if seed is not None and seed.path is not None and seed.path in r.notes and r.kind == "note":
                    continue
                if self.near_duplicate(r, shown):
                    s.duplicates += 1
                    continue
                r.mark = marks.get((r.lens, r.target))
                shown.append(r)
            s.results = shown
            if not shown and s.note is None:
                s.note = "everything it found was seen or dismissed before (--fresh)" if s.hidden else "nothing found"
        order = {n: i for i, n in enumerate(names)}
        sections.sort(key=lambda s: (-s.keep_score, order[s.lens.name]))
        for number, s in enumerate(sections, 1):
            for rank, r in enumerate(s.results, 1):
                r.ref = f"{number}.{rank}"
        cutoff = self.duplicate_cutoff
        self.stats.save()
        return {"start": start, "seed": seed, "sections": sections, "fresh": fresh,
                "duplicate_cutoff": cutoff, "meaning": self.meaning, "stats": self.stats.describe()}

    def budget(self, sections: list[Section], applying: list[Section]) -> None:
        """Equal space: each kind gets per_lens x (sections that apply) / 2, split over its sections, 3 to 5 each."""
        per = self.settings.per_lens
        kinds = {}
        for s in applying:
            kinds.setdefault(s.lens.kind if s.lens.kind != "map" else "discover", []).append(s)
        if len(kinds) <= 1:
            for s in sections:
                s.budget = per
            return
        total = per * len(applying)
        for group in kinds.values():
            share = total / 2 / len(group)
            for s in group:
                s.budget = max(3, min(5, int(share)))
        for s in sections:
            if not s.budget:
                s.budget = per


def space(sections: list[Section]) -> dict[str, int]:
    """Results shown per kind (deepen, discover)."""
    out = {"deepen": 0, "discover": 0}
    for s in sections:
        out["deepen" if s.lens.kind == "deepen" else "discover"] += len(s.results)
    return out
