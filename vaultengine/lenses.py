"""Deepen lenses: five ways to look around one starting note (`vault deepen`).

* neighbourhood  - ``related``'s notes (links, tags and text), kept when their score is at or
                   above the ``percentile`` of this note's scores over every other note;
* cited together - notes co-cited with it (``graphrank.Graph.cocitation``: notes that link to
                   both), kept at or above the ``percentile`` of all co-citation values in the
                   vault (every co-cited pair);
* same foundations - notes it is coupled with (``Graph.coupling``: what both link to), kept at
                   or above the ``percentile`` of all coupling values in the vault;
* backbone       - the main chain through its topic: in its community (``shape.py``), two notes
                   are joined by their direct links plus, for every missing note or tag both
                   attach to, 1 / ln(1 + that node's notes) (Adamic-Adar); the maximum spanning
                   tree of those weights keeps each note's strongest ties, and its longest path
                   is the backbone. The note is on it, or so many steps off it;
* cluster view   - its community: name, key phrases, hubs, topic pages; where the note stands
                   in it (its weight inside, as a percentile of the members'; the share of its
                   weight that goes to other communities); the communities next to it, with the
                   notes that carry the most weight there.

No number here is tuned on one vault: every cutoff is a percentile of the vault (the share of
notes or pairs that stands out), and the vault distributions for co-citation and coupling are
exact, or, past ``DISTRIBUTION_SOURCES`` notes, taken from a seeded sample of source notes.
"""

from __future__ import annotations

import math
import random
from collections import Counter, deque
from typing import Any

from vaultengine.related import Relatedness
from vaultengine.shape import Distribution, VaultShape, percentile, percentile_rank
from vaultengine.vault import link_name

DISTRIBUTION_SOURCES = 1000  # above this many notes, the vault distributions come from a seeded sample


class Lenses:
    def __init__(self, engine: Relatedness, shape: VaultShape, percentile_cut: float = 90.0, seed: int = 42,
                 stats=None) -> None:
        """``stats``: a ``stats.Stats`` that keeps the vault distributions between runs (`vault explore`)."""
        from vaultengine.stats import NoStats

        self.stats = stats if stats is not None else NoStats()
        self.engine = engine
        self.shape = shape
        self.graph = engine.graph
        self.p = percentile_cut
        self.seed = seed
        self._distributions: dict[str, list[float]] = {}

    def node(self, ref: str) -> int:
        from vaultengine.querylang import QueryError

        seed = self.engine.seed(ref)
        if seed.path is None:
            raise QueryError(f'"{ref}" is not a note yet (only links to it exist)', None, None,
                             "deepen starts from a note; `vault related` also takes a missing note")
        return seed.node

    def _name(self, node: int) -> str:
        g = self.graph
        return f"{g.labels[node]} (no note yet)" if g.is_missing(node) else link_name(g.labels[node])

    # -- the vault's distributions --

    def _candidates(self, a: int, kind: str) -> set[int]:
        g = self.graph
        if kind == "cocitation":
            found = {b for c in g.incoming[a] for b in g.outgoing[c]}
        else:
            found = {b for z in g.outgoing[a] for b in g.incoming[z]}
        return {b for b in found if b != a and not g.is_missing(b)}

    def distribution(self, kind: str) -> tuple[list[float], bool]:
        """(every nonzero value of ``kind`` between two notes in the vault, exact?)."""
        g = self.graph
        notes = self.shape.sg.notes
        exact = len(notes) <= DISTRIBUTION_SOURCES
        if kind not in self._distributions:
            def compute():
                sources = notes if exact else sorted(random.Random(self.seed).sample(notes, DISTRIBUTION_SOURCES))
                measure = g.cocitation if kind == "cocitation" else g.coupling
                values = []
                for a in sources:
                    for b in self._candidates(a, kind):
                        if not exact or a < b:  # exact: each pair once
                            value = measure(a, b)[0]
                            if value > 0:
                                values.append(value)
                return Distribution(values)

            self._distributions[kind] = self.stats.get(f"pairs:{kind}", compute, tolerant=True)
        return self._distributions[kind], exact

    # -- the lenses --

    def neighbourhood(self, node: int, limit: int) -> dict[str, Any]:
        engine = self.engine
        seed = engine.seed(self.graph.labels[node])
        everything = engine.related(seed, limit=None, words=False)
        scores = {h.path: h.score for h in everything.hits}
        others = [p for f, p in engine.paths.items() if f != self.graph.note_ids[node]]
        values = [scores.get(p, 0.0) for p in others]
        cut = percentile(values, self.p)
        kept = [h for h in everything.hits if h.score >= cut and h.score > 0]
        shown = engine.related(seed, limit=min(limit, len(kept)) if limit else len(kept), words=True).hits if kept else []
        return {"cutoff": cut, "percentile": self.p, "compared": len(values), "kept": len(kept),
                "text_unavailable": everything.text_unavailable, "uncached": everything.uncached,
                "hits": [{"path": h.path, "score": h.score, "linked": h.linked, "reason": h.reason} for h in shown]}

    def _pair_lens(self, node: int, kind: str, limit: int) -> dict[str, Any]:
        g = self.graph
        values, exact = self.distribution(kind)
        cut = percentile(values, self.p)
        measure = g.cocitation if kind == "cocitation" else g.coupling
        found = []
        for b in self._candidates(node, kind):
            value, evidence = measure(node, b)
            if value > 0:
                found.append((value, b, evidence))
        found.sort(key=lambda item: (-item[0], g.labels[item[1]]))
        kept = [item for item in found if item[0] >= cut]
        rows = kept if not limit else kept[:limit]
        weaker = found[len(kept): len(kept) + 3] if not kept else []
        row = lambda value, b, evidence: {
            "path": g.labels[b], "value": value, "percentile": percentile_rank(values, value),
            "evidence": [{"name": self._name(x), "weight": w} for x, w in evidence]}
        return {"cutoff": cut, "percentile": self.p, "pairs_in_vault": len(values), "exact": exact,
                "found": len(found), "kept": len(kept), "notes": [row(*item) for item in rows],
                "below_cutoff": [row(*item) for item in weaker]}

    def cited_together(self, node: int, limit: int) -> dict[str, Any]:
        return self._pair_lens(node, "cocitation", limit)

    def foundations(self, node: int, limit: int) -> dict[str, Any]:
        return self._pair_lens(node, "coupling", limit)

    def backbone(self, node: int) -> dict[str, Any]:
        shape, sg = self.shape, self.shape.sg
        community = shape.community_of(node)
        if community is None:
            return {"community": None, "chain": [], "note_on_chain": None, "steps_off": None, "joins_at": None}
        members = set(community.notes)
        number = community.number
        weight: Counter[tuple[int, int]] = Counter()
        via: dict[tuple[int, int], tuple[float, str]] = {}
        for v in community.notes:
            for u, w in sg.adj[v].items():
                if u in members and v < u:
                    weight[(v, u)] += w
                    via[(v, u)] = (math.inf, "links")
        for z in community.others:
            linked = sorted(u for u in sg.adj[z] if u in members)
            share = 1 / math.log(1 + len(sg.adj[z]))
            for i, a in enumerate(linked):
                for b in linked[i + 1:]:
                    weight[(a, b)] += share
                    if via.get((a, b), (0.0, ""))[0] < share:
                        via[(a, b)] = (share, f"both {sg.name(z) if sg.kinds[z] != 'missing' else '→ ' + sg.labels[z]}")
        # Maximum spanning tree (Kruskal; ties by path, so the same tree every run).
        parent = {v: v for v in members}

        def find(v):
            while parent[v] != v:
                parent[v] = parent[parent[v]]
                v = parent[v]
            return v

        tree: dict[int, list[int]] = {v: [] for v in members}
        for (a, b), w in sorted(weight.items(), key=lambda item: (-item[1], sg.labels[item[0][0]], sg.labels[item[0][1]])):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
                tree[a].append(b)
                tree[b].append(a)

        a, _, _ = tree_bfs(node, tree, sg)
        b, _, prev = tree_bfs(a, tree, sg)
        chain = []
        v = b
        while v is not None:
            chain.append(v)
            v = prev[v]
        steps = []
        for x, y in zip(chain, chain[1:]):
            key = (min(x, y), max(x, y))
            steps.append(via.get(key, (0.0, ""))[1])
        on = set(chain)
        result = {"community": number, "chain": [sg.labels[v] for v in chain], "steps": steps,
                  "note_on_chain": chain.index(node) + 1 if node in on else None, "steps_off": 0 if node in on else None,
                  "joins_at": None, "length": len(chain), "community_notes": len(members)}
        if node not in on:
            _, dist, _ = tree_bfs(node, tree, sg)
            target = min((v for v in on if v in dist), key=lambda v: (dist[v], sg.labels[v]), default=None)
            if target is not None:
                result["steps_off"], result["joins_at"] = dist[target], sg.labels[target]
        return result

    def cluster(self, node: int, limit: int) -> dict[str, Any]:
        shape, sg = self.shape, self.shape.sg
        community = shape.community_of(node)
        membership = shape.membership
        total = sum(w for u, w in sg.adj[node].items() if u != node)
        by_community: Counter[int] = Counter()
        for u, w in sg.adj[node].items():
            if u != node:
                by_community[membership[u]] += w
        participation = 1 - sum((w / total) ** 2 for w in by_community.values()) if total else 0.0
        if community is None:
            return {"community": None, "weight": total, "participation": participation}
        number = community.number
        inner = community.inner_weight
        # The communities next to this one: weight across, and the notes that carry most of it.
        across: Counter[int] = Counter()
        carriers: dict[int, Counter[int]] = {}
        for v in community.notes:
            for u, w in sg.adj[v].items():
                c = membership[u]
                if c != number:
                    across[c] += w
                    carriers.setdefault(c, Counter())[v] += w
        names = {c.number: c.name for c in shape.communities}
        neighbours = [{"community": c, "name": names.get(c, sg.name(next(u for u in range(len(membership)) if membership[u] == c))),
                       "weight": w, "through": [sg.labels[v] for v, _ in carriers[c].most_common(3)]}
                      for c, w in across.most_common(limit or None) if c in names]
        return {
            "community": number, "name": community.name, "notes": len(community.notes),
            "phrases": [{"phrase": p.phrase, "in_community": p.in_community, "in_vault": p.in_vault} for p in community.phrases],
            "hubs": [sg.labels[v] for v in community.hubs],
            "topic_pages": [{"path": sg.labels[v], "reaches": n} for v, n in community.topic_pages],
            "weight_inside": inner.get(node, 0.0),
            "inside_percentile": percentile_rank(list(inner.values()), inner.get(node, 0.0)),
            "weight": total, "outside_share": 1 - (by_community[number] / total) if total else 0.0,
            "participation": participation, "neighbours": neighbours,
        }


def tree_bfs(start: int, tree: dict[int, list[int]], sg) -> tuple[int, dict[int, int], dict[int, int | None]]:
    """(the farthest node from ``start`` in the tree, ties by path; hops to every node; the previous node on the way)."""
    dist: dict[int, int] = {start: 0}
    prev: dict[int, int | None] = {start: None}
    queue = deque([start])
    while queue:
        v = queue.popleft()
        for u in sorted(tree[v], key=lambda x: sg.labels[x]):
            if u not in dist:
                dist[u], prev[u] = dist[v] + 1, v
                queue.append(u)
    far = max(dist.values())
    return min((v for v in dist if dist[v] == far), key=lambda v: sg.labels[v]), dist, prev
