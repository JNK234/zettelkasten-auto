"""The shape of the vault from links and tags (no embeddings): communities, bridges, hubs and authorities.

The graph
---------
The graph ``related`` walks (``graphrank.build_graph``: notes, missing notes, tag links as edges
to what they tag with), plus one node per #tag joined to the notes that carry it. A #tag on
more than a third of the notes (``related.COMMON_TAG``; #baby) is left out: it would join
everything. Edge weight = the number of links between two nodes (a #tag edge weighs 1).

Communities
-----------
Louvain (Blondel et al. 2008) on that graph, in plain Python: move each node to the
neighbouring community with the largest modularity gain
``k_i,C - resolution * tot_C * k_i / 2m`` until nothing moves, merge every community into one
node, repeat until a level moves nothing. Nodes are visited in an order shuffled by a seeded
``random.Random(seed)``, a node stays put unless a move gains strictly more, and dictionaries
keep insertion order, so the same graph and seed give the same communities on every run.
Louvain can leave a community in pieces that only connected through nodes that later moved
away (the flaw Leiden fixes with its refinement step); each community is therefore split into
its connected components before it is reported. Communities are numbered by their number of
notes, largest first.

Each community is named by its *key phrases*: the tags, tag links and link targets of its
notes that are both common in it and concentrated in it, scored coverage x concentration =
(its notes that have the phrase / its notes) x (its notes that have it / all notes that have
it), at least two of its notes having it. Its *hubs* are the notes with the most weight to the
rest of the community. A note the format types as a topic page (its type is one of
``engine.explore.topic_types``, ``topic`` and ``page`` by default; a folder says nothing) is a
*topic page* of the community most of its direct links (either way) go to, its home, wherever
Louvain placed it; a page with no direct links to notes has its own community as home. A
community no page calls home has no topic page. Without a format there are no topic pages to
find, and the map says so.

Meaning edges (``vault map --meaning``)
---------------------------------------
A note with few links and tags can end up alone. With meaning edges, two notes that are each
among the other's ``engine.explore.meaning_neighbours`` nearest neighbours by embedding
(centered cosine, ``related.py``; 1 = each other's nearest) are joined by an edge of weight 1,
when their similarity is at or above ``engine.explore.meaning_percentile`` of every note's
nearest-neighbour similarity. With 1 neighbour each note gets at most one such edge. They only inform the
communities: bridges, hubs and authorities stay on links and tags.

Bridges
-------
Betweenness between notes: for every pair of notes, the share of their shortest paths (in
hops, through any node) that run through a note, summed (Brandes 2001, with only notes as
sources and targets), as a fraction of all pairs of other notes. Exact Brandes costs a
breadth-first search per note, O(notes x (nodes + edges)); when that exceeds
``engine.explore.path_budget`` edge visits, a seeded sample of source notes is used and the
sums are scaled up (Brandes & Pich 2007), and the result says so.

Burt's constraint (1992): c_i = sum over neighbours j of (p_ij + sum over neighbours q of
p_iq p_qj)^2 with p_ij = w_ij / (i's total weight). It is high when a note's contacts are all
tied to each other (one group) and low when they are not (it spans groups).

Hubs and authorities
--------------------
HITS (Kleinberg 1999) on the directed body links (tag links are tags, left out), missing
notes included: a good hub links to good authorities, a good authority is linked from good
hubs; power iteration to a fixed point, scores scaled so the largest is 1.

Every list below a threshold uses a percentile of the vault (``percentile``), never a number
tuned on one vault.
"""

from __future__ import annotations

import math
import random
import sqlite3
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass, field
from typing import Any

from vaultengine.graphrank import Graph, build_graph
from vaultengine.related import COMMON_TAG
from vaultengine.vault import link_name

DEFAULT_SEED = 42


class Distribution:
    """A vault distribution sorted once (numpy): its percentiles and the rank of a value by binary search."""

    def __init__(self, values) -> None:
        import numpy as np

        if isinstance(values, Distribution):
            values = values.sorted
        elif not isinstance(values, np.ndarray):
            values = list(values)
        self.sorted = np.sort(np.asarray(values, dtype=np.float64))

    def __len__(self) -> int:
        return len(self.sorted)


def _as_distribution(values) -> Distribution:
    return values if isinstance(values, Distribution) else Distribution(values)


def percentile(values, p: float) -> float:
    """The p-th percentile (0-100) of ``values``, linear between the two nearest ranks (0 for no values)."""
    ordered = _as_distribution(values).sorted
    if not len(ordered):
        return 0.0
    rank = (len(ordered) - 1) * p / 100
    low = math.floor(rank)
    high = min(low + 1, len(ordered) - 1)
    return float(ordered[low] + (ordered[high] - ordered[low]) * (rank - low))


def percentile_rank(values, x: float) -> float:
    """The share of ``values`` at or below ``x``, 0-100. Pass a ``Distribution`` when asking many times."""
    import numpy as np

    ordered = _as_distribution(values).sorted
    return 100 * float(np.searchsorted(ordered, x, side="right")) / len(ordered) if len(ordered) else 0.0


# ----- the graph -------------------------------------------------------------------------------


@dataclass
class ShapeGraph:
    graph: Graph
    labels: list[str]  # node -> note path, missing target, or #tag
    kinds: list[str]  # node -> "note" | "missing" | "tag"
    adj: list[dict[int, float]]  # undirected, weighted
    notes: list[int]  # the note nodes
    left_out_tags: list[tuple[str, int]] = field(default_factory=list)  # (#tag, notes) on more than a third

    def name(self, node: int) -> str:
        kind = self.kinds[node]
        if kind == "note":
            return link_name(self.labels[node])
        return self.labels[node] if kind == "tag" else f"{self.labels[node]} (no note yet)"

    @property
    def edges(self) -> int:
        return sum(len(a) for a in self.adj) // 2


def shape_graph(conn: sqlite3.Connection, tag_rule=None, graph: Graph | None = None) -> ShapeGraph:
    graph = graph or build_graph(conn, tag_rule)
    rows = conn.execute("SELECT file_id, name_key FROM tags ORDER BY name_key, file_id").fetchall()
    return shape_graph_from(graph, rows)


def shape_graph_from(graph: Graph, hashtags) -> ShapeGraph:
    """The map's graph from a link graph and the #tags ((file id, name key) rows); see ``shape_graph``."""
    n = len(graph.labels)
    adj = [dict(a) for a in graph.adjacency]
    labels = list(graph.labels)
    kinds = ["missing" if graph.is_missing(v) else "note" for v in range(n)]
    notes = [v for v in range(n) if kinds[v] == "note"]
    tags: dict[str, set[int]] = defaultdict(set)
    for file_id, name_key in sorted(hashtags, key=lambda row: (row[1], row[0])):
        if file_id in graph.by_file:
            tags[name_key].add(graph.by_file[file_id])
    left_out = []
    for name in sorted(tags):
        members = tags[name]
        if len(members) > COMMON_TAG * len(notes):
            left_out.append((f"#{name}", len(members)))
            continue
        node = len(labels)
        labels.append(f"#{name}")
        kinds.append("tag")
        adj.append({})
        for v in sorted(members):
            adj[node][v] = 1.0
            adj[v][node] = adj[v].get(node, 0.0) + 1.0
    return ShapeGraph(graph, labels, kinds, adj, notes, left_out)


# ----- meaning edges --------------------------------------------------------------------------------


def mutual_nearest(matrix, k: int, percentile_cut: float) -> tuple[list[tuple[int, int, float]], float]:
    """(row pairs i < j that are each among the other's k nearest by dot product, at or above the cutoff; the cutoff).

    ``matrix``: unit rows (centered embeddings). The cutoff is ``percentile_cut`` of every row's nearest-neighbour
    similarity. Computed in blocks of rows, never the whole rows x rows matrix.
    """
    import numpy as np

    n = len(matrix)
    if n < 2:
        return [], 0.0
    k = max(1, min(k, n - 1))
    nearest = np.empty((n, k), dtype=np.int64)
    sims = np.empty((n, k), dtype=np.float32)
    for start in range(0, n, 1024):
        block = matrix[start : start + 1024] @ matrix.T
        for i in range(block.shape[0]):
            block[i, start + i] = -np.inf
        top = np.argsort(-block, axis=1, kind="stable")[:, :k]
        nearest[start : start + 1024] = top
        sims[start : start + 1024] = np.take_along_axis(block, top, axis=1)
    cutoff = percentile([float(x) for x in sims[:, 0]], percentile_cut)
    near = [set(int(j) for j in row) for row in nearest]
    sim_of = {(i, int(j)): float(x) for i in range(n) for j, x in zip(nearest[i], sims[i])}
    pairs = [(i, j, sim_of[(i, j)]) for i in range(n) for j in sorted(near[i])
             if i < j and i in near[j] and sim_of[(i, j)] >= cutoff]
    return pairs, cutoff


# ----- communities -------------------------------------------------------------------------------


def louvain(adj: list[dict[int, float]], seed: int = DEFAULT_SEED, resolution: float = 1.0) -> list[int]:
    """A community number for every node (see the module docstring). Deterministic for a given seed."""
    rng = random.Random(seed)
    n = len(adj)
    of_node = list(range(n))  # original node -> community of the current level
    level = [dict(a) for a in adj]
    while True:
        size = len(level)
        degree = [sum(a.values()) for a in level]  # a self-loop holds its weight twice (both ends)
        m2 = sum(degree)
        if m2 == 0:
            break
        community = list(range(size))
        total = list(degree)
        order = list(range(size))
        rng.shuffle(order)
        moved = False
        improved = True
        while improved:
            improved = False
            for i in order:
                current = community[i]
                weights: dict[int, float] = {}
                for j, w in level[i].items():
                    if j != i:
                        weights[community[j]] = weights.get(community[j], 0.0) + w
                total[current] -= degree[i]
                best = current
                best_gain = weights.get(current, 0.0) - resolution * total[current] * degree[i] / m2
                for c, w in weights.items():
                    gain = w - resolution * total[c] * degree[i] / m2
                    if gain > best_gain + 1e-12:
                        best, best_gain = c, gain
                total[best] += degree[i]
                if best != current:
                    community[i] = best
                    improved = moved = True
        if not moved:
            break
        renumber: dict[int, int] = {}
        for c in community:
            renumber.setdefault(c, len(renumber))
        community = [renumber[c] for c in community]
        of_node = [community[c] for c in of_node]
        merged: list[dict[int, float]] = [{} for _ in renumber]
        for i in range(size):
            for j, w in level[i].items():
                a, b = community[i], community[j]
                merged[a][b] = merged[a].get(b, 0.0) + w
        level = merged
    return of_node


def connected_parts(adj: list[dict[int, float]], membership: list[int]) -> list[int]:
    """Split every community into its connected components (within the community)."""
    part = [-1] * len(adj)
    count = 0
    for start in range(len(adj)):
        if part[start] != -1:
            continue
        part[start] = count
        queue = deque([start])
        while queue:
            v = queue.popleft()
            for u in adj[v]:
                if part[u] == -1 and membership[u] == membership[start]:
                    part[u] = count
                    queue.append(u)
        count += 1
    return part


def modularity(adj: list[dict[int, float]], membership: list[int], resolution: float = 1.0) -> float:
    m2 = sum(sum(a.values()) for a in adj)
    if not m2:
        return 0.0
    inside: Counter[int] = Counter()
    total: Counter[int] = Counter()
    for i, a in enumerate(adj):
        total[membership[i]] += sum(a.values())
        for j, w in a.items():
            if membership[i] == membership[j]:
                inside[membership[i]] += w
    return sum(inside[c] / m2 - resolution * (total[c] / m2) ** 2 for c in total)


# ----- bridges -----------------------------------------------------------------------------------


@dataclass
class Betweenness:
    values: dict[int, float]  # note node -> share of note pairs whose shortest paths run through it
    sources: int  # notes the searches started from
    exact: bool
    visits: int  # the estimated cost that decided exact or sampled: notes x (nodes + edges)
    budget: int


def _csr(rows: list[list[int]]):
    """(indptr, indices, counts) of lists of node ids, each kept in its order."""
    import numpy as np

    counts = np.fromiter((len(r) for r in rows), dtype=np.int64, count=len(rows))
    indptr = np.zeros(len(rows) + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    indices = np.fromiter((x for r in rows for x in r), dtype=np.int64, count=int(indptr[-1]))
    return indptr, indices, counts


def betweenness(sg: ShapeGraph, budget: int, seed: int = DEFAULT_SEED) -> Betweenness:
    """Betweenness of the notes, between notes (see the module docstring).

    Brandes with one breadth-first search per source note, each level of the search done with numpy over
    the whole level (the pure-Python loop took 5 to 10 s at 5,000 notes). The arithmetic is the loop's,
    in the loop's order: nodes are discovered in queue order (neighbours sorted), a node's dependency adds
    its successors' shares in reversed discovery order, so the numbers are the same bit for bit (the tests
    compare them with the loop, kept in tests/shape_reference.py). Path counts are added as floats, exact
    below 2^53; above that it stops rather than round.
    """
    import numpy as np

    neighbours = [sorted(a) for a in sg.adj]
    size = len(sg.adj)
    notes = sg.notes
    per_search = size + sg.edges
    cost = len(notes) * per_search
    if cost <= budget or len(notes) < 3:
        sources, exact = list(notes), True
    else:
        k = max(1, min(len(notes), budget // max(1, per_search)))
        sources, exact = sorted(random.Random(seed).sample(notes, k)), False
    is_note = np.array([k == "note" for k in sg.kinds], dtype=np.float64)
    indptr, indices, degree = _csr(neighbours)
    empty = np.zeros(0, dtype=np.int64)

    def around(nodes):  # (each node repeated, its neighbours): rows in the order given, each sorted
        counts = degree[nodes]
        total = int(counts.sum())
        if not total:
            return empty, empty
        offsets = np.repeat(indptr[nodes] - (np.cumsum(counts) - counts), counts) + np.arange(total)
        return np.repeat(nodes, counts), indices[offsets]

    dist = np.full(size, -1, dtype=np.int64)
    sigma = np.zeros(size)
    delta = np.zeros(size)
    share = np.zeros(size)
    slot = np.zeros(size, dtype=np.int64)
    score = np.zeros(size)
    most_paths = 0.0
    for s in sources:
        dist[s], sigma[s] = 0, 1.0
        frontier = np.array([s], dtype=np.int64)
        levels, preds = [frontier], []
        depth = 0
        while True:
            _, reached = around(frontier)
            new = reached[dist[reached] < 0]
            if not len(new):
                break
            by_node = np.argsort(new, kind="stable")  # the first time each node is reached, in queue order
            grouped = new[by_node]
            first = np.ones(len(new), dtype=bool)
            first[1:] = grouped[1:] != grouped[:-1]
            level = new[np.sort(by_node[first])]
            dist[level] = depth + 1
            back = level[::-1]  # predecessors, from the reached side, in reversed queue order
            w, v = around(back)
            keep = dist[v] == depth
            w, v = w[keep], v[keep]
            slot[back] = np.arange(len(back))
            sigma[back] = np.bincount(slot[w], weights=sigma[v], minlength=len(back))
            preds.append((v, w))
            levels.append(level)
            frontier = level
            depth += 1
        for k in range(len(levels) - 1, 0, -1):
            here = levels[k]
            share[here] = (is_note[here] + delta[here]) / sigma[here]
            v, w = preds[k - 1]
            before = levels[k - 1]
            slot[before] = np.arange(len(before))
            delta[before] = np.bincount(slot[v], weights=sigma[v] * share[w], minlength=len(before))
        seen = np.concatenate(levels)
        most_paths = max(most_paths, float(sigma[seen].max()))
        delta[s] = 0.0
        score[seen] += delta[seen]
        dist[seen], sigma[seen], delta[seen] = -1, 0.0, 0.0
    if most_paths >= 2.0 ** 53:
        raise OverflowError("betweenness: more shortest paths than floats count exactly")
    n = len(notes)
    pairs = (n - 1) * (n - 2) / 2 if n > 2 else 1
    scale = (n / len(sources)) / 2 / pairs  # every pair is found from both ends; a sample stands for all notes
    return Betweenness({v: float(score[v]) * scale for v in notes}, len(sources), exact, cost, budget)


def constraint(sg: ShapeGraph) -> dict[int, float]:
    """Burt's constraint of every note that has neighbours (see the module docstring). The proportions are
    computed once per node; the sums are the same sum() calls, so the numbers are unchanged."""
    strength = [sum(a.values()) for a in sg.adj]
    share = [({j: w / strength[i] for j, w in a.items()} if strength[i] else {}) for i, a in enumerate(sg.adj)]
    result = {}
    for i in sg.notes:
        mine = share[i]
        contacts = [j for j in sg.adj[i] if j != i]
        if not contacts:
            continue
        total = 0.0
        for j in contacts:
            indirect = sum(mine.get(q, 0.0) * share[q].get(j, 0.0) for q in contacts if q != j)
            total += (mine.get(j, 0.0) + indirect) ** 2
        result[i] = total
    return result


def _sum_rows(values, lengths):
    """sum() of each row of a ragged array (given flat, row after row), vectorized across rows and equal to
    Python's sum() bit for bit: since 3.12 it adds floats with Neumaier's compensation, as done here."""
    import numpy as np

    lengths = np.asarray(lengths, dtype=np.int64)
    total, carry = np.zeros(len(lengths)), np.zeros(len(lengths))
    starts = np.cumsum(lengths) - lengths
    for k in range(int(lengths.max()) if len(lengths) else 0):
        rows = np.nonzero(lengths > k)[0]
        x = values[starts[rows] + k]
        t0 = total[rows]
        t = t0 + x
        carry[rows] += np.where(np.abs(t0) >= np.abs(x), (t0 - t) + x, (x - t) + t0)
        total[rows] = t
    fix = (carry != 0) & np.isfinite(carry)
    total[fix] += carry[fix]
    return total


def hits(graph: Graph, tol: float = 1e-9, max_iter: int = 1000) -> tuple[dict[int, float], dict[int, float]]:
    """(hub score of each node that links, authority score of each node linked), largest scaled to 1.

    Power iteration until no score moves by more than ``tol`` (scores are shown to two decimals; a
    tolerance on the sum of all changes would get stricter as the vault grows). Each step in numpy, with
    the additions in the order (and with the compensation) the pure-Python loop used, so the scores are
    the same bit for bit.
    """
    import numpy as np

    sources = [v for v in range(len(graph.outgoing)) if graph.outgoing[v]]
    if not sources:
        return {}, {}
    targets = sorted({t for v in sources for t in graph.outgoing[v]})
    position = {t: i for i, t in enumerate(targets)}
    out = [[position[t] for t in graph.outgoing[v]] for v in sources]
    lengths = [len(o) for o in out]
    rows = np.repeat(np.arange(len(sources)), lengths)
    cols = np.fromiter((t for o in out for t in o), dtype=np.int64, count=len(rows))
    hub = np.ones(len(sources))
    auth = np.zeros(len(targets))
    for _ in range(max_iter):
        new_auth = np.bincount(cols, weights=hub[rows], minlength=len(targets))  # += in source order
        norm = math.sqrt(sum((new_auth * new_auth).tolist())) or 1.0
        new_auth = new_auth / norm
        new_hub = _sum_rows(new_auth[cols], lengths)
        norm = math.sqrt(sum((new_hub * new_hub).tolist())) or 1.0
        new_hub = new_hub / norm
        change = max(float(np.abs(new_hub - hub).max()), float(np.abs(new_auth - auth).max()))
        hub, auth = new_hub, new_auth
        if change < tol:
            break
    hub_list, auth_list = hub.tolist(), auth.tolist()
    top_h, top_a = max(hub_list) or 1.0, max(auth_list) or 1.0
    return {v: x / top_h for v, x in zip(sources, hub_list)}, {t: x / top_a for t, x in zip(targets, auth_list)}


def compare(before: "VaultShape", after: "VaultShape") -> dict[str, Any]:
    """How the communities change from ``before`` to ``after`` (e.g. without and with meaning edges).

    Each community of ``after`` is matched with the ``before`` community it shares the most notes with; a note
    moved if its community's match is not the community it was in.
    """
    b_of = {v: c.number for c in before.communities for v in c.notes}
    match: dict[int, int | None] = {}
    for c in after.communities:
        shared = Counter(b_of[v] for v in c.notes if v in b_of)
        match[c.number] = min(shared, key=lambda k: (-shared[k], k)) if shared else None
    moved = [v for c in after.communities for v in c.notes if v in b_of and b_of[v] != match[c.number]]
    placed = sorted(set(before.stranded) - set(after.stranded), key=lambda v: after.sg.labels[v])
    merged = {m: [c.number for c in after.communities if match[c.number] == m] for m in set(match.values()) if m is not None}
    splits = {m: numbers for m, numbers in merged.items() if len(numbers) > 1}
    joined: dict[int, list[int]] = {}
    for c in after.communities:
        sources = sorted({b_of[v] for v in c.notes if v in b_of})
        if len(sources) > 1:
            joined[c.number] = sources
    return {"stranded_before": [after.sg.labels[v] for v in before.stranded],
            "stranded_after": [after.sg.labels[v] for v in after.stranded],
            "placed": [{"note": after.sg.labels[v], "community": after.membership[v]} for v in placed],
            "moved": [{"note": after.sg.labels[v], "from": b_of[v], "to": after.membership[v]} for v in moved],
            "match": match, "split": splits, "joined": joined,
            "communities_before": len(before.communities), "communities_after": len(after.communities)}


# ----- the map -----------------------------------------------------------------------------------


@dataclass
class Phrase:
    phrase: str  # "#tag", "[[tag link]]" or "→ link target"
    in_community: int  # its notes that have it
    in_vault: int  # all notes that have it
    score: float


@dataclass
class Community:
    number: int
    notes: list[int]  # note nodes, by weight inside the community (hubs first)
    others: list[int]  # missing notes and #tags placed in it
    phrases: list[Phrase]
    hubs: list[int]
    topic_pages: list[tuple[int, int]]  # (topic-folder note whose home this is, members it links to or is linked from)
    inner_weight: dict[int, float] = field(default_factory=dict)  # note -> weight to the rest of the community

    @property
    def name(self) -> str:
        words = [p.phrase.removeprefix("→ ") for p in self.phrases[:2]]
        return " · ".join(words) if words else "(no shared tag or link)"


class VaultShape:
    """Everything `vault map` and the deepen lenses need; each part computed on first use."""

    def __init__(self, conn: sqlite3.Connection, tag_rule=None, seed: int = DEFAULT_SEED, resolution: float = 1.0,
                 path_budget: int = 20_000_000, topic_page=None, graph: Graph | None = None, meaning=None,
                 meaning_percentile: float = 50.0, meaning_neighbours: int = 1, sg: ShapeGraph | None = None) -> None:
        """``topic_page``: path -> is it a topic page (from the format's types), None without a format.
        ``meaning``: a ``related.TextSimilarity`` to add meaning edges with (``--meaning``), else None."""
        self.conn = conn
        self.sg = sg if sg is not None else shape_graph(conn, tag_rule, graph)
        self.graph = self.sg.graph
        self.seed, self.resolution, self.path_budget = seed, resolution, path_budget
        self.topic_page = topic_page
        self.meaning = meaning
        self.meaning_percentile = meaning_percentile
        self.meaning_neighbours = meaning_neighbours
        self._meaning_edges: list[tuple[int, int, float]] | None = None
        self.meaning_cutoff: float | None = None
        self.meaning_unavailable: str | None = None
        self._membership: list[int] | None = None
        self._communities: list[Community] | None = None
        self._betweenness: Betweenness | None = None
        self._constraint: dict[int, float] | None = None
        self._hits = None
        from vaultengine.stats import NoStats

        self.stats = NoStats()  # `vault explore` sets a stats.Stats: these are then kept between runs (exact state only)

    # -- communities --

    @property
    def meaning_edges(self) -> list[tuple[int, int, float]]:
        """(note, note, centered cosine) for mutual nearest neighbours at or above the cutoff; [] without ``meaning``."""
        if self._meaning_edges is None:
            self._meaning_edges = []
            if self.meaning is not None:
                if not self.meaning.available():
                    self.meaning_unavailable = self.meaning.unavailable
                else:
                    self._meaning_edges = self._mutual_nearest()
        return self._meaning_edges

    def _mutual_nearest(self) -> list[tuple[int, int, float]]:
        import numpy as np

        text = self.meaning
        centered = text.centered
        rows = [i for i in range(len(centered)) if centered[i].any()]
        node_of = {i: self.graph.by_file[text.vectors.file_ids[i]] for i in rows}
        if len(rows) < 2:
            return []
        pairs, self.meaning_cutoff = mutual_nearest(centered[rows], self.meaning_neighbours, self.meaning_percentile)
        edges = [(node_of[rows[i]], node_of[rows[j]], sim) for i, j, sim in pairs]
        return sorted(edges, key=lambda e: (self.sg.labels[e[0]], self.sg.labels[e[1]]))

    @property
    def community_adj(self) -> list[dict[int, float]]:
        """The graph the communities are found on: links and tags, plus meaning edges when asked for."""
        if not self.meaning_edges:
            return self.sg.adj
        if getattr(self, "_community_adj", None) is None:
            adj = [dict(a) for a in self.sg.adj]
            for a, b, _ in self.meaning_edges:
                adj[a][b] = adj[a].get(b, 0.0) + 1.0
                adj[b][a] = adj[b].get(a, 0.0) + 1.0
            self._community_adj = adj
        return self._community_adj

    @property
    def membership(self) -> list[int]:
        """node -> community number (1 = most notes); singletons and note-less groups get their own numbers too."""
        if self._membership is None and not self.meaning_edges:
            self._membership = self.stats.get("membership", self._louvain)
        if self._membership is None:
            self._membership = self._louvain()
        return self._membership

    def _louvain(self) -> list[int]:
        adj = self.community_adj
        raw = connected_parts(adj, louvain(adj, self.seed, self.resolution))
        notes_in: Counter[int] = Counter(raw[v] for v in self.sg.notes)
        first = {}
        for v in range(len(raw)):
            first.setdefault(raw[v], self.sg.labels[v])
        order = sorted(set(raw), key=lambda c: (-notes_in[c], first[c]))
        number = {c: i + 1 for i, c in enumerate(order)}
        return [number[c] for c in raw]

    def features(self, note: int) -> set[str]:
        g = self.graph
        found = {f"→ {g.labels[t] if g.is_missing(t) else link_name(g.labels[t])}" for t in g.outgoing[note]}
        found |= {g.tag_name(t) for t in g.tag_links.get(note, ())}
        found |= {self.sg.labels[t] for t in self.sg.adj[note] if self.sg.kinds[t] == "tag"}
        return found

    @property
    def communities(self) -> list[Community]:
        """Communities with two or more notes, largest first."""
        if self._communities is None and not self.meaning_edges:
            self._communities = self.stats.get("communities", self._find_communities)
        if self._communities is None:
            self._communities = self._find_communities()
        return self._communities

    def _find_communities(self) -> list[Community]:
        """Communities with two or more notes, largest first (see ``communities``)."""
        membership = self.membership
        features = {v: self.features(v) for v in self.sg.notes}
        in_vault = Counter(f for fs in features.values() for f in fs)
        groups: dict[int, list[int]] = defaultdict(list)
        others: dict[int, list[int]] = defaultdict(list)
        for v in range(len(membership)):
            (groups if self.sg.kinds[v] == "note" else others)[membership[v]].append(v)
        result = []
        for number in sorted(groups):
            members = groups[number]
            if len(members) < 2:
                continue
            inner = {v: sum(w for u, w in self.community_adj[v].items() if membership[u] == number and u != v) for v in members}
            ranked = sorted(members, key=lambda v: (-inner[v], self.sg.labels[v]))
            counts = Counter(f for v in members for f in features[v])
            phrases = sorted(
                (Phrase(f, n, in_vault[f], (n / len(members)) * (n / in_vault[f])) for f, n in counts.items() if n >= 2),
                key=lambda p: (-p.score, -p.in_community, p.phrase),
            )
            pages = sorted(((v, n) for v, (home, n) in self.topic_homes().items() if home == number),
                           key=lambda item: (-item[1], self.sg.labels[item[0]]))
            result.append(Community(number, ranked, sorted(others[number]), phrases[:6], ranked[:3], pages, inner))
        return result

    def topic_homes(self) -> dict[int, tuple[int, int]]:
        """topic page -> (its home community, notes of it that it links to or is linked from)."""
        if self.topic_page is None:
            return {}
        membership = self.membership
        homes = {}
        for v in self.sg.notes:
            if not self.topic_page(self.sg.labels[v]):
                continue
            reach: Counter[int] = Counter(membership[u] for u in self.graph.adjacency[v]
                                          if self.sg.kinds[u] == "note" and u != v)
            if reach:
                home = max(reach, key=lambda c: (reach[c], c == membership[v], -c))
                homes[v] = (home, reach[home])
            else:
                homes[v] = (membership[v], 0)
        return homes

    def community_of(self, note: int) -> Community | None:
        number = self.membership[note]
        return next((c for c in self.communities if c.number == number), None)

    @property
    def modularity(self) -> float:
        return modularity(self.community_adj, self.membership, self.resolution)

    @property
    def stranded(self) -> list[int]:
        """Notes in no community of two or more notes."""
        big = {v for c in self.communities for v in c.notes}
        return [v for v in self.sg.notes if v not in big]

    # -- bridges, hubs --

    @property
    def betweenness(self) -> Betweenness:
        if self._betweenness is None:
            self._betweenness = self.stats.get("betweenness", lambda: betweenness(self.sg, self.path_budget, self.seed))
        return self._betweenness

    @property
    def constraint(self) -> dict[int, float]:
        if self._constraint is None:
            self._constraint = self.stats.get("constraint", lambda: constraint(self.sg))
        return self._constraint

    @property
    def hits(self) -> tuple[dict[int, float], dict[int, float]]:
        if self._hits is None:
            self._hits = self.stats.get("hits", lambda: hits(self.graph))
        return self._hits

    def spans(self, note: int) -> list[int]:
        """The communities (2+ notes) of a note's neighbours, most weight first."""
        big = {c.number for c in self.communities}
        weight: Counter[int] = Counter()
        for u, w in self.sg.adj[note].items():
            if self.membership[u] in big:
                weight[self.membership[u]] += w
        return [c for c, _ in weight.most_common()]

    def bridges(self) -> list[dict[str, Any]]:
        """Every note with its betweenness and constraint and their percentiles in the vault, most between first."""
        b = self.betweenness.values
        c = self.constraint
        b_all, c_all = Distribution(b.values()), Distribution(c.values())
        rows = []
        for v in self.sg.notes:
            rows.append({
                "node": v, "path": self.sg.labels[v], "betweenness": b[v],
                "betweenness_percentile": percentile_rank(b_all, b[v]),
                "constraint": c.get(v), "constraint_percentile": percentile_rank(c_all, c[v]) if v in c else None,
                "community": self.membership[v], "spans": self.spans(v),
            })
        rows.sort(key=lambda r: (-r["betweenness"], r["path"]))
        return rows

    def summary(self) -> dict[str, Any]:
        kinds = Counter(self.sg.kinds)
        in_big = sum(len(c.notes) for c in self.communities)
        return {
            "notes": kinds["note"], "missing_notes": kinds["missing"], "tags": kinds["tag"],
            "left_out_tags": [{"tag": t, "notes": n} for t, n in self.sg.left_out_tags],
            "nodes": len(self.sg.adj), "edges": self.sg.edges, "communities": len(self.communities),
            "notes_in_communities": in_big, "alone": kinds["note"] - in_big, "modularity": self.modularity,
            "seed": self.seed, "resolution": self.resolution, "tag_rule": self.graph.tag_rule,
            "meaning_edges": len(self.meaning_edges), "meaning_cutoff": self.meaning_cutoff,
            "meaning_unavailable": self.meaning_unavailable,
        }

    def to_dict(self, limit: int | None = None) -> dict[str, Any]:
        cut = (lambda items: items) if not limit else (lambda items: items[:limit])
        hub, auth = self.hits
        return {
            **self.summary(),
            "communities": [
                {"number": c.number, "name": c.name, "notes": [self.sg.labels[v] for v in c.notes],
                 "phrases": [asdict(p) for p in c.phrases], "hubs": [self.sg.labels[v] for v in c.hubs],
                 "topic_pages": [{"path": self.sg.labels[v], "reaches": n} for v, n in c.topic_pages]}
                for c in self.communities
            ],
            "betweenness": {"exact": self.betweenness.exact, "sources": self.betweenness.sources,
                            "visits": self.betweenness.visits, "budget": self.betweenness.budget},
            "bridges": [{k: v for k, v in r.items() if k != "node"} for r in cut(self.bridges())],
            "hubs": [{"path": self.graph.labels[v], "score": s} for v, s in cut(sorted(hub.items(), key=lambda x: (-x[1], x[0])))],
            "authorities": [{"target": self.graph.labels[v], "missing": self.graph.is_missing(v), "score": s}
                            for v, s in cut(sorted(auth.items(), key=lambda x: (-x[1], x[0])))],
        }
