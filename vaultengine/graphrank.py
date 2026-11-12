"""The link graph as a weighted undirected graph, and personalized PageRank on it.

Nodes are notes and *missing notes*: link targets that no file answers to yet
(``[[rl-environment-design]]`` with no such note). In vaults that use links as
topics, most of the structure is in links to notes that do not exist, so
leaving them out would drop most of the graph. Two notes that both link to the
same missing note are two hops apart through it.

Edge weight = the number of links between the two nodes, in either direction
(links and embeds; self-links and links to attachments are left out).

Personalized PageRank (random walk with restart) from a seed node s, with
restart probability ``alpha``: a walker follows a random edge (chosen in
proportion to its weight) and, with probability alpha at every step, jumps back
to s. The stationary distribution r satisfies

    r = alpha * e_s + (1 - alpha) * (M r + (mass on nodes without edges) * e_s)

where M[j][i] = w_ij / W_i (W_i: the weighted degree of i). ``personalized_pagerank``
finds r by power iteration until the L1 change is below ``tol``; the tests check
it against the exact solution of this linear system, computed with fractions.

For ranking related notes, ``related.py`` divides r_i by the weighted degree of i.
Raw PageRank favours hubs (a source note that links 60 things is "near" everything);
on an undirected graph r_s(i) / d_i = r_i(s) / d_s, so the normalised score is also
symmetric: the graph says A is as related to B as B is to A.

The same graph keeps the link directions too (``outgoing``, ``incoming``), for the pair
signals of ``relations.py``. For two notes a and b:

* co-citation - the notes that link to both. A citing note c weighs
  w_c = 1 / ln(1 + k_c), k_c = the nodes c links to (notes and missing notes): a list
  that links 80 things says less about any two of them than a note that links three.
  cocitation(a, b) = sum of w_c over the notes citing both
                     / sqrt((W_a + 1) * (W_b + 1)),  W_x = sum of w_c over all notes citing x;
* bibliographic coupling - the nodes both link to, missing notes included. A reference z
  weighs w_z = 1 / ln(1 + m_z), m_z = the notes linking to z: sharing a reference that
  50 notes cite says less than sharing one that two cite.
  coupling(a, b) = sum of w_z over the references of both
                   / sqrt((V_a + 1) * (V_b + 1)),  V_x = sum of w_z over all references of x.

Tag links (``tagrule.py``: ``Tags: [[topic]]``) are edges of the walk (``adjacency``) and count
as links between two notes (``direct``), but not in ``outgoing``/``incoming``: what two notes
are both tagged with is a shared tag (``tag_links``), not a shared reference.

Both are weighted cosines between 0 and 1, symmetric in a and b. The +1 keeps thin
evidence from looking perfect: two notes whose single citing note is the same one are
not 1.0 alike. Link counts do not matter here, only which nodes link to which; a and b
themselves are never their own citing note or reference (self-links are left out), so a
direct link between them counts only as a direct link.
"""

from __future__ import annotations

import math
import sqlite3
from collections import deque
from dataclasses import dataclass, field

ALPHA = 0.15  # restart probability (damping 0.85)


@dataclass
class Graph:
    labels: list[str]  # node -> note path, or the missing target as written
    note_ids: list[int | None]  # node -> files.id, None for a missing note
    adjacency: list[dict[int, float]]  # node -> {neighbour: weight}
    by_file: dict[int, int] = field(default_factory=dict)  # files.id -> node
    by_missing: dict[str, int] = field(default_factory=dict)  # lookup key of a missing target -> node
    direct: set[tuple[int, int]] = field(default_factory=set)  # (source node, target node) of resolved note links
    outgoing: list[dict[int, int]] = field(default_factory=list)  # node -> {node it links to: number of links}
    incoming: list[dict[int, int]] = field(default_factory=list)  # node -> {note linking to it: number of links}
    lines: dict[tuple[int, int], list[int]] = field(default_factory=dict)  # (source, target) -> lines of the links
    tag_lines: set[tuple[int, int, int]] = field(default_factory=set)  # (source, target, line) of tag links
    tag_links: dict[int, set[int]] = field(default_factory=dict)  # note node -> nodes it is tagged with
    tag_rule: str = ""  # which links are tags (tagrule.TagRule.description)

    def degree(self, node: int) -> float:
        return sum(self.adjacency[node].values())

    def walk(self) -> "Walk":
        """The random walk over ``adjacency``, built on first use (personalized PageRank from any node)."""
        if getattr(self, "_walk", None) is None:
            self._walk = Walk(self.adjacency)
        return self._walk

    def is_missing(self, node: int) -> bool:
        return self.note_ids[node] is None

    def tag_name(self, node: int) -> str:
        """A tag link's target as a tag: ``[[reward]]`` (lower case, the note's name or the missing target)."""
        from vaultengine.vault import link_name, lookup_key

        return f"[[{lookup_key(self.labels[node] if self.is_missing(node) else link_name(self.labels[node]))}]]"

    # -- directed signals between two nodes (see the module docstring) --

    def citer_weight(self, citer: int) -> float:
        return 1 / math.log(1 + len(self.outgoing[citer]))

    def reference_weight(self, reference: int) -> float:
        return 1 / math.log(1 + len(self.incoming[reference]))

    def cocitation(self, a: int, b: int) -> tuple[float, list[tuple[int, float]]]:
        """(value, [(note citing both, its weight)] heaviest first)."""
        both = set(self.incoming[a]) & set(self.incoming[b])
        return self._cosine(both, self.incoming[a], self.incoming[b], self.citer_weight)

    def coupling(self, a: int, b: int) -> tuple[float, list[tuple[int, float]]]:
        """(value, [(node both link to, its weight)] heaviest first)."""
        both = set(self.outgoing[a]) & set(self.outgoing[b])
        return self._cosine(both, self.outgoing[a], self.outgoing[b], self.reference_weight)

    def _cosine(self, both, of_a, of_b, weight) -> tuple[float, list[tuple[int, float]]]:
        if not both:
            return 0.0, []
        total_a = sum(weight(n) for n in of_a)
        total_b = sum(weight(n) for n in of_b)
        shared = sorted(((n, weight(n)) for n in both), key=lambda item: (-item[1], self.labels[item[0]]))
        return sum(w for _, w in shared) / math.sqrt((total_a + 1) * (total_b + 1)), shared


def link_rows(conn: sqlite3.Connection) -> list[tuple]:
    """Every link of a note (target not empty) as the row ``build_graph_from`` reads, in file and link order."""
    return conn.execute(
        """SELECT l.source_id, l.target_id, l.status, l.target_key, l.target, t.kind, l.line, s.path,
                  l.field_source, l.field_key
           FROM links l JOIN files s ON s.id = l.source_id LEFT JOIN files t ON t.id = l.target_id
           WHERE s.kind = 'note' AND l.target != '' ORDER BY l.source_id, l.ordinal"""
    ).fetchall()


def build_graph(conn: sqlite3.Connection, tag_rule=None) -> Graph:
    """The graph of the indexed vault; ``tag_rule`` (``tagrule.TagRule``, default: fields named tags) marks tag links."""
    notes = conn.execute("SELECT id, path FROM files WHERE kind = 'note' ORDER BY path").fetchall()
    return build_graph_from(notes, link_rows(conn), tag_rule)


def build_graph_from(notes, rows, tag_rule=None) -> Graph:
    """The graph of these notes ((file id, path), in path order) and link rows (``link_rows``): the experiment
    harness builds it from rows with links held out, so it is the same graph every command uses."""
    from vaultengine.tagrule import default_rule

    tag_rule = tag_rule or default_rule()
    graph = Graph([], [], [])
    graph.tag_rule = tag_rule.description

    def add(label: str, file_id: int | None) -> int:
        graph.labels.append(label)
        graph.note_ids.append(file_id)
        graph.adjacency.append({})
        graph.outgoing.append({})
        graph.incoming.append({})
        return len(graph.labels) - 1

    for file_id, path in notes:
        graph.by_file[file_id] = add(path, file_id)
    for source_id, target_id, status, target_key, target, target_kind, line, path, field_source, field_key in rows:
        if source_id not in graph.by_file:
            continue
        a = graph.by_file[source_id]
        if status == "resolved":
            if target_kind != "note" or target_id == source_id or target_id not in graph.by_file:
                continue
            b = graph.by_file[target_id]
            graph.direct.add((a, b))
        elif status == "unresolved":
            b = graph.by_missing.get(target_key)
            if b is None:
                b = graph.by_missing[target_key] = add(target, None)
        else:  # ambiguous: we cannot tell which note is meant
            continue
        graph.adjacency[a][b] = graph.adjacency[a].get(b, 0.0) + 1.0
        graph.adjacency[b][a] = graph.adjacency[b].get(a, 0.0) + 1.0
        graph.lines.setdefault((a, b), []).append(line)
        if tag_rule.is_tag(path, field_source, field_key):
            graph.tag_links.setdefault(a, set()).add(b)
            graph.tag_lines.add((a, b, line))
        else:
            graph.outgoing[a][b] = graph.outgoing[a].get(b, 0) + 1
            graph.incoming[b][a] = graph.incoming[b].get(a, 0) + 1
    return graph


def component(adjacency: list[dict[int, float]], seed: int) -> list[int]:
    seen = {seed}
    queue = deque([seed])
    while queue:
        node = queue.popleft()
        for other in adjacency[node]:
            if other not in seen:
                seen.add(other)
                queue.append(other)
    return sorted(seen)


class Walk:
    """The random walk of an adjacency as numpy arrays, built once: personalized PageRank from any seed."""

    def __init__(self, adjacency: list[dict[int, float]]) -> None:
        import numpy as np

        self.adjacency = adjacency
        src: list[int] = []
        dst: list[int] = []
        prob: list[float] = []
        stuck: list[int] = []
        for node, row in enumerate(adjacency):
            total = sum(row.values())
            if not total:
                stuck.append(node)
                continue
            for other, weight in row.items():
                src.append(node)
                dst.append(other)
                prob.append(weight / total)
        self.sources = np.array(src, dtype=np.intp)
        self.targets = np.array(dst, dtype=np.intp)
        self.prob = np.array(prob, dtype=np.float64)
        self.stuck = np.array(stuck, dtype=np.intp)

    def personalized_pagerank(self, seed: int, alpha: float = ALPHA, tol: float = 1e-10,
                              max_iter: int = 10_000) -> dict[int, float]:
        import numpy as np

        n = len(self.adjacency)
        keep = 1 - alpha
        moved = self.prob * keep
        rank = np.zeros(n)
        rank[seed] = 1.0
        for _ in range(max_iter):
            following = np.bincount(self.targets, weights=rank[self.sources] * moved, minlength=n)
            following[seed] += alpha + keep * float(rank[self.stuck].sum())
            change = float(np.abs(following - rank).sum())
            rank = following
            if change < tol:
                break
        return {node: float(rank[node]) for node in component(self.adjacency, seed)}


def personalized_pagerank(
    adjacency: list[dict[int, float]], seed: int, alpha: float = ALPHA, tol: float = 1e-10, max_iter: int = 10_000
) -> dict[int, float]:
    """Stationary distribution of the walk with restart to ``seed`` (only nodes reachable from it).

    Power iteration until the L1 change between two iterations is below ``tol``;
    the distance to the exact solution is then at most tol * (1 - alpha) / alpha.
    A node without edges sends its mass back to the seed (only the seed itself can be one: every
    other node it reaches has an edge). Each step is one pass over the edges in numpy (``Walk``;
    ``Graph.walk`` keeps one per graph): milliseconds at 5,000 notes, where the pure-Python loop
    took over a second.
    """
    return Walk(adjacency).personalized_pagerank(seed, alpha, tol, max_iter)


@dataclass
class Path:
    hops: int
    via: list[int]  # intermediate nodes for 2-hop paths (most-weighted first); the first step otherwise


def hop_paths(graph: Graph, seed: int, max_hops: int = 3) -> dict[int, Path]:
    """Shortest hop count from the seed to every node within ``max_hops``, with what it goes through."""
    distance = {seed: 0}
    first_step: dict[int, int] = {}
    queue = deque([seed])
    while queue:
        node = queue.popleft()
        if distance[node] == max_hops:
            continue
        for other in graph.adjacency[node]:
            if other not in distance:
                distance[other] = distance[node] + 1
                first_step[other] = other if node == seed else first_step[node]
                queue.append(other)
    paths = {}
    for node, hops in distance.items():
        if node == seed:
            continue
        if hops == 2:
            common = set(graph.adjacency[seed]) & set(graph.adjacency[node])
            via = sorted(common, key=lambda v: (-min(graph.adjacency[seed][v], graph.adjacency[node][v]),
                                                 graph.is_missing(v), graph.degree(v), graph.labels[v]))
            paths[node] = Path(2, via)
        else:
            paths[node] = Path(hops, [first_step[node]] if hops > 1 else [])
    return paths
