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
"""

from __future__ import annotations

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

    def degree(self, node: int) -> float:
        return sum(self.adjacency[node].values())

    def is_missing(self, node: int) -> bool:
        return self.note_ids[node] is None


def build_graph(conn: sqlite3.Connection) -> Graph:
    graph = Graph([], [], [])

    def add(label: str, file_id: int | None) -> int:
        graph.labels.append(label)
        graph.note_ids.append(file_id)
        graph.adjacency.append({})
        return len(graph.labels) - 1

    for file_id, path in conn.execute("SELECT id, path FROM files WHERE kind = 'note' ORDER BY path"):
        graph.by_file[file_id] = add(path, file_id)
    rows = conn.execute(
        """SELECT l.source_id, l.target_id, l.status, l.target_key, l.target, t.kind
           FROM links l JOIN files s ON s.id = l.source_id LEFT JOIN files t ON t.id = l.target_id
           WHERE s.kind = 'note' AND l.target != '' ORDER BY l.source_id, l.ordinal"""
    )
    for source_id, target_id, status, target_key, target, target_kind in rows:
        a = graph.by_file[source_id]
        if status == "resolved":
            if target_kind != "note" or target_id == source_id:
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


def personalized_pagerank(
    adjacency: list[dict[int, float]], seed: int, alpha: float = ALPHA, tol: float = 1e-10, max_iter: int = 10_000
) -> dict[int, float]:
    """Stationary distribution of the walk with restart to ``seed`` (only nodes reachable from it).

    Power iteration until the L1 change between two iterations is below ``tol``;
    the distance to the exact solution is then at most tol * (1 - alpha) / alpha.
    """
    nodes = component(adjacency, seed)
    local = {node: k for k, node in enumerate(nodes)}
    # Per node: (neighbour, transition probability); nodes without edges keep None.
    steps: list[list[tuple[int, float]] | None] = []
    for node in nodes:
        total = sum(adjacency[node].values())
        steps.append([(local[o], w / total) for o, w in adjacency[node].items()] if total else None)
    n, s = len(nodes), local[seed]
    rank = [0.0] * n
    rank[s] = 1.0
    keep = 1 - alpha
    for _ in range(max_iter):
        following = [0.0] * n
        stuck = 0.0
        for i, mass in enumerate(rank):
            if mass:
                out = steps[i]
                if out is None:
                    stuck += mass
                    continue
                mass *= keep
                for j, p in out:
                    following[j] += mass * p
        following[s] += alpha + keep * stuck
        change = sum(abs(a - b) for a, b in zip(following, rank))
        rank = following
        if change < tol:
            break
    return {node: rank[k] for k, node in enumerate(nodes)}


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
