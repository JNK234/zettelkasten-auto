"""The pure-Python betweenness, constraint and HITS that shape.py used before they were vectorized, kept to
prove the numpy versions give the same numbers, bit for bit (test_vaultengine_shape.ReferenceTests)."""

import math
import random
from collections import deque

from vaultengine.shape import DEFAULT_SEED, Betweenness, ShapeGraph
from vaultengine.graphrank import Graph


def betweenness(sg: ShapeGraph, budget: int, seed: int = DEFAULT_SEED) -> Betweenness:
    """Betweenness of the notes, between notes (see the module docstring)."""
    neighbours = [sorted(a) for a in sg.adj]
    is_note = [k == "note" for k in sg.kinds]
    notes = sg.notes
    per_search = len(sg.adj) + sg.edges
    cost = len(notes) * per_search
    if cost <= budget or len(notes) < 3:
        sources, exact = list(notes), True
    else:
        k = max(1, min(len(notes), budget // max(1, per_search)))
        sources, exact = sorted(random.Random(seed).sample(notes, k)), False
    score = [0.0] * len(sg.adj)
    for s in sources:
        order = []
        preds: list[list[int]] = [[] for _ in sg.adj]
        sigma = [0] * len(sg.adj)
        dist = [-1] * len(sg.adj)
        sigma[s], dist[s] = 1, 0
        queue = deque([s])
        while queue:
            v = queue.popleft()
            order.append(v)
            for w in neighbours[v]:
                if dist[w] < 0:
                    dist[w] = dist[v] + 1
                    queue.append(w)
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    preds[w].append(v)
        delta = [0.0] * len(sg.adj)
        for w in reversed(order):
            coefficient = ((1.0 if is_note[w] and w != s else 0.0) + delta[w]) / sigma[w]
            for v in preds[w]:
                delta[v] += sigma[v] * coefficient
            if w != s:
                score[w] += delta[w]
    n = len(notes)
    pairs = (n - 1) * (n - 2) / 2 if n > 2 else 1
    scale = (n / len(sources)) / 2 / pairs  # every pair is found from both ends; a sample stands for all notes
    return Betweenness({v: score[v] * scale for v in notes}, len(sources), exact, cost, budget)


def constraint(sg: ShapeGraph) -> dict[int, float]:
    """Burt's constraint of every note that has neighbours (see the module docstring)."""
    strength = [sum(a.values()) for a in sg.adj]
    p = lambda i, j: sg.adj[i].get(j, 0.0) / strength[i] if strength[i] else 0.0
    result = {}
    for i in sg.notes:
        contacts = [j for j in sg.adj[i] if j != i]
        if not contacts:
            continue
        total = 0.0
        for j in contacts:
            indirect = sum(p(i, q) * p(q, j) for q in contacts if q != j)
            total += (p(i, j) + indirect) ** 2
        result[i] = total
    return result


def hits(graph: Graph, tol: float = 1e-9, max_iter: int = 1000) -> tuple[dict[int, float], dict[int, float]]:
    """(hub score of each node that links, authority score of each node linked), largest scaled to 1.

    Power iteration until no score moves by more than ``tol`` (scores are shown to two decimals; a
    tolerance on the sum of all changes would get stricter as the vault grows).
    """
    sources = [v for v in range(len(graph.outgoing)) if graph.outgoing[v]]
    if not sources:
        return {}, {}
    targets = sorted({t for v in sources for t in graph.outgoing[v]})
    position = {t: i for i, t in enumerate(targets)}
    out = [[position[t] for t in graph.outgoing[v]] for v in sources]
    hub = [1.0] * len(sources)
    auth = [0.0] * len(targets)
    for _ in range(max_iter):
        new_auth = [0.0] * len(targets)
        for h, links in zip(hub, out):
            for t in links:
                new_auth[t] += h
        norm = math.sqrt(sum(x * x for x in new_auth)) or 1.0
        new_auth = [x / norm for x in new_auth]
        new_hub = [sum(new_auth[t] for t in links) for links in out]
        norm = math.sqrt(sum(x * x for x in new_hub)) or 1.0
        new_hub = [x / norm for x in new_hub]
        change = max(max(abs(a - b) for a, b in zip(new_hub, hub)), max(abs(a - b) for a, b in zip(new_auth, auth)))
        hub, auth = new_hub, new_auth
        if change < tol:
            break
    top_h, top_a = max(hub) or 1.0, max(auth) or 1.0
    return {v: x / top_h for v, x in zip(sources, hub)}, {t: x / top_a for t, x in zip(targets, auth)}
