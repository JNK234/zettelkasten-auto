"""Time the vault-shape algorithms (communities, betweenness, constraint, HITS, lens distributions) at scale.

    python scripts/bench_shape.py vault N DIR    write a generated vault of N notes to DIR, index it, time `vault map`'s parts
    python scripts/bench_shape.py graph N        time the algorithms on an in-memory graph of the same shape (no files)
    python scripts/bench_shape.py discover N DIR write a generated vault with words, index it, time `vault discover`'s lenses

The generated vault is seeded (the same every run): N // 60 topics; each note is tagged with its topic on a
`Tags:` line (a fifth with a second topic), links 4 to 8 notes (four in five in its own topic), links a missing
note one time in ten, and carries a #tag from a pool of N // 120 (seven in ten aligned with its topic).
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def generate(n: int, seed: int = 7):
    """[(name, topics, links, missing, hashtag)] for n notes."""
    rng = random.Random(seed)
    topics = max(2, n // 60)
    tags = max(2, n // 120)
    by_topic = [list(range(t, n, topics)) for t in range(topics)]
    notes = []
    for i in range(n):
        t = i % topics
        mine = [t] + ([rng.randrange(topics)] if rng.random() < 0.2 else [])
        links = set()
        for _ in range(rng.randint(4, 8)):
            pool = by_topic[t] if rng.random() < 0.8 else range(n)
            j = rng.choice(pool)
            if j != i:
                links.add(j)
        missing = f"concept-{rng.randrange(n // 10 + 1)}" if rng.random() < 0.1 else None
        tag = f"tag-{t % tags if rng.random() < 0.7 else rng.randrange(tags)}"
        notes.append((f"n{i:05d}", mine, sorted(links), missing, tag))
    return notes


def write_vault(n: int, folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for name, topics, links, missing, tag in generate(n):
        body = " ".join(f"[[n{j:05d}]]" for j in links) + (f" [[{missing}]]" if missing else "")
        text = "Tags: " + " ".join(f"[[topic-{t}]]" for t in topics) + f"\n\n# {name}\nSee {body}. #{tag}\n"
        (folder / f"{name}.md").write_text(text, encoding="utf-8")


def word_pools(topics: int, seed: int = 7) -> tuple[list[list[str]], list[str]]:
    """Words from the embedding model's own vocabulary: 40 per topic and 400 shared, seeded."""
    import json

    vocab = json.loads((Path(__file__).resolve().parent.parent / "models/potion-base-8M/tokenizer.json").read_text())["model"]["vocab"]
    words = sorted(w for w in vocab if w.isalpha() and w.isascii() and len(w) >= 4)
    rng = random.Random(seed)
    rng.shuffle(words)
    shared, rest = words[:400], words[400:]
    # Past len(rest) / 40 topics the pools wrap around: topics far apart share words.
    return [[rest[(t * 40 + k) % len(rest)] for k in range(40)] for t in range(topics)], shared


def write_discover_vault(n: int, folder: Path) -> None:
    """Like write_vault, with a body of 25 words (three in five from the note's topic) and missing notes named by words."""
    folder.mkdir(parents=True, exist_ok=True)
    topics = max(2, n // 60)
    pools, shared = word_pools(topics)
    rng = random.Random(11)
    for name, mine, links, missing, tag in generate(n):
        pool = pools[mine[0]]
        words = [rng.choice(pool) if rng.random() < 0.6 else rng.choice(shared) for _ in range(25)]
        if missing:
            missing = f"{rng.choice(pool)}-{rng.choice(pool)}"
        body = " ".join(f"[[n{j:05d}]]" for j in links) + (f" [[{missing}]]" if missing else "")
        text = ("Tags: " + " ".join(f"[[topic-{t}]]" for t in mine) + f"\n\n# {name}\n{' '.join(words)}.\nSee {body}. #{tag}\n")
        (folder / f"{name}.md").write_text(text, encoding="utf-8")


def bench_discover(n: int, folder: Path) -> None:
    from vaultengine.discover import Discover
    from vaultengine.embedding import load_provider
    from vaultengine.index import VaultIndex, default_db_path
    from vaultengine.related import Relatedness
    from vaultengine.shape import VaultShape

    root = Path(__file__).resolve().parent.parent
    if not folder.exists():
        timed(f"write {n} notes with words", lambda: write_discover_vault(n, folder))
    ix = VaultIndex(default_db_path(folder))
    timed("vault index", lambda: ix.update(folder))
    provider = load_provider("model2vec", root / "models/potion-base-8M")
    engine = Relatedness(ix.conn, provider, folder)
    timed("embed every note in memory (nothing cached)", lambda: engine.text.vectors)
    shape = VaultShape(ix.conn, graph=engine.graph)
    timed("communities", lambda: shape.communities)
    meaning = VaultShape(ix.conn, graph=engine.graph, meaning=engine.text)
    timed("map --meaning: mutual nearest neighbours", lambda: meaning.meaning_edges)
    timed("map --meaning: communities again", lambda: meaning.communities)
    d = timed("discover: key terms of every note", lambda: Discover(engine, shape))
    note = shape.sg.notes[len(shape.sg.notes) // 2]
    timed("hidden link: middles", lambda: d.middles)
    timed("hidden link: vault distribution", lambda: d.hidden_distribution())
    timed("hidden link: one note", lambda: d.hidden_link(note, 3))
    timed("same role: betweenness (budget)", lambda: shape.betweenness)
    timed("same role: features + distribution", lambda: d.same_role(note, 3))
    timed("bridge: one note", lambda: d.bridge(note, 3))
    timed("gap: with embeddings", lambda: d.gaps(True, None, 3))
    timed("gap: without", lambda: d.gaps(False, None, 3))
    timed("surprise: with embeddings", lambda: d.surprises(True, None, 3))
    timed("surprise: without", lambda: d.surprises(False, None, 3))
    timed("unwritten: with embeddings (names)", lambda: d.unwritten(True, None, 3))
    timed("unwritten: without (name words)", lambda: d.unwritten(False, None, 3))
    timed("wander: 20 steps", lambda: d.wander(note, 20))
    ix.close()


def timed(label: str, fn):
    started = time.perf_counter()
    value = fn()
    print(f"  {label:<44} {time.perf_counter() - started:8.2f} s", flush=True)
    return value


def bench_vault(n: int, folder: Path) -> None:
    from vaultengine.index import VaultIndex, default_db_path
    from vaultengine.lenses import Lenses
    from vaultengine.related import Relatedness
    from vaultengine.shape import VaultShape

    if not folder.exists():
        timed(f"write {n} notes", lambda: write_vault(n, folder))
    ix = VaultIndex(default_db_path(folder))
    report = timed("vault index (incremental; full on first run)", lambda: ix.update(folder))
    print(f"    {report.notes} notes, {len(report.parsed)} parsed")
    engine = timed("build the graph (graphrank.build_graph)", lambda: Relatedness(ix.conn, lambda: None, folder))
    shape = timed("shape graph (+ #tag nodes)", lambda: VaultShape(ix.conn, graph=engine.graph))
    sg = shape.sg
    print(f"    {len(sg.notes)} notes, {len(sg.adj)} nodes, {sg.edges} edges")
    timed("communities (Louvain + connectivity split)", lambda: shape.communities)
    print(f"    {len(shape.communities)} communities, modularity {shape.modularity:.3f}")
    per_search = len(sg.adj) + sg.edges
    for budget in (per_search * 200, per_search * 1000):
        s = VaultShape(ix.conn, graph=engine.graph, path_budget=budget)
        timed(f"betweenness, {budget // per_search} sampled sources", lambda: s.betweenness)
    exact = VaultShape(ix.conn, graph=engine.graph, path_budget=10**12)
    if n <= 3000:
        timed(f"betweenness, exact ({len(sg.notes)} sources)", lambda: exact.betweenness)
        approx = VaultShape(ix.conn, graph=engine.graph, path_budget=per_search * 200).betweenness.values
        top = lambda values: [v for v, _ in sorted(values.items(), key=lambda x: -x[1])[:20]]
        overlap = len(set(top(exact.betweenness.values)) & set(top(approx)))
        print(f"    top 20 bridges: {overlap} of 20 the same with 200 sampled sources")
    timed("Burt's constraint", lambda: shape.constraint)
    timed("HITS", lambda: shape.hits)
    lenses = Lenses(engine, shape)
    timed("co-citation distribution (lens cutoffs)", lambda: lenses.distribution("cocitation"))
    timed("coupling distribution (lens cutoffs)", lambda: lenses.distribution("coupling"))
    node = sg.notes[0]
    timed("backbone of one note's community", lambda: lenses.backbone(node))
    ix.close()


def bench_graph(n: int) -> None:
    from vaultengine.graphrank import Graph
    from vaultengine.shape import ShapeGraph, betweenness, constraint, hits, louvain

    notes = timed(f"generate {n} notes", lambda: generate(n))
    started = time.perf_counter()
    index = {}
    labels, kinds, adj = [], [], []

    def node(label, kind):
        if label not in index:
            index[label] = len(labels)
            labels.append(label)
            kinds.append(kind)
            adj.append({})
        return index[label]

    graph = Graph([], [], [])
    for name, *_ in notes:
        node(name, "note")
    outgoing = [dict() for _ in range(len(notes))]
    for name, topics, links, missing, tag in notes:
        a = index[name]
        targets = [(f"n{j:05d}", "note") for j in links] + ([(missing, "missing")] if missing else [])
        targets += [(f"topic-{t}", "missing") for t in topics] + [(f"#{tag}", "tag")]
        for label, kind in targets:
            b = node(label, kind)
            adj[a][b] = adj[a].get(b, 0.0) + 1.0
            adj[b][a] = adj[b].get(a, 0.0) + 1.0
            if kind == "note":
                outgoing[a][b] = 1
    graph.labels, graph.note_ids = labels, [i if k == "note" else None for i, k in enumerate(kinds)]
    graph.adjacency = adj
    graph.outgoing = outgoing + [{} for _ in range(len(labels) - len(notes))]
    graph.incoming = [{} for _ in labels]
    for a, targets in enumerate(graph.outgoing):
        for b in targets:
            graph.incoming[b][a] = 1
    sg = ShapeGraph(graph, labels, kinds, adj, [i for i, k in enumerate(kinds) if k == "note"])
    print(f"  {'build the graph':<44} {time.perf_counter() - started:8.2f} s")
    print(f"    {len(sg.notes)} notes, {len(adj)} nodes, {sg.edges} edges")
    membership = timed("Louvain", lambda: louvain(adj))
    print(f"    {len(set(membership))} communities")
    per_search = len(adj) + sg.edges
    for k in (50, 200):
        timed(f"betweenness, {k} sampled sources", lambda: betweenness(sg, per_search * k))
    timed("Burt's constraint", lambda: constraint(sg))
    timed("HITS", lambda: hits(graph))


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "vault":
        bench_vault(int(sys.argv[2]), Path(sys.argv[3]))
    elif len(sys.argv) >= 4 and sys.argv[1] == "discover":
        bench_discover(int(sys.argv[2]), Path(sys.argv[3]))
    elif len(sys.argv) >= 3 and sys.argv[1] == "graph":
        bench_graph(int(sys.argv[2]))
    else:
        print(__doc__)
        sys.exit(2)
