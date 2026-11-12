"""Discover lenses: what the vault does not say yet (`vault discover`).

Seven lenses. Every result has a one-line reason with its evidence, every cutoff is a percentile of
a vault distribution (``engine.explore.percentile``), and the same index, settings and seed give the
same output. Embeddings measure *distance* only (gap, surprise) and decide *merging* (unwritten);
they never rank anything by nearness, and those three lenses also say what they find without them.
Word-based evidence (key terms) comes from the vault's own statistics (``suggest.KeyTerms``: each
note's most distinctive terms by tf x idf over this vault); there is no word list.

hidden link (Swanson's A-B-C)
    Notes C that A has no link with (either way), reached through middles B: a note, missing note or
    tag both attach to (the map's graph), or a key term both have. Each middle weighs
    1 / ln(1 + the notes attached to it), so a rare shared middle counts more than a common one
    (Adamic-Adar); the score is the sum. Kept: two or more middles and a score at or above the
    percentile of every unlinked pair's score in the vault.
same role elsewhere
    Notes in another community that hold the same position. A note's role is ten numbers from its
    links, each turned into its percentile among the notes: notes linking to it, notes it links to,
    share of its links that go to missing notes, tags, weight inside its community as a z-score
    (Guimera & Amaral), participation (how evenly its weight spreads over communities), betweenness,
    constraint, hub and authority scores. Distance = mean absolute difference. Kept: at or below the
    (100 - percentile)th percentile of all pairwise distances.
gap
    Two communities close in meaning with few links between them. Closeness: cosine of the
    communities' mean embeddings (with the model) or of their key-term profiles (without). Links:
    the note-to-note links between them over the number the configuration model expects from their
    notes' link degrees (1 = as chance predicts). Kept: closeness at or above the percentile of all
    community pairs, and fewer links than chance; listed by fewest links for what chance expects,
    then by the most links expected: closeness admits a pair, it never orders them. The notes that
    would best join them are the unlinked pairs across with the highest hidden-link score.
surprise
    Links between notes far apart in meaning and in different communities. Kept: similarity of
    the two notes (embedding cosine, or key-term cosine without the model) at or below the
    (100 - percentile)th percentile of every linked pair's similarity, and different communities;
    listed by how rarely their two communities link (against chance), so the embedding only
    decides what counts as far.
bridge
    For one note: its betweenness and constraint among the vault's, the communities its links reach,
    and for each pair of communities the share of their notes' shortest paths that run through it.
unwritten
    Missing notes (link targets without a note) linked from the most communities. Missing notes that
    mean the same are merged first: by their name words (``suggest``'s overlapping names) without
    embeddings; with them, names that are each other's nearest neighbour by embedding at or above
    the percentile of all names' nearest-neighbour similarity. Kept: linked from at least the
    percentile of communities over all missing notes (or merged groups).
wander
    A random walk on links from the note, seeded: from each note to a note it links or that links to
    it, a note not yet visited when there is one, chosen by ``random.Random(seed)`` from the
    neighbours sorted by path. The same seed gives the same walk.
"""

from __future__ import annotations

import math
import random
from bisect import bisect_right
from collections import Counter, defaultdict, deque
from typing import Any

from vaultengine.related import Relatedness
from vaultengine.shape import Distribution, VaultShape, percentile, percentile_rank
from vaultengine.suggest import KeyTerms, Suggester
from vaultengine.vault import link_name

SOURCES = 1000  # vault distributions: exact up to this many notes, else from a seeded sample of source notes
PAIRS = 250_000  # role distances: every pair up to this many, else this many seeded random pairs (rank error ~0.1 points)
ROLE_FEATURES = ("linked from", "links to", "links to missing notes", "tags", "weight inside its community",
                 "participation", "betweenness", "constraint", "hub", "authority")


class Discover:
    def __init__(self, engine: Relatedness, shape: VaultShape, percentile_cut: float = 90.0, seed: int = 42,
                 provider: Any = None, stats: Any = None) -> None:
        """``stats``: a ``stats.Stats`` that keeps the vault-wide numbers between runs (`vault explore`)."""
        from vaultengine.connect import _linked_pairs
        from vaultengine.stats import NoStats

        self.engine, self.shape, self.sg, self.graph = engine, shape, shape.sg, engine.graph
        self.p = percentile_cut
        self.seed = seed
        self._provider = provider  # for embedding missing notes' names (unwritten), like engine.text's
        self.linked = _linked_pairs(engine.conn)
        self.notes = list(self.sg.notes)
        self.file_of = {v: self.graph.note_ids[v] for v in self.notes}
        self.node_of = {f: v for v, f in self.file_of.items()}
        self.stats = stats if stats is not None else NoStats()
        # Key terms that are words: formula pieces (o_t, 1.5b) are vault statistics too, but say nothing to a person.
        words = lambda terms: {t: w for t, w in terms.items() if t.isalpha()}
        path_of, file_of_path = engine.paths, {p: f for f, p in engine.paths.items()}
        self._key_terms_index: KeyTerms | None = None  # read from the index only when the key terms are computed

        def key_terms() -> KeyTerms:
            if self._key_terms_index is None:
                self._key_terms_index = KeyTerms(engine.conn, engine.n_notes)
            return self._key_terms_index

        by_path = self.stats.key_terms(
            lambda: {path_of[f]: words(t) for f, t in key_terms().all().items() if f in path_of},
            lambda path: words(key_terms().of(file_of_path[path])))
        self.key_terms = {file_of_path[p]: t for p, t in by_path.items() if p in file_of_path}
        self._cache: dict[str, Any] = {}

    # ----- shared pieces -------------------------------------------------------------------------

    def is_linked(self, a: int, b: int) -> bool:
        fa, fb = self.file_of[a], self.file_of[b]
        return fb in self.linked[fa] or fa in self.linked[fb]

    def name(self, node: int) -> str:
        return self.sg.name(node)

    def sample(self, items: list[int]) -> tuple[list[int], bool]:
        if len(items) <= SOURCES:
            return list(items), True
        return sorted(random.Random(self.seed).sample(items, SOURCES)), False

    @property
    def middles(self) -> tuple[dict[int, list[tuple[str, Any]]], dict[tuple[str, Any], list[int]], dict[tuple[str, Any], float]]:
        """(note -> its middles, middle -> its notes, middle -> weight). A middle is ("node", n) or ("term", t)."""
        if "middles" not in self._cache:
            self._cache["middles"] = self.stats.get("middles", self._middles)
        return self._cache["middles"]

    def _middles(self):
        of: dict[int, list[tuple[str, Any]]] = {}
        members: dict[tuple[str, Any], list[int]] = defaultdict(list)
        for v in self.notes:  # a note as a middle: its members are the notes linked with it
            mine = [("node", u) for u in sorted(self.sg.adj[v]) if u != v]
            mine += [("term", t) for t in sorted(self.key_terms.get(self.file_of[v], {}))]
            of[v] = mine
        for v, mine in of.items():
            for m in mine:
                members[m].append(v)
        weight = {m: 1 / math.log(1 + len(vs)) for m, vs in members.items() if len(vs) >= 2}
        return of, dict(members), weight

    def middle_name(self, m: tuple[str, Any]) -> str:
        kind, value = m
        if kind == "term":
            return f"'{value}'"
        return self.name(value)

    def hidden_scores(self, a: int, within: set[int] | None = None) -> dict[int, tuple[float, list[tuple[str, Any]]]]:
        """note -> (score, middles) for every unlinked note sharing a middle with ``a`` (only ``within`` if given)."""
        of, members, weight = self.middles
        scores: dict[int, float] = defaultdict(float)
        via: dict[int, list] = defaultdict(list)
        for m in of[a]:
            w = weight.get(m)
            if w is None:
                continue
            for c in members[m]:
                if c == a or (within is not None and c not in within):
                    continue
                scores[c] += w
                via[c].append(m)
        return {c: (s, sorted(via[c], key=lambda m: (-weight[m], self.middle_name(m))))
                for c, s in scores.items() if not self.is_linked(a, c)}

    # ----- hidden link ----------------------------------------------------------------------------

    def hidden_distribution(self) -> tuple[list[float], bool]:
        """Every unlinked pair's score (2+ middles): exact, or from a seeded sample of source notes as many as
        ``path_budget`` allows (each source visits the notes of each of its middles)."""
        if "hidden" not in self._cache:
            distribution, exact, sources = self.stats.get("hidden", self._hidden_distribution, tolerant=True)
            self._cache["hidden_sources"] = sources
            self._cache["hidden"] = (distribution, exact)
        return self._cache["hidden"]

    def _hidden_distribution(self) -> tuple[Any, bool, int]:
        """(every unlinked pair's score with 2+ middles, exact?, source notes) - see ``hidden_distribution``."""
        of, members, weight = self.middles
        cost = sum(len(members[m]) for v in self.notes for m in of[v] if m in weight) / max(1, len(self.notes))
        affordable = max(1, int(self.shape.path_budget // max(1.0, cost)))
        if len(self.notes) <= min(SOURCES, affordable):
            sources, exact = list(self.notes), True
        else:
            k = min(SOURCES, affordable, len(self.notes))
            sources, exact = sorted(random.Random(self.seed).sample(self.notes, k)), False
        values = []
        for a in sources:  # scores and counts only: the middles themselves are only listed for what is shown
            score: dict[int, float] = defaultdict(float)
            count: Counter[int] = Counter()
            for m in of[a]:
                w = weight.get(m)
                if w is None:
                    continue
                for c in members[m]:
                    if c != a:
                        score[c] += w
                        count[c] += 1
            for c, n in count.items():
                if n >= 2 and (not exact or a < c) and not self.is_linked(a, c):
                    values.append(score[c])
        return Distribution(values), exact, len(sources)


    def hidden_link(self, a: int, limit: int) -> dict[str, Any]:
        values, exact = self.hidden_distribution()
        cut = percentile(values, self.p)
        _, _, weight = self.middles
        found = [(score, c, via) for c, (score, via) in self.hidden_scores(a).items() if len(via) >= 2]
        found.sort(key=lambda item: (-item[0], self.sg.labels[item[1]]))
        kept = [item for item in found if item[0] >= cut]
        results = []
        for score, c, via in kept[: limit or None]:
            parts = ", ".join(f"{self.middle_name(m)} ({len(self.middles[1][m])} notes)" for m in via[:4])
            more = f" +{len(via) - 4}" if len(via) > 4 else ""
            results.append({
                "note": self.sg.labels[c], "score": score, "percentile": percentile_rank(values, score),
                "middles": [{"middle": self.middle_name(m), "kind": "key term" if m[0] == "term" else self.sg.kinds[m[1]],
                             "notes": len(self.middles[1][m]), "weight": weight[m]} for m in via],
                "reason": f"no link either way; {len(via)} middles in common: {parts}{more}; score {score:.2f}"
                          f" (p{percentile_rank(values, score):.0f} of {len(values)} unlinked pairs)",
            })
        return {"cutoff": cut, "pairs_in_vault": len(values), "exact": exact, "sources": self._cache["hidden_sources"],
                "found": len(found), "kept": len(kept), "results": results}

    # ----- same role elsewhere ---------------------------------------------------------------------

    @property
    def roles(self) -> dict[int, list[float]]:
        """note -> its ten role features as percentiles among the notes (0-1)."""
        if "roles" not in self._cache:
            self._cache["roles"], self._cache["roles_raw"] = self.stats.get("roles", self._roles)
        return self._cache["roles"]

    def _roles(self):
        """(note -> its ten role features as percentiles, note -> the raw features)."""
        shape, g = self.shape, self.graph
        membership = shape.membership
        adj = shape.community_adj
        raw: dict[int, list[float]] = {}
        inside: dict[int, float] = {}
        for v in self.notes:
            total = sum(w for u, w in adj[v].items() if u != v)
            by: Counter[int] = Counter()
            for u, w in adj[v].items():
                if u != v:
                    by[membership[u]] += w
            inside[v] = by[membership[v]]
            participation = 1 - sum((w / total) ** 2 for w in by.values()) if total else 0.0
            out = g.outgoing[v]
            missing = sum(1 for t in out if g.is_missing(t)) / len(out) if out else 0.0
            raw[v] = [len(g.incoming[v]), len(out), missing,
                      len(self.engine.note_tags.get(self.file_of[v], ())), 0.0, participation]
        groups: dict[int, list[int]] = defaultdict(list)
        for v in self.notes:
            groups[membership[v]].append(v)
        for members in groups.values():
            values = [inside[v] for v in members]
            mean = sum(values) / len(values)
            sd = math.sqrt(sum((x - mean) ** 2 for x in values) / len(values))
            for v in members:
                raw[v][4] = (inside[v] - mean) / sd if sd else 0.0
        b = shape.betweenness.values
        c = shape.constraint
        hub, auth = shape.hits
        for v in self.notes:
            raw[v] += [b.get(v, 0.0), c.get(v, 1.0), hub.get(v, 0.0), auth.get(v, 0.0)]
        columns = list(zip(*(raw[v] for v in self.notes)))
        ranked = []
        for column in columns:
            order = sorted(column)
            ranked.append({x: (bisect_right(order, x)) / len(order) for x in set(column)})
        return {v: [ranked[i][raw[v][i]] for i in range(len(ROLE_FEATURES))] for v in self.notes}, raw

    def role_matrix(self):
        """(notes in order, their role percentiles as a notes x 10 array)."""
        if "role_matrix" not in self._cache:
            import numpy as np

            self._cache["role_matrix"] = np.array([self.roles[v] for v in self.notes], dtype=np.float64)
        return self._cache["role_matrix"]

    def role_distance(self, a: int, b: int) -> float:
        ra, rb = self.roles[a], self.roles[b]
        return sum(abs(x - y) for x, y in zip(ra, rb)) / len(ra)

    def _role_distances(self, matrix) -> tuple[Any, bool]:
        """Role distances between notes: every pair up to ``PAIRS`` pairs, else a seeded sample of PAIRS pairs."""
        import numpy as np

        n = len(self.notes)
        exact = n * (n - 1) // 2 <= PAIRS
        values = []
        if exact:
            for i in range(n):  # one row of distances per note, vectorised
                values.append(np.abs(matrix[i + 1:] - matrix[i]).mean(axis=1))
        else:  # a seeded random sample of pairs: a percentile needs a sample, not all n^2 / 2 of them
            rng = np.random.default_rng(self.seed)
            x, y = rng.integers(0, n, PAIRS), rng.integers(0, n, PAIRS)
            keep = x != y
            x, y = x[keep], y[keep]
            total = np.zeros(len(x))
            for j in range(matrix.shape[1]):  # a feature at a time: never a pairs x features block in memory
                column = matrix[:, j]
                total += np.abs(column[x] - column[y])
            values.append(total / matrix.shape[1])
        return Distribution(np.concatenate(values) if values else []), exact

    def same_role(self, a: int, limit: int) -> dict[str, Any]:
        import numpy as np

        membership = self.shape.membership
        matrix = self.role_matrix()
        position = {v: i for i, v in enumerate(self.notes)}
        if "role_distances" not in self._cache:
            self._cache["role_distances"] = self.stats.get("role_distances", lambda: self._role_distances(matrix), tolerant=True)
        values, exact = self._cache["role_distances"]
        cut = percentile(values, 100 - self.p)
        mine = np.abs(matrix - matrix[position[a]]).mean(axis=1)
        found = sorted(((float(mine[position[b]]), b) for b in self.notes if b != a and membership[b] != membership[a]),
                       key=lambda item: (item[0], self.sg.labels[item[1]]))
        kept = [item for item in found if item[0] <= cut]
        roles, raw = self.roles, self._cache["roles_raw"]
        stand_out = sorted(range(len(ROLE_FEATURES)), key=lambda i: -abs(roles[a][i] - 0.5))[:3]
        results = []
        for distance, b in kept[: limit or None]:
            shown = "; ".join(f"{ROLE_FEATURES[i]} {_fmt(raw[a][i])} vs {_fmt(raw[b][i])} (p{roles[a][i] * 100:.0f} vs"
                              f" p{roles[b][i] * 100:.0f})" for i in stand_out)
            results.append({"note": self.sg.labels[b], "community": membership[b], "distance": distance,
                            "percentile": percentile_rank(values, distance),
                            "features": {ROLE_FEATURES[i]: [raw[a][i], raw[b][i]] for i in range(len(ROLE_FEATURES))},
                            "reason": f"community {membership[b]} (this note: {membership[a]}); role distance {distance:.3f}"
                                      f" (p{percentile_rank(values, distance):.0f}, lower is closer); {shown}"})
        return {"cutoff": cut, "pairs_in_vault": len(values), "exact": exact, "found": len(found), "kept": len(kept),
                "results": results}

    # ----- communities in meaning (gap) ----------------------------------------------------------------

    def _term_profile(self, notes: list[int]) -> Counter[str]:
        profile: Counter[str] = Counter()
        for v in notes:
            profile.update(self.key_terms.get(self.file_of[v], {}))
        return profile

    @staticmethod
    def _cosine(a: Counter[str], b: Counter[str]) -> float:
        dot = sum(w * b[t] for t, w in a.items() if t in b)
        norm = math.sqrt(sum(w * w for w in a.values())) * math.sqrt(sum(w * w for w in b.values()))
        return dot / norm if norm else 0.0

    def community_closeness(self, meaning: bool) -> dict[tuple[int, int], float] | None:
        """(c1, c2) -> closeness of every pair of communities; None when meaning is asked for and unavailable."""
        key = f"closeness:{meaning}"
        if key not in self._cache:
            communities = self.shape.communities
            if meaning:
                text = self.engine.text
                if not text.available():
                    self._cache[key] = None
                    return None
                import numpy as np

                centroid = {}
                for c in communities:
                    rows = [text.vectors.row_of[self.file_of[v]] for v in c.notes if self.file_of[v] in text.vectors.row_of]
                    mean = text.centered[rows].mean(axis=0) if rows else None
                    norm = float(np.linalg.norm(mean)) if mean is not None else 0.0
                    centroid[c.number] = mean / norm if norm else None
                result = {}
                for i, x in enumerate(communities):
                    for y in communities[i + 1:]:
                        cx, cy = centroid[x.number], centroid[y.number]
                        result[(x.number, y.number)] = float(cx @ cy) if cx is not None and cy is not None else 0.0
            else:  # key-term profiles as rows of one matrix: every pair's cosine in one product
                import numpy as np

                profiles = [self._term_profile(c.notes) for c in communities]
                terms = {t: i for i, t in enumerate(sorted({t for p in profiles for t in p}))}
                matrix = np.zeros((len(profiles), len(terms)), dtype=np.float64)
                for row, profile in enumerate(profiles):
                    for t, w in profile.items():
                        matrix[row, terms[t]] = w
                norms = np.linalg.norm(matrix, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                matrix /= norms
                cos = matrix @ matrix.T
                result = {(x.number, y.number): float(cos[i, j])
                          for i, x in enumerate(communities) for j, y in enumerate(communities) if j > i}
            self._cache[key] = result
        return self._cache[key]

    @property
    def link_ratio(self) -> dict[tuple[int, int], tuple[int, float, float]]:
        """(c1, c2) -> (note-to-note links between them, expected under the configuration model, observed / expected)."""
        if "ratio" not in self._cache:
            self._cache["ratio"] = self.stats.get("ratio", self._link_ratio)
        return self._cache["ratio"]

    def _link_ratio(self) -> dict[tuple[int, int], tuple[int, float, float]]:
        g, membership = self.graph, self.shape.membership
        degree: Counter[int] = Counter()
        between: Counter[tuple[int, int]] = Counter()
        total = 0.0
        for a in self.notes:
            for b, w in g.adjacency[a].items():
                if g.is_missing(b) or b == a:
                    continue
                degree[membership[a]] += w
                total += w
                ca, cb = membership[a], membership[b]
                if ca < cb:
                    between[(ca, cb)] += w
        numbers = [c.number for c in self.shape.communities]
        result = {}
        for i, x in enumerate(numbers):
            for y in numbers[i + 1:]:
                expected = degree[x] * degree[y] / total if total else 0.0
                result[(x, y)] = (int(between[(x, y)]), expected, between[(x, y)] / expected if expected else 0.0)
        return result

    def gaps(self, meaning: bool, only: int | None, limit: int, joiners: int = 3) -> dict[str, Any]:
        closeness = self.community_closeness(meaning)
        if closeness is None:
            return {"unavailable": self.engine.text.unavailable}
        values = Distribution(closeness.values())
        cut = percentile(values, self.p)
        names = {c.number: c for c in self.shape.communities}
        ratio = self.link_ratio
        # Meaning only admits a pair (the cutoff); the order is by links: fewest for what chance expects, then
        # the most links expected (0 where 4 are expected is a wider gap than 0 where 0.3 are), never by closeness.
        found = [(ratio[pair][2], -ratio[pair][1], pair) for pair, close in closeness.items()
                 if close >= cut and ratio[pair][2] < 1 and (only is None or only in pair)]
        found.sort()
        results = []
        for r, _, (x, y) in found[: limit or None]:
            close = closeness[(x, y)]
            links, expected, _ = ratio[(x, y)]
            pairs = []
            members_y = set(names[y].notes)
            for a in names[x].notes if joiners else ():  # joiners=0: the gaps alone (a hidden-link search per note)
                for b, (score, via) in self.hidden_scores(a, members_y).items():
                    if len(via) >= 2:
                        pairs.append((score, a, b, via))
            pairs.sort(key=lambda item: (-item[0], self.sg.labels[item[1]], self.sg.labels[item[2]]))
            join = [{"a": self.sg.labels[a], "b": self.sg.labels[b], "score": score,
                     "middles": [self.middle_name(m) for m in via[:4]]} for score, a, b, via in pairs[:joiners]]
            how = "mean embeddings" if meaning else "key-term profiles"
            results.append({
                "communities": [x, y], "names": [names[x].name, names[y].name], "closeness": close,
                "closeness_percentile": percentile_rank(values, close), "links": links, "expected": expected, "ratio": r,
                "join": join,
                "reason": f"closeness of their {how} {close:.2f} (p{percentile_rank(values, close):.0f} of {len(values)}"
                          f" community pairs); {links} links between them where chance predicts {expected:.1f}",
            })
        return {"cutoff": cut, "pairs": len(values), "found": len(found), "results": results}

    # ----- surprise --------------------------------------------------------------------------------------

    def note_similarity(self, meaning: bool):
        """A function (a, b) -> similarity of two notes, or None when meaning is asked for and unavailable."""
        if meaning:
            text = self.engine.text
            if not text.available():
                return None
            rows = text.vectors.row_of
            centered = text.centered

            def by_embedding(a: int, b: int) -> float | None:
                fa, fb = self.file_of[a], self.file_of[b]
                if fa not in rows or fb not in rows or not centered[rows[fa]].any() or not centered[rows[fb]].any():
                    return None
                return float(centered[rows[fa]] @ centered[rows[fb]])

            return by_embedding

        def by_terms(a: int, b: int) -> float | None:
            ta, tb = self.key_terms.get(self.file_of[a]), self.key_terms.get(self.file_of[b])
            return self._cosine(Counter(ta), Counter(tb)) if ta and tb else None

        return by_terms

    def linked_pairs(self) -> list[tuple[int, int, list[int]]]:
        """(source, target, lines) of every body link between two notes (tag links are tags)."""
        g = self.graph
        pairs = []
        for a in self.notes:
            for b in sorted(g.outgoing[a]):
                if not g.is_missing(b) and b != a:
                    pairs.append((a, b, g.lines.get((a, b), [])))
        return pairs

    def surprises(self, meaning: bool, only: int | None, limit: int, among: set[int] | None = None) -> dict[str, Any]:
        """``only``: links with this note; ``among``: links with one of these notes (a missing note's linkers)."""
        similarity = self.note_similarity(meaning)
        if similarity is None:
            return {"unavailable": self.engine.text.unavailable}
        membership = self.shape.membership
        pairs = self.linked_pairs()
        # The vault distribution (every linked pair's similarity) is kept between explore runs; the pairs asked
        # about are scored now.
        values = self.stats.get(f"surprise:{meaning}", lambda: Distribution(
            [s for a, b, _ in pairs if (s := similarity(a, b)) is not None]), tolerant=True)
        wanted = [(a, b, lines) for a, b, lines in pairs if (only is None or only in (a, b))
                  and (among is None or a in among or b in among)]
        scored = [(s, a, b, lines) for a, b, lines in wanted if (s := similarity(a, b)) is not None]
        cut = percentile(values, 100 - self.p)
        ratio = self.link_ratio
        found = [item for item in scored if item[0] <= cut and membership[item[1]] != membership[item[2]]]
        # Meaning only admits a link (the cutoff); the order is by how rarely the two communities link (fewest for
        # what chance expects first), then by path: never by the embedding.
        def rarity(item):
            ca, cb = membership[item[1]], membership[item[2]]
            between = ratio.get((min(ca, cb), max(ca, cb)))
            return (between[2] if between else math.inf, self.sg.labels[item[1]], self.sg.labels[item[2]])

        found.sort(key=rarity)
        results = []
        for s, a, b, lines in found[: limit or None]:
            ca, cb = membership[a], membership[b]
            pair = (min(ca, cb), max(ca, cb))
            between = ratio.get(pair)
            terms = lambda v: ", ".join(list(self.key_terms.get(self.file_of[v], {}))[:3]) or "—"
            how = "embedding cosine" if meaning else "key-term cosine"
            results.append({
                "a": self.sg.labels[a], "b": self.sg.labels[b], "lines": lines, "similarity": s,
                "percentile": percentile_rank(values, s), "communities": [ca, cb],
                "ratio": between[2] if between else None,
                "reason": f"{link_name(self.sg.labels[a])} links it (line {lines[0] if lines else '?'}); {how} {s:.2f}"
                          f" (p{percentile_rank(values, s):.0f} of {len(values)} linked pairs); communities {ca} and {cb}"
                          + (f" ({between[0]} links between them where chance predicts {between[1]:.1f})" if between else
                             " (one of them in no community)")
                          + f"; about: {terms(a)} / {terms(b)}",
            })
        return {"cutoff": cut, "linked_pairs": len(values), "found": len(found), "results": results}

    # ----- bridge ------------------------------------------------------------------------------------------

    def bridge(self, v: int, limit: int) -> dict[str, Any]:
        shape, sg = self.shape, self.sg
        membership = shape.membership
        neighbours = [sorted(a) for a in sg.adj]
        is_note = set(self.notes)

        def bfs(s: int) -> tuple[list[int], list[int]]:
            dist, sigma = [-1] * len(neighbours), [0] * len(neighbours)
            dist[s], sigma[s] = 0, 1
            queue = deque([s])
            while queue:
                x = queue.popleft()
                for y in neighbours[x]:
                    if dist[y] < 0:
                        dist[y] = dist[x] + 1
                        queue.append(y)
                    if dist[y] == dist[x] + 1:
                        sigma[y] += sigma[x]
            return dist, sigma

        per_search = len(sg.adj) + sg.edges
        others = [s for s in self.notes if s != v]
        if len(others) * per_search <= shape.path_budget:
            sources, exact = others, True
        else:
            k = max(1, min(len(others), shape.path_budget // per_search))
            sources, exact = sorted(random.Random(self.seed).sample(others, k)), False
        dv, sv = bfs(v)
        through: Counter[tuple[int, int]] = Counter()
        pairs: Counter[tuple[int, int]] = Counter()
        for s in sources:
            ds, ss = bfs(s)
            if ds[v] < 0:
                continue
            for t in self.notes:
                if t in (s, v) or ds[t] < 0 or (exact and t < s):  # exact: each pair once, from its smaller end
                    continue
                key = tuple(sorted((membership[s], membership[t])))
                pairs[key] += 1
                if ds[v] + dv[t] == ds[t]:
                    through[key] += ss[v] * sv[t] / ss[t]
        big = {c.number for c in shape.communities}
        shares = sorted(((through[k] / pairs[k], k) for k in pairs if through[k] and k[0] in big and k[1] in big),
                        key=lambda item: (-item[0], item[1]))
        reach: Counter[int] = Counter()
        for u, w in sg.adj[v].items():
            if u != v and membership[u] in big:
                reach[membership[u]] += w
        row = next(r for r in shape.bridges() if r["node"] == v)
        brokered = [{"communities": list(k), "share": share, "pairs": pairs[k],
                     "reason": f"{share:.0%} of the shortest paths between notes of "
                               + (f"communities {k[0]} and {k[1]}" if k[0] != k[1] else f"community {k[0]}")
                               + f" ({pairs[k]} pairs{'' if exact else ', sampled'}) run through it"}
                    for share, k in shares[: limit or None]]
        return {"betweenness": row["betweenness"], "betweenness_percentile": row["betweenness_percentile"],
                "constraint": row["constraint"], "constraint_percentile": row["constraint_percentile"],
                "community": membership[v], "reaches": [{"community": c, "weight": w} for c, w in reach.most_common()],
                "exact": exact, "sources": len(sources), "brokers": brokered}

    # ----- unwritten ---------------------------------------------------------------------------------------

    def missing_groups(self, meaning: bool) -> tuple[list[list[str]], dict[str, Any]] | None:
        """Groups of missing-note keys that mean the same (singletons included), and how they were merged."""
        key = f"groups:{meaning}"
        if key not in self._cache:
            self._cache[key] = self.stats.get(key, lambda: self._missing_groups(meaning))
        return self._cache[key]

    def _missing_groups(self, meaning: bool) -> tuple[list[list[str]], dict[str, Any]] | None:
        suggester = self._suggester
        keys = sorted(suggester.linkers)
        parent = {k: k for k in keys}

        def find(k):
            while parent[k] != k:
                parent[k] = parent[parent[k]]
                k = parent[k]
            return k

        how: dict[str, Any] = {}
        if meaning:
            text = self.engine.text
            if not text.available():
                return None
            import numpy as np

            names = [suggester.name(k).replace("-", " ").replace("_", " ") for k in keys]
            vectors = np.asarray(text.provider.embed(names), dtype=np.float64)
            used = np.abs(vectors).sum(axis=1) > 0
            centered = vectors - vectors[used].mean(axis=0)
            norms = np.linalg.norm(centered, axis=1, keepdims=True)
            norms[(norms == 0) | ~used[:, None]] = np.inf
            centered /= norms
            rows = [i for i in range(len(keys)) if used[i]]
            nearest, best = {}, {}
            matrix = centered[rows]
            for start in range(0, len(rows), 1024):
                block = matrix[start : start + 1024] @ matrix.T
                for k in range(block.shape[0]):
                    block[k, start + k] = -np.inf
                for k in range(block.shape[0]):
                    nearest[rows[start + k]] = rows[int(block[k].argmax())]
                    best[rows[start + k]] = float(block[k].max())
            cut = percentile(list(best.values()), self.p)
            how = {"by": "embedding of the names", "cutoff": cut, "pairs": []}
            for i, j in nearest.items():
                if i < j and nearest.get(j) == i and best[i] >= cut:
                    parent[find(keys[i])] = find(keys[j])
                    how["pairs"].append((keys[i], keys[j], best[i]))
        else:
            how = {"by": "name words"}
            for group in suggester.same_idea():
                first = group[0][0]
                for name, _ in group[1:]:
                    parent[find(self._key_of_name[name])] = find(self._key_of_name[first])
        groups: dict[str, list[str]] = defaultdict(list)
        for k in keys:
            groups[find(k)].append(k)
        return sorted(groups.values(), key=lambda g: g[0]), how

    @property
    def _suggester(self) -> Suggester:
        if "suggester" not in self._cache:
            s = Suggester(self.engine.conn, self.engine)
            self._cache["suggester"] = s
            self._key_of_name = {s.name(k): k for k in s.linkers}
        return self._cache["suggester"]

    def unwritten(self, meaning: bool, only: int | None, limit: int, among: set[int] | None = None) -> dict[str, Any]:
        """``only``: missing notes this note links; ``among``: missing notes one of these notes links."""
        grouped = self.missing_groups(meaning)
        if grouped is None:
            return {"unavailable": self.engine.text.unavailable}
        groups, how = grouped
        suggester = self._suggester
        membership = self.shape.membership
        scored = []
        for group in groups:
            linkers = set().union(*(suggester.linkers[k] for k in group))
            communities = Counter(membership[self.node_of[f]] for f in linkers if f in self.node_of)
            scored.append((len(communities), len(linkers), group, communities, linkers))
        values = [s[0] for s in scored]
        cut = percentile(values, self.p)
        only_file = self.file_of[only] if only is not None else None
        among_files = {self.file_of[v] for v in among} if among is not None else None
        found = [s for s in scored if s[0] >= cut and s[0] >= 2 and (only_file is None or only_file in s[4])
                 and (among_files is None or among_files & s[4])]
        found.sort(key=lambda s: (-s[0], -s[1], suggester.name(s[2][0])))
        pairs = {(a, b): c for a, b, c in how.get("pairs", [])}
        results = []
        for spread, n, group, communities, _ in found[: limit or None]:
            names = [suggester.name(k) for k in sorted(group, key=lambda k: (-len(suggester.linkers[k]), k))]
            spread_text = ", ".join(f"{c}: {k}" for c, k in sorted(communities.items(), key=lambda x: (-x[1], x[0])))
            merged = ""
            if len(group) > 1:
                if how["by"] == "name words":
                    merged = f"; merged by name words: {', '.join(f'[[{x}]]' for x in names)}"
                else:
                    sims = [f"{c:.2f}" for (a, b), c in pairs.items() if a in group and b in group]
                    merged = f"; merged by embedding ({', '.join(sims)}): {', '.join(f'[[{x}]]' for x in names)}"
            results.append({
                "names": names, "linking_notes": n, "communities": dict(communities), "spread": spread,
                "reason": f"{n} notes in {spread} communities link it ({spread_text}){merged}",
            })
        return {"cutoff": cut, "groups": len(values), "found": len(found), "by": how["by"],
                "merge_cutoff": how.get("cutoff"), "merged": sum(1 for g in groups if len(g) > 1), "results": results}

    # ----- wander --------------------------------------------------------------------------------------------

    def wander(self, start: int, steps: int, seed: int | None = None) -> dict[str, Any]:
        g, membership = self.graph, self.shape.membership
        rng = random.Random(self.seed if seed is None else seed)
        walk = [start]
        visited = {start}
        hops = []
        current = start
        for _ in range(steps):
            around = sorted((u for u in g.adjacency[current] if not g.is_missing(u) and u != current),
                            key=lambda u: g.labels[u])
            if not around:
                break
            fresh = [u for u in around if u not in visited]
            nxt = rng.choice(fresh or around)
            out, back = g.lines.get((current, nxt), []), g.lines.get((nxt, current), [])

            def kind(a, b, line):
                return "a tag link, " if (a, b, line) in g.tag_lines else ""

            how = (f"it links to {link_name(g.labels[nxt])} ({kind(current, nxt, out[0])}line {out[0]})" if out else
                   f"{link_name(g.labels[nxt])} links to it ({kind(nxt, current, back[0])}line {back[0]})")
            change = membership[nxt] != membership[current]
            hops.append({"from": g.labels[current], "to": g.labels[nxt], "lines_out": out, "lines_back": back,
                         "community": membership[nxt], "new_community": change, "revisit": nxt in visited,
                         "reason": how + (f"; into community {membership[nxt]}" if change else "")
                         + ("; every neighbour already visited" if not fresh else "")})
            walk.append(nxt)
            visited.add(nxt)
            current = nxt
        return {"seed": self.seed if seed is None else seed, "steps": len(hops), "walk": [g.labels[v] for v in walk],
                "hops": hops, "communities": len({membership[v] for v in walk})}


def _fmt(x: float) -> str:
    return f"{x:.0f}" if float(x).is_integer() else f"{x:.2f}"


