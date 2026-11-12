"""`vault experiment`: compare the options on the vault's own notes, seeded, and let the results pick defaults.

Nothing is labelled by hand. Three questions:

relatedness
    Which signals rank related notes (``related.py``: graph, cocitation, coupling, tags, text) and with which
    weights. Scored by recovering hidden links: a seeded share of the linked note pairs is hidden (``--hide``,
    15%); for each hidden pair every link between the two notes goes (both ways, tag links too), and with it
    everything derived from it: the graph, co-citation, coupling and tags are rebuilt from the links that
    remain, and the links' own text ("[[B]]", its display text, a Tags: entry) is blanked in both notes
    before they are embedded. From the note that wrote the link, every note it is not still linked with is
    ranked by the option, with the same functions ``related`` uses; the hidden partner's rank gives recall at
    5 and 10 (is it in the top k) and MRR (the mean of 1 / rank). Ties count at their expected rank.
communities
    Louvain with and without meaning edges (neighbour counts 1, 2, 3 and cutoffs at the 25th, 50th and 75th
    percentile), at resolutions 0.5, 1 and 1.5. Scored against the vault's own structure, each held out of
    the graph it is compared with: the Tags: topics (communities found without tag links and #tags; each
    note labelled by its most widely used tag, tags on one note left out) and the topic pages (communities found without the pages' links; each
    note labelled by the page that links it). NMI and ARI, notes in no community (stranded, on the map's
    full graph), stability (ARI between runs with different seeds) and time.
text
    What a note is embedded as (``embedding/text.py``: whole, no_template, first_section), scored by the
    hidden links again with text alone, and with today's related mix. A diagnostic says how much of each
    note's five nearest neighbours share its note type (the format's type, else its folder), against the
    share by chance: a high ratio means neighbours by template, not by idea.

Seeds: ``--seeds`` runs (hidden links and Louvain), each seed fixed, so the report is the same every run. A
winner is *clear* over another option when it is ahead on every seed and the mean of the per-seed
differences is more than twice its standard error; otherwise the difference is within the noise between
seeds. Datasets: the whole vault, and the vault without each ``--without`` prefix.
"""

from __future__ import annotations

import math
import random
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from vaultengine.graphrank import build_graph_from, link_rows
from vaultengine.related import combine, link_signals, note_tag_sets, tag_similarity

HIDE = 0.15
SEEDS = 5
QUERIES = 200  # hidden links scored per seed at most (a seeded sample beyond)
RESOLUTIONS = (0.5, 1.0, 1.5)
NEIGHBOURS = (1, 2, 3)
MEANING_PERCENTILES = (25.0, 50.0, 75.0)
TEXT_VARIANTS = ("whole", "no_template", "first_section")


# ----- statistics ----------------------------------------------------------------------------------


def nmi(a: list, b: list) -> float:
    """Normalised mutual information of two labelings (arithmetic-mean normalisation); 1 when both are one block."""
    n = len(a)
    if n == 0:
        return 0.0
    ca, cb, joint = Counter(a), Counter(b), Counter(zip(a, b))
    mi = sum(c / n * math.log(c * n / (ca[x] * cb[y])) for (x, y), c in joint.items())
    ha = -sum(c / n * math.log(c / n) for c in ca.values())
    hb = -sum(c / n * math.log(c / n) for c in cb.values())
    if ha == 0 and hb == 0:
        return 1.0
    return mi / ((ha + hb) / 2) if ha + hb else 0.0


def ari(a: list, b: list) -> float:
    """Adjusted Rand index of two labelings (Hubert & Arabie)."""
    n = len(a)
    if n < 2:
        return 0.0
    pairs = lambda x: x * (x - 1) / 2
    joint = sum(pairs(c) for c in Counter(zip(a, b)).values())
    sa = sum(pairs(c) for c in Counter(a).values())
    sb = sum(pairs(c) for c in Counter(b).values())
    expected = sa * sb / pairs(n)
    top = (sa + sb) / 2
    return (joint - expected) / (top - expected) if top != expected else 1.0


@dataclass
class Comparison:
    """How option ``a`` compares with ``b`` over the same seeds (higher is better)."""

    mean: float  # mean per-seed difference a - b
    se: float  # its standard error
    ahead: int  # seeds where a is ahead
    seeds: int

    @property
    def clear(self) -> bool:
        return self.ahead == self.seeds and self.seeds > 1 and self.mean > 2 * self.se

    def words(self) -> str:
        verdict = "bigger than the seed-to-seed noise" if self.clear else "within the seed-to-seed noise"
        return f"{self.mean:+.3f} (± {self.se:.3f}), ahead in {self.ahead} of {self.seeds} seeds: {verdict}"


def compare(a: list[float], b: list[float]) -> Comparison:
    diffs = [x - y for x, y in zip(a, b)]
    se = statistics.stdev(diffs) / math.sqrt(len(diffs)) if len(diffs) > 1 else 0.0
    return Comparison(statistics.fmean(diffs) if diffs else 0.0, se, sum(d > 0 for d in diffs), len(diffs))


def mean_sd(values: list[float]) -> tuple[float, float]:
    return (statistics.fmean(values), statistics.stdev(values) if len(values) > 1 else 0.0) if values else (0.0, 0.0)


# ----- the data -------------------------------------------------------------------------------------


@dataclass
class Dataset:
    name: str
    root: Path
    notes: list[tuple[int, str]]  # (file id, path), path order
    rows: list[tuple]  # graphrank.link_rows, links into left-out notes turned into links to missing notes
    hashtags: list[tuple[int, str]]
    spans: dict[frozenset, list[tuple[int, int, int]]]  # linked pair -> (file id of the linking note, start, end)
    writer: dict[frozenset, int]  # linked pair -> the note that wrote a link (the hidden link's query note)
    left_out: int = 0
    path_of: dict[int, str] = field(default_factory=dict)

    @property
    def pairs(self) -> list[frozenset]:
        return sorted(self.spans, key=lambda p: tuple(sorted(self.path_of[f] for f in p)))


def load_dataset(conn, root: str | Path, name: str, without: list[str] = ()) -> Dataset:
    def left_out(path: str) -> bool:
        return any(path == w.strip("/") or path.startswith(w.strip("/") + "/") for w in without)

    everything = conn.execute("SELECT id, path FROM files WHERE kind = 'note' ORDER BY path").fetchall()
    notes = [(f, p) for f, p in everything if not left_out(p)]
    keep = {f for f, _ in notes}
    rows = []
    for row in link_rows(conn):
        source_id, target_id, status, target_key, target, kind, line, path, fs, fk = row
        if source_id not in keep:
            continue
        if status == "resolved" and kind == "note" and target_id not in keep:
            row = (source_id, None, "unresolved", target_key, target, None, line, path, fs, fk)
        rows.append(row)
    hashtags = [(f, k) for f, k in conn.execute("SELECT file_id, name_key FROM tags") if f in keep]
    spans: dict[frozenset, list[tuple[int, int, int]]] = defaultdict(list)
    writer: dict[frozenset, int] = {}
    path_of = dict(notes)
    for s, t, start, end in conn.execute(
        """SELECT l.source_id, l.target_id, l.start_offset, l.end_offset FROM links l JOIN files f ON f.id = l.target_id
           WHERE l.status = 'resolved' AND f.kind = 'note' AND l.source_id != l.target_id ORDER BY l.source_id, l.ordinal"""
    ):
        if s in keep and t in keep:
            pair = frozenset((s, t))
            spans[pair].append((s, start, end))
            if pair not in writer or path_of[s] < path_of[writer[pair]]:
                writer[pair] = s
    return Dataset(name, Path(root), notes, rows, hashtags, dict(spans), writer, len(everything) - len(notes), path_of)


class Texts:
    """What each note is embedded as, per variant, with given link spans blanked; vectors cached by text."""

    def __init__(self, data: Dataset, provider, template: Callable[[str], Callable[[str], bool]]) -> None:
        self.data, self.provider, self.template = data, provider, template
        self._raw: dict[int, str] = {}
        self._texts: dict[tuple[int, str], str] = {}  # (file id, variant) -> text, nothing blanked
        self._vectors: dict[tuple[str, str], Any] = {}

    def raw(self, file_id: int) -> str:
        if file_id not in self._raw:
            from vaultengine.parser import normalize_text

            try:
                data = (self.data.root / self.data.path_of[file_id]).read_bytes()
            except OSError:
                data = b""
            self._raw[file_id] = normalize_text(data.decode("utf-8", errors="replace"))
        return self._raw[file_id]

    def text(self, file_id: int, variant: str, blank: list[tuple[int, int]] = ()) -> str:
        from vaultengine.embedding.text import embedding_text
        from vaultengine.parser import parse_note_with_layout

        if not blank and (file_id, variant) in self._texts:
            return self._texts[(file_id, variant)]
        raw = self.raw(file_id)
        if blank:  # the held-out links' own text: gone before anything is derived from the note
            chars = list(raw)
            for start, end in blank:
                chars[start:end] = [c if c == "\n" else " " for c in chars[start:end]]
            raw = "".join(chars)
        path = self.data.path_of[file_id]
        note, layout = parse_note_with_layout(raw, path=path)
        text = embedding_text(path, note, layout, variant, self.template(path))
        if not blank:
            self._texts[(file_id, variant)] = text
        return text

    def matrix(self, variant: str, blanked: dict[int, list[tuple[int, int]]]):
        """(file ids, centered unit rows) for every note of the dataset."""
        import numpy as np

        ids = [f for f, _ in self.data.notes]
        texts = [self.text(f, variant, blanked.get(f, ())) for f in ids]
        todo = [t for t in dict.fromkeys(texts) if (variant, t) not in self._vectors]
        for start in range(0, len(todo), 256):
            batch = todo[start:start + 256]
            for t, v in zip(batch, self.provider.embed(batch)):
                self._vectors[(variant, t)] = v
        m = np.array([self._vectors[(variant, t)] for t in texts], dtype=np.float64)
        used = np.abs(m).sum(axis=1) > 0
        centered = m - (m[used].mean(axis=0) if used.any() else 0.0)
        norms = np.linalg.norm(centered, axis=1, keepdims=True)
        norms[(norms == 0) | ~used[:, None]] = np.inf
        return ids, centered / norms


def template_rule(conn, data: Dataset, spec=None) -> Callable[[str], Callable[[str], bool]]:
    """path -> (heading text -> is it a template heading of that note): the headings the format requires for
    the note's type, else a heading written the same way in 3 or more of the dataset's notes (the rule of
    ``suggest.KeyTerms``)."""
    from vaultengine.vault import lookup_key

    if spec is not None:
        def by_type(path: str) -> Callable[[str], bool]:
            t = spec.type_for(path)
            patterns = t.headings if t else ()
            return lambda text: any(p.matches(text) for p in patterns)

        return by_type
    keep = set(data.path_of)
    written: dict[str, set[int]] = defaultdict(set)
    for file_id, text in conn.execute("SELECT file_id, text FROM headings"):
        if file_id in keep:
            written[lookup_key(text)].add(file_id)
    common = frozenset(k for k, notes in written.items() if len(notes) >= 3)
    return lambda path: (lambda text: lookup_key(text) in common)


# ----- relatedness and text: hidden links ----------------------------------------------------------------


@dataclass
class Option:
    name: str
    weights: dict[str, float]
    text: str = "whole"  # the text variant for the text signal
    question: str = "relatedness"


def relatedness_options(current: dict[str, float], variant: str = "whole") -> list[Option]:
    """The mixes compared: each signal set asked about, today's related, and a few weightings of all five (round
    numbers, so a winner can go into the config as it is)."""
    options = [
        Option("links only", {"graph": 1.0}),
        Option("links + co-citation + coupling", {"graph": 0.5, "cocitation": 0.25, "coupling": 0.25}),
        Option("links + co-citation + coupling + tags", {"graph": 0.4, "cocitation": 0.2, "coupling": 0.2, "tags": 0.2}),
        Option("text only", {"text": 1.0}),
        Option("tags + text", {"tags": 0.3, "text": 0.7}),
        Option("today's related (" + ", ".join(f"{k} {v:g}" for k, v in current.items() if v) + ")", dict(current)),
        Option("links 0.3, tags 0.2, text 0.5", {"graph": 0.3, "tags": 0.2, "text": 0.5}),
        Option("all five", {"graph": 0.3, "cocitation": 0.1, "coupling": 0.1, "tags": 0.2, "text": 0.3}),
        Option("all five, text 0.5", {"graph": 0.2, "cocitation": 0.1, "coupling": 0.1, "tags": 0.1, "text": 0.5}),
        Option("all five, text 0.7", {"graph": 0.1, "cocitation": 0.05, "coupling": 0.05, "tags": 0.1, "text": 0.7}),
        Option("links 0.1, tags 0.1, text 0.8", {"graph": 0.1, "tags": 0.1, "text": 0.8}),
        Option("all five, tags 0.3", {"graph": 0.2, "cocitation": 0.1, "coupling": 0.1, "tags": 0.3, "text": 0.3}),
    ]
    for o in options:
        o.text = variant
    return options


def text_options(current: dict[str, float]) -> list[Option]:
    out = []
    for v in TEXT_VARIANTS:
        out.append(Option(f"text only · {v}", {"text": 1.0}, v, "text"))
        out.append(Option(f"today's related · {v}", dict(current), v, "text"))
    return out


def expected_rank(scores: dict[int, float], target: int, candidates: list[int]) -> float:
    """1 + the candidates above the target + half of those tied with it (the rank under random tie-breaking)."""
    mine = scores.get(target, 0.0)
    above = sum(1 for c in candidates if c != target and scores.get(c, 0.0) > mine)
    tied = sum(1 for c in candidates if c != target and scores.get(c, 0.0) == mine)
    return 1 + above + tied / 2


@dataclass
class LinkResult:
    option: Option
    mrr: list[float] = field(default_factory=list)  # per seed
    r5: list[float] = field(default_factory=list)
    r10: list[float] = field(default_factory=list)
    seconds: float = 0.0
    queries: int = 0


def hide_pairs(data: Dataset, seed: int, hide: float = HIDE) -> set[frozenset]:
    """The linked note pairs a seed hides: a seeded sample of ``hide`` of them."""
    pairs = data.pairs
    return set(random.Random(seed).sample(pairs, max(1, round(hide * len(pairs))))) if pairs else set()


@dataclass
class HeldOut:
    """The vault as it would be had the hidden links never been written: everything the signals use."""

    graph: Any
    note_tags: dict[int, set[str]]
    tag_idf: dict[str, float]
    blanked: dict[int, list[tuple[int, int]]]  # file id -> the hidden links' spans, blanked in its text
    linked: dict[int, set[int]] = field(default_factory=dict)  # file id -> the notes it is still linked with


def held_out(data: Dataset, hidden: set[frozenset], tag_rule) -> HeldOut:
    """Every link between the notes of a hidden pair goes, both ways, tag links too; the graph, co-citation,
    coupling and tags are rebuilt from the rest, and the links' own text is blanked for embedding."""
    rows = [r for r in data.rows if not (r[2] == "resolved" and r[1] is not None and frozenset((r[0], r[1])) in hidden)]
    graph = build_graph_from(data.notes, rows, tag_rule)
    note_tags, tag_idf, _ = note_tag_sets(data.hashtags, graph, len(data.notes))
    blanked: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for pair in hidden:
        for f, start, end in data.spans[pair]:
            blanked[f].append((start, end))
    linked: dict[int, set[int]] = defaultdict(set)
    for x, y in graph.direct:
        linked[graph.note_ids[x]].add(graph.note_ids[y])
        linked[graph.note_ids[y]].add(graph.note_ids[x])
    return HeldOut(graph, note_tags, tag_idf, dict(blanked), dict(linked))


def query_signals(data: Dataset, held: HeldOut, query: int, want: set[str],
                  vectors: dict[str, Any] | None = None, timings: Counter | None = None
                  ) -> tuple[dict[str, dict[int, float]], list[int]]:
    """(signal -> {file id: raw value}, the candidates) from the note ``query``: the link signals in ``want``,
    tags, and text per variant in ``vectors`` (variant -> (ids, centered unit rows)) under ``text:<variant>``.
    The candidates are the notes it is not linked with once the hidden links are gone. ``timings``: seconds per
    signal are added to it."""
    import numpy as np

    timings = timings if timings is not None else Counter()
    clock = time.perf_counter

    graph = held.graph
    node = graph.by_file[query]
    linked = held.linked.get(query, set())
    candidates = [f for f, _ in data.notes if f != query and f not in linked]
    raw: dict[str, dict[int, float]] = {}
    for name in sorted(want & {"graph", "cocitation", "coupling"}):
        started = clock()
        raw.update(link_signals(graph, node, {name}))
        timings[name] += clock() - started
    started = clock()
    mine = held.note_tags.get(query, set())
    raw["tags"] = {f: v for f, tags in held.note_tags.items() if f != query
                   for v in [tag_similarity(mine, tags, held.tag_idf)] if v > 0}
    timings["tags"] += clock() - started
    for variant, got in (vectors or {}).items():
        started = clock()
        if got is not None:
            ids, m = got
            values = m @ m[ids.index(query)]
            raw[f"text:{variant}"] = {f: float(v) for f, v in zip(ids, values) if f != query and v > 0 and np.isfinite(v)}
        timings[f"text:{variant}"] += clock() - started
    return raw, candidates


def hidden_link_runs(data: Dataset, options: list[Option], texts: Texts | None, tag_rule, seeds: list[int],
                     hide: float = HIDE, queries: int = QUERIES, progress: Callable[[str], None] = lambda s: None) -> dict:
    """Score every option on every seed's hidden links. Returns {"results": {option name: LinkResult}, "hidden":
    links hidden per seed, "pairs": linked pairs}."""
    results = {o.name: LinkResult(o) for o in options}
    signal_time: Counter[str] = Counter()
    hidden_counts = []
    want = {k for o in options for k, w in o.weights.items() if w}
    for seed in seeds:
        hidden = hide_pairs(data, seed, hide)
        hidden_counts.append(len(hidden))
        started = time.perf_counter()
        held = held_out(data, hidden, tag_rule)
        signal_time["held out"] += time.perf_counter() - started
        vectors = {}
        for variant in sorted({o.text for o in options if o.weights.get("text")}):
            started = time.perf_counter()
            vectors[variant] = texts.matrix(variant, held.blanked) if texts is not None else None
            signal_time[f"text:{variant}"] += time.perf_counter() - started
        ordered = sorted(hidden, key=lambda p: tuple(sorted(data.path_of[f] for f in p)))
        chosen = ordered if len(ordered) <= queries else sorted(random.Random(seed + 1).sample(ordered, queries),
                                                                key=ordered.index)
        ranks: dict[str, list[float]] = defaultdict(list)
        for pair in chosen:
            a = data.writer[pair]
            b = next(iter(pair - {a}))
            raw, candidates = query_signals(data, held, a, want, vectors, signal_time)
            for o in options:
                started = time.perf_counter()
                signals = {k: raw.get(k, {}) for k in ("graph", "cocitation", "coupling", "tags")}
                signals["text"] = raw.get(f"text:{o.text}", {})
                scores = combine(signals, o.weights)
                ranks[o.name].append(expected_rank(scores, b, candidates))
                results[o.name].seconds += time.perf_counter() - started
        for o in options:
            rs = ranks[o.name]
            res = results[o.name]
            res.mrr.append(statistics.fmean(1 / r for r in rs) if rs else 0.0)
            res.r5.append(statistics.fmean(r <= 5 for r in rs) if rs else 0.0)
            res.r10.append(statistics.fmean(r <= 10 for r in rs) if rs else 0.0)
            res.queries += len(rs)
        progress(f"{data.name}: seed {seed}: {len(hidden)} links hidden, {len(chosen)} scored")
    for o in options:  # the held-out graph split evenly; each signal's cost among the options that use it
        results[o.name].seconds += signal_time["held out"] / len(options)
        for k, w in o.weights.items():
            if w:
                key = f"text:{o.text}" if k == "text" else k
                users = sum(1 for x in options if x.weights.get(k) and (k != "text" or x.text == o.text))
                results[o.name].seconds += signal_time[key] / users
    return {"results": results, "hidden": hidden_counts, "pairs": len(data.pairs)}


def same_type_share(data: Dataset, texts: Texts, variant: str, type_of: Callable[[str], str]) -> tuple[float, float]:
    """(share of each note's 5 nearest notes by text that have its type, that share by chance)."""
    import numpy as np

    ids, m = texts.matrix(variant, {})
    m = np.where(np.isfinite(m), m, 0.0)
    has = np.abs(m).sum(axis=1) > 0
    kinds = [type_of(data.path_of[f]) for f in ids]
    counts = Counter(kinds)
    n = len(ids)
    same = []
    for start in range(0, n, 512):  # in blocks: a 5,000 x 5,000 similarity matrix would be 200 MB
        sims = m[start:start + 512] @ m.T
        sims[np.arange(sims.shape[0]), np.arange(start, start + sims.shape[0])] = -np.inf
        sims[:, ~has] = -np.inf
        for row in range(sims.shape[0]):
            i = start + row
            if not has[i]:
                continue
            order = np.lexsort((np.arange(n), -sims[row]))[:5]  # the 5 most similar, ties by position
            top = [j for j in order if np.isfinite(sims[row, j])]
            if top:
                same.append(sum(kinds[j] == kinds[i] for j in top) / len(top))
    chance = sum(c * (c - 1) for c in counts.values()) / (n * (n - 1)) if n > 1 else 0.0
    return (statistics.fmean(same) if same else 0.0), chance


# ----- communities --------------------------------------------------------------------------------------


@dataclass
class CommunityOption:
    name: str
    resolution: float
    neighbours: int | None  # None: no meaning edges
    percentile: float | None


def community_options() -> list[CommunityOption]:
    out = [CommunityOption(f"no meaning edges · resolution {r:g}", r, None, None) for r in RESOLUTIONS]
    for k in NEIGHBOURS:
        for p in MEANING_PERCENTILES:
            for r in RESOLUTIONS:
                out.append(CommunityOption(f"meaning {k} nn ≥ p{p:g} · resolution {r:g}", r, k, p))
    return out


@dataclass
class CommunityResult:
    option: CommunityOption
    nmi_tags: list[float] = field(default_factory=list)
    ari_tags: list[float] = field(default_factory=list)
    nmi_pages: list[float] = field(default_factory=list)
    ari_pages: list[float] = field(default_factory=list)
    stranded: list[float] = field(default_factory=list)
    communities: list[float] = field(default_factory=list)
    stability: float = 0.0
    seconds: float = 0.0


def community_runs(data: Dataset, options: list[CommunityOption], tag_rule, seeds: list[int], texts: Texts | None,
                   variant: str, is_page: Callable[[str], bool] | None, progress=lambda s: None) -> dict:
    from vaultengine.shape import connected_parts, louvain, shape_graph_from

    def graph_of(rows, tags):
        return shape_graph_from(build_graph_from(data.notes, rows, tag_rule), tags)

    full = graph_of(data.rows, data.hashtags)
    no_tags = graph_of([r for r in data.rows if not tag_rule.is_tag(r[7], r[8], r[9])], [])
    pages = {f for f, p in data.notes if is_page and is_page(p)}
    no_pages = graph_of([r for r in data.rows if r[0] not in pages], data.hashtags)
    # References. Tags: each note's most widely used topic (its #tags and tag links, the ones on most notes left
    # out as related does, and the ones on a single note too: they group nothing). A note with none is unlabelled.
    note_tags, _, tag_notes = note_tag_sets(data.hashtags, build_graph_from(data.notes, data.rows, tag_rule), len(data.notes))
    tag_label = {f: max(shared, key=lambda t: (tag_notes[t], t)) for f, tags in note_tags.items()
                 if (shared := [t for t in tags if tag_notes[t] >= 2])}
    # Topic pages: each note linked from a page gets the page that links the fewest notes; a page is its own.
    page_label: dict[int, str] = {}
    graph = build_graph_from(data.notes, data.rows, tag_rule)
    targets = {p: sorted(graph.note_ids[t] for t in graph.adjacency[graph.by_file[p]]
                         if not graph.is_missing(t) and (graph.by_file[p], t) in graph.direct) for p in pages}
    for p in sorted(pages, key=lambda p: (len(targets[p]), data.path_of[p])):
        page_label.setdefault(p, data.path_of[p])
        for f in targets[p]:
            page_label.setdefault(f, data.path_of[p])
    # Meaning edges, once per neighbour count and cutoff.
    edges: dict[tuple[int, float], list[tuple[int, int]]] = {}
    if texts is not None and any(o.neighbours for o in options):
        from vaultengine.shape import mutual_nearest

        ids, m = texts.matrix(variant, {})
        rows = [i for i in range(len(ids)) if m[i].any()]  # a note with no known word has no vector
        for o in options:
            if o.neighbours and (o.neighbours, o.percentile) not in edges:
                pairs, _ = mutual_nearest(m[rows], o.neighbours, o.percentile)
                edges[(o.neighbours, o.percentile)] = [(ids[rows[i]], ids[rows[j]]) for i, j, _ in pairs]

    def partition(sg, option, seed):
        adj = [dict(a) for a in sg.adj]
        for fa, fb in edges.get((option.neighbours, option.percentile), []) if option.neighbours else []:
            a, b = sg.graph.by_file[fa], sg.graph.by_file[fb]
            adj[a][b] = adj[a].get(b, 0.0) + 1.0
            adj[b][a] = adj[b].get(a, 0.0) + 1.0
        parts = connected_parts(adj, louvain(adj, seed, option.resolution))
        return {sg.graph.note_ids[v]: parts[v] for v in sg.notes}

    results = {o.name: CommunityResult(o) for o in options}
    for o in options:
        res = results[o.name]
        started = time.perf_counter()
        runs = []
        for seed in seeds:
            member = partition(full, o, seed)
            runs.append(member)
            size = Counter(member.values())
            res.stranded.append(sum(1 for f in member if size[member[f]] < 2))
            res.communities.append(sum(1 for c, n in size.items() if n >= 2))
            held = partition(no_tags, o, seed)
            labelled = sorted(tag_label)
            res.nmi_tags.append(nmi([held[f] for f in labelled], [tag_label[f] for f in labelled]))
            res.ari_tags.append(ari([held[f] for f in labelled], [tag_label[f] for f in labelled]))
            if page_label:
                held = partition(no_pages, o, seed)
                labelled = sorted(page_label)
                res.nmi_pages.append(nmi([held[f] for f in labelled], [page_label[f] for f in labelled]))
                res.ari_pages.append(ari([held[f] for f in labelled], [page_label[f] for f in labelled]))
        keys = sorted(runs[0])
        pair_scores = [ari([x[f] for f in keys], [y[f] for f in keys]) for i, x in enumerate(runs) for y in runs[i + 1:]]
        res.stability = statistics.fmean(pair_scores) if pair_scores else 1.0
        res.seconds = (time.perf_counter() - started) / max(1, len(seeds))
    progress(f"{data.name}: {len(options)} community options × {len(seeds)} seeds")
    return {"results": results, "tag_labelled": len(tag_label), "tag_labels": len(set(tag_label.values())),
            "page_labelled": len(page_label), "pages": len(pages), "meaning_edges": {f"{k}/{p:g}": len(v) for (k, p), v in edges.items()}}


# ----- the whole experiment and its report -------------------------------------------------------------


def run(conn, root, *, tag_rule, spec, provider, related_weights: dict[str, float], text_variant: str,
        resolution: float, without: list[str], seeds: list[int], meaning: tuple[int, float] | None = None,
        hide: float = HIDE, queries: int = QUERIES,
        questions: tuple[str, ...] = ("relatedness", "communities", "text"), progress=lambda s: None) -> dict:
    """Every question on every dataset (the whole vault, then without each ``without`` prefix)."""
    is_page = None
    if spec is not None:
        pages = [t.name for t in spec.types if t.name in ("topic", "page")]
        is_page = (lambda path: (t := spec.type_for(path)) is not None and t.name in pages) if pages else None
    datasets = [load_dataset(conn, root, "whole vault")]
    if without:
        datasets.append(load_dataset(conn, root, "without " + ", ".join(without), without))
    out: dict[str, Any] = {"seeds": seeds, "hide": hide, "queries": queries, "datasets": [],
                           "related_weights": related_weights, "text_variant": text_variant, "resolution": resolution,
                           "meaning": meaning,
                           "model": getattr(provider, "model_id", None)}
    for data in datasets:
        entry: dict[str, Any] = {"name": data.name, "notes": len(data.notes), "left_out": data.left_out,
                                 "linked_pairs": len(data.pairs), "timings": {}}
        texts = Texts(data, provider, template_rule(conn, data, spec)) if provider is not None else None
        # Relatedness and text are scored on the same hidden links, in one pass: the link signals once per query.
        options = relatedness_options(related_weights, text_variant) if "relatedness" in questions else []
        if texts is None:
            options = [o for o in options if not o.weights.get("text")]
        elif "text" in questions:
            options += text_options(related_weights)
        if options:
            started = time.perf_counter()
            runs = hidden_link_runs(data, options, texts, tag_rule, seeds, hide, queries, progress)
            entry["timings"]["hidden links (relatedness and text)"] = time.perf_counter() - started
            for question in ("relatedness", "text"):
                mine = {name: r for name, r in runs["results"].items() if r.option.question == question}
                if mine:
                    entry[question] = {**runs, "results": mine}
        if "text" in entry:
            started = time.perf_counter()
            type_of = (lambda p: (t.name if (t := spec.type_for(p)) else "(untyped)")) if spec is not None else \
                (lambda p: p.rsplit("/", 1)[0] if "/" in p else "")
            entry["text"]["same_type"] = {v: same_type_share(data, texts, v, type_of) for v in TEXT_VARIANTS}
            entry["timings"]["same-type diagnostic"] = time.perf_counter() - started
        if "communities" in questions:
            started = time.perf_counter()
            entry["communities"] = community_runs(data, community_options(), tag_rule, seeds, texts, text_variant,
                                                  is_page, progress)
            entry["timings"]["communities"] = time.perf_counter() - started
        out["datasets"].append(entry)
    out["decisions"] = decide(out, related_weights, text_variant, resolution, meaning)
    return out


def _best(results: dict, key: Callable) -> list:
    return sorted(results.values(), key=lambda r: (-statistics.fmean(key(r)), r.option.name))


def is_today(option: CommunityOption, resolution: float, meaning: tuple[int, float] | None) -> bool:
    """Is this community option today's setting (``meaning``: (neighbours, percentile) when meaning edges are on)?"""
    if option.resolution != resolution:
        return False
    return option.neighbours is None if meaning is None else (option.neighbours, option.percentile) == tuple(meaning)


def decide(out: dict, related_weights: dict, text_variant: str, resolution: float,
           meaning: tuple[int, float] | None = None) -> list[dict]:
    """Which defaults the data supports changing: an option must beat today's setting clearly in every dataset, on
    every score the question is judged by; of those, the best mean of the first score over the datasets is taken.

    relatedness: MRR. text: MRR with text alone (what the embedded text is worth by itself; the related mix
    dilutes it). communities: ARI and NMI with the Tags: topics (both, since NMI alone rises with the number of
    communities; the topic pages label too few notes in a small vault to decide anything).
    """
    decisions = []

    def choose(question: str, keys: list[Callable], today: Callable, setting: str, value_of: Callable, current,
               eligible: Callable = lambda o: True) -> None:
        entries = [d for d in out["datasets"] if question in d]
        if not entries:
            return
        names = [n for n, r in entries[0][question]["results"].items() if eligible(r.option)]
        todays = [next(r for r in d[question]["results"].values() if today(r.option)) for d in entries]
        clear = []
        for name in names:
            comps = [[compare(key(d[question]["results"][name]), key(t)) for key in keys] for d, t in zip(entries, todays)]
            if all(c.clear for per in comps for c in per):
                score = statistics.fmean(statistics.fmean(keys[0](d[question]["results"][name])) for d in entries)
                clear.append((score, name, [per[0] for per in comps]))
        if clear:
            score, name, comps = max(clear)
            value = value_of(entries[0][question]["results"][name].option)
            decisions.append({"setting": setting, "from": current, "to": value, "option": name, "change": value != current,
                              "why": f"{name} beats today's setting in every dataset: "
                                     + "; ".join(f"{d['name']}: {c.words()}" for d, c in zip(entries, comps))})
        else:
            winners = [next(r for r in _best(d[question]["results"], keys[0]) if eligible(r.option)).option.name
                       for d in entries]
            decisions.append({"setting": setting, "from": current, "to": current, "option": None, "change": False,
                              "why": "no option beats today's setting clearly in every dataset (winners: "
                                     + "; ".join(f"{d['name']}: {w}" for d, w in zip(entries, winners)) + ")"})

    signals = ("graph", "cocitation", "coupling", "tags", "text")
    choose("relatedness", [lambda r: r.mrr], lambda o: o.name.startswith("today's related ("), "related.unlinked_weights",
           lambda o: {k: round(o.weights.get(k, 0.0), 4) for k in signals},
           {k: related_weights.get(k, 0.0) for k in signals})
    choose("text", [lambda r: r.mrr], lambda o: o.name == f"text only · {text_variant}", "embedding.text",
           lambda o: o.text, text_variant, eligible=lambda o: o.name.startswith("text only"))
    choose("communities", [lambda r: r.ari_tags, lambda r: r.nmi_tags], lambda o: is_today(o, resolution, meaning),
           "explore (meaning edges, resolution)",
           lambda o: {"meaning": o.neighbours is not None, "meaning_neighbours": o.neighbours,
                      "meaning_percentile": o.percentile, "resolution": o.resolution},
           {"meaning": meaning is not None, "meaning_neighbours": meaning[0] if meaning else None,
            "meaning_percentile": meaning[1] if meaning else None, "resolution": resolution})
    return decisions


def _fmt(values: list[float], digits: int = 3) -> str:
    m, sd = mean_sd(values)
    return f"{m:.{digits}f} ± {sd:.{digits}f}"


def _ranking(out: dict, question: str, q: dict) -> tuple[list, Any, Callable, str]:
    """(options best first, today's result, the score, its name) for a question's results on one dataset."""
    if question == "communities":
        ranked = _best(q["results"], lambda r: r.ari_tags)
        today = next(r for r in ranked if is_today(r.option, out["resolution"], out.get("meaning")))
        return ranked, today, lambda r: r.ari_tags, "ARI with the Tags: topics"
    ranked = _best(q["results"], lambda r: r.mrr)
    if question == "text":
        ranked = [r for r in ranked if r.option.name.startswith("text only")] + \
                 [r for r in ranked if not r.option.name.startswith("text only")]
        today = next(r for r in ranked if r.option.name == f"text only · {out['text_variant']}")
        return ranked, today, lambda r: r.mrr, "MRR, text alone"
    today = next(r for r in ranked if r.option.name.startswith("today's related ("))
    return ranked, today, lambda r: r.mrr, "MRR"


TITLES = {"relatedness": "Which signals rank related notes", "text": "What text gets embedded", "communities": "Communities"}


def markdown(out: dict, vault: str, date: str, seconds: float, command: str | None = None) -> str:
    meaning = out.get("meaning")
    lines = [f"# Experiment, {date}", "", *([f"`{command}`", ""] if command else []),
             f"Vault `{vault}`, model `{out['model']}`, seeds {', '.join(map(str, out['seeds']))}, "
             f"{out['hide']:.0%} of the linked note pairs hidden per seed (at most {out['queries']} scored). "
             f"Today's settings: related.unlinked_weights "
             + ", ".join(f"{k} {v:g}" for k, v in out["related_weights"].items() if v)
             + f"; embedded text `{out['text_variant']}`; resolution {out['resolution']:g}, "
             + (f"meaning edges {meaning[0]} nn ≥ p{meaning[1]:g}" if meaning else "no meaning edges") + ". "
             f"Whole run: {seconds:.0f} s. Made by `vault experiment` (`vaultengine/experiment.py` says how every number "
             "is computed); the same vault, settings and seeds give the same report, timings aside.", "",
             "A difference is *clear* when the option is ahead on every seed and the mean per-seed difference is more than "
             "twice its standard error; otherwise it is within the seed-to-seed noise.", "",
             "Time: " + "; ".join(f"{d['name']}: " + ", ".join(f"{k} {v:.0f} s" for k, v in d["timings"].items())
                                  for d in out["datasets"]) + ".", "",
             "## Summary", "",
             "| question | dataset | score | winner | margin over the runner-up | over today's setting |",
             "|---|---|---|---|---|---|"]
    for question in TITLES:
        for d in out["datasets"]:
            if question not in d:
                continue
            ranked, today, key, score = _ranking(out, question, d[question])
            pool = [r for r in ranked if question != "text" or r.option.name.startswith("text only")]
            first, second = pool[0], pool[1]
            versus = "it is today's setting" if first is today else compare(key(first), key(today)).words()
            lines.append(f"| {question} | {d['name']} | {score} | {first.option.name} | "
                         f"{compare(key(first), key(second)).words()} ({second.option.name}) | {versus} |")
    lines.append("")
    for question, title in TITLES.items():
        entries = [d for d in out["datasets"] if question in d]
        if not entries:
            continue
        lines += [f"## {title}", ""]
        for d in entries:
            q = d[question]
            ranked, today, key, score = _ranking(out, question, q)
            lines += [f"### {d['name']} ({d['notes']} notes, {d['linked_pairs']} linked pairs"
                      + (f"; {d['left_out']} notes left out" if d["left_out"] else "") + ")", ""]
            if question in ("relatedness", "text"):
                hidden = q["hidden"]
                span = f"{min(hidden)}" if min(hidden) == max(hidden) else f"{min(hidden)}–{max(hidden)}"
                scored = ranked[0].queries // len(out["seeds"])
                lines.append(f"Hidden per seed: {span} of {q['pairs']} linked pairs; {scored} scored per seed."
                             + (f" With {scored} hidden links a seed, one of them moving from rank 1 to rank 2 changes "
                                f"MRR by {0.5 / scored:.3f}." if scored < 100 else ""))
                if question == "text":
                    lines.append("Judged by the text-only rows (what the embedded text is worth by itself); the rows "
                                 "with today's related mix show how much of it survives the mix.")
                lines += ["", "| option | MRR | recall@5 | recall@10 | s per seed |"
                          + (" same-type neighbours (chance) |" if question == "text" else ""),
                          "|---|---|---|---|---|" + ("---|" if question == "text" else "")]
                for r in ranked:
                    extra = ""
                    if question == "text":
                        share, chance = q["same_type"][r.option.text]
                        extra = f" {share:.2f} ({chance:.2f}) |"
                    lines.append(f"| {r.option.name} | {_fmt(r.mrr)} | {_fmt(r.r5, 2)} | {_fmt(r.r10, 2)} | "
                                 f"{r.seconds / len(out['seeds']):.2f} |{extra}")
            else:
                lines.append(f"References: Tags: topics label {q['tag_labelled']} notes with {q['tag_labels']} topics; "
                             f"topic pages ({q['pages']}) label {q['page_labelled']} notes"
                             + (" (too few to judge by)" if q["page_labelled"] < 30 else "")
                             + ". Meaning edges (neighbours/percentile → edges): "
                             + (", ".join(f"{k} → {v}" for k, v in q["meaning_edges"].items()) or "none (no model)") + ".")
                lines += ["", "| option | ARI tags | NMI tags | ARI pages | NMI pages | stranded | communities | stability | s per run |",
                          "|---|---|---|---|---|---|---|---|---|"]
                for r in ranked:
                    pages = (_fmt(r.ari_pages), _fmt(r.nmi_pages)) if r.nmi_pages else ("—", "—")
                    lines.append(f"| {r.option.name} | {_fmt(r.ari_tags)} | {_fmt(r.nmi_tags)} | {pages[0]} | {pages[1]} | "
                                 f"{statistics.fmean(r.stranded):.1f} | {statistics.fmean(r.communities):.1f} | "
                                 f"{r.stability:.3f} | {r.seconds:.2f} |")
                lines.append("")
                for res in sorted({r.option.resolution for r in ranked}):
                    plain = next((r for r in ranked if r.option.resolution == res and r.option.neighbours is None), None)
                    best = next((r for r in ranked if r.option.resolution == res and r.option.neighbours), None)
                    if plain and best:
                        lines.append(f"- Resolution {res:g}: the best meaning edges ({best.option.name}) against none: "
                                     f"ARI {compare(best.ari_tags, plain.ari_tags).words()}; "
                                     f"NMI {compare(best.nmi_tags, plain.nmi_tags).words()}.")
                fewest = min(ranked, key=lambda r: (statistics.fmean(r.stranded), r.option.name))
                lines += ["", f"Fewest stranded notes: {fewest.option.name} ({statistics.fmean(fewest.stranded):.1f}; "
                              f"today {statistics.fmean(today.stranded):.1f}). Stability: ARI between the partitions of "
                              "different seeds (1 = the same communities whatever the seed)."]
            lines.append("")
    lines += ["## Defaults", "",
              "A default changes only when an option beats today's setting clearly in every dataset, on every score the "
              "question is judged by (relatedness: MRR; text: MRR with text alone; communities: ARI and NMI with the "
              "Tags: topics); then the one with the best mean over the datasets is taken.", ""]
    for d in out["decisions"]:
        verdict = f"change to `{d['to']}`" if d["change"] else "keep"
        lines.append(f"- `{d['setting']}`: {verdict}. {d['why']}.")
    return "\n".join(lines) + "\n"
