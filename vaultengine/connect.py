"""Connect notes: plain-text mentions that are not links yet, and pairs of notes that probably belong linked.

Rules both follow, and the tests check:

* never a link that already exists: a note that links to another (a link or an
  embed, with or without a #heading, or an ambiguous link that could mean it) is
  never suggested to link it again, in either direction for pairs;
* never a note to itself;
* a mention only counts in plain text: not inside code, comments, frontmatter, a
  tag, a [[link]] or ![[embed]] (including its display text), a markdown link or
  image (text and URL), or a bare URL.

Unlinked mentions
-----------------
A note is mentioned where its title (file name) or one of its aliases appears as
words: both are split into words exactly like the search index, so the kebab-case
title ``reward-hacking`` is mentioned by "reward hacking", "Reward Hacking" and
"reward-hacking", but not by "reward hackings" or "reward_hacking". The words must
be consecutive in one stretch of text (not across a heading, and a link between
them breaks the run). Where several names match at the same place, the longest
wins: "reward hacking as specification gaming" is a mention of that note, not also
of a note called "reward".

Links to add
------------
Pairs of notes with no link between them, scored from three kinds of evidence,
each a similarity from 0 to 1 and each shown in the reason:

* shared neighbours - the notes (or missing notes) both are linked with, in either
  direction, except notes that link to both (those are the next kind). Each
  neighbour z weighs w_z = 1 / ln(1 + its number of links), so a topic linked
  from 52 notes counts a quarter of a note linked from two (Adamic-Adar), and the
  sum over shared neighbours is divided by sqrt((W_a + 1) * (W_b + 1)), where W is
  the total weight of a note's neighbours. That is a cosine: two notes that each
  link 60 things share many neighbours by chance and are not rewarded for it. The
  +1 keeps thin evidence from looking perfect: two notes whose one neighbour is
  the same are 0.5 alike, not 1;
* cited together - the notes that link to both (co-citation), each citing note c
  weighing 1 / ln(1 + the notes c links to), divided the same way (+1 included)
  by the total weight of the notes citing each of the two;
* similar text - the centered cosine of the two notes' embeddings (``related.py``:
  the configured embedding provider, the vault's mean vector subtracted), 0 if negative.
  The words named are those in both notes that add the most to it.

* shared tags - weighted Jaccard of the two notes' tags, #tags and tag links
  (``Tags: [[topic]]``, ``tagrule.py``), exactly as ``related`` and ``explain`` score it.
  Tag links are tags here too: they are not neighbours, and a note that tags two
  notes does not cite them together, so ``suggest``, ``related`` and ``explain``
  read the same links the same way.

score = (0.4 * shared + 0.3 * cited + 0.2 * tags + 0.3 * text) / 1.2, i.e. 1/3, 1/4, 1/6
and 1/4; without the embedding model (missing or broken), the text evidence is left out
and the other weights are rescaled to sum to 1. Candidates are the pairs with a
shared neighbour, a co-citation or a shared tag through a node with at most ``HUB_LINKS``
notes, plus each note's 5 closest notes by embedding. Hubs (a topic linked from hundreds of
notes, a note that links hundreds) still add their evidence to those candidates,
but do not make candidates by themselves: that would be every pair of their notes,
on the weakest evidence there is.
"""

from __future__ import annotations

import bisect
import json
import math
import re
import sqlite3
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

from vaultengine.parser import normalize_text, parse_note_with_layout
from vaultengine.related import Relatedness, active_weights, tag_label
from vaultengine.search import field_segments
from vaultengine.snippets import _tokens
from vaultengine.tokenize import tokenize
from vaultengine.vault import link_name, lookup_key

WEIGHTS = {"shared": 0.4 / 1.2, "cited": 0.3 / 1.2, "tags": 0.2 / 1.2, "text": 0.3 / 1.2}
HUB_LINKS = 50
TEXT_NEIGHBOURS = 5
_BARE_URL = re.compile(r"\b(?:https?|ftp)://[^\s<>\"'\]\)]+|\bwww\.[^\s<>\"'\]\)]+", re.IGNORECASE)


def _linked_pairs(conn: sqlite3.Connection) -> dict[int, set[int]]:
    """note id -> ids of the files it links to: resolved links and embeds, and every candidate of an ambiguous link."""
    ids = {path: file_id for file_id, path in conn.execute("SELECT id, path FROM files")}
    linked: dict[int, set[int]] = defaultdict(set)
    for source, target, status, candidates in conn.execute("SELECT source_id, target_id, status, candidates FROM links"):
        if status == "resolved" and target is not None:
            linked[source].add(target)
        elif status == "ambiguous":
            linked[source].update(ids[path] for path in json.loads(candidates) if path in ids)
    return linked


# ----- unlinked mentions -----------------------------------------------------------------


@dataclass
class Mention:
    source: str  # the note that mentions
    target: str  # the note mentioned
    name: str  # the title or alias as written in the target
    line: int
    start: int  # offsets in the source note's text
    end: int
    before: str  # text around the mention, for display
    text: str
    after: str


class MentionFinder:
    def __init__(self, conn: sqlite3.Connection, root: str | Path, min_words: int = 1) -> None:
        self.conn = conn
        self.root = Path(root)
        self.paths = {row[0]: row[1] for row in conn.execute("SELECT id, path FROM files WHERE kind = 'note'")}
        self.linked = _linked_pairs(conn)
        # first word -> [(words, target id, name)], longest names first
        self.names: dict[str, list[tuple[tuple[str, ...], int, str]]] = defaultdict(list)
        self.names_of: dict[int, list[str]] = defaultdict(list)
        entries = [(file_id, link_name(path)) for file_id, path in self.paths.items()]
        entries += [(file_id, alias) for file_id, alias in conn.execute("SELECT file_id, alias FROM aliases") if file_id in self.paths]
        for file_id, name in entries:
            words = tuple(tokenize(name))
            if words and len(words) >= min_words:
                self.names[words[0]].append((words, file_id, name))
                self.names_of[file_id].append(name)
        for options in self.names.values():
            options.sort(key=lambda option: (-len(option[0]), self.paths[option[1]]))

    def scan(self, source_id: int, only: int | None = None) -> list[Mention]:
        """Unlinked mentions in one note (of any note, or only of ``only``)."""
        path = self.paths[source_id]
        try:
            raw = (self.root / path).read_bytes()
        except OSError:
            return []
        text = unicodedata.normalize("NFC", normalize_text(raw.decode("utf-8", errors="replace")))
        note, layout = parse_note_with_layout(text, path)
        excluded = sorted(layout.links + [m.span() for m in _BARE_URL.finditer(layout.masked)])
        starts = [a for a, _ in excluded]

        def hidden(start: int, end: int) -> bool:
            i = bisect.bisect_right(starts, end - 1) - 1  # the last excluded span starting before `end`
            while i >= 0 and excluded[i][0] < end:
                if excluded[i][1] > start:
                    return True
                i -= 1
            return False

        segments = field_segments(path, note, layout)
        found = []
        for field_name in ("body", "headings"):
            tokens = _tokens(segments[field_name])
            i = 0
            while i < len(tokens):
                match = None
                for words, target, name in self.names.get(tokens[i].term, ()):
                    n = len(words)
                    window = tokens[i : i + n]
                    if (
                        len(window) == n
                        and all(t.term == w for t, w in zip(window, words))
                        and window[-1].position - window[0].position == n - 1
                        and not any(hidden(t.start, t.end) for t in window)
                    ):
                        match = (n, target, name)
                        break
                if match is None:
                    i += 1
                    continue
                n, target, name = match
                start, end = tokens[i].start, tokens[i + n - 1].end
                i += n  # the longest name at this place takes it, whoever it is
                if target == source_id or target in self.linked[source_id] or (only is not None and target != only):
                    continue
                line_start = text.rfind("\n", 0, start) + 1
                line_end = text.find("\n", end)
                line_end = len(text) if line_end == -1 else line_end
                before = text[max(line_start, start - 60) : start]
                after = text[end : min(line_end, end + 60)]
                found.append(Mention(path, self.paths[target], name, text.count("\n", 0, start) + 1, start, end,
                                     ("…" if start - 60 > line_start else "") + before.lstrip(),
                                     text[start:end], after.rstrip() + ("…" if end + 60 < line_end else "")))
        return sorted(found, key=lambda m: m.start)  # in reading order (headings were scanned after the body)

    def all(self) -> list[Mention]:
        """Every unlinked mention. Only notes that the index says contain all the words of some name are read."""
        notes_with: dict[str, set[int]] = defaultdict(set)
        for term, file_id in self.conn.execute("SELECT DISTINCT term, file_id FROM postings WHERE field IN ('body', 'headings')"):
            notes_with[term].add(file_id)
        sources: set[int] = set()
        for options in self.names.values():
            for words, target, _ in options:
                holders = sorted((notes_with.get(w, set()) for w in set(words)), key=len)
                found = set(holders[0]).intersection(*holders[1:]) if holders else set()
                sources |= found - {target}
        return [m for source in sorted(sources, key=self.paths.get) if source in self.paths for m in self.scan(source)]

    def of(self, target: int) -> list[Mention]:
        """Unlinked mentions of one note; reads only the notes whose index has one of its names as a phrase."""
        from vaultengine.querylang import Phrase, Word
        from vaultengine.search import _Searcher

        searcher = _Searcher(self.conn, fuzzy=False)
        candidates: set[int] = set()
        for name in self.names_of[target]:
            words = tuple(tokenize(name))
            clause = Word(words[0], exact=True) if len(words) == 1 else Phrase(words)
            _, matches = searcher.word(clause) if len(words) == 1 else searcher.phrase(clause)
            candidates |= set(matches)
        candidates.discard(target)
        return [m for source in sorted(candidates, key=self.paths.get) if source in self.paths for m in self.scan(source, only=target)]


# ----- links to add --------------------------------------------------------------------------


@dataclass
class LinkSuggestion:
    a: str  # note paths, a < b
    b: str
    score: float
    shared: float
    cited: float
    text: float
    tags: float = 0.0
    shared_with: list[str] = field(default_factory=list)  # neighbours, strongest first ("(no note yet)" marked)
    shared_tags: list[str] = field(default_factory=list)  # #tags and [[tag links]] both have, rarest first
    cited_by: list[str] = field(default_factory=list)  # notes linking both, strongest first
    terms: list[str] = field(default_factory=list)  # words in both notes that add most to their embedding cosine
    reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class LinkSuggester:
    def __init__(self, conn: sqlite3.Connection, engine: Relatedness | None = None) -> None:
        self.conn = conn
        self.engine = engine or Relatedness(conn)
        self.paths = self.engine.paths
        self.weights = dict(WEIGHTS)  # the weights of the last suggestions() (text left out without a model)
        self.linked = _linked_pairs(conn)  # every link, tag links too: a tagged pair is already linked
        # Undirected neighbours of notes and missing notes ("m:" + key), and what each note cites, from the
        # graph's link layers, which leave tag links out (they are the tags evidence).
        self.neighbours: dict[str, set[str]] = defaultdict(set)
        self.cites: dict[int, set[int]] = defaultdict(set)  # note -> notes it links to
        self.missing_name: dict[str, str] = {}
        graph = self.engine.graph
        name = {}
        for v in range(len(graph.labels)):
            if graph.is_missing(v):
                name[v] = f"m:{lookup_key(graph.labels[v])}"
                self.missing_name[name[v]] = graph.labels[v]
            else:
                name[v] = f"n:{graph.note_ids[v]}"
        for a, targets in enumerate(graph.outgoing):
            if graph.is_missing(a):
                continue
            for b in targets:
                self.neighbours[name[a]].add(name[b])
                self.neighbours[name[b]].add(name[a])
                if not graph.is_missing(b):
                    self.cites[graph.note_ids[a]].add(graph.note_ids[b])

    def is_linked(self, a: int, b: int) -> bool:
        return b in self.linked[a] or a in self.linked[b]

    def _name(self, node: str) -> str:
        if node.startswith("m:"):
            return f"{self.missing_name[node]} (no note yet)"
        return link_name(self.paths[int(node[2:])])

    def text_similarity(self) -> dict[tuple[int, int], float]:
        """Centered embedding cosine for each note's 5 closest notes (both directions kept; positive only)."""
        import numpy as np

        text = self.engine.text
        if not text.available():
            return {}
        ids = text.vectors.file_ids
        if not ids:
            return {}
        centered = text.centered
        result = {}
        k = min(TEXT_NEIGHBOURS, len(ids) - 1)
        block = 1024  # rows at a time: all the cosines at once would be notes^2 numbers (20 GB at 50,000 notes)
        for start in range(0, len(ids), block):
            cosines = centered[start:start + block] @ centered.T
            cosines[np.arange(cosines.shape[0]), np.arange(start, start + cosines.shape[0])] = -np.inf
            for offset in range(cosines.shape[0]):
                row = start + offset
                if k <= 0 or not centered[row].any():
                    continue
                nearest = np.argpartition(-cosines[offset], k - 1)[:k]
                for other in nearest:
                    cosine = float(cosines[offset, other])
                    if cosine > 0:
                        a, b = ids[row], ids[other]
                        result[(min(a, b), max(a, b))] = cosine
        return result

    def suggestions(self, limit: int | None = 10, for_note: int | None = None, path: str | None = None) -> list[LinkSuggestion]:
        """Pairs, best first; only those involving ``for_note``, or with both paths containing ``path``."""
        # Shared neighbours: weighted cosine of the two notes' neighbour sets.
        weight = {node: 1 / math.log(1 + len(around)) for node, around in self.neighbours.items()}
        total = {f: sum(weight[z] for z in self.neighbours.get(f"n:{f}", ())) for f in self.paths}
        citer_weight = {c: 1 / math.log(1 + len(targets)) for c, targets in self.cites.items()}
        cited_total: Counter[int] = Counter()
        for citer, targets in self.cites.items():
            for note in targets:
                cited_total[note] += citer_weight[citer]
        # Candidates from specific evidence only (see HUB_LINKS).
        candidates: set[tuple[int, int]] = set()
        for node, around in self.neighbours.items():
            if len(around) <= HUB_LINKS:
                notes = sorted(int(n[2:]) for n in around if n.startswith("n:"))
                candidates.update((a, b) for i, a in enumerate(notes) for b in notes[i + 1 :])
        for targets in self.cites.values():
            if len(targets) <= HUB_LINKS:
                notes = sorted(targets)
                candidates.update((a, b) for i, a in enumerate(notes) for b in notes[i + 1 :])
        engine = self.engine
        tag_notes: dict[str, list[int]] = defaultdict(list)
        for f, tags in engine.note_tags.items():  # common tags already left out
            for t in tags:
                tag_notes[t].append(f)
        for t, notes in tag_notes.items():
            if len(notes) <= HUB_LINKS:
                notes = sorted(notes)
                candidates.update((a, b) for i, a in enumerate(notes) for b in notes[i + 1 :])
        text = self.text_similarity()
        self.weights = active_weights(WEIGHTS, set() if self.engine.text.available() else {"text"})
        weights = self.weights
        candidates |= set(text)
        candidates = {p for p in candidates if p[0] != p[1] and not self.is_linked(*p)}
        if for_note is not None:
            candidates = {p for p in candidates if for_note in p}
        if path:
            needle = unicodedata.normalize("NFC", path).casefold()
            candidates = {p for p in candidates if all(needle in self.paths[f].casefold() for f in p)}
        # All evidence for each candidate, hubs included.
        citers_of: dict[int, set[int]] = defaultdict(set)
        for citer, targets in self.cites.items():
            for note in targets:
                citers_of[note].add(citer)
        shared: dict[tuple[int, int], float] = {}
        cited: dict[tuple[int, int], float] = {}
        shared_with: dict[tuple[int, int], list[tuple[float, str]]] = {}
        cited_by: dict[tuple[int, int], list[tuple[float, int]]] = {}
        for a, b in candidates:
            both_cite = citers_of[a] & citers_of[b]
            common = [z for z in self.neighbours.get(f"n:{a}", set()) & self.neighbours.get(f"n:{b}", set())
                      if not (z.startswith("n:") and int(z[2:]) in both_cite)]  # a citer of both counts as "cited together"
            if common:
                shared[(a, b)] = sum(weight[z] for z in common) / math.sqrt((total[a] + 1) * (total[b] + 1))
                shared_with[(a, b)] = [(weight[z], z) for z in common]
            if both_cite:
                cited[(a, b)] = sum(citer_weight[c] for c in both_cite) / math.sqrt((cited_total[a] + 1) * (cited_total[b] + 1))
                cited_by[(a, b)] = [(citer_weight[c], c) for c in both_cite]
        tags = {p: engine.tag_score(engine.note_tags.get(p[0], set()), engine.note_tags.get(p[1], set())) for p in candidates}
        scores = {
            p: weights["shared"] * shared.get(p, 0.0) + weights["cited"] * cited.get(p, 0.0) + weights["tags"] * tags[p]
            + weights["text"] * text.get(p, 0.0)
            for p in candidates
        }
        ranked = sorted(candidates, key=lambda p: (-scores[p], self.paths[p[0]], self.paths[p[1]]))
        found = []
        for a, b in ranked if limit is None else ranked[:limit]:  # details only for what is returned
            x = text.get((a, b), 0.0)
            terms = self.engine.text.words(self.engine.text.bag(a, self.paths[a]), self.engine.text.bag(b, self.paths[b])) if x > 0 else []
            suggestion = LinkSuggestion(
                a=self.paths[a], b=self.paths[b], score=scores[(a, b)],
                shared=shared.get((a, b), 0.0), cited=cited.get((a, b), 0.0), text=x, tags=tags[(a, b)],
                shared_tags=[tag_label(t) for t in sorted(engine.note_tags.get(a, set()) & engine.note_tags.get(b, set()),
                                                         key=lambda t: (-engine.tag_idf[t], t))],
                shared_with=[self._name(n) for _, n in sorted(shared_with.get((a, b), []), key=lambda item: (-item[0], item[1]))],
                cited_by=[link_name(self.paths[n]) for _, n in sorted(cited_by.get((a, b), []), key=lambda item: (-item[0], self.paths[item[1]]))],
                terms=terms,
            )
            suggestion.reason = self.reason(suggestion, self.engine.text.explains)
            found.append(suggestion)
        return found

    @staticmethod
    def reason(s: LinkSuggestion, explains: bool = True) -> str:
        """``explains``: the model names words; then "similar text" needs one (as in ``related.py``)."""
        parts = []

        def listing(names: list[str], count: int = 2) -> str:
            shown = [n if len(n) <= 45 else n[:44].rstrip() + "…" for n in names[:count]]
            return ", ".join(shown) + (f" +{len(names) - count}" if len(names) > count else "")

        if s.shared_with:
            parts.append("shared neighbours: " + listing(s.shared_with))
        if s.shared_tags:
            parts.append("shares " + listing(s.shared_tags, 3))
        if s.cited_by:
            parts.append("cited together by " + listing(s.cited_by))
        if s.terms:
            parts.append("similar text (" + ", ".join(s.terms) + ")")
        elif s.text > 0 and not explains:
            parts.append("similar text")
        return " · ".join(parts)
