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
* similar text - cosine similarity of the two notes' distinctive-term profiles
  (``related.py``: top 30 terms by sublinear tf x idf).

score = 0.4 * shared + 0.3 * cited + 0.3 * text. Candidates are the pairs with a
shared neighbour or a co-citation through a note with at most ``HUB_LINKS`` links,
plus each note's 5 textually closest notes. Hubs (a topic linked from hundreds of
notes, a note that links hundreds) still add their evidence to those candidates,
but do not make candidates by themselves: that would be every pair of their notes,
on the weakest evidence there is.
"""

from __future__ import annotations

import bisect
import heapq
import json
import math
import re
import sqlite3
import unicodedata
from array import array
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

from vaultengine.parser import normalize_text, parse_note_with_layout
from vaultengine.related import Relatedness
from vaultengine.search import field_segments
from vaultengine.snippets import _tokens
from vaultengine.tokenize import tokenize
from vaultengine.vault import link_name

WEIGHTS = {"shared": 0.4, "cited": 0.3, "text": 0.3}
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
    shared_with: list[str] = field(default_factory=list)  # neighbours, strongest first ("(no note yet)" marked)
    cited_by: list[str] = field(default_factory=list)  # notes linking both, strongest first
    terms: list[str] = field(default_factory=list)  # shared distinctive terms
    reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class LinkSuggester:
    """``counts(source path, field kind)``: whether a link counts as evidence (default: every link; a
    format leaves out links in metadata values). Any existing link still means the pair is linked."""

    def __init__(self, conn: sqlite3.Connection, engine: Relatedness | None = None, counts=None) -> None:
        self.conn = conn
        self.engine = engine or Relatedness(conn)
        self.paths = self.engine.paths
        self.linked = _linked_pairs(conn)
        # Undirected neighbours of notes and missing notes ("m:" + key), and what each note cites, with lines.
        self.neighbours: dict[str, set[str]] = defaultdict(set)
        self.cites: dict[int, set[int]] = defaultdict(set)  # note -> notes it links to
        self.missing_name: dict[str, str] = {}
        rows = conn.execute(
            """SELECT l.source_id, l.target_id, l.status, l.target_key, l.target, l.line, t.kind, s.path, l.field_kind
               FROM links l JOIN files s ON s.id = l.source_id LEFT JOIN files t ON t.id = l.target_id
               WHERE s.kind = 'note' AND l.target != ''"""
        )
        for source, target, status, key, written, line, kind, source_path, field_kind in rows:
            if counts is not None and not counts(source_path, field_kind):
                continue
            if status == "resolved" and kind == "note" and target != source:
                a, b = f"n:{source}", f"n:{target}"
                self.cites[source].add(target)
            elif status == "unresolved":
                a, b = f"n:{source}", f"m:{key}"
                self.missing_name.setdefault(f"m:{key}", written)
            else:
                continue
            self.neighbours[a].add(b)
            self.neighbours[b].add(a)

    def is_linked(self, a: int, b: int) -> bool:
        return b in self.linked[a] or a in self.linked[b]

    def _name(self, node: str) -> str:
        if node.startswith("m:"):
            return f"{self.missing_name[node]} (no note yet)"
        return link_name(self.paths[int(node[2:])])

    def text_similarity(self) -> dict[tuple[int, int], tuple[float, list[str]]]:
        """Cosine of the top-30 term profiles for each note's 5 closest notes (both directions kept).

        Exact, and without comparing a note with every note that shares one of its terms:

        * notes with the same profile (same terms, same weights, so the same order) have the same cosine
          with any note, to the last bit: the work is done once per distinct profile (a "class"), and a
          note's 5 closest are the class's 6 closest notes less itself (ties broken by the larger id, as
          ``heapq.nlargest`` over (cosine, id) does);
        * a class's terms are taken rarest first, and once the 6th-best similarity found so far beats what
          the terms left could give a class not seen yet (Cauchy-Schwarz: the norm of the profile over those
          terms, over the whole norm), no other class can enter;
        * the cosines are then computed in the profile's own term order, as the pair-by-pair version did,
          so the values are the same to the last bit.
        """
        profiles = {f: self.engine._top(profile) for f, profile in self.engine.all_profiles().items()}
        class_of: dict[int, int] = {}
        members: list[list[int]] = []  # class -> its notes
        class_profiles: list[dict[str, float]] = []
        by_key: dict[tuple, int] = {}
        for f, profile in profiles.items():
            key = tuple(profile.items())
            c = by_key.get(key)
            if c is None:
                c = by_key[key] = len(members)
                members.append([])
                class_profiles.append(profile)
            members[c].append(f)
            class_of[f] = c
        norms = [math.sqrt(sum(w * w for w in p.values())) for p in class_profiles]
        best_ids = [sorted(m, reverse=True)[:TEXT_NEIGHBOURS + 1] for m in members]
        sizes = [len(m) for m in members]
        by_term: dict[str, list[tuple[int, float]]] = defaultdict(list)
        for c, profile in enumerate(class_profiles):
            for term, weight in profile.items():
                by_term[term].append((c, weight))
        want = TEXT_NEIGHBOURS + 1
        nearest: dict[int, list[tuple[float, int]]] = {}  # class -> its 6 closest notes, members included
        for c, profile in enumerate(class_profiles):
            norm_c = norms[c]
            if not norm_c:
                continue
            order = sorted(profile, key=lambda t: (len(by_term[t]), t))  # rarest first
            left = [0.0] * (len(order) + 1)  # squared norm of the terms not yet taken
            for i in range(len(order) - 1, -1, -1):
                left[i] = left[i + 1] + profile[order[i]] ** 2
            partial: dict[int, float] = {}
            for i, term in enumerate(order):
                if partial:
                    bound = math.sqrt(left[i]) / norm_c  # the best a class sharing none of the terms so far can do
                    counted, sixth = 0, None
                    for cosine, d in heapq.nlargest(want, ((dot / (norm_c * norms[d]), d) for d, dot in partial.items() if norms[d])):
                        counted += sizes[d]
                        if counted >= want:
                            sixth = cosine
                            break
                    if sixth is not None and sixth > bound * (1 + 1e-9) + 1e-12:
                        break
                weight_c = profile[term]
                for d, w in by_term[term]:
                    partial[d] = partial.get(d, 0.0) + weight_c * w
            scored = []
            for d in partial:
                if not norms[d]:
                    continue
                theirs = class_profiles[d]
                dot = 0
                for term, weight in profile.items():  # the order the pair-by-pair version summed in
                    w = theirs.get(term)
                    if w is not None:
                        dot += weight * w
                cosine = dot / (norm_c * norms[d])
                scored.extend((cosine, o) for o in best_ids[d])
            nearest[c] = heapq.nlargest(want, scored)
        result = {}
        shared_of: dict[tuple[int, int], list[str]] = {}
        for f in profiles:
            c = class_of[f]
            if c not in nearest:
                continue
            mine = class_profiles[c]
            for cosine, other in [item for item in nearest[c] if item[1] != f][:TEXT_NEIGHBOURS]:
                d = class_of[other]
                shared = shared_of.get((c, d))
                if shared is None:
                    theirs = class_profiles[d]
                    shared = shared_of[(c, d)] = sorted(set(mine) & set(theirs), key=lambda t: (-mine[t] * theirs[t], t))[:3]
                result[(min(f, other), max(f, other))] = (cosine, shared)
        return result

    def suggestions(self, limit: int | None = 10, for_note: int | None = None, path: str | None = None,
                    min_score: float | None = None, structural_percentile: float | None = None) -> list[LinkSuggestion]:
        """Pairs, best first; only those involving ``for_note``, or with both paths containing ``path``.
        ``structural_percentile``: only pairs with structural evidence (shared neighbours or citers) scoring at
        least that nearest-rank percentile of such pairs, and at least ``min_score`` (the floor); the threshold
        used is left in ``self.threshold``.

        Candidates are the pairs that share a neighbour (or a citer) with at most HUB_LINKS links, plus the
        pairs ``text_similarity`` finds. Their evidence is accumulated note by note: each shared neighbour adds
        its weight once, so no pair's neighbour sets are intersected, and the names and terms that explain a
        pair are worked out only for the pairs returned. The scores are those of the pair-by-pair definition
        (weighted shared neighbours, weighted co-citers, text), summed exactly (``math.fsum``), so they do not
        depend on the order sets happen to iterate in.
        """
        weight = {node: 1 / math.log(1 + len(around)) for node, around in self.neighbours.items()}
        total = {f: math.fsum(weight[z] for z in self.neighbours.get(f"n:{f}", ())) for f in self.paths}
        citer_weight = {c: 1 / math.log(1 + len(targets)) for c, targets in self.cites.items()}
        cited_parts: dict[int, list[float]] = defaultdict(list)
        for citer, targets in self.cites.items():
            for note in targets:
                cited_parts[note].append(citer_weight[citer])
        cited_total = {note: math.fsum(parts) for note, parts in cited_parts.items()}
        hub_of: dict[int, list[str]] = defaultdict(list)  # note -> its neighbours with more than HUB_LINKS links
        for node, around in self.neighbours.items():
            if len(around) > HUB_LINKS:
                for other in around:
                    if other.startswith("n:"):
                        hub_of[int(other[2:])].append(node)
        text = self.text_similarity()
        needle = unicodedata.normalize("NFC", path).casefold() if path else None
        in_path = (lambda f: needle in self.paths[f].casefold()) if needle else (lambda f: True)

        shift = max(max(self.paths, default=0).bit_length(), 1)  # a pair (x, y) is kept as one int: x << shift | y
        mask = (1 << shift) - 1
        keys = array("q") if 2 * shift <= 62 else []  # plain ints when two ids do not fit in 63 bits
        scores = array("d")
        no_structure: set[int] = set()  # indices of (text) pairs with no shared neighbour and no citer
        linked = self.linked
        w_shared, w_cited, w_text = WEIGHTS["shared"], WEIGHTS["cited"], WEIGHTS["text"]
        sqrt, fsum = math.sqrt, math.fsum
        paths = self.paths
        in_path_ok = {f for f in paths if in_path(f)} if needle else None

        # Per neighbour node: its notes in id order, and (when it is a note) what it cites.
        node_notes: dict[str, list[int]] = {}
        for node, around in self.neighbours.items():
            node_notes[node] = sorted(int(m[2:]) for m in around if m.startswith("n:"))
        node_cites = {f"n:{c}": targets for c, targets in self.cites.items()}
        shared_total = {f: total[f] + 1 for f in paths}
        cited_plus = {f: cited_total.get(f, 0.0) + 1 for f in paths}
        empty: tuple = ()

        def add(parts: dict, b: int, value: float) -> None:
            held = parts.get(b)
            if held is None:
                parts[b] = value
            elif held.__class__ is list:
                held.append(value)
            else:
                parts[b] = [held, value]

        def total_of(held) -> float:
            return held if held.__class__ is float else fsum(held)

        notes = [for_note] if for_note is not None else sorted(paths)
        kept_text = set()
        for a in notes:
            node_a = f"n:{a}"
            around_a = self.neighbours.get(node_a)
            if not around_a or (in_path_ok is not None and a not in in_path_ok):
                continue
            shared_parts: dict[int, object] = {}  # b -> its weight, or the list of them
            cited_here: dict[int, object] = {}
            for z in around_a:
                z_notes = node_notes[z]
                z_cites = node_cites.get(z, empty)
                start_at = bisect.bisect_right(z_notes, a) if for_note is None else 0
                if len(self.neighbours[z]) <= HUB_LINKS:
                    if a in z_cites:
                        cw = citer_weight[int(z[2:])]
                        wz = weight[z]
                        for b in z_notes[start_at:]:
                            if b == a:
                                continue
                            if b in z_cites:
                                add(cited_here, b, cw)
                                if b not in shared_parts:
                                    shared_parts[b] = None
                            else:
                                add(shared_parts, b, wz)
                    else:
                        wz = weight[z]
                        get = shared_parts.get
                        for b in z_notes[start_at:]:
                            if b == a:
                                continue
                            held = get(b)
                            if held is None:
                                shared_parts[b] = wz
                            elif held.__class__ is list:
                                held.append(wz)
                            else:
                                shared_parts[b] = [held, wz]
                elif a in z_cites and len(z_cites) <= HUB_LINKS:
                    cw = citer_weight[int(z[2:])]
                    for b in z_cites:  # a specific citer with many links: candidates through what it cites
                        if b != a and (for_note is not None or b > a):
                            add(cited_here, b, cw)
                            if b not in shared_parts:
                                shared_parts[b] = None
            hubs = hub_of.get(a, empty)
            if hubs:
                for b in list(shared_parts):  # hubs: their evidence counts for every candidate they connect
                    node_b = f"n:{b}"
                    for hub in hubs:
                        if node_b not in self.neighbours[hub]:
                            continue
                        hub_cites = node_cites.get(hub, empty)
                        if a in hub_cites and b in hub_cites:
                            if len(hub_cites) > HUB_LINKS:
                                add(cited_here, b, citer_weight[int(hub[2:])])
                        else:
                            held = shared_parts[b]
                            if held is None:
                                shared_parts[b] = weight[hub]
                            else:
                                add(shared_parts, b, weight[hub])
            linked_a = linked[a]
            for b, held in shared_parts.items():
                x, y = (a, b) if a < b else (b, a)
                if b in linked_a or a in linked[b]:
                    continue
                if in_path_ok is not None and b not in in_path_ok:
                    continue
                shared = total_of(held) / sqrt(shared_total[x] * shared_total[y]) if held is not None else 0.0
                cited_held = cited_here.get(b)
                cited = total_of(cited_held) / sqrt(cited_plus[x] * cited_plus[y]) if cited_held is not None else 0.0
                found_text = text.get((x, y))
                if found_text is not None:
                    kept_text.add((x, y))
                    scores.append(w_shared * shared + w_cited * cited + w_text * found_text[0])
                else:
                    scores.append(w_shared * shared + w_cited * cited + w_text * 0.0)
                keys.append(x << shift | y)
        for (x, y), (cosine, _) in text.items():  # text finds pairs that share nothing specific: their evidence, pair by pair
            if (x, y) in kept_text or (for_note is not None and for_note not in (x, y)):
                continue
            if self.is_linked(x, y) or (in_path_ok is not None and not (x in in_path_ok and y in in_path_ok)):
                continue
            shared_sum, cited_sum = self._pair_evidence(x, y, weight, citer_weight)
            shared = shared_sum / sqrt(shared_total[x] * shared_total[y]) if shared_sum else 0.0
            cited = cited_sum / sqrt(cited_plus[x] * cited_plus[y]) if cited_sum else 0.0
            if not (shared_sum or cited_sum):
                no_structure.add(len(scores))
            keys.append(x << shift | y)
            scores.append(w_shared * shared + w_cited * cited + w_text * cosine)

        self.threshold = min_score
        candidates = range(len(scores))
        if structural_percentile is not None:
            # The nearest-rank percentile of the structural scores: the k-th largest, k = n - rank + 1. Only the
            # scores at or above the floor are ranked (when fewer than k are, the floor is the threshold).
            n = len(scores) - len(no_structure)
            floor = min_score or 0.0
            if n:
                rank = int(max(1, -(-n * structural_percentile // 100)))
                k = n - rank + 1
                above = [scores[i] for i in candidates if scores[i] >= floor and i not in no_structure]
                self.threshold = heapq.nlargest(k, above)[-1] if len(above) >= k else floor
                if min_score is not None:
                    self.threshold = max(min_score, self.threshold)
            candidates = [i for i in candidates if i not in no_structure]
        if self.threshold is not None:
            threshold = self.threshold
            candidates = [i for i in candidates if scores[i] >= threshold]
        if limit is not None and len(candidates) > limit:  # the limit-th best score; ties with it are ranked by path
            cutoff = heapq.nlargest(limit, (scores[i] for i in candidates))[-1]
            candidates = [i for i in candidates if scores[i] >= cutoff]
        key = lambda i: (-scores[i], self.paths[keys[i] >> shift], self.paths[keys[i] & mask])  # noqa: E731
        ranked = sorted(candidates, key=key)[:limit]
        found = []
        for i in ranked:  # the evidence, for what is returned only
            a, b = keys[i] >> shift, keys[i] & mask
            shared_sum, cited_sum = self._pair_evidence(a, b, weight, citer_weight)
            shared_with, cited_by = self._pair_details(a, b, weight, citer_weight)
            x, terms = text.get((a, b), (0.0, []))
            suggestion = LinkSuggestion(
                a=self.paths[a], b=self.paths[b], score=scores[i],
                shared=shared_sum / math.sqrt(shared_total[a] * shared_total[b]) if shared_sum else 0.0,
                cited=cited_sum / math.sqrt(cited_plus[a] * cited_plus[b]) if cited_sum else 0.0, text=x,
                shared_with=shared_with, cited_by=cited_by,
                terms=[t for t in terms if t.isalpha()],  # shown to a person: words (all terms count in the score)
            )
            suggestion.reason = self.reason(suggestion)
            found.append(suggestion)
        return found

    def _citers_of(self) -> dict[int, set[int]]:
        if not hasattr(self, "_citers"):
            citers: dict[int, set[int]] = defaultdict(set)
            for citer, targets in self.cites.items():
                for note in targets:
                    citers[note].add(citer)
            self._citers = citers
        return self._citers

    def _common(self, a: int, b: int) -> tuple[list[str], set[int]]:
        both_cite = self._citers_of()[a] & self._citers_of()[b]
        common = [z for z in self.neighbours.get(f"n:{a}", set()) & self.neighbours.get(f"n:{b}", set())
                  if not (z.startswith("n:") and int(z[2:]) in both_cite)]  # a citer of both counts as "cited together"
        return common, both_cite

    def _pair_evidence(self, a: int, b: int, weight, citer_weight) -> tuple[float, float]:
        common, both_cite = self._common(a, b)
        return math.fsum(weight[z] for z in common), math.fsum(citer_weight[c] for c in both_cite)

    def _pair_details(self, a: int, b: int, weight, citer_weight) -> tuple[list[str], list[str]]:
        common, both_cite = self._common(a, b)
        shared_with = [self._name(n) for _, n in sorted(((weight[z], z) for z in common), key=lambda item: (-item[0], item[1]))]
        cited_by = [link_name(self.paths[n]) for _, n in sorted(((citer_weight[c], c) for c in both_cite),
                                                                 key=lambda item: (-item[0], self.paths[item[1]]))]
        return shared_with, cited_by

    @staticmethod
    def reason(s: LinkSuggestion) -> str:
        parts = []

        def listing(names: list[str], count: int = 2) -> str:
            shown = [n if len(n) <= 45 else n[:44].rstrip() + "…" for n in names[:count]]
            return ", ".join(shown) + (f" +{len(names) - count}" if len(names) > count else "")

        if s.shared_with:
            parts.append("shared neighbours: " + listing(s.shared_with))
        if s.cited_by:
            parts.append("cited together by " + listing(s.cited_by))
        if s.terms:
            parts.append("similar text (" + ", ".join(s.terms) + ")")
        return " · ".join(parts)
