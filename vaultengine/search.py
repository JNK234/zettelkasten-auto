"""Full-text search over the SQLite index: fields, postings, query language, ranking.

Indexing
--------
Every note is split into five fields, and each word occurrence is counted in
exactly one of them:

* ``title``    - the file name without ``.md`` (what ``[[links]]`` use);
* ``aliases``  - the ``aliases`` frontmatter values;
* ``headings`` - the text of every ``#`` heading;
* ``tags``     - each distinct tag name (frontmatter and inline), ``#ai-safety`` -> ``ai``, ``safety``;
* ``body``     - everything else after the frontmatter, minus the heading lines, the
                 inline ``#tags``, the ``^block-id`` markers and the keys of header and
                 inline fields (``Status:``, ``supports::``; their values stay). Link text
                 counts as written, so ``[[reward-model|RM]]`` gives ``reward``, ``model``, ``rm``.

Text inside fenced code, inline code and ``%% comments %%`` is not indexed in any
field. Other frontmatter properties are not indexed.
Terms come from ``vaultengine.tokenize`` (UAX #29 words, lowercased, no stemming).

Each (term, note, field) is one row of ``postings`` with the term frequency and
the token positions of every occurrence. A field is a list of values: one per
alias, heading or tag, and one per stretch of body text between two headings.
Consecutive values are ``VALUE_GAP`` positions apart, so neither a phrase nor a
NEAR can match across two headings, two aliases, or a heading that interrupts
the body. ``field_lengths`` holds the token count of each non-empty field, and
``terms`` the vocabulary with document frequencies (for spelling variants).
All three are rewritten for a note only when that note is re-parsed.

Query language
--------------
Parsed by ``querylang.py`` (syntax, errors) and evaluated here: every part of
the tree is a set of notes. Words forgive spelling mistakes (see ``spelling.py``:
1 edit for 4-7 characters, 2 for longer words, none for shorter words or words
containing digits), except under a negation. Phrases, quoted words and NEAR
pairs are exact. Filters (tag:, path:, [property], links-to:, linked-from:,
within:N of:, has:, missing:, modified:) narrow the set and never change a score;
the words, phrases and NEAR pairs that are not negated rank it.

Ranking
-------
Every clause (word, quoted word, phrase, NEAR pair) is scored with BM25F
(Robertson, Zaragoza & Taylor 2004; "simple BM25F"):

    tf~(c, d) = sum over fields f of  w_f * tf(c, d, f) / ((1 - b) + b * len(d, f) / avglen(f))
    score(c, d) = idf(c) * tf~(c, d) * (k1 + 1) / (k1 + tf~(c, d))
    idf(c)    = ln(1 + (N - df(c) + 0.5) / (df(c) + 0.5))

* w: title 3, headings 2, aliases 2, tags 2, body 1.  k1 = 1.2, b = 0.75 for every field.
* N is the number of notes; df(c) the number of notes where c matches in any field.
  For a phrase, tf counts its occurrences; for a NEAR pair, the pairs of occurrences
  within the distance.
* avglen(f) is the mean length of field f over the notes where f is not empty
  (Lucene's convention); averaging over all notes would drown alias matches.
* (k1 + 1) is the classic Robertson scaling; it does not change the order.
* A word that occurs exactly in a note is scored on its exact occurrences only.
  Otherwise its best spelling variant in the note counts, times ``FUZZY_WEIGHT``
  per edit, with the idf of the most common form among the word and its variants
  (a typo that occurs in one note must not look more informative than the word).

Notes are ordered by the number of text clauses they match exactly, then by
the sum of their clause scores. So a note that contains a word as typed always
ranks above one that only has a near miss of it, and among notes matching the
same number of clauses exactly, BM25F decides. Notes that match the filters but
none of the words come last; without any words, results are ordered by path.
Every result set comes with counts of its tags and folders (``Facets``).
"""

from __future__ import annotations

import bisect
import json
import math
import posixpath
import sqlite3
import unicodedata
from collections import Counter, deque
from datetime import datetime
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from vaultengine.model import ParsedNote
from vaultengine.parser import NoteLayout
from vaultengine.properties import PROPERTIES_VERSION, body_fields
from vaultengine.querylang import (
    And, Has, LinkedFrom, LinksTo, Modified, Near, Node, Not, Or, Path, Phrase, Prop, QueryError, Tag, Within, Word,
    parse, text_clauses,
)
from vaultengine import properties as props
from vaultengine.spelling import close_terms, edit_budget
from vaultengine.tokenize import TOKENIZER_VERSION, tokenize
from vaultengine.vault import link_name, lookup_key

FIELDS = ("title", "aliases", "headings", "tags", "body")
FIELD_WEIGHTS = {"title": 3.0, "aliases": 2.0, "headings": 2.0, "tags": 2.0, "body": 1.0}
K1 = 1.2
B = 0.75
VALUE_GAP = 100
FUZZY_WEIGHT = 0.5  # per edit: a one-edit variant counts half, a two-edit variant a quarter

# Bump when field extraction changes. Stored in the index meta together with
# TOKENIZER_VERSION; a different value makes the next `vault index` re-read every note.
# 2: body text is split into values at headings. 3: property keys are not body text.
FIELDS_VERSION = 3
SEARCH_VERSION = f"{FIELDS_VERSION}.{TOKENIZER_VERSION}.{PROPERTIES_VERSION}"


# ----- indexing --------------------------------------------------------------


def _blank(text: str, spans: list[tuple[int, int]]) -> str:
    """``text`` with each span replaced by spaces (same length, and a word boundary)."""
    if not spans:
        return text
    parts, position = [], 0
    for start, end in sorted(spans):
        start = max(start, position)  # overlapping spans
        if end <= start:
            continue
        parts.append(text[position:start])
        parts.append(" " * (end - start))
        position = end
    parts.append(text[position:])
    return "".join(parts)


def field_segments(path: str, note: ParsedNote, layout: NoteLayout) -> dict[str, list[tuple[str, int | None]]]:
    """The values of each field as (text, offset in ``layout.text``), offset None when not in the text.

    Tokenising each value gives the field's terms; the offsets let snippets point back into the note.
    """
    # Same length as layout.text, so every offset below is valid in the note text.
    # Property keys ("Status:", "supports::") are structure, not text; their values stay.
    keys = [f.key_span for f in body_fields(layout.text, layout.masked)]
    masked = _blank(layout.masked, layout.inline_tags + layout.block_ids + keys)
    body: list[tuple[str, int | None]] = []
    position = 0
    for heading in layout.headings:
        body.append((masked[position : heading.start], position))
        position = heading.end
    body.append((masked[position:], position))
    return {
        "title": [(link_name(path), None)],
        "aliases": [(alias, None) for alias in note.aliases],
        # Closing '#'s of a heading are punctuation, which the tokenizer skips anyway.
        "headings": [(masked[h.text_start : h.text_end], h.text_start) for h in layout.headings],
        "tags": [(name, None) for name in note.tag_names],
        "body": body,
    }


def note_fields(path: str, note: ParsedNote, layout: NoteLayout) -> dict[str, list[list[str]]]:
    """Terms of each field, as one list per value."""
    return {name: [tokenize(text) for text, _ in values] for name, values in field_segments(path, note, layout).items()}


def postings(fields: dict[str, list[list[str]]]) -> tuple[dict[tuple[str, str], list[int]], dict[str, int]]:
    """({(term, field): positions}, {field: length}) for the non-empty fields."""
    positions: dict[tuple[str, str], list[int]] = {}
    lengths: dict[str, int] = {}
    for name, values in fields.items():
        position = 0
        for value in values:
            for term in value:
                positions.setdefault((term, name), []).append(position)
                position += 1
            if value:  # values without words (an empty stretch of body) add no gap
                position += VALUE_GAP
        length = sum(len(value) for value in values)
        if length:
            lengths[name] = length
    return positions, lengths


# ----- query language --------------------------------------------------------
# The syntax tree and parser are in querylang.py; these names stay importable from here.

Clause = Word | Phrase | Near


def parse_query(query: str) -> list[Clause]:
    """The words, phrases and NEAR pairs of ``query`` that rank results (not negated), in order."""
    return text_clauses(parse(query))


# ----- ranking -----------------------------------------------------------------


def idf(n_notes: int, df: int) -> float:
    return math.log(1 + (n_notes - df + 0.5) / (df + 0.5))


@dataclass
class FieldMatch:
    tf: int
    length: int
    avg_length: float
    weighted_tf: float  # w_f * tf / ((1 - b) + b * length / avg_length)


@dataclass
class Match:
    """How one clause matched one note."""

    clause: str  # the clause label
    exact: bool
    term: str | None  # for words: the indexed term that matched (a variant when not exact)
    edits: int  # 0 unless a spelling variant matched
    weight: float  # FUZZY_WEIGHT ** edits
    tf: float  # tf~: sum of the fields' weighted_tf
    score: float  # weight * idf * tf~ * (k1 + 1) / (k1 + tf~)
    fields: dict[str, FieldMatch] = field(default_factory=dict)


@dataclass
class Hit:
    path: str
    title: str
    exact: int  # clauses matched exactly (the first sort key)
    score: float
    matches: dict[str, Match] = field(default_factory=dict)  # by clause label, in query order


@dataclass
class ClauseStats:
    kind: str  # "word" | "phrase" | "near"
    required: bool
    df: int  # notes matching exactly
    idf: float
    variants: dict[str, int] = field(default_factory=dict)  # spelling variants in the index: term -> edits
    variant_df: dict[str, int] = field(default_factory=dict)
    fuzzy_idf: float | None = None  # idf used for variant matches


@dataclass
class SearchResult:
    query: str
    clauses: dict[str, ClauseStats]
    notes: int  # N
    avg_lengths: dict[str, float]
    total: int  # matching notes
    hits: list[Hit]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _bm25f(fields: dict[str, tuple[int, int]], avg_lengths: dict[str, float], idf_value: float) -> tuple[float, float, dict[str, FieldMatch]]:
    """(tf~, score, per-field detail) for {field: (tf, field length)}."""
    detail = {}
    for name in FIELDS:
        if name in fields:
            tf, length = fields[name]
            avg = avg_lengths[name]
            detail[name] = FieldMatch(tf, length, avg, FIELD_WEIGHTS[name] * tf / ((1 - B) + B * length / avg))
    total = sum(f.weighted_tf for f in detail.values())
    return total, idf_value * total * (K1 + 1) / (K1 + total), detail


class _Searcher:
    def __init__(self, conn: sqlite3.Connection, fuzzy: bool) -> None:
        self.conn = conn
        self.fuzzy = fuzzy
        self.n_notes = conn.execute("SELECT COUNT(*) FROM files WHERE kind = 'note'").fetchone()[0]
        self.avg_lengths = {
            row[0]: row[1] / row[2]
            for row in conn.execute("SELECT field, SUM(length), COUNT(*) FROM field_lengths GROUP BY field")
        }
        self.paths: dict[int, str] = {}
        self._vocabulary: list[str] | None = None

    def vocabulary(self) -> list[str]:
        if self._vocabulary is None:
            self._vocabulary = [row[0] for row in self.conn.execute("SELECT term FROM terms ORDER BY term")]
        return self._vocabulary

    def df(self, terms: Iterable[str]) -> dict[str, int]:
        terms = list(terms)
        found = dict.fromkeys(terms, 0)
        marks = ",".join("?" * len(terms))
        found.update(self.conn.execute(f"SELECT term, df FROM terms WHERE term IN ({marks})", terms).fetchall())
        return found

    def rows(self, terms: Iterable[str], positions: bool = False) -> list[tuple]:
        """(file_id, term, field, tf, field length[, positions]) for every posting of ``terms``."""
        terms = sorted(set(terms))
        marks = ",".join("?" * len(terms))
        extra = ", p.positions" if positions else ""
        result = []
        for row in self.conn.execute(
            f"""SELECT p.file_id, f.path, p.term, p.field, p.tf, l.length{extra}
                FROM postings p
                JOIN field_lengths l ON l.file_id = p.file_id AND l.field = p.field
                JOIN files f ON f.id = p.file_id
                WHERE p.term IN ({marks})""",
            terms,
        ):
            self.paths[row[0]] = row[1]
            result.append((row[0], *row[2:]))
        return result

    # -- clauses --

    def word(self, clause: Word) -> tuple[ClauseStats, dict[int, Match]]:
        budget = 0 if clause.exact or not self.fuzzy else edit_budget(clause.term)
        variants = close_terms(clause.term, budget, self.vocabulary()) if budget else {}
        variants.pop(clause.term, None)
        dfs = self.df([clause.term, *variants])
        stats = ClauseStats("word", clause.required, dfs[clause.term], idf(self.n_notes, dfs[clause.term]))
        if variants:
            stats.variants = dict(sorted(variants.items(), key=lambda item: (item[1], -dfs[item[0]], item[0])))
            stats.variant_df = {term: dfs[term] for term in stats.variants}
            stats.fuzzy_idf = idf(self.n_notes, max(dfs.values()))
        by_note: dict[int, dict[str, dict[str, tuple[int, int]]]] = {}
        for file_id, term, name, tf, length in self.rows([clause.term, *variants]):
            by_note.setdefault(file_id, {}).setdefault(term, {})[name] = (tf, length)
        matches = {}
        for file_id, terms in by_note.items():
            if clause.term in terms:
                tf, score, detail = _bm25f(terms[clause.term], self.avg_lengths, stats.idf)
                matches[file_id] = Match(clause.label, True, clause.term, 0, 1.0, tf, score, detail)
                continue
            best = None
            for term, fields in terms.items():
                edits = variants[term]
                weight = FUZZY_WEIGHT**edits
                tf, score, detail = _bm25f(fields, self.avg_lengths, stats.fuzzy_idf)
                candidate = Match(clause.label, False, term, edits, weight, tf, weight * score, detail)
                if best is None or (candidate.score, -edits, term) > (best.score, -best.edits, best.term):
                    best = candidate
            matches[file_id] = best
        return stats, matches

    def _positional(self, label: str, kind: str, terms: list[str], count) -> tuple[ClauseStats, dict[int, Match]]:
        """Shared by phrases and NEAR: ``count(positions by term)`` gives occurrences in one field."""
        occurrences: dict[int, dict[str, tuple[int, int]]] = {}
        if all(self.df(terms).values()):
            grouped: dict[tuple[int, str], dict[str, list[int]]] = {}
            lengths: dict[tuple[int, str], int] = {}
            for file_id, term, name, _tf, length, positions in self.rows(terms, positions=True):
                grouped.setdefault((file_id, name), {})[term] = [int(p) for p in positions[1:-1].split(",")]
                lengths[(file_id, name)] = length
            for (file_id, name), by_term in grouped.items():
                if len(by_term) == len(set(terms)):
                    n = count(by_term)
                    if n:
                        occurrences.setdefault(file_id, {})[name] = (n, lengths[(file_id, name)])
        stats = ClauseStats(kind, True, len(occurrences), idf(self.n_notes, len(occurrences)))
        matches = {}
        for file_id, fields in occurrences.items():
            tf, score, detail = _bm25f(fields, self.avg_lengths, stats.idf)
            matches[file_id] = Match(label, True, None, 0, 1.0, tf, score, detail)
        return stats, matches

    def phrase(self, clause: Phrase) -> tuple[ClauseStats, dict[int, Match]]:
        def count(by_term: dict[str, list[int]]) -> int:
            later = [set(by_term[term]) for term in clause.terms[1:]]
            return sum(all(start + i + 1 in positions for i, positions in enumerate(later)) for start in by_term[clause.terms[0]])

        return self._positional(clause.label, "phrase", list(clause.terms), count)

    def near(self, clause: Near) -> tuple[ClauseStats, dict[int, Match]]:
        def count(by_term: dict[str, list[int]]) -> int:
            return near_pairs(by_term[clause.left], by_term[clause.right], clause.distance, same=clause.left == clause.right)

        return self._positional(clause.label, "near", [clause.left, clause.right], count)


def near_pairs(left: list[int], right: list[int], distance: int, same: bool = False) -> int:
    """Pairs (l, r) of positions with |l - r| <= distance; for the same word, pairs of distinct occurrences."""
    right = sorted(right)
    pairs = sum(bisect.bisect_right(right, p + distance) - bisect.bisect_left(right, p - distance) for p in left)
    return (pairs - len(left)) // 2 if same else pairs


@dataclass
class Facets:
    tags: list[tuple[str, int]]  # (tag, notes), nested tags also counted under every parent
    folders: list[tuple[str, int]]  # (folder, notes), every ancestor folder; "" is the vault root


class _Evaluator:
    """Turns a syntax tree into the set of matching note ids."""

    def __init__(self, conn: sqlite3.Connection, searcher: "_Searcher") -> None:
        self.conn = conn
        self.searcher = searcher
        self.notes = {row[0]: row[1] for row in conn.execute("SELECT id, path FROM files WHERE kind = 'note'")}
        self.universe = set(self.notes)
        self.warnings: list[str] = []
        self.text: dict[str, tuple[ClauseStats, dict[int, Match]]] = {}
        self._files: list[tuple] | None = None
        self._properties: dict[str, list[tuple[int, str]]] | None = None
        self._graph: dict[int, set[int]] | None = None

    # -- text --

    def text_matches(self, clause: Word | Phrase | Near) -> tuple[ClauseStats, dict[int, Match]]:
        if clause.label not in self.text:
            if isinstance(clause, Word):
                self.text[clause.label] = self.searcher.word(clause)
            elif isinstance(clause, Phrase):
                self.text[clause.label] = self.searcher.phrase(clause)
            else:
                self.text[clause.label] = self.searcher.near(clause)
        return self.text[clause.label]

    # -- tree --

    def eval(self, node: Node, positive: bool = True) -> set[int]:
        if isinstance(node, Word):
            if not positive and not node.exact:  # negated words match exactly
                node = Word(node.term, True, node.span)
            return set(self.text_matches(node)[1])
        if isinstance(node, (Phrase, Near)):
            return set(self.text_matches(node)[1])
        if isinstance(node, And):
            result = set(self.universe)
            for child in node.required:
                result &= self.eval(child, positive)
            if node.any:
                result &= set().union(*(self.eval(word, positive) for word in node.any))
            return result
        if isinstance(node, Or):
            return set().union(*(self.eval(child, positive) for child in node.items))
        if isinstance(node, Not):
            return self.universe - self.eval(node.item, not positive)
        return getattr(self, "_" + type(node).__name__.lower())(node) & self.universe

    # -- filters --

    def _tag(self, node: Tag) -> set[int]:
        key = lookup_key(node.name)
        rows = self.conn.execute(
            "SELECT DISTINCT file_id FROM tags WHERE name_key = ? OR substr(name_key, 1, ?) = ?",
            (key, len(key) + 1, key + "/"),
        )
        return {row[0] for row in rows}

    def _path(self, node: Path) -> set[int]:
        needle = unicodedata.normalize("NFC", node.text).casefold()
        return {file_id for file_id, path in self.notes.items() if needle in path.casefold()}

    def properties(self) -> dict[str, list[tuple[int, str]]]:
        if self._properties is None:
            self._properties = {}
            for file_id, key, value_json in self.conn.execute("SELECT file_id, key, value_json FROM properties"):
                self._properties.setdefault(lookup_key(key), []).append((file_id, value_json))
        return self._properties

    def _prop(self, node: Prop) -> set[int]:
        rows = self.properties().get(lookup_key(node.key), [])
        if node.op is None:
            return {file_id for file_id, _ in rows}
        return {file_id for file_id, value_json in rows if _property_matches(node, value_json)}

    def _has(self, node: Has) -> set[int]:
        sql = {
            "tags": "SELECT DISTINCT file_id FROM tags",
            "aliases": "SELECT DISTINCT file_id FROM aliases",
            "headings": "SELECT DISTINCT file_id FROM headings",
            "links": "SELECT DISTINCT l.source_id FROM links l JOIN files t ON t.id = l.target_id "
                     "WHERE l.status = 'resolved' AND t.kind = 'note' AND l.target_id != l.source_id",
            "backlinks": "SELECT DISTINCT target_id FROM links WHERE status = 'resolved' AND target_id != source_id",
            "unresolved": "SELECT DISTINCT source_id FROM links WHERE status = 'unresolved'",
            "frontmatter": "SELECT file_id FROM notes WHERE has_frontmatter",
            "properties": "SELECT DISTINCT file_id FROM properties",
        }[node.what]
        return {row[0] for row in self.conn.execute(sql)}

    def _modified(self, node: Modified) -> set[int]:
        found = set()
        for file_id, modified_ns in self.conn.execute("SELECT id, modified_ns FROM files WHERE kind = 'note'"):
            when = datetime.fromtimestamp(modified_ns / 1e9)
            if node.op == "range":
                ok = node.start.start <= when < node.end.end
            else:
                ok = props.compare(node.op, when, node.start)
            if ok:
                found.add(file_id)
        return found

    def resolve(self, ref: str, span: tuple[int, int]) -> int | None:
        """The file a [[reference]] points to, like a link would; None if there is none."""
        if self._files is None:
            self._files = [(row[0], row[1], lookup_key(row[1]), row[2]) for row in self.conn.execute("SELECT id, path, name_key FROM files")]
        key = lookup_key(ref)
        with_md = key if key.endswith(".md") else key + ".md"
        levels = [
            [f for f in self._files if f[2] in (key, with_md)],
            [f for f in self._files if f[2].endswith("/" + key) or f[2].endswith("/" + with_md)],
            [f for f in self._files if f[3] == key],
        ]
        aliases = {row[0] for row in self.conn.execute("SELECT file_id FROM aliases WHERE alias_key = ?", (key,))}
        levels.append([f for f in self._files if f[0] in aliases])
        for candidates in levels:
            if len(candidates) == 1:
                return candidates[0][0]
            if candidates:
                paths = sorted(f[1] for f in candidates)
                raise QueryError(
                    f"[[{ref}]] could be {len(paths)} notes: " + ", ".join(paths), *span,
                    f"write the path, e.g. [[{paths[0][:-3] if paths[0].endswith('.md') else paths[0]}]]",
                )
        return None

    def _unresolved_sources(self, ref: str) -> set[int]:
        key = lookup_key(ref)
        name_key = lookup_key(posixpath.basename(ref))
        rows = self.conn.execute(
            "SELECT DISTINCT source_id FROM links WHERE status != 'resolved' AND (target_key = ? OR target_name_key = ?)",
            (key, name_key),
        )
        found = {row[0] for row in rows}
        self.warnings.append(
            f"[[{ref}]] is not a file in the vault; using the links written to it ({len(found)} notes)"
            if found else f"[[{ref}]] is not a file in the vault, and no note links to it"
        )
        return found

    def _linksto(self, node: LinksTo) -> set[int]:
        target = self.resolve(node.ref, node.span)
        if target is None:
            return self._unresolved_sources(node.ref)
        rows = self.conn.execute("SELECT DISTINCT source_id FROM links WHERE target_id = ? AND source_id != ?", (target, target))
        return {row[0] for row in rows}

    def _linkedfrom(self, node: LinkedFrom) -> set[int]:
        source = self.resolve(node.ref, node.span)
        if source is None or source not in self.notes:
            raise QueryError(f"[[{node.ref}]] is not a note in the vault", *node.span, "linked-from: needs an existing note")
        rows = self.conn.execute(
            "SELECT DISTINCT target_id FROM links WHERE source_id = ? AND status = 'resolved' AND target_id != ?", (source, source)
        )
        return {row[0] for row in rows}

    def graph(self) -> dict[int, set[int]]:
        """Notes and the resolved links between them, in both directions."""
        if self._graph is None:
            self._graph = {file_id: set() for file_id in self.notes}
            rows = self.conn.execute(
                """SELECT DISTINCT l.source_id, l.target_id FROM links l JOIN files t ON t.id = l.target_id
                   WHERE l.status = 'resolved' AND t.kind = 'note' AND l.source_id != l.target_id"""
            )
            for a, b in rows:
                self._graph[a].add(b)
                self._graph[b].add(a)
        return self._graph

    def _within(self, node: Within) -> set[int]:
        origin = self.resolve(node.ref, node.span)
        graph = self.graph()
        if origin in self.notes:
            distance = {origin: 0}
            frontier = deque([origin])
        else:  # a missing note or an attachment: the notes linking to it are one hop away
            if origin is None:
                first = self._unresolved_sources(node.ref)
            else:
                first = {row[0] for row in self.conn.execute("SELECT DISTINCT source_id FROM links WHERE target_id = ?", (origin,))}
            distance = {file_id: 1 for file_id in first}
            frontier = deque(first)
        while frontier:
            current = frontier.popleft()
            if distance[current] == node.hops:
                continue
            for neighbour in graph[current]:
                if neighbour not in distance:
                    distance[neighbour] = distance[current] + 1
                    frontier.append(neighbour)
        return {file_id for file_id, d in distance.items() if d >= 1}

    # -- facets --

    def facets(self, ids: set[int], top: int | None) -> Facets:
        tags: Counter[str] = Counter()
        spelling: dict[str, str] = {}
        per_note: dict[int, set[str]] = {}
        for file_id, name, name_key in self.conn.execute("SELECT file_id, name, name_key FROM tags"):
            if file_id in ids:
                parts = name_key.split("/")
                written = name.split("/")
                for depth in range(1, len(parts) + 1):
                    key = "/".join(parts[:depth])
                    spelling.setdefault(key, "/".join(written[:depth]))
                    per_note.setdefault(file_id, set()).add(key)
        for keys in per_note.values():
            tags.update(keys)
        folders: Counter[str] = Counter()
        for file_id in ids:
            parts = self.notes[file_id].split("/")[:-1]
            folders.update("/".join(parts[:depth]) for depth in range(1, len(parts) + 1))
            if not parts:
                folders[""] += 1

        def ordered(counter: Counter, names=lambda k: k):
            rows = sorted(((names(k), n) for k, n in counter.items()), key=lambda item: (-item[1], item[0]))
            return rows if top is None else rows[:top]

        return Facets(ordered(tags, lambda k: spelling[k]), ordered(folders))


def _property_matches(node: Prop, value_json: str) -> bool:
    """Does one property value satisfy ``[key op value]``? (See properties.py.)"""
    texts = props.value_texts(value_json)
    for text in texts:
        when = props.parse_date(text) if node.period is not None else None
        if when is not None:
            if isinstance(node.period, tuple):
                if node.period[0].start <= when < node.period[1].end:
                    return True
            elif props.compare(node.op if node.op in (">", ">=", "<", "<=") else "=", when, node.period):
                return True
            continue
        value = props.number(text) if node.number is not None else None
        if value is not None and props.compare(node.op if node.op != ":" else "=", value, node.number):
            return True
    if node.op == ":":
        needle = node.value.casefold()
        return any(needle in text.casefold() for text in texts)
    if node.op == "=":
        return props.normalize_item(node.value) in props.items(value_json)
    return False


@dataclass
class QueryResult(SearchResult):
    tree: str = ""
    warnings: list[str] = field(default_factory=list)
    facets: Facets | None = None


def search(
    conn: sqlite3.Connection, query: str, limit: int | None = 10, fuzzy: bool = True, facets: int | None = 10
) -> QueryResult:
    """Run ``query`` (see querylang.py): filters narrow, text ranks. Raises QueryError."""
    tree = parse(query)
    searcher = _Searcher(conn, fuzzy)
    if tree is None:
        return QueryResult(query, {}, searcher.n_notes, searcher.avg_lengths, 0, [], "", [], Facets([], []))
    evaluator = _Evaluator(conn, searcher)
    matched = evaluator.eval(tree)

    clauses = text_clauses(tree)
    stats: dict[str, ClauseStats] = {}
    per_clause = []
    for clause in clauses:
        stats[clause.label], matches = evaluator.text_matches(clause)
        per_clause.append((clause, matches))
    hits = []
    for file_id in matched:
        found = {clause.label: matches[file_id] for clause, matches in per_clause if file_id in matches}
        path = evaluator.notes[file_id]
        hits.append(
            Hit(
                path=path,
                title=link_name(path),
                exact=sum(m.exact for m in found.values()),
                score=sum(m.score for m in found.values()),
                matches=found,
            )
        )
    if clauses:
        hits.sort(key=lambda hit: (-hit.exact, -hit.score, hit.path))
    else:
        hits.sort(key=lambda hit: hit.path)
    return QueryResult(
        query, stats, searcher.n_notes, searcher.avg_lengths, len(hits), hits if limit is None else hits[:limit],
        tree.sexpr(), evaluator.warnings, evaluator.facets(matched, facets),
    )
