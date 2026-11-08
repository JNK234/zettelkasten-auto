"""Notes that do not exist yet but should: evidence from links, text and tags, and drafts.

Every link to a note that does not exist is a vote for writing it. ``suggest``
ranks those missing notes by how many notes link to them and, for each, says
where the linking notes live, what they are about (the terms they share most,
as in ``related.py``), and how many other notes mention the name without linking
it (a phrase search for the name's words). "About" lists the terms that come
up most among the linking notes' distinctive terms.

It also flags two kinds of cleanup:

* overlapping names: a missing note whose name words (ignoring small words like
  "a", "as", "of", and punctuation inside numbers) contain another missing
  note's words plus at most two more, e.g. ``llm-as-judge`` and
  ``llm-as-a-judge-framework``. That is the same idea under two names, or a
  topic and a narrower one; either way, one note should probably link the other;
* probably an existing note: a missing note whose name words (at least two) are
  all in an existing note's title or alias and make up at least three quarters
  of it, e.g. ``[[on-policy-vs-off-policy-learning]]`` for the existing
  ``on-policy-versus-off-policy-learning``.

``draft`` writes a starting point in this project's note format: the linking
notes with what each says around its link, the terms they share, the notes
``related`` finds for the missing note (so the new note starts connected), and
the linking notes from a sources folder as references. It never invents content:
every line of it comes from the vault.
"""

from __future__ import annotations

import math
import re
import sqlite3
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from vaultengine.parser import normalize_text
from vaultengine.querylang import Phrase, Word
from vaultengine.related import Relatedness
from vaultengine.tokenize import tokenize
from vaultengine.vault import link_name, lookup_key

SMALL_WORDS = {"a", "an", "and", "as", "at", "by", "for", "from", "in", "is", "of", "on", "or", "the", "to", "versus", "vs", "with"}
MAX_EXTRA_WORDS = 2  # a name extending another by up to this many words is grouped with it
ABOUT_SAMPLE = 30  # linking notes read to say what a missing note is about
EXISTING_COVERAGE = 0.75  # share of an existing title's words a link must have to be "probably that note"
_HEADER_LINE = re.compile(r"[ \t]*[A-Za-z][A-Za-z0-9_-]*:(?:[ \t].*)?")
_SPACE = re.compile(r"\s+")


def name_words(name: str) -> frozenset[str]:
    words = set()
    for term in tokenize(name):
        if any(char.isdigit() for char in term):
            term = "".join(char for char in term if char.isalnum())
        if term not in SMALL_WORDS:
            words.add(term)
    return frozenset(words)


@dataclass
class Missing:
    name: str  # the most common spelling of the link target
    key: str
    linking_notes: int
    links: int
    mentions: int  # notes that contain the name as a phrase but do not link it
    folders: list[tuple[str, int]] = field(default_factory=list)
    about: list[str] = field(default_factory=list)


@dataclass
class Suggestions:
    notes: list[Missing]  # missing notes to write, most wanted first
    same_idea: list[list[tuple[str, int]]]  # groups of (missing note, linking notes)
    existing: list[tuple[str, int, str]]  # (missing note, linking notes, existing note path)
    total_missing: int

    def to_dict(self) -> dict:
        return asdict(self)


class Suggester:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn
        self.engine = Relatedness(conn)
        self.spellings: dict[str, Counter[str]] = defaultdict(Counter)
        self.linkers: dict[str, set[int]] = defaultdict(set)
        self.link_count: Counter[str] = Counter()
        rows = conn.execute(
            """SELECT l.target_key, l.target, l.source_id FROM links l JOIN files f ON f.id = l.source_id
               WHERE l.status = 'unresolved' AND f.kind = 'note' AND l.target != ''"""
        )
        for key, target, source in rows:
            if "." in target.rsplit("/", 1)[-1] and not target.lower().endswith(".md") and not re.search(r"\d\.\d", target):
                continue  # a missing attachment (image.png), not a note
            self.spellings[key][target] += 1
            self.linkers[key].add(source)
            self.link_count[key] += 1

    def name(self, key: str) -> str:
        return self.spellings[key].most_common(1)[0][0]

    def mentions(self, key: str) -> int:
        terms = tokenize(self.name(key))
        if not terms:
            return 0
        searcher = self.engine.searcher
        _, matches = searcher.word(Word(terms[0], exact=True)) if len(terms) == 1 else searcher.phrase(Phrase(tuple(terms)))
        return len(set(matches) - self.linkers[key])

    def about(self, key: str, count: int = 6) -> list[str]:
        """Terms that recur among the linking notes' most distinctive terms (``related.py`` profiles).

        At most ``ABOUT_SAMPLE`` linking notes are read (the first by path), which bounds the cost
        for topics linked from hundreds of notes.
        """
        combined: Counter[str] = Counter()
        for linker in sorted(self.linkers[key], key=lambda f: self.engine.paths[f])[:ABOUT_SAMPLE]:
            for term, weight in self.engine._top(self.engine.term_profile(linker)).items():
                combined[term] += weight
        own = set(tokenize(self.name(key)))
        # Shown to a person: words only (no model sizes, versions or formula fragments like 7b, qwen2.5, pi_).
        return [term for term, _ in combined.most_common() if term not in own and term.isalpha()][:count]

    def folders(self, key: str) -> list[tuple[str, int]]:
        counts = Counter(self.engine.paths[f].rsplit("/", 1)[0] if "/" in self.engine.paths[f] else "" for f in self.linkers[key])
        return sorted(counts.items(), key=lambda item: (-item[1], item[0]))

    def suggestions(self, limit: int | None = 15, min_links: int = 2) -> Suggestions:
        keys = [k for k, notes in self.linkers.items() if len(notes) >= min_links]
        keys.sort(key=lambda k: (-len(self.linkers[k]), -self.link_count[k], k))
        shown = keys if limit is None else keys[:limit]
        notes = [
            Missing(self.name(k), k, len(self.linkers[k]), self.link_count[k], self.mentions(k), self.folders(k)[:3], self.about(k))
            for k in shown
        ]
        return Suggestions(notes, self.same_idea(), self.close_to_existing(min_links), len(self.linkers))

    def same_idea(self) -> list[list[tuple[str, int]]]:
        """Groups around a core name (2+ words): missing notes whose names add at most two words to it.

        Not transitive: "evaluation-methodology" and "llm-as-judge" do not end up in one group
        just because "llm-as-judge-evaluation-methodology" contains both. A name joins the core
        it shares the most words with (then the most linked core).
        """
        words = {k: name_words(self.name(k)) for k in self.linkers}
        cores = sorted((k for k in words if len(words[k]) >= 2), key=lambda k: (-len(self.linkers[k]), k))
        groups: dict[str, list[str]] = defaultdict(list)
        for key in words:
            best = None
            for core in cores:
                if core == key or not words[core] <= words[key] or len(words[key] - words[core]) > MAX_EXTRA_WORDS:
                    continue
                # Same words spelled differently: the less linked spelling joins the other, not both ways.
                if words[core] == words[key] and (len(self.linkers[core]), key) < (len(self.linkers[key]), core):
                    continue
                candidate = (len(words[core]), len(self.linkers[core]))
                if best is None or candidate > best[0]:
                    best = (candidate, core)
            if best is not None:
                groups[best[1]].append(key)
        result = []
        for core, members in groups.items():
            members.sort(key=lambda k: (-len(self.linkers[k]), k))
            result.append([(self.name(k), len(self.linkers[k])) for k in [core, *members]])
        result.sort(key=lambda group: (-sum(n for _, n in group), group[0][0]))
        return result

    def close_to_existing(self, min_links: int) -> list[tuple[str, int, str]]:
        titles: dict[int, set[frozenset[str]]] = defaultdict(set)
        for file_id, path in self.engine.paths.items():
            titles[file_id].add(name_words(link_name(path)))
        for file_id, alias in self.conn.execute("SELECT file_id, alias FROM aliases"):
            titles[file_id].add(name_words(alias))
        by_word: dict[str, set[int]] = defaultdict(set)
        for file_id, names in titles.items():
            for words in names:
                for word in words:
                    by_word[word].add(file_id)
        found = []
        for key, linkers in self.linkers.items():
            words = name_words(self.name(key))
            if len(linkers) < min_links or len(words) < 2:
                continue
            candidates = set.intersection(*(by_word.get(word, set()) for word in words))
            matches = [file_id for file_id in candidates
                       if any(words <= n and len(words) >= EXISTING_COVERAGE * len(n) for n in titles[file_id])]
            if len(matches) == 1:
                found.append((self.name(key), len(linkers), self.engine.paths[matches[0]]))
        found.sort(key=lambda item: (-item[1], item[0]))
        return found

    # -- drafts --

    def resolve_missing(self, target: str) -> str:
        key = lookup_key(target)
        if key not in self.linkers:
            raise KeyError(target)
        return key

    def contexts(self, key: str, limit: int = 8) -> list[tuple[str, str]]:
        """(linking note path, what it says around its first link in prose), best first."""
        seed = self.engine.seed(self.name(key))
        order = {hit.path: i for i, hit in enumerate(self.engine.related(seed, limit=None).hits)}
        rows = self.conn.execute(
            """SELECT f.path, l.start_offset, l.end_offset FROM links l JOIN files f ON f.id = l.source_id
               WHERE l.status = 'unresolved' AND l.target_key = ? ORDER BY f.path, l.start_offset""",
            (key,),
        )
        per_note: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for path, start, end in rows:
            per_note[path].append((start, end))
        found = []
        root = Path(self.conn.execute("SELECT value FROM meta WHERE key = 'vault_root'").fetchone()[0])
        for path in sorted(per_note, key=lambda p: (order.get(p, len(order)), p)):
            try:
                text = normalize_text((root / path).read_bytes().decode("utf-8", errors="replace"))
            except OSError:
                continue
            for start, end in per_note[path]:
                line_start = text.rfind("\n", 0, start) + 1
                line_end = text.find("\n", end)
                line = text[line_start : len(text) if line_end == -1 else line_end]
                if _HEADER_LINE.fullmatch(line) or "::" in line:
                    continue  # a Tags:/Status: line or a field list: no sentence to quote
                found.append((path, _quote(text, start, end, line_start, line_start + len(line))))
                break
        return found[:limit]

    def draft(self, target: str, now: datetime | None = None) -> tuple[str, str]:
        """(suggested file path, note text) for a missing note."""
        key = self.resolve_missing(target)
        name = self.name(key)
        paths = self.engine.paths
        linkers = sorted(paths[f] for f in self.linkers[key])
        # The other link targets most linking notes share become the note's Tags: links.
        shared = Counter()
        for file_id in self.linkers[key]:
            targets = {t for (t,) in self.conn.execute("SELECT DISTINCT target FROM links WHERE source_id = ? AND target != ''", (file_id,))}
            shared.update(t for t in targets if lookup_key(t) != key)
        tags = [t for t, n in shared.most_common() if n >= max(2, 0.3 * len(linkers))][:4]
        related = [h for h in self.engine.related(self.engine.seed(name), limit=None).hits if not h.linked][:5]
        sources = [p for p in linkers if "source" in p.split("/")[0].lower()]
        when = (now or datetime.now()).strftime("%d-%m-%Y %H:%M")

        lines = [f"Created: {when}", "", "Tags: " + " ".join(f"[[{t}]]" for t in tags), "", f"# {name}", ""]
        lines.append(f"> Drafted by `vault suggest` from the {len(linkers)} notes that link here. "
                     "Every line below comes from them: rewrite it in your own words, then delete this line.")
        lines += ["", "## What the linking notes say", ""]
        contexts = self.contexts(key)
        for path, quote in contexts:
            lines.append(f"- [[{link_name(path)}]]: {quote}")
        quoted = {path for path, _ in contexts}
        rest = [p for p in linkers if p not in quoted]
        if rest:
            lines.append(f"- also linked from: " + ", ".join(f"[[{link_name(p)}]]" for p in rest))
        lines += ["", "## Key terms", "", ", ".join(self.about(key, 10)), "", "# See Also", ""]
        lines += [f"- [[{h.title}]]: {h.reason}" for h in related] or ["- (nothing beyond the linking notes)"]
        lines += ["", "# References", ""]
        lines += [f"{i}. [[{link_name(p)}]]" for i, p in enumerate(sources, 1)]
        folder = Counter(p.rsplit("/", 1)[0] if "/" in p else "" for p in linkers).most_common(1)[0][0]
        file_name = name if name.lower().endswith(".md") else name + ".md"
        path = file_name if "/" in name else (f"{folder}/{file_name}" if folder else file_name)
        return unicodedata.normalize("NFC", path), "\n".join(lines).rstrip() + "\n"


def _quote(text: str, start: int, end: int, line_start: int, line_end: int, room: int = 110) -> str:
    """The sentence piece around text[start:end] within its line, cut at word boundaries."""
    left = max(line_start, start - room)
    right = min(line_end, end + room)
    if left > line_start:
        left = text.find(" ", left, start) + 1 or left
    if right < line_end:
        cut = text.rfind(" ", end, right)
        right = cut if cut > end else right
    piece = _SPACE.sub(" ", text[left:right].strip())
    return ("… " if left > line_start else "") + f'"{piece}"' + (" …" if right < line_end else "")
