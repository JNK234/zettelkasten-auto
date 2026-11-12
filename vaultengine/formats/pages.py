"""`vault page`: pages (the format's pipeline ``compile`` type) made from zettels.

A page the engine manages has two generated blocks between HTML-comment markers,
and everything else in it is yours::

    Created: …
    Tags: …

    # Title

    <!-- vault:zettels -->
    ## Zettels
    ### <tag>
    - [[zettel]]: its first line
    <!-- /vault:zettels -->

    <!-- vault:sources -->
    ## Sources
    - [[source]] (zettel, zettel)
    <!-- /vault:sources -->

    (your text)

* **The zettels on a page** are the zettels it links to now, anywhere in it
  (``TypedVault.note_links``): the block's own lines, and any zettel you link in your
  text. To drop one, delete its line (and your links to it); to add one, link it and
  refresh. Excerpts in the blocks are plain text (links turned into their text), so
  the blocks link only the zettels and sources they list.
* **Grouped by tag**: each zettel goes under the one of its tags that the most zettels
  on the page share (ties: the tag it lists first); zettels without tags come last.
* **Sources**: every source note those zettels link to, with the zettels citing it;
  zettels that rest on no source note are named at the end.
* **State** (``.vaultengine/pages.json``, engine-owned, never in the note): for each
  page, when it was last written or refreshed, the SHA-256 of each of its zettels then,
  the zettels excluded from it (`vault page exclude`), and a fingerprint of the page's
  text. `vault page check` compares the zettels with the record. A renamed page keeps its
  record: a recorded page that is gone is paired with an unrecorded page of the same or
  nearly the same text, as the baseline pairs renamed notes (``check.pair_renames``).
* **Behind**: zettels not on the page and not excluded from it that have at least
  ``MIN_IN_COMMON`` (2) things in common with it, counting each tag of the page's they
  carry and each note the page links to (one of its zettels or sources) that they link
  to too.
* The index note gets one line per page between ``<!-- vault:pages -->`` markers,
  rewritten on every page write; the log note gets a line appended.

Writes go through the safe write path (``run_write``): refused if they would add a
violation, atomic together with the state file. Refresh replaces only the lines
between markers; every other byte of the page stays as it was.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vaultengine.formats.check import fingerprint, pair_renames
from vaultengine.formats.pipeline import Pipeline
from vaultengine.formats.typed import TypedVault
from vaultengine.formats.write import FileChange, WriteError, WritePlan, find_note, plan_new, read_text
from vaultengine.vault import link_name, lookup_key

STATE_PATH = ".vaultengine/pages.json"
STATE_VERSION = 1
MIN_IN_COMMON = 2  # tags and links a zettel shares with a page before the page is behind on it
BLOCKS = ("zettels", "sources")
INDEX_BLOCK = "pages"
EXCERPT_LENGTH = 160

_WIKILINK = re.compile(r"(!?)\[\[([^\]|]*?)(?:\|([^\]]*))?\]\]")
_MD_LINK = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
_FIELD = re.compile(r"^[\w][\w\s-]*::(\s|$)")
_LIST_MARK = re.compile(r"^(?:[-*+]\s+|\d+[.)]\s+|>\s*)+")


class PageError(WriteError):
    """The page cannot be made or refreshed as asked (bad markers, not a page, no zettels). Nothing was written."""


def open_marker(name: str) -> str:
    return f"<!-- vault:{name} -->"


def close_marker(name: str) -> str:
    return f"<!-- /vault:{name} -->"


@dataclass(frozen=True)
class Block:
    name: str
    open: int  # 0-based index of the opening marker line
    close: int  # 0-based index of the closing marker line


def find_blocks(lines: list[str], names: tuple[str, ...]) -> dict[str, Block]:
    """The marked blocks present in ``lines`` (a missing block is simply absent). Raises
    PageError for a marker that appears twice, a close without an open, or blocks that overlap."""
    found: dict[str, Block] = {}
    for name in names:
        opens = [i for i, line in enumerate(lines) if line.strip() == open_marker(name)]
        closes = [i for i, line in enumerate(lines) if line.strip() == close_marker(name)]
        if not opens and not closes:
            continue
        if len(opens) != 1 or len(closes) != 1 or closes[0] < opens[0]:
            raise PageError(f"the {name} block's markers are not one {open_marker(name)} followed by one "
                            f"{close_marker(name)} (found {len(opens)} and {len(closes)}); fix them by hand")
        found[name] = Block(name, opens[0], closes[0])
    spans = sorted(found.values(), key=lambda b: b.open)
    for a, b in zip(spans, spans[1:]):
        if b.open < a.close:
            raise PageError(f"the {a.name} and {b.name} blocks overlap; fix the markers by hand")
    return found


def plain(text: str) -> str:
    """Links turned into their text, so an excerpt links nothing."""
    text = _WIKILINK.sub(lambda m: "" if m[1] else (m[3] or m[2].split("#")[0]), text)
    return _MD_LINK.sub(lambda m: m[1], text)


def excerpt(text: str, body_start: int, limit: int = EXCERPT_LENGTH) -> str:
    """The first line of prose at or after line ``body_start`` (1-based): not a heading, a
    fence or what it fences, a comment, a rule, a table or a ``key::`` field. Plain text,
    cut at a word boundary."""
    fenced = False
    for raw in text.splitlines()[max(body_start - 1, 0):]:
        line = raw.strip()
        if line.startswith(("```", "~~~")):
            fenced = not fenced
            continue
        if fenced or not line or line.startswith(("#", "<!--", "|", "---", "***", "___")) or _FIELD.match(line):
            continue
        # No comment markers (they could swallow a block's closing marker), no links.
        line = " ".join(plain(_LIST_MARK.sub("", line)).replace("<!--", "<!- -").replace("-->", "- ->").split())
        if not line:
            continue
        if len(line) > limit:
            line = line[:limit].rsplit(" ", 1)[0].rstrip(" ,;:.-")
            if line.count("`") % 2:  # cut inside inline code: drop the unfinished span
                line = line[:line.rindex("`")].rstrip(" ,;:.-(")
            line += " …"
        return line
    return ""


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="surrogateescape")).hexdigest()


class Pages:
    def __init__(self, typed: TypedVault) -> None:
        spec = typed.spec
        if spec.pages is None or spec.pipeline is None:
            raise PageError(f"format {spec.name!r} has no `pages:` block (see docs/formats.md)")
        self.typed = typed
        self.config = spec.pages
        self.pipeline = Pipeline(typed)
        self.page_types = self.pipeline.compile_types
        self.zettel_types = self.pipeline.extract_types
        # Read once per run (a check of every page must not re-read them per page).
        self._texts: dict[str, str | None] = {}
        self._tag_index: dict[str, list[str]] | None = None
        self._state: dict[str, Any] | None = None
        self._index_part: tuple[set[str], dict[str, str]] | None = None
        self._index_read = False
        self._names: dict[str, str] = {}
        self._renames: dict[str, dict[str, Any]] | None = None

    # ----- reading --------------------------------------------------------------------------

    def text(self, path: str) -> str | None:
        if path not in self._texts:
            root = self.typed.vault.root
            try:
                self._texts[path] = None if root is None else read_text(Path(root), path)
            except (OSError, WriteError):
                self._texts[path] = None
        return self._texts[path]

    def is_page(self, path: str) -> bool:
        return self.typed.type_of(path) in self.page_types

    def is_zettel(self, path: str) -> bool:
        return self.typed.type_of(path) in self.zettel_types

    def pages(self) -> list[str]:
        return sorted(p for p in self.typed.vault.notes if self.is_page(p))

    def tags(self, path: str) -> list[tuple[str, str, str]]:
        """(key, name, written as) for each tag of the note: ``#x`` and ``[[x]]`` are the same tag."""
        return note_tags(self.typed, self.config.tags, path)

    def title(self, path: str) -> str:
        note = self.typed.vault.notes[path]
        heading = next((h for h in note.headings if h.level == 1), None)
        return heading.text if heading else link_name(path)

    def first_line(self, path: str) -> str:
        text = self.text(path)
        if text is None:
            return ""
        note = self.typed.vault.notes[path]
        start = max([note.body_start_line] + [f.line + 1 for f in note.header_fields])
        return excerpt(text, start)

    def page_zettels(self, page: str) -> list[str]:
        """The zettels the page links to now, anywhere in it."""
        return [p for p in self.typed.note_links(page) if self.is_zettel(p)]

    def sources_of(self, zettels: list[str]) -> tuple[dict[str, list[str]], list[str]]:
        """Source note -> the zettels (of these) citing it; and the zettels that cite no source note."""
        cited: dict[str, list[str]] = {}
        bare = []
        for zettel in zettels:
            sources = [p for p in self.typed.note_links(zettel) if self.typed.type_of(p) in self.pipeline.source_types]
            if not sources:
                bare.append(zettel)
            for source in sources:
                cited.setdefault(source, []).append(zettel)
        return cited, bare

    # ----- rendering ------------------------------------------------------------------------

    def _name(self, path: str) -> str:
        """What to write in [[...]]: the name, or the path when another file has that name
        (also right for a page that does not exist yet)."""
        if path not in self._names:
            name = link_name(path)
            others = [f for f in self.typed.vault.files_named(name) if f != path]
            self._names[path] = path[:-3] if others and path.lower().endswith(".md") else (path if others else name)
        return self._names[path]

    def render_zettels(self, zettels: list[str]) -> list[str]:
        counts: Counter[str] = Counter()
        names: dict[str, str] = {}
        for zettel in zettels:
            for key, name, _ in self.tags(zettel):
                counts[key] += 1
                names.setdefault(key, name)
        groups: dict[str | None, list[str]] = {}
        for zettel in zettels:
            keys = [key for key, _, _ in self.tags(zettel)]
            best = max(keys, key=lambda k: (counts[k], -keys.index(k))) if keys else None
            groups.setdefault(best, []).append(zettel)
        order = sorted((k for k in groups if k is not None), key=lambda k: (-len(groups[k]), names[k].casefold()))
        lines = ["## Zettels"]
        if not zettels:
            lines.extend(["", "- (none yet: link zettels from the page and refresh)"])
        for key in order + ([None] if None in groups else []):
            lines.extend(["", f"### {names[key] if key is not None else '(no tags)'}"])
            for zettel in sorted(groups[key], key=lambda p: self._name(p).casefold()):
                first = self.first_line(zettel)
                lines.append(f"- [[{self._name(zettel)}]]" + (f": {first}" if first else ""))
        return lines

    def render_sources(self, zettels: list[str]) -> list[str]:
        cited, bare = self.sources_of(zettels)
        lines = ["## Sources", ""]
        if not cited:
            lines.append("- (none yet)")
        for source in sorted(cited, key=lambda p: self._name(p).casefold()):
            names = ", ".join(sorted((link_name(z) for z in cited[source]), key=str.casefold))
            lines.append(f"- [[{self._name(source)}]] ({names})")
        if bare:
            names = ", ".join(sorted((link_name(z) for z in bare), key=str.casefold))
            lines.extend(["", f"Resting on no source note: {names}"])
        return lines

    def render_block(self, name: str, zettels: list[str]) -> list[str]:
        return self.render_zettels(zettels) if name == "zettels" else self.render_sources(zettels)

    def blocks_text(self, zettels: list[str]) -> str:
        parts = []
        for name in BLOCKS:
            parts.append("\n".join([open_marker(name), *self.render_block(name, zettels), close_marker(name)]))
        return "\n\n".join(parts)

    def default_tags(self, zettels: list[str]) -> str:
        """The tags at least half of the zettels (and at least two) share, at most five, most shared first;
        else the most common one. What the page is about, not every tag two zettels happen to share:
        the page's tags also decide which zettels it is behind on."""
        return shared_tags(self.typed, self.config.tags, zettels)

    # ----- index and log ----------------------------------------------------------------------

    def index_line(self, page: str, zettels: list[str] | None = None, title: str | None = None) -> str:
        zettels = self.page_zettels(page) if zettels is None else zettels
        cited, _ = self.sources_of(zettels)
        return (f"- [[{self._name(page)}]]: {title or self.title(page)} "
                f"({len(zettels)} zettel{'' if len(zettels) == 1 else 's'}, {len(cited)} source{'' if len(cited) == 1 else 's'})")

    def index_lines(self, override: dict[str, str] | None = None) -> list[str]:
        """One line per page, by path; ``override`` gives the line of a page being written."""
        lines = dict(override or {})
        for page in self.pages():
            lines.setdefault(page, self.index_line(page))
        return [lines[p] for p in sorted(lines)]

    def index_change(self, override: dict[str, str]) -> FileChange | None:
        path = self.config.index
        old = self.text(path) if path in self.typed.vault.files else None
        lines = self.index_lines(override)
        if old is None:
            new = "\n".join(["# Index", "", open_marker(INDEX_BLOCK), *lines, close_marker(INDEX_BLOCK)]) + "\n"
            return FileChange(path, None, new, [f"new index: {len(lines)} page{'' if len(lines) == 1 else 's'}"])
        new = replace_block(old, INDEX_BLOCK, lines, append_if_missing=True)
        if new == old:
            return None
        return FileChange(path, old, new, [f"pages part: {len(lines)} line{'' if len(lines) == 1 else 's'}"])

    def log_change(self, line: str) -> FileChange:
        path = self.config.log
        old = self.text(path) if path in self.typed.vault.files else None
        if old is None:
            return FileChange(path, None, f"# Log\n\n{line}\n", ["new log"])
        newline = "\r\n" if "\r\n" in old else "\n"
        sep = "" if not old or old.endswith(("\n", "\r")) else newline
        return FileChange(path, old, old + sep + line + newline, ["appended a line"])

    # ----- state ---------------------------------------------------------------------------

    def state(self) -> dict[str, Any]:
        """The engine state as on disk when first asked (a copy: callers may change it)."""
        if self._state is None:
            self._state = self._read_state()
        return json.loads(json.dumps(self._state)) if self._state["pages"] else {"version": STATE_VERSION, "pages": {}}

    def recorded(self, page: str) -> dict[str, Any] | None:
        """The page's record, also when the page was renamed since (then ``renamed_from`` says from where)."""
        if self._state is None:
            self._state = self._read_state()
        record = self._state["pages"].get(page)
        if record is not None:
            return record
        rename = self.renames().get(page)
        return None if rename is None else {**self._state["pages"][rename["from"]], "renamed_from": rename["from"],
                                            "rename": rename}

    def renames(self) -> dict[str, dict[str, Any]]:
        """New page path -> the rename that gives it a gone page's record."""
        if self._renames is None:
            if self._state is None:
                self._state = self._read_state()
            records = self._state["pages"]
            gone = {p: r["fingerprint"] for p, r in records.items() if p not in self.typed.vault.notes and r.get("fingerprint")}
            fresh = {}
            if gone:
                for page in self.pages():
                    text = self.text(page)
                    if page not in records and text is not None and find_blocks_safe(text):
                        fresh[page] = fingerprint(text)
            self._renames = {r["to"]: r for r in pair_renames(gone, fresh)}
        return self._renames

    def _read_state(self) -> dict[str, Any]:
        root = self.typed.vault.root
        if root is None:
            return {"version": STATE_VERSION, "pages": {}}
        try:
            data = json.loads((Path(root) / STATE_PATH).read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {"version": STATE_VERSION, "pages": {}}
        except (OSError, json.JSONDecodeError) as exc:
            raise PageError(f"{STATE_PATH} cannot be read ({exc}); move it away and refresh the pages") from None
        if not isinstance(data, dict) or data.get("version") != STATE_VERSION or not isinstance(data.get("pages"), dict):
            raise PageError(f"{STATE_PATH} is not a page state file of version {STATE_VERSION}")
        return data

    def state_change(self, page: str, zettels: list[str] | None, now: dt.datetime, text: str | None = None,
                     excluded: list[str] | None = None) -> FileChange | None:
        """The state file with this page's record updated: its zettels now (None: keep the recorded ones),
        its excluded zettels (None: keep them), a fingerprint of ``text`` (the page after the write).
        A record found under the page's old path moves to the new one. None when nothing changes."""
        root = self.typed.vault.root
        if root is None:
            return None
        file = Path(root) / STATE_PATH
        old = file.read_text(encoding="utf-8") if file.exists() else None
        data = self.state()
        previous = self.recorded(page)
        entry = {
            "refreshed_at": now.isoformat(timespec="seconds"),
            "zettels": ({z: sha256_text(self.text(z) or "") for z in sorted(zettels)} if zettels is not None
                        else dict((previous or {}).get("zettels", {}))),
            "excluded": sorted(excluded if excluded is not None else (previous or {}).get("excluded", [])),
            "fingerprint": fingerprint(text if text is not None else (self.text(page) or "")),
        }
        renamed_from = (previous or {}).get("renamed_from")
        if (previous is not None and renamed_from is None and previous.get("zettels") == entry["zettels"]
                and previous.get("excluded", []) == entry["excluded"]
                and (previous.get("fingerprint") or {}).get("sha256") == entry["fingerprint"]["sha256"]):
            return None
        if renamed_from is not None:
            del data["pages"][renamed_from]
        data["pages"][page] = entry
        new = json.dumps(data, indent=1, ensure_ascii=False, sort_keys=True) + "\n"
        what = [f"{len(entry['zettels'])} zettels recorded"]
        if entry["excluded"]:
            what.append(f"{len(entry['excluded'])} excluded")
        if renamed_from is not None:
            what.append(f"record moved from {renamed_from} (renamed)")
        return FileChange(STATE_PATH, old, new, [f"{page}: {', '.join(what)}"])

    # ----- new and refresh ---------------------------------------------------------------

    def plan_new(self, title: str, zettels: list[str], *, folder: str | None = None, tags: str | None = None,
                 now: dt.datetime | None = None) -> WritePlan:
        now = now or dt.datetime.now()
        if not zettels:
            raise PageError("no zettels to make the page from")
        for zettel in zettels:
            if not self.is_zettel(zettel):
                kind = self.typed.type_of(zettel) or "note of no type"
                raise PageError(f"{zettel} is a {kind}, not a {'/'.join(sorted(self.zettel_types))}")
        zettels = list(dict.fromkeys(zettels))
        page_type = sorted(self.page_types)[0]
        values = {self.config.tags: tags if tags is not None else self.default_tags(zettels)}
        plan = plan_new(self.typed, page_type, title, folder=folder, values=values, now=now,
                        body_text=self.blocks_text(zettels))
        page = plan.files[0]
        page.new += "\n"  # an empty line after the blocks: the body is yours
        page.changes = [f"new {page_type}: {title} ({len(zettels)} zettels)"]
        cited, _ = self.sources_of(zettels)
        line = self.index_line(page.path, zettels, title)
        files = [page]
        index = self.index_change({page.path: line})
        if index is not None:
            files.append(index)
        files.append(self.log_change(self._log_line(now, "new", page.path, len(zettels), len(cited))))
        state = self.state_change(page.path, zettels, now, text=page.new)
        return WritePlan("page new", files, [], [state] if state else [])

    def plan_refresh(self, page_ref: str, now: dt.datetime | None = None) -> WritePlan:
        now = now or dt.datetime.now()
        page = find_note(self.typed, page_ref)
        if not self.is_page(page):
            raise PageError(f"{page} is a {self.typed.type_of(page) or 'note of no type'}, not a page")
        old = self.text(page)
        if old is None:
            raise PageError(f"{page} cannot be read")
        lines = old.splitlines()
        blocks = find_blocks(lines, BLOCKS)
        missing = [name for name in BLOCKS if name not in blocks]
        if missing:
            raise PageError(f"{page} has no {' or '.join(missing)} block; add "
                            + " and ".join(f"{open_marker(n)} … {close_marker(n)}" for n in missing)
                            + " where it should go, then refresh")
        zettels = self.page_zettels(page)
        new = old
        for name in BLOCKS:
            new = replace_block(new, name, self.render_block(name, zettels))
        files = []
        notes = []
        if new != old:
            changed = [name for name in BLOCKS if _inner(old, name) != _inner(new, name)]
            files.append(FileChange(page, old, new, [f"rebuilt the {' and '.join(changed)} block{'s' if len(changed) > 1 else ''} "
                                                     f"({len(zettels)} zettels)"]))
            index = self.index_change({page: self.index_line(page, zettels)})
            if index is not None:
                files.append(index)
            cited, _ = self.sources_of(zettels)
            files.append(self.log_change(self._log_line(now, "refresh", page, len(zettels), len(cited))))
        else:
            notes.append(f"{page}: blocks already up to date")
            index = self.index_change({page: self.index_line(page, zettels)})
            if index is not None:
                files.append(index)
        state = self.state_change(page, zettels, now, text=new)
        return WritePlan("page refresh", files, notes, [state] if state else [])

    def plan_exclude(self, page_ref: str, zettel_refs: list[str], undo: bool = False,
                     now: dt.datetime | None = None) -> WritePlan:
        """Record that these zettels are not for this page (``undo``: that they may be again). Only the
        engine state changes; the page is not touched."""
        now = now or dt.datetime.now()
        page = find_note(self.typed, page_ref)
        if not self.is_page(page):
            raise PageError(f"{page} is a {self.typed.type_of(page) or 'note of no type'}, not a page")
        record = self.recorded(page)
        if record is None:
            raise PageError(f"{page} has no record yet: `vault page refresh … --write` it first")
        excluded = list(record.get("excluded", []))
        on_page = set(self.page_zettels(page))
        notes = []
        for ref in zettel_refs:
            zettel = find_note(self.typed, ref)
            if not self.is_zettel(zettel):
                raise PageError(f"{zettel} is a {self.typed.type_of(zettel) or 'note of no type'}, not a zettel")
            if undo:
                if zettel in excluded:
                    excluded.remove(zettel)
                else:
                    notes.append(f"{zettel} was not excluded")
            elif zettel in on_page:
                raise PageError(f"{zettel} is on the page; to take it off, delete its line (and your links to it) and refresh")
            elif zettel in excluded:
                notes.append(f"{zettel} is already excluded")
            else:
                excluded.append(zettel)
        state = self.state_change(page, None, now, excluded=excluded)
        return WritePlan("page exclude", [], notes, [state] if state else [])

    def _log_line(self, now: dt.datetime, action: str, page: str, zettels: int, sources: int) -> str:
        return (f"- {now:%Y-%m-%d %H:%M} page {action} [[{self._name(page)}]] "
                f"({zettels} zettel{'' if zettels == 1 else 's'}, {sources} source{'' if sources == 1 else 's'})")

    # ----- check --------------------------------------------------------------------------

    def _index_lines(self) -> tuple[set[str], dict[str, str]] | None:
        """The index's pages part, read once; None when there is no index note."""
        if not self._index_read:
            text = self.text(self.config.index) if self.config.index in self.typed.vault.files else None
            self._index_part = None if text is None else _index_part(text)
            self._index_read = True
        return self._index_part

    def _zettels_with_tag(self, key: str) -> list[str]:
        """Zettels carrying the tag ``key`` in their tags key. From the index when the vault has one and
        zettels read their metadata inline (the index holds header fields, not frontmatter); else from
        every zettel's metadata."""
        vault = self.typed.vault
        types = [self.typed.spec.type_named(name) for name in self.zettel_types]
        if hasattr(vault, "notes_with_header_item") and all(t.metadata_source == "inline" for t in types):
            return [z for z in vault.notes_with_header_item(self.config.tags, key) if self.is_zettel(z)]
        if self._tag_index is None:
            index: dict[str, list[str]] = {}
            for zettel in sorted(self.pipeline.extracts):
                for tag_key, _, _ in self.tags(zettel):
                    index.setdefault(tag_key, []).append(zettel)
            self._tag_index = index
        return self._tag_index.get(key, [])

    def behind(self, page: str, on_page: list[str] | None = None) -> list[dict[str, Any]]:
        """Zettels not on the page and not excluded from it with at least MIN_IN_COMMON things in
        common with it: the page's tags they carry, and the notes the page links to that they link to."""
        on_page = set(self.page_zettels(page) if on_page is None else on_page)
        record = self.recorded(page) or {}
        excluded = set(record.get("excluded", []))
        page_tags = {key: name for key, name, _ in self.tags(page)}
        page_links = self.typed.note_links(page)
        found: dict[str, dict[str, list[str]]] = {}
        for key, name in page_tags.items():
            for zettel in self._zettels_with_tag(key):
                found.setdefault(zettel, {"tags": [], "links": []})["tags"].append(name)
        for target in page_links:
            for zettel in self.typed.linked_from(target):
                if self.is_zettel(zettel):
                    found.setdefault(zettel, {"tags": [], "links": []})["links"].append(self._name(target))
        rows = [{"path": z, **why} for z, why in found.items()
                if z not in on_page and z != page and z not in excluded and len(why["tags"]) + len(why["links"]) >= MIN_IN_COMMON]
        rows.sort(key=lambda r: (-(len(r["tags"]) + len(r["links"])), r["path"]))
        return rows

    def record_staleness(self, page: str) -> list[str]:
        """Why a managed page needs a refresh, from the engine record alone (no block is rendered): no record;
        zettels changed, deleted, added or dropped since it; its index line. Cheap enough for every page at once
        (`vault next`); `check` shows the same with the block lines."""
        record = self.recorded(page)
        if record is None:
            return ["no record of its zettels"]
        reasons = []
        zettels = self.page_zettels(page)
        recorded = record.get("zettels", {})
        changed = [z for z in zettels if z in recorded and z in self.typed.vault.notes
                   and sha256_text(self.text(z) or "") != recorded[z]]
        added = [z for z in zettels if z not in recorded]
        dropped = [z for z in recorded if z not in zettels]
        if changed:
            reasons.append(f"{len(changed)} zettel{'' if len(changed) == 1 else 's'} changed since the last refresh")
        if added:
            reasons.append(f"{len(added)} zettel{'' if len(added) == 1 else 's'} linked but not in its block")
        if dropped:
            reasons.append(f"{len(dropped)} zettel{'' if len(dropped) == 1 else 's'} in its record but no longer linked")
        part = self._index_lines()
        if part is None or self.index_line(page, zettels) not in part[0]:
            reasons.append("its index line is missing or out of date")
        return reasons

    def managed(self, page: str) -> bool:
        text = self.text(page)
        return text is not None and bool(find_blocks_safe(text))

    def behind_all(self, pages: list[str]) -> dict[str, list[dict[str, Any]]]:
        """``behind`` for many pages in one pass (the same rows). Each tag's zettels and each link target's
        zettels are looked up once, and a page's candidates are counted (``Counter.update``) before anything
        else is built, so only zettels with two or more things in common are looked at twice."""
        if hasattr(self.typed.vault.notes, "prefetch"):
            self.typed.vault.notes.prefetch(pages)
        by_tag: dict[str, list[str]] = {}
        by_target: dict[str, list[str]] = {}
        result = {}
        for page in pages:
            page_tags = {key: name for key, name, _ in self.tags(page)}
            page_links = self.typed.note_links(page)
            for key in page_tags:
                if key not in by_tag:
                    by_tag[key] = self._zettels_with_tag(key)
            for target in page_links:
                if target not in by_target:
                    by_target[target] = [z for z in self.typed.linked_from(target) if self.is_zettel(z)]
            counts: Counter[str] = Counter()
            for key in page_tags:
                counts.update(by_tag[key])
            for target in page_links:
                counts.update(by_target[target])
            on_page = set(self.page_zettels(page))
            excluded = set((self.recorded(page) or {}).get("excluded", []))
            rows = []
            for zettel, count in counts.items():
                if count < MIN_IN_COMMON or zettel in on_page or zettel == page or zettel in excluded:
                    continue
                tags = [name for key, name in page_tags.items() if zettel in by_tag[key]]
                links = [self._name(t) for t in page_links if zettel in by_target[t]]
                rows.append({"path": zettel, "tags": tags, "links": links})
            rows.sort(key=lambda r: (-(len(r["tags"]) + len(r["links"])), r["path"]))
            result[page] = rows
        return result

    def check(self, page: str) -> dict[str, Any]:
        text = self.text(page) or ""
        lines = text.splitlines()
        result: dict[str, Any] = {"path": page, "title": self.title(page), "managed": False, "blocks": [],
                                  "changed": [], "behind": [], "excluded": [], "violations": [], "index": None,
                                  "refreshed_at": None, "renamed_from": None, "ready": False, "problems": []}
        try:
            blocks = find_blocks(lines, BLOCKS)
        except PageError as exc:
            result["problems"].append(str(exc))
            return result
        if not blocks:
            result["problems"].append("no vault blocks: not made by `vault page new`")
            return result
        result["managed"] = True
        zettels = self.page_zettels(page)
        if hasattr(self.typed.vault.notes, "prefetch"):
            self.typed.vault.notes.prefetch(zettels)  # index-backed: the page's zettels in one batch, not one by one
        for name in BLOCKS:
            if name not in blocks:
                result["blocks"].append({"name": name, "status": "missing", "removed": [], "added": []})
                continue
            block = blocks[name]
            current = [line.rstrip("\r") for line in lines[block.open + 1:block.close]]
            wanted = self.render_block(name, zettels)
            status = "current" if current == wanted else "out-of-date"
            removed = [line for line in current if line not in wanted]
            added = [line for line in wanted if line not in current]
            result["blocks"].append({"name": name, "status": status, "removed": removed, "added": added})

        record = self.recorded(page)
        result["renamed_from"] = (record or {}).get("renamed_from")
        result["excluded"] = list((record or {}).get("excluded", []))
        if record is None:
            result["changed"] = None  # nothing recorded: never written or refreshed by the engine
        else:
            result["refreshed_at"] = record.get("refreshed_at")
            recorded = record.get("zettels", {})
            for zettel in sorted(set(recorded) | set(zettels)):
                if zettel not in recorded:
                    result["changed"].append({"path": zettel, "status": "not recorded"})
                elif zettel not in self.typed.vault.notes:
                    result["changed"].append({"path": zettel, "status": "deleted"})
                elif sha256_text(self.text(zettel) or "") != recorded[zettel]:
                    result["changed"].append({"path": zettel, "status": "changed"})

        result["behind"] = self.behind(page, zettels)
        result["violations"] = [issue.to_dict() for issue in self.typed.issues(page)]
        part = self._index_lines()
        wanted_line = self.index_line(page, zettels)
        if part is None:
            result["index"] = "missing"
        else:
            lines_now, by_link = part
            link = f"- [[{self._name(page)}]]:"
            if wanted_line in lines_now:
                result["index"] = "current"
            elif link in by_link:
                result["index"] = "out-of-date"
            else:
                result["index"] = "missing"
        result["ready"] = (all(b["status"] == "current" for b in result["blocks"]) and result["changed"] == []
                           and not result["behind"] and not result["violations"] and result["index"] == "current")
        return result


def note_tags(typed: TypedVault, key: str, path: str) -> list[tuple[str, str, str]]:
    """(key, name, written as) for each tag in the note's ``key`` metadata: ``#x`` and ``[[x]]`` are the same tag."""
    value = typed.metadata(path).get(key)
    if value is None:
        return []
    found, seen = [], set()
    for item in value.items:
        if item.kind not in ("tag", "link"):
            continue
        tag_key = lookup_key(item.value)
        if tag_key not in seen:
            seen.add(tag_key)
            found.append((tag_key, item.value, f"#{item.value}" if item.kind == "tag" else f"[[{item.value}]]"))
    return found


def shared_tags(typed: TypedVault, key: str, paths: list[str]) -> str:
    """The tags at least half of the notes (and at least two) carry in ``key``, at most five, most shared
    first; else the most common one; "" when none has a tag. Written as the vault mostly writes them."""
    counts: Counter[str] = Counter()
    written: dict[str, str] = {}
    first_seen: dict[str, int] = {}
    for path in paths:
        for tag_key, _, form in note_tags(typed, key, path):
            counts[tag_key] += 1
            if tag_key not in written or form.startswith("[["):
                written[tag_key] = form  # a link wins over a #tag: the vault mostly links tags
            first_seen.setdefault(tag_key, len(first_seen))
    ranked = sorted(counts, key=lambda k: (-counts[k], first_seen[k]))
    shared = [k for k in ranked if counts[k] >= max(2, (len(paths) + 1) // 2)][:5] or ranked[:1]
    return " ".join(written[k] for k in shared)


def find_blocks_safe(text: str) -> list[str]:
    """Names of the well-formed blocks in ``text`` (none when the markers are broken)."""
    try:
        return list(find_blocks(text.splitlines(), BLOCKS))
    except PageError:
        return []


def _index_part(text: str) -> tuple[set[str], dict[str, str]]:
    """The lines of the index's pages part, as a set and by their leading "- [[name]]:"."""
    lines = text.splitlines()
    try:
        part = find_blocks(lines, (INDEX_BLOCK,)).get(INDEX_BLOCK)
    except PageError:
        part = None
    inner = [l.rstrip("\r") for l in lines[part.open + 1:part.close]] if part else []
    by_link = {}
    for line in inner:
        head, sep, _ = line.partition("]]:")
        if sep:
            by_link[head + sep] = line
    return set(inner), by_link


def _inner(text: str, name: str) -> list[str]:
    lines = text.splitlines()
    block = find_blocks(lines, (name,)).get(name)
    return lines[block.open + 1:block.close] if block else []


def replace_block(text: str, name: str, inner: list[str], append_if_missing: bool = False) -> str:
    """``text`` with the lines between ``name``'s markers replaced by ``inner``. Every other
    byte stays: the marker lines, line endings elsewhere, a missing final newline. The new
    lines use the file's line ending (the first one found, else \\n)."""
    newline = "\r\n" if "\r\n" in text else "\n"
    kept = text.splitlines(keepends=True)
    block = find_blocks([line.rstrip("\r\n") for line in kept], (name,)).get(name)
    body = "".join(line + newline for line in inner)
    if block is None:
        if not append_if_missing:
            raise PageError(f"no {name} block")
        sep = "" if not text or text.endswith(("\n", "\r")) else newline
        return text + sep + (newline if text else "") + open_marker(name) + newline + body + close_marker(name) + newline
    return "".join(kept[:block.open + 1]) + body + "".join(kept[block.close:])
