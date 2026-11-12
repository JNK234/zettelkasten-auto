"""`vault link-it` and `vault unlink`: a result of `vault explore` made into a real link, and undone.

Link it writes, in both notes, one line under the section ``engine.explore.link_section`` ("See Also"):

    - [[other note]] — <one-line reason>

at the end of that section's text, or in a new ``## See Also`` section at the end of the note when it has
none (at the level of the note's last heading, at most 2, so it is not nested under it). The reason is the
result's lens and evidence (``note_reason``: without "no link either way", which the link makes false, and
without percentiles, which date), cut to one line of at most ``REASON_CHARS`` characters,
with nothing in it that would become a link, a tag or math (``[[``, ``]]``, ``#``, ``$``, ``%%``). A note that
already links the other note in that section is left alone.

Both go through the safe write path (``formats/write.py``): the gate refuses the write if it would add a
violation of the format, both files are committed atomically, the index is updated. Without ``--write`` it is
a dry run: the diff, and nothing written.

Unlink removes, in both notes, the list lines in that section that link the other note, and the section's
heading too when link-it created it and nothing else is left in it (with the blank line before it when link-it
added that line). Lines elsewhere are never touched. Both keep a note's line endings (LF, CRLF) and its ending: a
note without a final line break gets none, so link-it then unlink gives back the note byte for byte.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from vaultengine.formats.edit import Lines
from vaultengine.formats.write import FileChange, WriteError, WritePlan, find_note, read_text

REASON_CHARS = 160
_BULLET = re.compile(r"^\s*[-*+]\s")


def note_reason(lens_title: str, evidence: str) -> str:
    """The reason link-it writes for an explore result: its lens and evidence, without what stops being true once
    the link is there ("no link either way") and without the vault statistics (percentiles), which date fast."""
    text = re.sub(r"^(no link either way|does not link it|not linked); ", "", evidence)
    text = re.sub(r"\s*\[(meaning|key terms)\]$", "", text)
    if "; join " in text:  # a gap: the pair is this link; what it rests on is the middles the two share
        text, _, join = text.partition("; join ")
        via = join.partition("(via ")[2]
        text += f"; shared: {via[:-1] if via.endswith(')') else via}" if via else ""
    text = re.sub(r"\d+ links? between them where chance predicts [\d.]+",
                  "fewer links between their communities than chance predicts", text)
    text = re.sub(r"\s*\(p\d+[^()]*(\([^()]*\)[^()]*)*\)", "", text)
    text = re.sub(r"; score \d+(\.\d+)?", "", text)
    text = re.sub(r"; linked$", "", text)
    return f"{lens_title.lower()}: {text}"


def clean_reason(text: str) -> str:
    """One line, nothing that would parse as a link, tag, math or comment, at most REASON_CHARS characters."""
    text = " ".join(str(text).split())
    text = text.replace("[[", "").replace("]]", "").replace("%%", "").replace("$", "")
    text = re.sub(r"#(?=\w)", "", text)
    if len(text) > REASON_CHARS:
        cut = text[: REASON_CHARS - 1]
        text = (cut[: cut.rfind(" ")] if " " in cut else cut).rstrip(" ,;:") + "…"
    return text


@dataclass
class Written:
    """What link-it writes (or would write) in one note."""

    path: str
    line: str
    created: bool  # the section is new
    blank: bool = False  # a blank line was added before the new section's heading


def section_span(note, section: str, last_line: int) -> tuple[int, int] | None:
    """(heading line, last line of the section) of the first heading named ``section`` (any level, any case)."""
    want = section.strip().lower()
    headings = list(note.headings)
    for i, h in enumerate(headings):
        if h.text.strip().lower() == want:
            end = last_line
            for later in headings[i + 1:]:
                if later.level <= h.level:
                    end = later.line - 1
                    break
            return h.line, end
    return None


def _links_to(typed, path: str, other: str, lines: range | None = None) -> list[int]:
    """Lines of ``path`` (within ``lines``) with a link that resolves to ``other``."""
    from vaultengine.resolver import Resolver

    resolver = Resolver(typed.vault)
    note = typed.vault.notes[path]
    return sorted({l.line for l in note.links if (lines is None or l.line in lines)
                   and resolver.resolve(path, l).target_path == other})


def plan_link_it(typed, a_ref: str, b_ref: str, reason_a: str, reason_b: str, section: str) -> tuple[WritePlan, list[Written]]:
    a, b = find_note(typed, a_ref), find_note(typed, b_ref)
    if a == b:
        raise WriteError("a note cannot be linked to itself")
    plan = WritePlan("link-it")
    written = []
    for path, other, reason in ((a, b, reason_a), (b, a, reason_b)):
        raw = read_text(typed.vault.root, path)
        lines = Lines(raw, keep_ending=True)
        note = typed.vault.notes[path]
        span = section_span(note, section, lines.real)
        text = f"- [[{typed.link_text(other)}]] — {clean_reason(reason)}"
        if span is not None and _links_to(typed, path, other, range(span[0], span[1] + 1)):
            plan.notes.append(f"{path} already links {typed.link_text(other)} under {section}")
            continue
        added_blank = False
        if span is not None:
            heading, end = span
            last = max((n for n in range(heading + 1, end + 1) if lines.text(n).strip()), default=heading)
            lines.after[last].append(text)
            where = f"after line {last}, under {section}"
        else:
            added_blank = lines.real > 0 and bool(lines.text(lines.real).strip())  # a blank line before the heading
            level = note.headings[-1].level if note.headings and note.headings[-1].level <= 2 else 2
            heading_text = "#" * level + f" {section}"
            lines.at_end.extend(([""] if added_blank else []) + [heading_text, text])
            where = f"new section {heading_text} at the end (the level of the note's last heading, at most 2)"
        plan.files.append(FileChange(path, raw, lines.render(), [f"{where}: {text}"]))
        written.append(Written(path, text, span is None, added_blank))
    return plan, written


def plan_unlink(typed, a_ref: str, b_ref: str, section: str, created: dict[str, bool] | None = None,
                blank: dict[str, bool | None] | None = None) -> WritePlan:
    """Remove the list lines under ``section`` that link the other note, in both notes. ``created``: path -> did
    link-it create the section (then an emptied section goes too). ``blank``: path -> did link-it add the blank
    line before that section's heading (then it goes with it); None when not recorded (links made before link-it
    recorded it): a blank line right before the heading goes."""
    a, b = find_note(typed, a_ref), find_note(typed, b_ref)
    plan = WritePlan("unlink")
    for path, other in ((a, b), (b, a)):
        raw = read_text(typed.vault.root, path)
        lines = Lines(raw, keep_ending=True)
        note = typed.vault.notes[path]
        span = section_span(note, section, lines.real)
        if span is None:
            plan.notes.append(f"{path} has no {section} section")
            continue
        heading, end = span
        found = [n for n in _links_to(typed, path, other, range(heading + 1, end + 1)) if _BULLET.match(lines.text(n))]
        if not found:
            plan.notes.append(f"{path} has no line linking {typed.link_text(other)} under {section}")
            continue
        lines.delete.update(found)
        described = [f"line {n}: removed {lines.text(n).strip()}" for n in found]
        left = [n for n in range(heading + 1, end + 1) if n not in lines.delete and lines.text(n).strip()]
        if not left and (created or {}).get(path):
            lines.delete.add(heading)
            lines.delete.update(n for n in range(heading + 1, end + 1))
            added_blank = (blank or {}).get(path)
            if added_blank is None:  # not recorded: the blank line right before the heading, if there is one
                added_blank = heading > 1 and not lines.text(heading - 1).strip()
            if added_blank and heading > 1 and not lines.text(heading - 1).strip():
                lines.delete.add(heading - 1)  # the blank line link-it put before the heading
            described.append(f"line {heading}: removed the {section} heading link-it had added (nothing else under it)")
        plan.files.append(FileChange(path, raw, lines.render(), described))
    return plan
