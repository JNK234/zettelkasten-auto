"""`vault fix`: repair the violations that have exactly one safe repair.

Repaired:
* missing-mirror: A ``supports:: [[B]]`` and B lacks the mirror. B gets
  ``supported-by:: [[A]]``: appended to its existing ``supported-by::`` field, or as a
  new field line after its last relation field, or right under the type's
  ``relations_heading``. With none of these there is no safe place and it is skipped.
* missing-heading: the heading is inserted, with an empty section, before the next
  required heading that is present (or at the end of the note), at the level it has
  in most notes of the type. Skipped when the note's headings are out of order or the
  required heading is a /regex/.

The CLI writes the repairs through the safe write path (see ``write.py``): one atomic
group, the same "no new violations" gate, and an index update for the notes written.

Everything else is left alone, and so is every line the repair does not need:
edits are made on the raw file, line by line, so untouched lines keep their bytes
(line endings, BOM, trailing spaces). Before anything is written, the edited notes
are re-checked in memory together with the rest of the vault. A file is only
changed if its repairs clear the violations they target and no (note, issue code)
gains a violation; otherwise it is reported as skipped.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from vaultengine.formats.edit import Lines
from vaultengine.formats.typed import Issue, TypedVault
from vaultengine.parser import _INLINE_FIELD, parse_note
from vaultengine.vault import Vault

FIXABLE = ("missing-mirror", "missing-heading")


@dataclass
class FileFix:
    path: str
    original: str  # decoded file content, byte-exact (BOM and line endings kept)
    new: str
    changes: list[str]  # one line per repair, for people
    issues: list[Issue]  # the violations repaired


@dataclass
class FixPlan:
    files: list[FileFix] = field(default_factory=list)
    skipped: list[tuple[Issue, str]] = field(default_factory=list)  # fixable kind, but not safe here
    untouched: Counter[str] = field(default_factory=Counter)  # violations fix never repairs, by code (before)
    remaining: Counter[str] = field(default_factory=Counter)  # all violations once the plan is applied

    @property
    def fixes(self) -> int:
        return sum(len(f.issues) for f in self.files)


def plan_fixes(typed: TypedVault, only: tuple[str, ...] = FIXABLE) -> FixPlan:
    plan = FixPlan()
    wanted: dict[str, list[Issue]] = defaultdict(list)
    for issue in typed.issues():
        if issue.code in FIXABLE and issue.code in only:
            wanted[issue.path].append(issue)
        else:
            plan.untouched[issue.code] += 1

    root = typed.vault.root
    drafts: dict[str, FileFix] = {}
    for path, issues in sorted(wanted.items()):
        draft, skipped = _draft(typed, path, issues, root)
        plan.skipped.extend(skipped)
        if draft is not None:
            drafts[path] = draft

    # Re-check the edited vault; drop files whose repairs do not hold, until all do.
    while drafts:
        rejected = _verify(typed, drafts)
        if not rejected:
            break
        for path, reason in rejected.items():
            plan.skipped.extend((issue, reason) for issue in drafts.pop(path).issues)
    plan.files = [drafts[path] for path in sorted(drafts)]
    after = _recheck(typed, drafts) if drafts else typed
    plan.remaining = Counter(issue.code for issue in after.issues())
    return plan


def _draft(typed: TypedVault, path: str, issues: list[Issue], root: Path | None) -> tuple[FileFix | None, list[tuple[Issue, str]]]:
    skipped: list[tuple[Issue, str]] = []
    if root is None:
        return None, [(issue, "vault was not loaded from disk") for issue in issues]
    try:
        raw = (root / path).read_bytes().decode("utf-8")
    except UnicodeDecodeError:
        return None, [(issue, "file is not valid UTF-8; not touched") for issue in issues]
    lines = Lines(raw)
    note_type = typed.spec.type_for(path)
    changes: list[str] = []
    repaired: list[Issue] = []
    out_of_order = any(issue.code == "heading-order" for issue in typed.issues(path))

    mirrors: dict[str, list[Issue]] = defaultdict(list)
    for issue in issues:
        if issue.code == "missing-mirror":
            mirrors[issue.detail["relation"]].append(issue)
    for relation, group in mirrors.items():
        if note_type is not None and not note_type.allows_relation(relation):
            skipped.extend((issue, f"`{relation}::` is not a relation of {note_type.name}") for issue in group)
            continue
        anchor = typed.relation_anchor(path, relation)
        if anchor is None:
            reason = "no relation field to extend and no relations_heading to put one under"
            skipped.extend((issue, reason) for issue in group)
            continue
        links = ", ".join(f"[[{issue.detail['link']}]]" for issue in group)
        if anchor.kind == "append":
            lines.append[anchor.line].append((" " if anchor.empty_value else ", ") + links)
            changes.append(f"line {anchor.line}: added {links} to `{relation}::`")
        else:
            match = _INLINE_FIELD.match(lines.text(anchor.line))
            prefix = match["prefix"] if match else ""
            lines.after[anchor.line].append(f"{prefix}{relation}:: {links}")
            changes.append(f"after line {anchor.line}: new field `{relation}:: {links}`")
        repaired.extend(group)

    for issue in issues:
        if issue.code != "missing-heading":
            continue
        detail = issue.detail
        if out_of_order:
            skipped.append((issue, "the note's headings are out of order; fix by hand"))
            continue
        if not detail["fixable"]:
            skipped.append((issue, "a required heading is a pattern, not a name to write"))
            continue
        new_lines: list[str] = []
        for name, level in zip(detail["headings"], detail["levels"]):
            new_lines.extend(["#" * level + " " + name, ""])
        shown = ", ".join(repr(name) for name in detail["headings"])
        before = detail["before_line"]
        if before is not None:
            if before > 1 and lines.text(before - 1).strip():
                new_lines.insert(0, "")
            lines.before[before].extend(new_lines)
            changes.append(f"before line {before}: added {shown}")
        else:
            new_lines.pop()  # nothing follows at the end of the note
            if lines.real and lines.text(lines.real).strip():
                new_lines.insert(0, "")
            lines.at_end.extend(new_lines)
            changes.append(f"end of note: added {shown}")
        repaired.append(issue)

    if not repaired:
        return None, skipped
    return FileFix(path, raw, lines.render(), changes, repaired), skipped


def _verify(typed: TypedVault, drafts: dict[str, FileFix]) -> dict[str, str]:
    """Re-check with the drafts applied. Returns {path: reason} for drafts to drop.

    All drafts are checked together; if that fails, each one alone, so one bad
    repair does not hold back the others.
    """
    problems = _problems(typed, drafts)
    if not problems:
        return {}
    rejected = {}
    for path in drafts:
        alone = _problems(typed, {path: drafts[path]})
        if alone:
            rejected[path] = alone[0]
    return rejected or {path: problems[0] for path in drafts}  # only the combination fails: drop all


def _recheck(typed: TypedVault, drafts: dict[str, FileFix]) -> TypedVault:
    notes = dict(typed.vault.notes)
    for path, draft in drafts.items():
        notes[path] = parse_note(draft.new, path=path)
    return TypedVault(typed.spec, Vault(typed.vault.files, notes, typed.vault.root))


def _problems(typed: TypedVault, drafts: dict[str, FileFix]) -> list[str]:
    after = _recheck(typed, drafts)

    problems = []
    before_counts = Counter((i.path, i.code) for i in typed.issues())
    for (path, code), count in sorted(Counter((i.path, i.code) for i in after.issues()).items()):
        if count > before_counts.get((path, code), 0):
            problems.append(f"the repair would add a {code} violation to {path}")
    for path, draft in drafts.items():
        if _targets(draft.issues) & _targets(after.issues(path)):
            problems.append(f"the repair would not clear its violations in {path}")
    return problems


def _targets(issues: list[Issue]) -> set[tuple[str, str]]:
    """What each fixable violation is about: the missing heading names, the mirror's source note."""
    found = set()
    for issue in issues:
        if issue.code == "missing-heading":
            found.update(("heading", name) for name in issue.detail["headings"])
        elif issue.code == "missing-mirror":
            found.add(("mirror", f"{issue.detail['relation']} {issue.detail['source']}"))
    return found


def unified_diff(plan: "FixPlan") -> str:
    from vaultengine.formats.write import FileChange, diff_text

    return diff_text([FileChange(f.path, f.original, f.new) for f in plan.files])


def apply_fixes(plan: FixPlan, root: Path) -> tuple[list[FileFix], list[tuple[FileFix, str]]]:
    """Write the planned files as one atomic group (no index update; the CLI does that).

    If any file changed on disk since planning, nothing is written.
    """
    from vaultengine.formats.write import FileChange, WriteFailed, commit, recover, write_lock

    changes = [FileChange(f.path, f.original, f.new, f.changes) for f in plan.files]
    with write_lock(root):
        recover(root)
        try:
            commit(root, changes)
        except WriteFailed as exc:
            return [], [(f, str(exc)) for f in plan.files]
    return list(plan.files), []
