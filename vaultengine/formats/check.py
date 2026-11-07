"""`vault check`: the violations of a format, for the whole vault or for what changed.

"Changed" means added or modified since the last `vault index` (the index's own
change detection: size/mtime, then content hash). The whole vault is still read,
because links, mirrors and target types depend on other notes. A change can
break a note that did not change, so ``--changed`` reports violations

* in added or modified notes;
* caused by them elsewhere: a missing mirror implied by a changed note's field
  (A now ``supports:: [[B]]``, B lacks ``supported-by:: [[A]]``);
* caused by deleted notes: relation links that pointed at a deleted note's name.

Checking never writes anything, not even the index: after a failed check,
fix the notes and check again; run `vault index` once the notes are right.
"""

from __future__ import annotations

import posixpath
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Literal

from vaultengine.formats.typed import Issue, TypedVault
from vaultengine.vault import link_name, lookup_key


@dataclass
class CheckScope:
    mode: Literal["all", "changed"]
    checked: list[str] = field(default_factory=list)  # notes checked
    added: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    last_indexed_at: str | None = None
    reason: str | None = None  # why a --changed run checked everything


def scope_all(typed: TypedVault) -> CheckScope:
    return CheckScope("all", checked=sorted(typed.vault.notes))


def scope_changed(typed: TypedVault, added: list[str], changed: list[str], deleted: list[str],
                  last_indexed_at: str | None, reason: str | None = None) -> CheckScope:
    """Scope for ``--changed``. With a ``reason`` (the index must be rebuilt) every note is checked."""
    notes = typed.vault.notes
    if reason:
        return CheckScope("changed", sorted(notes), last_indexed_at=last_indexed_at, reason=reason)
    added_notes = sorted(p for p in added if p in notes)
    changed_notes = sorted(p for p in changed if p in notes)
    deleted_notes = sorted(p for p in deleted if p.lower().endswith(".md"))
    return CheckScope(
        "changed", sorted(added_notes + changed_notes), added_notes, changed_notes, deleted_notes, last_indexed_at,
    )


def select_issues(typed: TypedVault, scope: CheckScope) -> list[Issue]:
    issues = typed.issues()
    if scope.mode == "all" or scope.reason:
        return issues
    checked = set(scope.checked)
    gone = {lookup_key(link_name(path)) for path in scope.deleted} | {lookup_key(path[:-3]) for path in scope.deleted}

    def involved(issue: Issue) -> bool:
        if issue.path in checked or issue.related in checked:
            return True
        if issue.code == "unresolved-target" and issue.detail:
            target = issue.detail["target"]
            return lookup_key(target) in gone or lookup_key(posixpath.basename(target)) in gone
        return False

    return [issue for issue in issues if involved(issue)]


def check_report(typed: TypedVault, scope: CheckScope, issues: list[Issue]) -> dict[str, Any]:
    by_code = Counter(issue.code for issue in issues)
    paths = sorted({issue.path for issue in issues})
    checked = set(scope.checked)
    return {
        "format": {"name": typed.spec.name, "path": typed.spec.path},
        "scope": {
            "mode": scope.mode,
            "checked": len(scope.checked),
            "notes": scope.checked if scope.mode == "changed" else None,
            "added": scope.added,
            "changed": scope.changed,
            "deleted": scope.deleted,
            "last_indexed_at": scope.last_indexed_at,
            "reason": scope.reason,
        },
        "ok": not issues,
        "summary": {
            "violations": len(issues),
            "notes": len(paths),
            "by_code": [{"code": code, "count": count} for code, count in sorted(by_code.items(), key=lambda kv: (-kv[1], kv[0]))],
        },
        "notes": [
            {
                "path": path,
                "type": typed.type_of(path),
                "checked": path in checked,  # False: affected by a changed or deleted note
                "violations": [issue.to_dict() for issue in issues if issue.path == path],
            }
            for path in paths
        ],
    }
