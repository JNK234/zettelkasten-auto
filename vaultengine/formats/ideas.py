"""`vault idea new` and `vault idea set`: ideas made from findings (``vaultengine/findings.py``).

An idea is a note of the format's ``ideas.type``::

    Created: …
    Tags: …                 (the tags the finding's notes share, or --tags)
    Status: #open

    # Title

    <!-- vault:finding -->
    ## Finding

    - id: link-5ec0974343
    - method: link (two notes that probably belong linked)
    - notes: [[a]], [[b]]
    - evidence:
      - both link to: …
    - found: 2026-11-10 12:00
    <!-- /vault:finding -->

    ## Claim

    (yours)

The block is what the format requires of an idea (``required_blocks``), so an idea without one
fails `vault check`, and `vault new --type idea` is refused by the write gate. The status is one
of the key's values, set with `vault idea set` through the same gate. Default titles say what the
finding is about without taking a missing note's name (the idea would then *be* that note).
"""

from __future__ import annotations

import datetime as dt

from vaultengine.findings import METHODS, Finding
from vaultengine.formats.pages import shared_tags
from vaultengine.formats.typed import TypedVault
from vaultengine.formats.write import WriteError, WritePlan, find_note, plan_new, plan_set


class IdeaError(WriteError):
    """The idea cannot be made or changed as asked. Nothing was written."""


def _config(typed: TypedVault):
    if typed.spec.ideas is None:
        raise IdeaError(f"format {typed.spec.name!r} has no `ideas:` block (see docs/formats.md)")
    return typed.spec.ideas


def default_title(finding: Finding) -> str:
    names = [note["name"] for note in finding.notes]
    if finding.method == "missing-note":
        return f"Missing note {names[0]}"
    if finding.method == "same-idea":
        return f"{names[0]} or {names[1]}"
    if finding.method == "existing-name":
        return f"{names[0]} means {names[1]}"
    return f"{names[0]} and {names[1]}"


def finding_block(finding: Finding, block: str, now: dt.datetime) -> str:
    notes = ", ".join(f"[[{note['name']}]]" for note in finding.notes)
    lines = [
        f"<!-- vault:{block} -->",
        "## Finding",
        "",
        f"- id: {finding.id}",
        f"- method: {finding.method} ({METHODS[finding.method]})",
        f"- notes: {notes}",
        "- evidence:",
        *[f"  - {line}" for line in finding.evidence],
        f"- found: {now:%Y-%m-%d %H:%M}",
        f"<!-- /vault:{block} -->",
    ]
    return "\n".join(lines)


def finding_paths(typed: TypedVault, finding: Finding) -> list[str]:
    """The existing notes a finding is about, or for a missing note the notes linking to it: where its tags come from."""
    paths = [note["path"] for note in finding.notes if note["path"] in typed.vault.notes]
    if finding.method in ("missing-note", "same-idea", "existing-name"):
        linking = finding.data.get("linking_notes", [])
        if isinstance(linking, dict):
            linking = [p for group in linking.values() for p in group]
        paths.extend(p for p in linking if p in typed.vault.notes)
    return list(dict.fromkeys(paths))


def plan_idea_new(typed: TypedVault, finding: Finding, *, title: str | None = None, tags: str | None = None,
                  now: dt.datetime | None = None) -> WritePlan:
    config = _config(typed)
    now = now or dt.datetime.now()
    title = title or default_title(finding)
    if tags is None:
        tags = shared_tags(typed, config.tags, finding_paths(typed, finding))
        if not tags:
            raise IdeaError("the finding's notes have no tags to give the idea; give --tags")
    body = finding_block(finding, config.block, now) + "\n\n## Claim\n"
    plan = plan_new(typed, config.type, title, values={config.tags: tags}, now=now, body_text=body)
    plan.files[0].new += "\n"  # an empty line under Claim: the claim is yours
    plan.files[0].changes = [f"new {config.type} from {finding.id}: {title}"]
    plan.operation = "idea new"
    return plan


def plan_idea_set(typed: TypedVault, idea_ref: str, status: str) -> WritePlan:
    config = _config(typed)
    idea = find_note(typed, idea_ref)
    if typed.type_of(idea) != config.type:
        raise IdeaError(f"{idea} is a {typed.type_of(idea) or 'note of no type'}, not a {config.type}")
    value = status.strip()
    value = value if value.startswith("#") else f"#{value}"
    plan = plan_set(typed, idea, config.status, value)
    plan.operation = "idea set"
    return plan
