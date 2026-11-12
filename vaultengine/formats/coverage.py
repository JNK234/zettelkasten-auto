"""`vault coverage`: how well sources and notes are connected, per the format's
``coverage:`` block.

* Sources nothing links to: notes of a ``sources`` type with no resolved link or
  embed from any other note.
* Notes that cite no source: notes of a ``citing`` type without a resolved link to a
  note of a ``sources`` type. Each one says why:
  - ``missing``: it has citations (links in a ``citation_roles`` role), but their targets
    do not exist;
  - ``not-a-source``: its citations resolve, but to notes that are not sources;
  - ``none``: nothing it links to could be a citation.

Notes in the format's ``ignored_folders`` are outside it: their links do not make a
source linked. ``citation_gap`` (the reasons above) is shared with `vault pipeline`.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from vaultengine.formats.typed import RoledLink, TypedVault


def citation_gap(typed: TypedVault, path: str, roles: set[str], links: list[RoledLink] | None = None
                 ) -> tuple[str, list[RoledLink]]:
    """Why a note that cites no source cites none: ``missing`` (citations whose targets do
    not exist), ``not-a-source`` (they resolve to notes that are not sources) or ``none``;
    with the citations that show it."""
    links = typed.links(path) if links is None else links
    citations = [l for l in links if l.role in roles]
    missing = [l for l in citations if l.edge.status != "resolved"]
    elsewhere = [l for l in citations if l.edge.status == "resolved"]
    return ("missing" if missing else "not-a-source" if elsewhere else "none"), (missing or elsewhere)


def citation_dict(typed: TypedVault, link: RoledLink) -> dict[str, Any]:
    return {"line": link.edge.link.line, "raw": link.edge.link.raw, "role": link.role, "status": link.edge.status,
            "target_path": link.edge.target_path,
            "target_type": typed.type_of(link.edge.target_path) if link.edge.target_path else None}


def coverage(typed: TypedVault) -> dict[str, Any]:
    config = typed.spec.coverage
    if config is None:
        raise ValueError(f"format {typed.spec.name!r} has no `coverage:` block")
    source_types = set(config.sources)
    roles = set(config.citation_roles)
    sources = sorted(p for p in typed.vault.notes if typed.type_of(p) in source_types)

    unlinked = []
    linked = 0
    for path in sources:
        incoming = [p for p in typed.graph.backlink_sources(path) if not typed.is_ignored(p)]
        if incoming:
            linked += 1
        else:
            unlinked.append(path)

    citing_notes = sorted(p for p in typed.vault.notes if typed.type_of(p) in set(config.citing))
    without = []
    counts: Counter[str] = Counter()
    cited_by: Counter[str] = Counter()
    for path in citing_notes:
        links = typed.links(path)
        cites = [l for l in links if l.edge.status == "resolved" and typed.type_of(l.edge.target_path) in source_types
                 and l.edge.target_path != path]
        if cites:
            counts["cites"] += 1
            cited_by.update({l.edge.target_path for l in cites})
            continue
        reason, shown = citation_gap(typed, path, roles, links)
        counts[reason] += 1
        without.append({
            "path": path,
            "type": typed.type_of(path),
            "reason": reason,
            "citations": [citation_dict(typed, l) for l in shown],
        })
    return {
        "format": {"name": typed.spec.name, "path": typed.spec.path},
        "source_types": sorted(source_types),
        "citing_types": list(config.citing),
        "citation_roles": sorted(roles),
        "sources": {"total": len(sources), "linked": linked, "unlinked": unlinked},
        "citing": {
            "total": len(citing_notes),
            "cite_a_source": counts["cites"],
            "without_source": len(without),
            "by_reason": {"missing": counts["missing"], "not_a_source": counts["not-a-source"], "none": counts["none"]},
            "notes": without,
        },
        "most_cited": [{"path": p, "citing_notes": n} for p, n in sorted(cited_by.items(), key=lambda kv: (-kv[1], kv[0]))[:5]],
    }
