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
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from vaultengine.formats.typed import TypedVault


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
        incoming = typed.graph.backlink_sources(path)
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
        citations = [l for l in links if l.role in roles]
        missing = [l for l in citations if l.edge.status != "resolved"]
        elsewhere = [l for l in citations if l.edge.status == "resolved"]
        reason = "missing" if missing else "not-a-source" if elsewhere else "none"
        counts[reason] += 1
        shown = missing or elsewhere
        without.append({
            "path": path,
            "type": typed.type_of(path),
            "reason": reason,
            "citations": [
                {"line": l.edge.link.line, "raw": l.edge.link.raw, "role": l.role, "status": l.edge.status,
                 "target_path": l.edge.target_path, "target_type": typed.type_of(l.edge.target_path) if l.edge.target_path else None}
                for l in shown
            ],
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
