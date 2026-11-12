"""Map history: `vault map --record` saves a small snapshot of the map; `vault map --since DATE` says what changed.

A snapshot (``memory.py``, compressed JSON) holds what the comparison needs and nothing else:

* the communities: number, name, their notes (paths), their topic pages;
* the gaps the gap lens finds (with embeddings when the model is there, else key terms; the basis is kept and
  used again): the two communities, the links between their notes then (as note pairs), what chance predicted;
* the missing notes: name and how many notes link them;
* counts (notes, links).

What changed, against the vault now:

gaps closed
    A recorded gap's two sides are the notes its two communities had then. They are measured now as the gap lens
    measures two communities (links between them against what the configuration model expects from their notes'
    link degrees). The gap is *closed* when the links are no longer fewer than chance predicts, or when both sides
    are one community now; otherwise it is *open*. The links added between the sides since are listed. (Measuring
    the recorded sides, not today's communities, keeps renumbering and reshuffled communities out of it.)
communities formed / gone
    Each community now is matched with the recorded community it shares the most notes with (Jaccard). One whose
    best match is below ``SAME`` (one half) was *formed*; a recorded one that no community now matches at
    ``SAME`` is *gone* (merged or split).
missing notes written
    A recorded missing note whose name now resolves to a note.
topic pages
    A recorded community without a topic page whose match now has one.
"""

from __future__ import annotations

from typing import Any

SAME = 0.5


class SideLinks:
    """Links between groups of notes (each note in at most one group: the communities of a snapshot), measured as the
    gap lens measures two communities (``Discover.link_ratio``: link weight between them against what the
    configuration model expects from their notes' link degrees). One pass over the graph for every gap; adding in
    the order a pass per gap did, so the numbers are the same."""

    def __init__(self, graph, group_of: dict[str, int]) -> None:
        from collections import Counter, defaultdict

        self.degree: Counter[int] = Counter()
        self.between: Counter[tuple[int, int]] = Counter()  # (group of a, group of b) -> weight, a -> b
        self.total = 0.0
        group = {v: group_of[label] for v, label in enumerate(graph.labels) if label in group_of}
        for a in range(len(graph.labels)):
            if graph.is_missing(a):
                continue
            for b, w in graph.adjacency[a].items():
                if graph.is_missing(b) or b == a:
                    continue
                self.total += w
                if a in group:
                    self.degree[group[a]] += w
                    if b in group:
                        self.between[(group[a], group[b])] += w
        self.pairs: dict[tuple[int, int], set[tuple[str, str]]] = defaultdict(set)
        for a, b in graph.direct:
            if a in group and b in group:
                x, y = sorted((group[a], group[b]))
                self.pairs[(x, y)].add(tuple(sorted((graph.labels[a], graph.labels[b]))))

    def measure(self, x: int, y: int) -> tuple[float, float]:
        """(link weight from group x to group y, what chance predicts)."""
        return float(self.between[(x, y)]), (self.degree[x] * self.degree[y] / self.total if self.total else 0.0)

    def links(self, x: int, y: int) -> list[tuple[str, str]]:
        """The linked note pairs between the two groups (paths, each pair sorted)."""
        return sorted(self.pairs[(min(x, y), max(x, y))])


def gaps_now(discover, meaning: bool) -> dict[str, Any]:
    """Every gap the gap lens finds, without the notes that would join them (not needed here, and the costly part)."""
    r = discover.gaps(meaning, None, 0, joiners=0)
    if "unavailable" in r:
        return discover.gaps(False, None, 0, joiners=0) | {"basis": "key terms"}
    return r | {"basis": "meaning" if meaning else "key terms"}


def snapshot(shape, discover, meaning: bool) -> dict[str, Any]:
    sg, graph = shape.sg, shape.graph
    communities = [{"number": c.number, "name": c.name, "notes": sorted(sg.labels[v] for v in c.notes),
                    "topic_pages": [sg.labels[v] for v, _ in c.topic_pages]} for c in shape.communities]
    sides = SideLinks(graph, {path: c["number"] for c in communities for path in c["notes"]})
    g = gaps_now(discover, meaning)
    gaps = [{"communities": row["communities"], "names": row["names"],
             **dict(zip(("links", "expected"), sides.measure(*row["communities"]))),
             "pairs": sides.links(*row["communities"])}
            for row in g["results"]]
    missing = {}
    for v in range(len(graph.labels)):
        if graph.is_missing(v):
            linkers = sum(1 for u in graph.adjacency[v] if not graph.is_missing(u))
            if linkers:
                missing[graph.labels[v]] = linkers
    return {"version": 1, "notes": len(sg.notes), "links": len(graph.direct), "topic_pages_known": shape.topic_page is not None,
            "communities": communities, "gaps": gaps, "gap_basis": g["basis"], "missing": missing}


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def changes(old: dict[str, Any], shape, discover, engine) -> dict[str, Any]:
    from vaultengine.vault import lookup_key

    sg, graph = shape.sg, shape.graph
    now = [{"number": c.number, "name": c.name, "notes": {sg.labels[v] for v in c.notes},
            "topic_pages": [sg.labels[v] for v, _ in c.topic_pages]} for c in shape.communities]
    before = [dict(c, notes=set(c["notes"])) for c in old["communities"]]
    best_old = {c["number"]: max(((_jaccard(c["notes"], o["notes"]), o["number"]) for o in before), default=(0.0, None)) for c in now}
    best_now = {o["number"]: max(((_jaccard(o["notes"], c["notes"]), c["number"]) for c in now), default=(0.0, None)) for o in before}
    formed = [{"community": c["number"], "name": c["name"], "notes": len(c["notes"]),
               "best_match": best_old[c["number"]][1], "overlap": best_old[c["number"]][0]}
              for c in now if best_old[c["number"]][0] < SAME]
    gone = [{"community": o["number"], "name": o["name"], "notes": len(o["notes"]), "overlap": best_now[o["number"]][0]}
            for o in before if best_now[o["number"]][0] < SAME]

    # Gaps: the recorded sides (their notes then), measured now as the gap lens measures two communities.
    membership = shape.membership
    where = {sg.labels[v]: membership[v] for v in sg.notes}
    old_members = {o["number"]: o["notes"] for o in before}
    sides = SideLinks(graph, {path: o["number"] for o in before for path in o["notes"]})
    gaps = []
    for gap in old["gaps"]:
        x, y = gap["communities"]
        xs, ys = old_members.get(x, set()), old_members.get(y, set())
        links, expected = sides.measure(x, y)
        pairs_now = sides.links(x, y)
        then = {tuple(p) for p in gap["pairs"]}
        added = [list(p) for p in pairs_now if p not in then]
        homes = lambda side: {where[p] for p in side if p in where}
        entry = {"communities": [x, y], "names": gap["names"], "links_then": gap["links"], "expected_then": gap["expected"],
                 "links_now": links, "expected_now": expected, "added": added}
        if not homes(xs) or not homes(ys):
            entry.update(status="gone", why="the notes of one side are gone")
        elif len(homes(xs) | homes(ys)) == 1:
            entry.update(status="closed", why=f"both sides are one community now ({next(iter(homes(xs)))})")
        elif expected and links >= expected:
            entry.update(status="closed", why=f"{links:g} links between the two sides where chance predicts {expected:.1f}:"
                                               " no longer fewer than chance")
        else:
            entry.update(status="open", why=f"{links:g} links between the two sides where chance predicts {expected:.1f}")
        gaps.append(entry)

    names_now = {lookup_key(graph.labels[v][:-3] if graph.labels[v].endswith(".md") else graph.labels[v]): graph.labels[v]
                 for v in sg.notes}
    from vaultengine.vault import link_name

    names_now.update({lookup_key(link_name(graph.labels[v])): graph.labels[v] for v in sg.notes})
    written = [{"name": name, "linked_from": n, "note": names_now[lookup_key(name)]}
               for name, n in sorted(old["missing"].items(), key=lambda item: (-item[1], item[0]))
               if lookup_key(name) in names_now]
    pages = []
    for o in before:
        if o["topic_pages"] or not old.get("topic_pages_known"):
            continue
        score, number = best_now[o["number"]]
        match = next((c for c in now if c["number"] == number), None)
        if match is not None and score >= SAME and match["topic_pages"]:
            pages.append({"community_then": o["number"], "name": o["name"], "community": number, "topic_pages": match["topic_pages"]})
    return {"notes": [old["notes"], len(sg.notes)], "links": [old["links"], len(graph.direct)],
            "communities": [len(before), len(now)], "formed": formed, "gone": gone, "gaps": gaps,
            "gap_basis": old.get("gap_basis"), "written": written, "topic_pages": pages}
