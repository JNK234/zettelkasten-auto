# ABOUTME: Bundle assembly for exporting connected knowledge.
# ABOUTME: Collects source + zettels + semantically and explicitly linked notes.

import re
from collections import deque
from datetime import datetime
from pathlib import Path


def extract_tags(content: str) -> list[str]:
    """Extract [[links]] from the Tags: line."""
    match = re.search(r"^Tags:(.*)$", content, re.MULTILINE)
    if not match:
        return []
    return re.findall(r"\[\[([^\]]+)\]\]", match.group(1))


def extract_see_also(content: str) -> list[str]:
    """Extract [[links]] from the # See Also section."""
    match = re.search(r"(?ms)^# See Also\s*\n(.*?)(?=^# |\Z)", content)
    if not match:
        return []
    return re.findall(r"\[\[([^\]]+)\]\]", match.group(1))


def extract_references(content: str) -> list[str]:
    """Extract [[links]] from the # References section."""
    match = re.search(r"(?ms)^# References\s*\n(.*?)(?=^# |\Z)", content)
    if not match:
        return []
    return re.findall(r"\[\[([^\]]+)\]\]", match.group(1))


def normalize_link(title: str) -> str:
    """Normalize title to filename format for consistent linking."""
    if "|" in title:
        title = title.split("|")[-1]
    normalized = re.sub(r"[^\w\s-]", "", title)
    normalized = re.sub(r"[\s_]+", "-", normalized.strip())
    normalized = re.sub(r"-+", "-", normalized)
    return normalized[:100].lower()


def sanitize_filename(title: str) -> str:
    """Convert title to safe filename."""
    return normalize_link(title)


def find_zettels_from_source(zettel_dir: Path, source_name: str) -> list[Path]:
    """Find all zettels that reference a source in their References section."""
    normalized_source = normalize_link(source_name)
    matching: list[Path] = []

    for zettel_path in sorted(zettel_dir.glob("*.md")):
        refs = extract_references(zettel_path.read_text())
        if any(normalize_link(ref) == normalized_source for ref in refs):
            matching.append(zettel_path)
    return matching


def traverse_tag_connections(
    zettel_dir: Path,
    seed_titles: list[str],
    depth: int,
) -> dict[int, list[str]]:
    """Traverse tag relationships using BFS, grouped by degree."""
    normalized_to_path = {
        normalize_link(path.stem): path
        for path in sorted(zettel_dir.glob("*.md"))
    }
    visited = {normalize_link(title) for title in seed_titles}
    queue = deque()
    result: dict[int, list[str]] = {}

    for title in seed_titles:
        normalized = normalize_link(title)
        path = normalized_to_path.get(normalized)
        if not path:
            continue
        for tag in extract_tags(path.read_text()):
            tag_normalized = normalize_link(tag)
            if tag_normalized not in visited and tag_normalized in normalized_to_path:
                visited.add(tag_normalized)
                queue.append((tag_normalized, 1))

    while queue:
        current_title, current_depth = queue.popleft()
        if current_depth > depth:
            continue

        result.setdefault(current_depth, []).append(current_title)
        if current_depth == depth:
            continue

        current_path = normalized_to_path.get(current_title)
        if not current_path:
            continue

        for tag in extract_tags(current_path.read_text()):
            tag_normalized = normalize_link(tag)
            if tag_normalized not in visited and tag_normalized in normalized_to_path:
                visited.add(tag_normalized)
                queue.append((tag_normalized, current_depth + 1))

    return result


def build_bundle_content(
    source_content: str | None,
    source_name: str,
    zettel_dir: Path,
    connected_by_degree: dict[int, list[str]],
    title: str,
    metadata_summary: dict[str, int],
) -> str:
    """Assemble the final bundle markdown."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"# Bundle: {title}",
        f"Generated: {timestamp}",
        "",
    ]

    if source_content is not None:
        lines.extend([
            "## Source Content",
            "",
            source_content,
            "",
        ])

    lines.extend([
        "## Connected Knowledge",
        "",
    ])

    included_titles: set[str] = set()
    for degree in sorted(connected_by_degree):
        titles = [title for title in connected_by_degree[degree] if title not in included_titles]
        if not titles:
            continue

        lines.extend([
            f"### Degree {degree}",
            "",
        ])
        for zettel_title in titles:
            zettel_path = zettel_dir / f"{zettel_title}.md"
            if not zettel_path.exists():
                continue
            included_titles.add(zettel_title)
            lines.extend([
                f"#### {zettel_title}",
                "",
                zettel_path.read_text().rstrip(),
                "",
            ])

    lines.extend([
        "## Metadata",
        "",
        f"Source: [[{source_name}]]",
        f"Source notes: {metadata_summary.get('source_notes', 0)}",
        f"Direct tags: {metadata_summary.get('direct_tags', 0)}",
        f"Direct similar notes: {metadata_summary.get('direct_similar_notes', 0)}",
        f"Traversed notes: {metadata_summary.get('traversed_notes', 0)}",
        f"Total notes included: {len(included_titles)}",
        f"Max depth: {max(connected_by_degree.keys()) if connected_by_degree else 0}",
    ])

    return "\n".join(lines).rstrip() + "\n"


def bundle_from_source(
    source_path: Path,
    zettel_dir: Path,
    bundles_dir: Path,
    client,
    config: dict,
    dry_run: bool = False,
) -> tuple[Path | None, str]:
    """Bundle around a source file."""
    from src.db import find_similar

    source_name = source_path.stem
    source_content = source_path.read_text()
    extracted_zettels = find_zettels_from_source(zettel_dir, source_name)
    source_note_titles = [normalize_link(path.stem) for path in extracted_zettels]

    bundle_config = config.get("bundle", {})
    embedding_config = config.get("embeddings", {})
    depth = bundle_config.get("default_depth", 1)
    top_k = bundle_config.get("default_top_k", 10)
    max_distance = embedding_config.get("max_distance", 0.5)
    embedding_provider = embedding_config.get("provider", "openai")
    embedding_model = embedding_config.get("model")

    similar_titles = [
        normalize_link(title)
        for title in find_similar(
            client,
            source_content,
            top_k,
            max_distance,
            embedding_provider,
            embedding_model,
            create=not dry_run,
        )
    ]
    direct_similar = [title for title in similar_titles if title not in source_note_titles]
    degree_zero = source_note_titles + [title for title in direct_similar if title not in source_note_titles]

    connections = traverse_tag_connections(zettel_dir, degree_zero, depth)
    connected_by_degree: dict[int, list[str]] = {0: degree_zero}
    connected_by_degree.update(connections)

    traversed_notes = sum(len(titles) for degree, titles in connected_by_degree.items() if degree > 0)
    content = build_bundle_content(
        source_content=source_content,
        source_name=source_name,
        zettel_dir=zettel_dir,
        connected_by_degree=connected_by_degree,
        title=source_name,
        metadata_summary={
            "source_notes": len(source_note_titles),
            "direct_tags": 0,
            "direct_similar_notes": len(direct_similar),
            "traversed_notes": traversed_notes,
        },
    )

    if dry_run:
        return None, content

    bundles_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = bundles_dir / f"{sanitize_filename(source_name)}-{timestamp}.md"
    output_path.write_text(content)
    return output_path, content


def bundle_from_note(
    zettel_title: str,
    zettel_dir: Path,
    bundles_dir: Path,
    client,
    config: dict,
    dry_run: bool = False,
) -> tuple[Path | None, str]:
    """Bundle around an existing zettel."""
    del client

    normalized_title = normalize_link(zettel_title)
    zettel_path = zettel_dir / f"{normalized_title}.md"
    if not zettel_path.exists():
        for candidate in sorted(zettel_dir.glob("*.md")):
            if normalize_link(candidate.stem) == normalized_title:
                zettel_path = candidate
                break

    if not zettel_path.exists():
        raise FileNotFoundError(f"Zettel not found: {zettel_title}")

    zettel_content = zettel_path.read_text()
    bundle_config = config.get("bundle", {})
    depth = bundle_config.get("default_depth", 1)

    direct_tags = [normalize_link(tag) for tag in extract_tags(zettel_content)]
    direct_similar = [normalize_link(title) for title in extract_see_also(zettel_content)]

    degree_zero = [normalized_title]
    for title in direct_tags + direct_similar:
        if title not in degree_zero:
            degree_zero.append(title)

    connections = traverse_tag_connections(zettel_dir, degree_zero, depth)
    connected_by_degree: dict[int, list[str]] = {0: degree_zero}
    connected_by_degree.update(connections)

    traversed_notes = sum(len(titles) for degree, titles in connected_by_degree.items() if degree > 0)
    content = build_bundle_content(
        source_content=None,
        source_name=zettel_title,
        zettel_dir=zettel_dir,
        connected_by_degree=connected_by_degree,
        title=zettel_title,
        metadata_summary={
            "source_notes": 1,
            "direct_tags": len(direct_tags),
            "direct_similar_notes": len(direct_similar),
            "traversed_notes": traversed_notes,
        },
    )

    if dry_run:
        return None, content

    bundles_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = bundles_dir / f"{sanitize_filename(zettel_title)}-{timestamp}.md"
    output_path.write_text(content)
    return output_path, content
