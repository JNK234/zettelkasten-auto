# ABOUTME: Main orchestration module for Zettelkasten automation.
# ABOUTME: Processes source files, extracts concepts, maintains index state, and creates bundles.

import argparse
import math
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv

from src.bundle import bundle_from_note, bundle_from_source
from src.db import (
    delete_ids,
    find_similar,
    get_client,
    get_collection,
    get_embedding_function,
    get_index_drift,
    index_zettel,
)
from src.llm import (
    InvalidLLMOutputError,
    ProviderResponseError,
    extract_concepts_with_diagnostics,
)

load_dotenv()


DEFAULT_CONFIG = {
    "vault": {
        "root": ".",
        "sources_dir": "10_SOURCES",
        "zettel_dir": "20_ZETTLEKASTEN",
        "bundles_dir": "110_BUNDLES",
        "template_path": "60_TEMPLATES/Note Template.md",
    },
    "llm": {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "base_url": "http://localhost:11434",
    },
    "embeddings": {
        "db_path": "chromastores/zettledb",
        "provider": "sentence-transformer",
        "model": "all-MiniLM-L6-v2",
        "top_k": 5,
        "max_distance": 0.5,
    },
    "bundle": {
        "default_depth": 1,
        "default_top_k": 10,
    },
    "processing": {
        "min_source_chars": 100,
        "duplicate_max_distance": 0.15,
        "chunk_target_chars": 4000,
        "chunk_max_chars": 6000,
    },
}


def _deep_merge(defaults: dict, overrides: dict) -> dict:
    """Recursively merge configuration dictionaries."""
    merged = dict(defaults)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config() -> dict:
    """Load YAML configuration from the config directory."""
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(config_path, encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    return _deep_merge(DEFAULT_CONFIG, loaded)


def resolve_path(vault_root: Path, configured_path: str) -> Path:
    """Resolve config paths relative to the vault root unless already absolute."""
    path = Path(configured_path).expanduser()
    if path.is_absolute():
        return path
    return vault_root / path


def read_file_with_retry(path: Path, max_retries: int = 3, delay: float = 0.5) -> str:
    """Read file with retry logic for transient iCloud sync issues."""
    for attempt in range(max_retries):
        try:
            return path.read_text(encoding="utf-8")
        except OSError as exc:
            if exc.errno == 11 and attempt < max_retries - 1:
                time.sleep(delay * (attempt + 1))
                continue
            raise
    return path.read_text(encoding="utf-8")


def list_markdown_files(directory: Path) -> list[Path]:
    """Return markdown files in a directory in deterministic order."""
    return sorted(directory.glob("*.md"))


def sanitize_filename(title: str) -> str:
    """Convert title to safe filename format."""
    sanitized = re.sub(r"[^\w\s-]", "", title)
    sanitized = re.sub(r"[\s_]+", "-", sanitized.strip())
    sanitized = re.sub(r"-+", "-", sanitized)
    return sanitized[:100].lower()


def normalize_link(title: str) -> str:
    """Normalize titles and Obsidian links to the filename format."""
    if "|" in title:
        title = title.split("|")[-1]
    return sanitize_filename(title)


def extract_tags(content: str) -> list[str]:
    """Extract [[links]] from the Tags line."""
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


def remove_see_also_section(content: str) -> str:
    """Remove the # See Also section from a note."""
    return re.sub(r"(?ms)\n*^# See Also\s*\n.*?(?=^# |\Z)", "\n", content).strip() + "\n"


def _replace_markdown_section(content: str, heading: str, section_body: str) -> str:
    """Replace a markdown section body or append the section if it does not exist."""
    section_text = f"# {heading}\n\n{section_body.strip()}".rstrip() if section_body.strip() else f"# {heading}"
    pattern = rf"(?ms)^# {re.escape(heading)}\s*\n.*?(?=^# |\Z)"
    if re.search(pattern, content):
        updated = re.sub(pattern, section_text.rstrip() + "\n\n", content).rstrip()
        return updated + "\n"
    if heading != "References" and re.search(r"(?m)^# References\b", content):
        updated = re.sub(r"(?m)^# References\b", f"{section_text}\n\n# References", content, count=1).rstrip()
        return updated + "\n"
    return content.rstrip() + f"\n\n{section_text}\n"


def _processed_status_match(content: str) -> re.Match[str] | None:
    return re.search(r"^Status:(.*)$", content, re.MULTILINE)


def _has_legacy_processed_marker(content: str) -> bool:
    nonempty_lines = [line.strip() for line in content.splitlines() if line.strip()]
    return bool(nonempty_lines and nonempty_lines[-1] == "#processed")


def is_source_processed(content: str) -> bool:
    """Treat a source as processed only via Status or legacy terminal marker."""
    match = _processed_status_match(content)
    if match and re.search(r"(^|\s)#processed(\s|$)", match.group(1)):
        return True
    return _has_legacy_processed_marker(content)


def mark_processed(source_path: Path) -> None:
    """Mark a source as processed via the Status line."""
    content = source_path.read_text(encoding="utf-8")
    if is_source_processed(content):
        return

    match = _processed_status_match(content)
    if match:
        updated = re.sub(
            r"^(Status:.*)$",
            lambda found: f"{found.group(1)} #processed",
            content,
            count=1,
            flags=re.MULTILINE,
        )
    else:
        updated = content.rstrip() + "\n\nStatus: #processed\n"
    source_path.write_text(updated, encoding="utf-8")


def load_template(template_path: Path) -> str:
    """Load the zettel template from disk."""
    return template_path.read_text(encoding="utf-8")


def write_zettel(
    output_dir: Path,
    title: str,
    content: str,
    tags: list[str],
    similar: list[str],
    source: str,
    template: str,
) -> Path:
    """Write a zettel file using the configured template."""
    output_path = output_dir / f"{sanitize_filename(title)}.md"
    date_str = datetime.now().strftime("%d-%m-%Y")
    time_str = datetime.now().strftime("%H:%M")

    normalized_tags = []
    for tag in tags:
        normalized = normalize_link(tag)
        if normalized and normalized not in normalized_tags:
            normalized_tags.append(normalized)

    normalized_similar = []
    for related in similar:
        normalized = normalize_link(related)
        if normalized and normalized not in normalized_similar:
            normalized_similar.append(normalized)

    references_body = f"1. [[{source}]]"
    see_also_body = "\n".join(f"- [[{title}]]" for title in normalized_similar)

    zettel_content = template
    zettel_content = zettel_content.replace("{{date}}", date_str)
    zettel_content = zettel_content.replace("{{time}}", time_str)
    zettel_content = re.sub(
        r"^Tags:.*$",
        f"Tags: {' '.join(f'[[{tag}]]' for tag in normalized_tags)}".rstrip(),
        zettel_content,
        flags=re.MULTILINE,
    )

    if "# {{Title}}" in zettel_content:
        zettel_content = zettel_content.replace("# {{Title}}", f"# {title}\n\n{content.strip()}")
    else:
        zettel_content = zettel_content.replace("{{Title}}", title)
        zettel_content = zettel_content.rstrip() + f"\n\n{content.strip()}\n"

    zettel_content = _replace_markdown_section(zettel_content, "See Also", see_also_body)
    zettel_content = _replace_markdown_section(zettel_content, "References", references_body)
    output_path.write_text(zettel_content.rstrip() + "\n", encoding="utf-8")
    return output_path


def zettel_exists(
    zettel_dir: Path,
    title: str,
    content: str,
    client,
    embedding_provider: str,
    embedding_model: str | None,
    duplicate_threshold: float,
) -> tuple[bool, str | None]:
    """Check if a zettel already exists by filename or semantic similarity."""
    filename = sanitize_filename(title) + ".md"
    if (zettel_dir / filename).exists():
        return True, f"file exists: {filename}"

    similar = find_similar(
        client,
        content,
        top_k=1,
        max_distance=duplicate_threshold,
        provider=embedding_provider,
        model_name=embedding_model,
        create=True,
    )
    if similar:
        return True, f"too similar to: {similar[0]}"
    return False, None


def cosine_distance(vector_a: list[float], vector_b: list[float]) -> float:
    """Compute cosine distance without requiring numpy."""
    dot = sum(a * b for a, b in zip(vector_a, vector_b))
    norm_a = math.sqrt(sum(a * a for a in vector_a))
    norm_b = math.sqrt(sum(b * b for b in vector_b))
    if norm_a == 0 or norm_b == 0:
        return 1.0
    similarity = dot / (norm_a * norm_b)
    similarity = max(min(similarity, 1.0), -1.0)
    return 1.0 - similarity


def dedupe_candidate_concepts(
    concepts: list[dict],
    embedding_provider: str,
    embedding_model: str | None,
    duplicate_threshold: float,
) -> list[dict]:
    """Deduplicate concepts by normalized title, then by semantic similarity."""
    title_deduped: list[dict] = []
    seen_titles: set[str] = set()
    for concept in concepts:
        normalized_title = normalize_link(concept["title"])
        if normalized_title in seen_titles:
            continue
        seen_titles.add(normalized_title)
        title_deduped.append(concept)

    if len(title_deduped) < 2:
        return title_deduped

    try:
        embedding_fn = get_embedding_function(embedding_provider, embedding_model)
        embeddings = embedding_fn([concept["content"] for concept in title_deduped])
    except Exception as exc:
        print(f"  Warning: semantic dedupe skipped ({exc})")
        return title_deduped

    kept_concepts: list[dict] = []
    kept_embeddings: list[list[float]] = []
    for concept, embedding in zip(title_deduped, embeddings):
        is_duplicate = any(
            cosine_distance(embedding, existing_embedding) <= duplicate_threshold
            for existing_embedding in kept_embeddings
        )
        if is_duplicate:
            continue
        kept_concepts.append(concept)
        kept_embeddings.append(embedding)
    return kept_concepts


def split_large_paragraph(paragraph: str, max_chars: int) -> list[str]:
    """Split a large paragraph into sentence-ish chunks when necessary."""
    if len(paragraph) <= max_chars:
        return [paragraph]

    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        candidate = sentence if not current else f"{current} {sentence}"
        if current and len(candidate) > max_chars:
            chunks.append(current.strip())
            current = sentence
        else:
            current = candidate
    if current.strip():
        chunks.append(current.strip())

    final_chunks: list[str] = []
    for chunk in chunks:
        if len(chunk) <= max_chars:
            final_chunks.append(chunk)
            continue
        for index in range(0, len(chunk), max_chars):
            final_chunks.append(chunk[index:index + max_chars].strip())
    return [chunk for chunk in final_chunks if chunk]


def split_section_by_paragraphs(section: str, max_chars: int) -> list[str]:
    """Split a markdown section on paragraph boundaries."""
    paragraphs = re.split(r"\n\s*\n", section.strip())
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        parts = split_large_paragraph(paragraph.strip(), max_chars)
        for part in parts:
            candidate = part if not current else f"{current}\n\n{part}"
            if current and len(candidate) > max_chars:
                chunks.append(current.strip())
                current = part
            else:
                current = candidate
    if current.strip():
        chunks.append(current.strip())
    return chunks


def split_source_content(content: str, chunk_target_chars: int, chunk_max_chars: int) -> list[str]:
    """Split source content by headings first, then by paragraphs."""
    stripped = content.strip()
    if len(stripped) <= chunk_max_chars:
        return [stripped]

    heading_matches = list(re.finditer(r"(?m)^#{1,6}\s+.*$", stripped))
    sections: list[str] = []
    if not heading_matches:
        sections = split_section_by_paragraphs(stripped, chunk_max_chars)
    else:
        last_index = 0
        for match in heading_matches:
            if match.start() > last_index:
                preamble = stripped[last_index:match.start()].strip()
                if preamble:
                    sections.append(preamble)
            last_index = match.start()
        for index, match in enumerate(heading_matches):
            end = heading_matches[index + 1].start() if index + 1 < len(heading_matches) else len(stripped)
            section = stripped[match.start():end].strip()
            if section:
                sections.append(section)

    normalized_sections: list[str] = []
    for section in sections:
        if len(section) > chunk_max_chars:
            normalized_sections.extend(split_section_by_paragraphs(section, chunk_max_chars))
        else:
            normalized_sections.append(section)

    chunks: list[str] = []
    current = ""
    for section in normalized_sections:
        candidate = section if not current else f"{current}\n\n{section}"
        if current and len(candidate) > chunk_target_chars:
            chunks.append(current.strip())
            current = section
        else:
            current = candidate
    if current.strip():
        chunks.append(current.strip())
    return [chunk for chunk in chunks if chunk]


def looks_non_extractable_source(content: str) -> bool:
    """Apply a lightweight heuristic to classify non-knowledge sources."""
    lowered = content.lower()
    keyword_hits = sum(
        keyword in lowered
        for keyword in [
            "todo",
            "to-do",
            "meeting",
            "journal",
            "daily note",
            "progress update",
            "changelog",
            "build log",
            "status update",
        ]
    )
    nonempty_lines = [line.strip() for line in content.splitlines() if line.strip()]
    list_lines = sum(
        line.startswith(("- ", "* ", "1. ", "2. ", "3. ", "[ ]", "[x]"))
        for line in nonempty_lines
    )
    list_ratio = (list_lines / len(nonempty_lines)) if nonempty_lines else 0
    return keyword_hits >= 2 or list_ratio >= 0.6


def extract_source_concepts(
    content: str,
    llm_provider: str,
    llm_model: str | None,
    llm_kwargs: dict,
    chunk_target_chars: int,
    chunk_max_chars: int,
) -> tuple[list[dict], dict[str, int | str | None]]:
    """Extract concepts from source content with chunking and diagnostics."""
    chunks = split_source_content(content, chunk_target_chars, chunk_max_chars)
    concepts: list[dict] = []
    diagnostics: dict[str, int | str | None] = {
        "chunk_count": len(chunks),
        "invalid_chunks": 0,
        "invalid_items": 0,
        "provider_error": None,
        "had_structured_output": 0,
    }

    for index, chunk in enumerate(chunks, start=1):
        try:
            chunk_concepts, invalid_items = extract_concepts_with_diagnostics(
                chunk,
                llm_provider,
                llm_model,
                **llm_kwargs,
            )
        except ProviderResponseError as exc:
            diagnostics["invalid_chunks"] += 1
            print(f"  Chunk {index}/{len(chunks)} provider error: {exc}")
            continue
        except InvalidLLMOutputError as exc:
            diagnostics["invalid_chunks"] += 1
            print(f"  Chunk {index}/{len(chunks)} invalid output: {exc}")
            continue

        if chunk_concepts or invalid_items:
            diagnostics["had_structured_output"] = 1
        if invalid_items:
            diagnostics["invalid_items"] += invalid_items
            print(f"  Chunk {index}/{len(chunks)} dropped {invalid_items} invalid concept(s).")

        concepts.extend(chunk_concepts)
    return concepts, diagnostics


def collect_zettel_entries(zettel_dir: Path) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Collect zettel contents keyed by normalized ID, merging collisions."""
    grouped_contents: dict[str, list[str]] = {}
    grouped_files: dict[str, list[str]] = {}

    for zettel_path in list_markdown_files(zettel_dir):
        normalized_id = normalize_link(zettel_path.stem)
        grouped_files.setdefault(normalized_id, []).append(zettel_path.name)
        content = zettel_path.read_text(encoding="utf-8").strip()
        if content:
            grouped_contents.setdefault(normalized_id, []).append(content)

    entries = {
        entry_id: "\n\n---\n\n".join(contents)
        for entry_id, contents in grouped_contents.items()
        if contents
    }
    collisions = {
        entry_id: names
        for entry_id, names in grouped_files.items()
        if len(names) > 1
    }
    return entries, collisions


def refresh_active_index(
    zettel_dir: Path,
    client,
    embedding_provider: str,
    embedding_model: str | None,
    dry_run: bool,
) -> dict[str, list[str]]:
    """Refresh or validate the active index against current zettel files."""
    entries, collisions = collect_zettel_entries(zettel_dir)
    drift = get_index_drift(
        client,
        entries,
        embedding_provider,
        embedding_model,
        create=not dry_run,
    )

    if collisions:
        print("Index collisions detected:")
        for entry_id, names in sorted(collisions.items()):
            print(f"  {entry_id}: {names}")

    if not drift["missing_or_stale_ids"] and not drift["extra_ids"]:
        return drift

    if dry_run:
        raise RuntimeError(
            "Active index is stale: "
            f"{len(drift['missing_or_stale_ids'])} missing/stale entry(s), "
            f"{len(drift['extra_ids'])} extra entry(s). Run `zettel rebuild-index` first."
        )

    if drift["extra_ids"]:
        delete_ids(client, drift["extra_ids"], embedding_provider, embedding_model)
    for entry_id in drift["missing_or_stale_ids"]:
        index_zettel(
            client,
            entry_id,
            entries[entry_id],
            embedding_provider,
            embedding_model,
            force=True,
        )
    return drift


def sync_backlinks(zettel_dir: Path, dry_run: bool = False) -> None:
    """Legacy maintenance: migrate See Also sections to Tags and ensure backlinks."""
    zettels = list_markdown_files(zettel_dir)
    print(f"Scanning {len(zettels)} zettels...")
    if dry_run:
        print("*** DRY RUN MODE ***\n")

    see_also_to_remove = 0
    tags_to_add_from_see_also = 0
    backlinks_to_add = 0
    normalized_to_path = {normalize_link(path.stem): path for path in zettels}
    zettel_contents = {path: path.read_text(encoding="utf-8") for path in zettels}
    pending_backlinks: dict[Path, set[str]] = {path: set() for path in zettels}

    migrations: list[tuple[Path, list[str]]] = []
    for zettel_path, content in zettel_contents.items():
        see_also_links = extract_see_also(content)
        if see_also_links:
            migrations.append((zettel_path, see_also_links))
            see_also_to_remove += 1
            tags_to_add_from_see_also += len(see_also_links)

    if migrations:
        print("Migrating See Also to Tags:")
        for zettel_path, links in migrations:
            print(f"  {zettel_path.name}: +{len(links)} tags")
            if dry_run:
                continue

            content = zettel_contents[zettel_path]
            existing_tags = extract_tags(content)
            merged_tags = [normalize_link(tag) for tag in existing_tags]
            for link in links:
                normalized = normalize_link(link)
                if normalized not in merged_tags:
                    merged_tags.append(normalized)

            updated = re.sub(
                r"^Tags:.*$",
                f"Tags: {' '.join(f'[[{tag}]]' for tag in merged_tags)}".rstrip(),
                content,
                flags=re.MULTILINE,
            )
            updated = remove_see_also_section(updated)
            zettel_contents[zettel_path] = updated
            zettel_path.write_text(updated, encoding="utf-8")
    else:
        print("No See Also sections found to migrate.")

    print("\nAnalyzing backlinks:")
    for zettel_path in zettels:
        content = zettel_contents[zettel_path]
        zettel_title = normalize_link(zettel_path.stem)
        for linked_title in extract_tags(content):
            linked_normalized = normalize_link(linked_title)
            target_path = normalized_to_path.get(linked_normalized)
            if not target_path:
                continue
            target_tags = {normalize_link(tag) for tag in extract_tags(zettel_contents[target_path])}
            if zettel_title not in target_tags:
                pending_backlinks[target_path].add(zettel_title)

    print("\nAdding backlinks:")
    files_modified = 0
    for target_path, new_links in pending_backlinks.items():
        if not new_links:
            continue

        for link in sorted(new_links):
            print(f"  + [[{link}]] -> {target_path.name}")
            backlinks_to_add += 1

        if dry_run:
            continue

        content = zettel_contents[target_path]
        existing_tags = [normalize_link(tag) for tag in extract_tags(content)]
        for link in sorted(new_links):
            if link not in existing_tags:
                existing_tags.append(link)
        updated = re.sub(
            r"^Tags:.*$",
            f"Tags: {' '.join(f'[[{tag}]]' for tag in existing_tags)}".rstrip(),
            content,
            flags=re.MULTILINE,
        )
        target_path.write_text(updated, encoding="utf-8")
        files_modified += 1

    print("\n=== Summary ===")
    print(f"See Also sections removed: {see_also_to_remove}")
    print(f"Tags added (from See Also): {tags_to_add_from_see_also}")
    print(f"Backlinks added: {backlinks_to_add}")
    if not dry_run:
        print(f"Files modified for backlinks: {files_modified}")


def normalize_links(zettel_dir: Path, dry_run: bool = False) -> None:
    """Normalize all [[links]] in Tags lines to filename format."""
    zettels = list_markdown_files(zettel_dir)
    print(f"Scanning {len(zettels)} zettels...")
    if dry_run:
        print("*** DRY RUN MODE ***\n")

    files_to_update = 0
    links_normalized = 0
    duplicates_removed = 0

    for zettel_path in zettels:
        content = zettel_path.read_text(encoding="utf-8")
        existing_tags = extract_tags(content)
        if not existing_tags:
            continue

        normalized_tags: list[str] = []
        changes: list[tuple[str, str]] = []
        for tag in existing_tags:
            normalized = normalize_link(tag)
            if normalized != tag:
                changes.append((tag, normalized))
            if normalized not in normalized_tags:
                normalized_tags.append(normalized)
            else:
                duplicates_removed += 1

        if existing_tags == normalized_tags and len(existing_tags) == len(normalized_tags):
            continue

        files_to_update += 1
        links_normalized += len(changes)
        print(f"  {zettel_path.name}: {len(changes)} links normalized")
        for old, new in changes:
            print(f"    [[{old}]] -> [[{new}]]")

        if dry_run:
            continue

        updated = re.sub(
            r"^Tags:.*$",
            f"Tags: {' '.join(f'[[{tag}]]' for tag in normalized_tags)}".rstrip(),
            content,
            flags=re.MULTILINE,
        )
        zettel_path.write_text(updated, encoding="utf-8")

    print("\n=== Summary ===")
    print(f"Files to update: {files_to_update}")
    print(f"Links normalized: {links_normalized}")
    print(f"Duplicates removed: {duplicates_removed}")


def rename_files(zettel_dir: Path, dry_run: bool = False) -> None:
    """Rename zettel files to normalized format, merging duplicates."""
    zettels = list_markdown_files(zettel_dir)
    print(f"Scanning {len(zettels)} zettels...")
    if dry_run:
        print("*** DRY RUN MODE ***\n")

    normalized_to_files: dict[str, list[Path]] = {}
    for zettel_path in zettels:
        normalized_to_files.setdefault(normalize_link(zettel_path.stem), []).append(zettel_path)

    files_renamed = 0
    files_merged = 0
    files_deleted = 0

    for normalized_name, files in sorted(normalized_to_files.items()):
        target_path = zettel_dir / f"{normalized_name}.md"
        if len(files) == 1:
            source = files[0]
            if source.name == target_path.name:
                continue
            print(f"  Rename: {source.name} -> {target_path.name}")
            if not dry_run:
                source.rename(target_path)
            files_renamed += 1
            continue

        print(f"  Merge {len(files)} files -> {target_path.name}")
        if dry_run:
            files_merged += 1
            continue

        merged_content = []
        for path in sorted(files, key=lambda candidate: candidate.stat().st_mtime):
            content = path.read_text(encoding="utf-8").strip()
            if content:
                merged_content.append(content)
        target_path.write_text("\n\n---\n\n".join(merged_content).rstrip() + "\n", encoding="utf-8")
        for path in files:
            if path != target_path:
                path.unlink()
                files_deleted += 1
        files_merged += 1

    print("\n=== Summary ===")
    print(f"Files renamed: {files_renamed}")
    print(f"Merge operations: {files_merged}")
    print(f"Files deleted (merged): {files_deleted}")


def rebuild_index(
    zettel_dir: Path,
    client,
    embedding_provider: str,
    embedding_model: str | None,
    dry_run: bool = False,
) -> None:
    """Rebuild the active Chroma collection from current zettel files."""
    entries, collisions = collect_zettel_entries(zettel_dir)
    print(f"Found {len(entries)} normalized zettel entrie(s) to index...")
    if dry_run:
        print("*** DRY RUN MODE ***\n")

    collection = get_collection(client, embedding_provider, embedding_model)
    existing = collection.get(include=[])
    existing_ids = set(existing["ids"] or [])
    expected_ids = set(entries)
    stale_ids = sorted(existing_ids - expected_ids)

    if collisions:
        print("Collisions detected:")
        for entry_id, names in sorted(collisions.items()):
            print(f"  {entry_id}: {names}")

    if stale_ids:
        print(f"Stale entries to remove: {len(stale_ids)}")
        for entry_id in stale_ids[:10]:
            print(f"  - {entry_id}")
        if not dry_run:
            delete_ids(client, stale_ids, embedding_provider, embedding_model)

    indexed = 0
    for entry_id, content in sorted(entries.items()):
        if dry_run:
            print(f"  + {entry_id}")
            indexed += 1
            continue
        index_zettel(
            client,
            entry_id,
            content,
            embedding_provider,
            embedding_model,
            force=True,
        )
        indexed += 1

    print("\n=== Summary ===")
    print(f"Stale entries removed: {len(stale_ids)}")
    print(f"Zettels indexed: {indexed}")
    print(f"Collisions (merged): {len(collisions)}")


def cmd_process(args) -> None:
    """Process sources, extract concepts, and create zettels."""
    config = load_config()
    vault_root = Path(config["vault"]["root"]).expanduser()
    sources_dir = resolve_path(vault_root, config["vault"]["sources_dir"])
    zettel_dir = resolve_path(vault_root, config["vault"]["zettel_dir"])
    template_path = resolve_path(vault_root, config["vault"]["template_path"])

    llm_provider = config["llm"].get("provider", "openai")
    llm_model = config["llm"].get("model")
    llm_base_url = config["llm"].get("base_url")

    embedding_provider = config["embeddings"].get("provider", "openai")
    embedding_model = config["embeddings"].get("model")
    top_k = config["embeddings"].get("top_k", 5)
    max_distance = config["embeddings"].get("max_distance", 0.5)
    db_path = resolve_path(vault_root, config["embeddings"]["db_path"])

    processing_config = config.get("processing", {})
    min_source_chars = processing_config.get("min_source_chars", 100)
    duplicate_threshold = processing_config.get("duplicate_max_distance", 0.15)
    chunk_target_chars = processing_config.get("chunk_target_chars", 4000)
    chunk_max_chars = processing_config.get("chunk_max_chars", 6000)

    print(f"Vault root: {vault_root}")
    print(f"Sources dir: {sources_dir}")
    print(f"Zettel dir: {zettel_dir}")
    print(f"Template: {template_path}")
    print(f"LLM provider: {llm_provider}")
    if llm_model:
        print(f"LLM model: {llm_model}")
    print(f"ChromaDB path: {db_path}")
    print(f"Embedding provider: {embedding_provider}")
    if embedding_model:
        print(f"Embedding model: {embedding_model}")
    if args.dry_run:
        print("*** DRY RUN MODE - No files or index entries will be written ***")

    template = load_template(template_path)
    client = get_client(str(db_path))

    if not args.dry_run:
        drift = refresh_active_index(zettel_dir, client, embedding_provider, embedding_model, dry_run=False)
        print(
            "Index refresh complete. "
            f"Updated: {len(drift['missing_or_stale_ids'])}, "
            f"Removed stale: {len(drift['extra_ids'])}"
        )

    llm_kwargs = {"base_url": llm_base_url} if llm_provider == "ollama" and llm_base_url else {}
    result_counts: Counter[str] = Counter()
    source_files = list_markdown_files(sources_dir)
    limit = args.limit if args.limit > 0 else None
    started_unprocessed = 0
    total_created = 0
    total_duplicates_skipped = 0

    already_processed = sum(
        1 for f in source_files if is_source_processed(read_file_with_retry(f))
    )
    total_sources = len(source_files)
    unprocessed = total_sources - already_processed
    print(f"\nSources: {total_sources} total, {already_processed} processed, {unprocessed} unprocessed")
    if limit:
        print(f"Limit: {limit} (of {unprocessed} unprocessed)")

    for source_path in source_files:
        content = read_file_with_retry(source_path).strip()
        print(f"\n--- Processing: {source_path.name} ---")

        if is_source_processed(content):
            print("  Result: processed")
            result_counts["processed"] += 1
            continue

        if limit is not None and started_unprocessed >= limit:
            break
        started_unprocessed += 1

        if len(content) < min_source_chars:
            print(f"  Skipping: content too short ({len(content)} chars, min {min_source_chars})")
            print("  Result: too_short")
            result_counts["too_short"] += 1
            continue

        concepts, diagnostics = extract_source_concepts(
            content=content,
            llm_provider=llm_provider,
            llm_model=llm_model,
            llm_kwargs=llm_kwargs,
            chunk_target_chars=chunk_target_chars,
            chunk_max_chars=chunk_max_chars,
        )
        print(f"  Chunked into {diagnostics['chunk_count']} part(s).")

        if diagnostics["provider_error"]:
            print("  Result: provider_error")
            result_counts["provider_error"] += 1
            continue

        if not concepts:
            if diagnostics["invalid_chunks"] or diagnostics["invalid_items"]:
                print("  Result: invalid_output")
                result_counts["invalid_output"] += 1
                continue
            if looks_non_extractable_source(content):
                if not args.dry_run:
                    mark_processed(source_path)
                print("  Result: non_extractable")
                result_counts["non_extractable"] += 1
            else:
                if not args.dry_run:
                    mark_processed(source_path)
                print("  Result: no_valid_concepts")
                result_counts["no_valid_concepts"] += 1
            continue

        deduped_concepts = dedupe_candidate_concepts(
            concepts,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            duplicate_threshold=duplicate_threshold,
        )
        print(f"  Found {len(deduped_concepts)} candidate concept(s) after dedupe.")

        if not deduped_concepts:
            if not args.dry_run:
                mark_processed(source_path)
            print("  Result: no_valid_concepts")
            result_counts["no_valid_concepts"] += 1
            continue

        created_for_source = 0
        duplicates_for_source = 0
        for concept in deduped_concepts:
            similar_titles = find_similar(
                client,
                concept["content"],
                top_k,
                max_distance,
                embedding_provider,
                embedding_model,
                create=not args.dry_run,
            )
            print(f"    [CONCEPT] {concept['title']}")
            print(f"      Tags: {concept.get('suggested_tags', [])}")
            if similar_titles:
                print(f"      Similar notes: {similar_titles[:3]}")

            if args.dry_run:
                print(f"      Type: {concept['concept_type']}")
                print(f"      Level: {concept['abstraction_level']}")
                continue

            duplicate, reason = zettel_exists(
                zettel_dir,
                concept["title"],
                concept["content"],
                client,
                embedding_provider,
                embedding_model,
                duplicate_threshold,
            )
            if duplicate:
                print(f"      SKIPPED (duplicate): {reason}")
                duplicates_for_source += 1
                total_duplicates_skipped += 1
                continue

            zettel_path = write_zettel(
                output_dir=zettel_dir,
                title=concept["title"],
                content=concept["content"],
                tags=concept.get("suggested_tags", []),
                similar=similar_titles,
                source=source_path.stem,
                template=template,
            )
            print(f"      Written: {zettel_path.name}")
            index_zettel(
                client,
                normalize_link(concept["title"]),
                zettel_path.read_text(encoding="utf-8"),
                embedding_provider,
                embedding_model,
                force=True,
            )
            created_for_source += 1
            total_created += 1

        if args.dry_run:
            print("  Result: created")
            result_counts["created"] += 1
            continue

        if created_for_source > 0 or duplicates_for_source > 0:
            mark_processed(source_path)

        if created_for_source > 0:
            print("  Result: created")
            result_counts["created"] += 1
        else:
            print("  Result: no_valid_concepts")
            result_counts["no_valid_concepts"] += 1

    print("\n=== Summary ===")
    print(f"Created zettels: {total_created}")
    print(f"Duplicate concepts skipped: {total_duplicates_skipped}")
    for reason in [
        "created",
        "processed",
        "too_short",
        "non_extractable",
        "provider_error",
        "invalid_output",
        "no_valid_concepts",
    ]:
        print(f"{reason}: {result_counts[reason]}")


def cmd_sync_backlinks(args) -> None:
    """Sync backlinks command handler."""
    config = load_config()
    vault_root = Path(config["vault"]["root"]).expanduser()
    zettel_dir = resolve_path(vault_root, config["vault"]["zettel_dir"])
    print(f"Zettel dir: {zettel_dir}")
    sync_backlinks(zettel_dir, dry_run=args.dry_run)


def cmd_normalize_links(args) -> None:
    """Normalize links command handler."""
    config = load_config()
    vault_root = Path(config["vault"]["root"]).expanduser()
    zettel_dir = resolve_path(vault_root, config["vault"]["zettel_dir"])
    print(f"Zettel dir: {zettel_dir}")
    normalize_links(zettel_dir, dry_run=args.dry_run)


def cmd_rename_files(args) -> None:
    """Rename files command handler."""
    config = load_config()
    vault_root = Path(config["vault"]["root"]).expanduser()
    zettel_dir = resolve_path(vault_root, config["vault"]["zettel_dir"])
    print(f"Zettel dir: {zettel_dir}")
    rename_files(zettel_dir, dry_run=args.dry_run)


def cmd_rebuild_index(args) -> None:
    """Rebuild index command handler."""
    config = load_config()
    vault_root = Path(config["vault"]["root"]).expanduser()
    zettel_dir = resolve_path(vault_root, config["vault"]["zettel_dir"])
    db_path = resolve_path(vault_root, config["embeddings"]["db_path"])
    embedding_provider = config["embeddings"].get("provider", "openai")
    embedding_model = config["embeddings"].get("model")

    print(f"Zettel dir: {zettel_dir}")
    print(f"ChromaDB path: {db_path}")
    print(f"Embedding provider: {embedding_provider}")
    if embedding_model:
        print(f"Embedding model: {embedding_model}")

    client = get_client(str(db_path))
    rebuild_index(zettel_dir, client, embedding_provider, embedding_model, dry_run=args.dry_run)


def cmd_bundle(args) -> None:
    """Bundle command handler."""
    config = load_config()
    vault_root = Path(config["vault"]["root"]).expanduser()
    zettel_dir = resolve_path(vault_root, config["vault"]["zettel_dir"])
    bundles_dir = resolve_path(vault_root, config["vault"].get("bundles_dir", "110_BUNDLES"))
    db_path = resolve_path(vault_root, config["embeddings"]["db_path"])
    client = get_client(str(db_path))

    if args.depth is not None:
        config.setdefault("bundle", {})["default_depth"] = args.depth
    if args.top_k is not None:
        config.setdefault("bundle", {})["default_top_k"] = args.top_k

    print(f"Zettel dir: {zettel_dir}")
    print(f"Bundles dir: {bundles_dir}")
    if args.dry_run:
        print("*** DRY RUN MODE ***")

    if not args.source and not args.note:
        print("Error: Must specify --source or --note")
        sys.exit(1)

    embedding_provider = config["embeddings"].get("provider", "openai")
    embedding_model = config["embeddings"].get("model")

    if args.source:
        try:
            refresh_active_index(zettel_dir, client, embedding_provider, embedding_model, dry_run=args.dry_run)
        except RuntimeError as exc:
            print(f"Error: {exc}")
            sys.exit(1)

        sources_dir = resolve_path(vault_root, config["vault"]["sources_dir"])
        source_path = sources_dir / args.source
        if not source_path.exists():
            source_path = sources_dir / f"{args.source}.md"
        if not source_path.exists():
            print(f"Error: Source file not found: {args.source}")
            sys.exit(1)

        print(f"\nBundling from source: {source_path.name}")
        output_path, content = bundle_from_source(
            source_path=source_path,
            zettel_dir=zettel_dir,
            bundles_dir=bundles_dir,
            client=client,
            config=config,
            dry_run=args.dry_run,
        )
    else:
        print(f"\nBundling around note: {args.note}")
        try:
            output_path, content = bundle_from_note(
                zettel_title=args.note,
                zettel_dir=zettel_dir,
                bundles_dir=bundles_dir,
                client=client,
                config=config,
                dry_run=args.dry_run,
            )
        except FileNotFoundError as exc:
            print(f"Error: {exc}")
            sys.exit(1)

    if args.dry_run:
        print("\n=== Bundle Preview ===")
        print(content[:2000])
        if len(content) > 2000:
            print(f"\n... ({len(content) - 2000} more characters)")
    else:
        print(f"\nBundle written to: {output_path}")


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI parser."""
    parser = argparse.ArgumentParser(
        description="Zettelkasten automation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    process_parser = subparsers.add_parser(
        "process",
        help="Process source files into zettels",
    )
    process_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Extract concepts and preview results without writing notes or index state",
    )
    process_parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit the number of unprocessed source files to handle",
    )

    sync_parser = subparsers.add_parser(
        "sync-backlinks",
        help="Legacy maintenance: migrate See Also to Tags and ensure backlinks",
    )
    sync_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without writing",
    )

    normalize_parser = subparsers.add_parser(
        "normalize-links",
        help="Normalize all Tags links to the filename format",
    )
    normalize_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without writing",
    )

    rebuild_parser = subparsers.add_parser(
        "rebuild-index",
        help="Rebuild the active embedding collection from current zettels",
    )
    rebuild_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview index changes without writing",
    )

    rename_parser = subparsers.add_parser(
        "rename-files",
        help="Rename zettel files to normalized format, merging duplicates",
    )
    rename_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without writing",
    )

    bundle_parser = subparsers.add_parser(
        "bundle",
        help="Assemble source or note context for export",
    )
    bundle_parser.add_argument(
        "--source",
        help="Source file to bundle (looked up in sources_dir)",
    )
    bundle_parser.add_argument(
        "--note",
        help="Zettel title to bundle around",
    )
    bundle_parser.add_argument(
        "--depth",
        type=int,
        help="Max traversal depth (default from config)",
    )
    bundle_parser.add_argument(
        "--top-k",
        type=int,
        dest="top_k",
        help="Max semantic seeds for source bundles (default from config)",
    )
    bundle_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview the bundle without writing files; source mode also requires a fresh index",
    )
    return parser


def main() -> None:
    """Main CLI entry point with subcommands."""
    parser = build_parser()

    if len(sys.argv) > 1 and sys.argv[1] in {"-h", "--help"}:
        parser.parse_args()
        return

    if len(sys.argv) > 1 and sys.argv[1].startswith("-"):
        args = parser.parse_args(["process"] + sys.argv[1:])
    else:
        args = parser.parse_args()

    if args.command is None:
        args = parser.parse_args(["process"])

    if args.command == "process":
        cmd_process(args)
    elif args.command == "sync-backlinks":
        cmd_sync_backlinks(args)
    elif args.command == "normalize-links":
        cmd_normalize_links(args)
    elif args.command == "rename-files":
        cmd_rename_files(args)
    elif args.command == "rebuild-index":
        cmd_rebuild_index(args)
    elif args.command == "bundle":
        cmd_bundle(args)


if __name__ == "__main__":
    main()
