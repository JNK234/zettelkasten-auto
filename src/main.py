# ABOUTME: Main orchestration module for Zettelkasten automation.
# ABOUTME: Processes source files, extracts concepts via LLM, and creates linked zettels.

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterator

import yaml
from dotenv import load_dotenv

load_dotenv()

from src.db import get_client, index_zettel, find_similar
from src.llm import extract_concepts


def load_config() -> dict:
    """Load YAML configuration from config directory."""
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_sources(sources_dir: Path) -> Iterator[Path]:
    """Yield .md files, skipping those already marked as #processed."""
    for path in sources_dir.glob("*.md"):
        content = path.read_text()
        if "#processed" not in content:
            yield path


def sanitize_filename(title: str) -> str:
    """Convert title to safe filename."""
    sanitized = re.sub(r"[^\w\s-]", "", title)
    sanitized = re.sub(r"[\s_]+", "-", sanitized.strip())
    sanitized = re.sub(r"-+", "-", sanitized)
    return sanitized[:100].lower()


def normalize_link(title: str) -> str:
    """Normalize title to filename format for consistent linking.

    Handles Obsidian alias syntax [[path|display]] by using the display text.
    Must match sanitize_filename() logic exactly so links match filenames.
    """
    # Handle Obsidian alias syntax: [[path|display]] -> use display text
    if "|" in title:
        title = title.split("|")[-1]

    normalized = re.sub(r"[^\w\s-]", "", title)
    normalized = re.sub(r"[\s_]+", "-", normalized.strip())
    normalized = re.sub(r"-+", "-", normalized)
    return normalized[:100].lower()


def zettel_exists(
    zettel_dir: Path,
    title: str,
    content: str,
    client,
    embedding_provider: str,
    embedding_model: str | None,
    duplicate_threshold: float = 0.15,
) -> tuple[bool, str | None]:
    """Check if a zettel already exists by filename or semantic similarity.

    Returns:
        (exists, reason) - True if duplicate found, with reason string
    """
    # Check 1: Exact filename match
    filename = sanitize_filename(title) + ".md"
    if (zettel_dir / filename).exists():
        return True, f"file exists: {filename}"

    # Check 2: Semantic similarity (very tight threshold for duplicates)
    similar = find_similar(
        client, content, top_k=1, max_distance=duplicate_threshold,
        provider=embedding_provider, model_name=embedding_model
    )
    if similar:
        return True, f"too similar to: {similar[0]}"

    return False, None


def load_template(template_path: Path) -> str:
    """Load zettel template from file."""
    return template_path.read_text()


def write_zettel(
    output_dir: Path,
    title: str,
    content: str,
    tags: list,
    similar: list,
    source: str,
    template: str,
) -> Path:
    """Write a zettel file using the loaded template.

    Tags and similar notes are merged into the Tags line.
    """
    filename = sanitize_filename(title) + ".md"
    output_path = output_dir / filename

    date_str = datetime.now().strftime("%d-%m-%Y")
    time_str = datetime.now().strftime("%H:%M")

    # Normalize all links to filename format and merge (deduplicated)
    all_links = [normalize_link(t) for t in tags] if tags else []
    if similar:
        for s in similar:
            normalized = normalize_link(s)
            if normalized not in all_links:
                all_links.append(normalized)
    tags_str = " ".join(f"[[{link}]]" for link in all_links) if all_links else ""

    # Build references section
    references = f"# References\n\n1. [[{source}]]"

    # Replace Obsidian Templater placeholders
    zettel_content = template
    zettel_content = zettel_content.replace("{{date}}", date_str)
    zettel_content = zettel_content.replace("{{time}}", time_str)
    zettel_content = zettel_content.replace("Tags:", f"Tags: {tags_str}" if tags_str else "Tags:")

    # Replace title placeholder and insert content after it
    zettel_content = zettel_content.replace("# {{Title}}", f"# {title}\n\n{content}")

    # No See Also section - all links go in Tags
    zettel_content = zettel_content.replace("# References", references)

    output_path.write_text(zettel_content)
    return output_path


def mark_processed(source_path: Path) -> None:
    """Add #processed tag to source file's Status line, or append if no Status line exists."""
    content = source_path.read_text()
    if "#processed" in content:
        return  # Already marked

    # Try to add to existing Status line
    if "Status:" in content:
        # Add #processed to the Status line
        import re
        updated = re.sub(
            r"(Status:.*?)(\n|$)",
            r"\1 #processed\2",
            content,
            count=1
        )
        source_path.write_text(updated)
    else:
        # No Status line, append at end
        source_path.write_text(content.rstrip() + "\n\n#processed\n")


def extract_tags(content: str) -> list[str]:
    """Extract [[links]] from the Tags: line."""
    match = re.search(r"^Tags:(.*)$", content, re.MULTILINE)
    if not match:
        return []
    tags_line = match.group(1)
    return re.findall(r"\[\[([^\]]+)\]\]", tags_line)


def extract_see_also(content: str) -> list[str]:
    """Extract [[links]] from the # See Also section."""
    # Match from "# See Also" to next heading or end
    match = re.search(r"# See Also\s*\n(.*?)(?=\n#|\Z)", content, re.DOTALL)
    if not match:
        return []
    section = match.group(1)
    return re.findall(r"\[\[([^\]]+)\]\]", section)


def remove_see_also_section(content: str) -> str:
    """Remove the # See Also section from content."""
    # Remove "# See Also" section up to next heading or end
    return re.sub(r"\n*# See Also\s*\n.*?(?=\n#|\Z)", "", content, flags=re.DOTALL)


def add_tag_link(path: Path, title: str, dry_run: bool) -> bool:
    """Add [[title]] to the Tags line of a zettel file.

    Returns True if a change was made (or would be made in dry_run).
    """
    content = path.read_text()
    existing_tags = extract_tags(content)

    if title in existing_tags:
        return False

    # Find and update the Tags line
    match = re.search(r"^(Tags:.*)$", content, re.MULTILINE)
    if not match:
        return False

    old_tags_line = match.group(1)
    new_tags_line = f"{old_tags_line} [[{title}]]"

    if not dry_run:
        updated = content.replace(old_tags_line, new_tags_line, 1)
        path.write_text(updated)

    return True


def sync_backlinks(zettel_dir: Path, dry_run: bool = False) -> None:
    """Migrate See Also sections to Tags and ensure bidirectional backlinks."""
    zettels = list(zettel_dir.glob("*.md"))
    print(f"Scanning {len(zettels)} zettels...")

    if dry_run:
        print("*** DRY RUN MODE ***\n")

    # Track stats
    see_also_to_remove = 0
    tags_to_add_from_see_also = 0
    backlinks_to_add = 0

    # Build maps for lookup (normalized title -> path)
    # This handles both "Agentic AI Systems" and "agentic-ai-systems" formats
    normalized_to_path: dict[str, Path] = {}
    for z in zettels:
        normalized_to_path[normalize_link(z.stem)] = z

    # Load all zettel contents into memory (single read per file)
    zettel_contents = {z: z.read_text() for z in zettels}

    # Track pending changes per file: path -> set of tags to add
    pending_backlinks: dict[Path, set[str]] = {z: set() for z in zettels}

    # First pass: migrate See Also to Tags
    migrations = []  # (path, see_also_links)
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

            if not dry_run:
                content = zettel_contents[zettel_path]
                existing_tags = extract_tags(content)

                # Normalize existing tags and see_also links, then merge (dedupe)
                new_tags = [normalize_link(t) for t in existing_tags]
                for link in links:
                    normalized = normalize_link(link)
                    if normalized not in new_tags:
                        new_tags.append(normalized)

                # Rebuild Tags line with normalized links
                new_tags_str = " ".join(f"[[{t}]]" for t in new_tags)
                content = re.sub(r"^Tags:.*$", f"Tags: {new_tags_str}", content, flags=re.MULTILINE)

                # Remove See Also section
                content = remove_see_also_section(content)

                # Update in-memory content and write to disk
                zettel_contents[zettel_path] = content
                zettel_path.write_text(content)
    else:
        print("No See Also sections found to migrate.")

    # Second pass: collect all backlinks needed (no writes yet)
    print("\nAnalyzing backlinks:")
    for zettel_path in zettels:
        content = zettel_contents[zettel_path]
        zettel_title = zettel_path.stem
        tags = extract_tags(content)

        for linked_title in tags:
            # Look up by normalized link to handle format differences
            normalized = normalize_link(linked_title)
            if normalized in normalized_to_path:
                target_path = normalized_to_path[normalized]
                target_content = zettel_contents[target_path]
                target_tags = extract_tags(target_content)

                # Check if backlink exists (use normalize_link for comparison)
                target_tags_normalized = {normalize_link(t) for t in target_tags}
                normalized_zettel = normalize_link(zettel_title)
                if normalized_zettel not in target_tags_normalized:
                    pending_backlinks[target_path].add(normalized_zettel)

    # Third pass: apply backlinks (single write per file)
    print("\nAdding backlinks:")
    files_modified = 0
    for target_path, new_links in pending_backlinks.items():
        if not new_links:
            continue

        for link in sorted(new_links):
            print(f"  + [[{link}]] -> {target_path.name}")
            backlinks_to_add += 1

        if not dry_run:
            content = zettel_contents[target_path]
            existing_tags = extract_tags(content)

            # Normalize existing tags and add new backlinks (dedupe)
            all_tags = [normalize_link(t) for t in existing_tags]
            for link in new_links:
                if link not in all_tags:
                    all_tags.append(link)

            # Rebuild Tags line with normalized links
            new_tags_str = " ".join(f"[[{t}]]" for t in all_tags)
            content = re.sub(r"^Tags:.*$", f"Tags: {new_tags_str}", content, flags=re.MULTILINE)
            target_path.write_text(content)
            files_modified += 1

    if backlinks_to_add == 0:
        print("  No backlinks to add.")

    # Summary
    print(f"\n=== Summary ===")
    print(f"See Also sections removed: {see_also_to_remove}")
    print(f"Tags added (from See Also): {tags_to_add_from_see_also}")
    print(f"Backlinks added: {backlinks_to_add}")
    if not dry_run and backlinks_to_add > 0:
        print(f"Files modified for backlinks: {files_modified}")

    if dry_run:
        print("\nRun without --dry-run to apply changes.")


def cmd_process(args) -> None:
    """Process sources, extract concepts, create zettels."""
    config = load_config()

    # Resolve paths from config structure
    vault_root = Path(config["vault"]["root"])
    zettel_dir = vault_root / config["vault"]["zettel_dir"]
    sources_dir = vault_root / config["vault"]["sources_dir"]

    # LLM configuration
    llm_provider = config["llm"].get("provider", "openai")
    llm_model = config["llm"].get("model", None)
    llm_base_url = config["llm"].get("base_url", None)

    template_path = vault_root / config["vault"]["template_path"]

    print(f"Vault root: {vault_root}")
    print(f"Sources dir: {sources_dir}")
    print(f"Zettel dir: {zettel_dir}")
    print(f"Template: {template_path}")
    print(f"LLM provider: {llm_provider}")
    if llm_model:
        print(f"LLM model: {llm_model}")
    if args.dry_run:
        print("*** DRY RUN MODE - No files will be written ***")

    # Load template once
    template = load_template(template_path)

    # Initialize ChromaDB client with configured path
    db_path = str(vault_root / config["embeddings"]["db_path"])
    top_k = config["embeddings"].get("top_k", 5)
    max_distance = config["embeddings"].get("max_distance", 0.5)
    embedding_provider = config["embeddings"].get("provider", "openai")
    embedding_model = config["embeddings"].get("model", None)

    print(f"ChromaDB path: {db_path}")
    print(f"Embedding provider: {embedding_provider}")
    if embedding_model:
        print(f"Embedding model: {embedding_model}")
    print(f"Similarity threshold: max_distance={max_distance}")
    client = get_client(db_path)

    # Index existing zettels (skip unchanged files)
    existing_zettels = list(zettel_dir.glob("*.md"))
    print(f"Checking {len(existing_zettels)} existing zettels...")
    indexed_count = 0
    skipped_count = 0
    for zettel_path in existing_zettels:
        zettel_content = zettel_path.read_text().strip()
        zettel_title = zettel_path.stem
        if not zettel_content:
            skipped_count += 1
            continue
        try:
            was_indexed = index_zettel(client, zettel_title, zettel_content, embedding_provider, embedding_model)
            if was_indexed:
                indexed_count += 1
            else:
                skipped_count += 1
        except Exception as e:
            print(f"  Error indexing {zettel_path.name}: {e}")
    print(f"Indexing complete. Indexed: {indexed_count}, Skipped (unchanged): {skipped_count}")

    sources_processed = 0
    zettels_created = 0
    duplicates_skipped = 0

    source_files = list(get_sources(sources_dir))
    if args.limit > 0:
        source_files = source_files[:args.limit]
    print(f"Found {len(source_files)} source files to process.")

    MIN_CONTENT_LENGTH = 100  # Skip files with less than 100 characters

    for source_path in source_files:
        print(f"\n--- Processing: {source_path.name} ---")
        content = source_path.read_text().strip()
        source_name = source_path.stem

        # Skip empty or too short files
        if len(content) < MIN_CONTENT_LENGTH:
            print(f"  Skipping: content too short ({len(content)} chars, min {MIN_CONTENT_LENGTH})")
            continue

        print(f"  Extracting concepts via LLM...")
        llm_kwargs = {"base_url": llm_base_url} if llm_base_url else {}
        concepts = extract_concepts(content, llm_provider, llm_model, **llm_kwargs)
        print(f"  Found {len(concepts)} concepts.")

        for concept in concepts:
            print(f"    [CONCEPT] {concept['title']}")
            print(f"      Tags: {concept.get('suggested_tags', [])}")

            # find_similar returns list of title strings
            similar_titles = find_similar(
                client, concept["content"], top_k, max_distance,
                embedding_provider, embedding_model
            )
            if similar_titles:
                print(f"      Similar notes: {similar_titles[:3]}...")

            if args.dry_run:
                print(f"      Type: {concept.get('concept_type', 'N/A')}")
                print(f"      Level: {concept.get('abstraction_level', 'N/A')}")
                print(f"      --- Content ---")
                print(f"      {concept['content']}")
                print(f"      --- End Content ---")
                continue

            # Check for duplicates before writing
            is_duplicate, reason = zettel_exists(
                zettel_dir, concept["title"], concept["content"],
                client, embedding_provider, embedding_model
            )
            if is_duplicate:
                print(f"      SKIPPED (duplicate): {reason}")
                duplicates_skipped += 1
                continue

            zettel_path = write_zettel(
                zettel_dir,
                concept["title"],
                concept["content"],
                concept.get("suggested_tags", []),
                similar_titles,
                source_name,
                template,
            )
            print(f"      Written: {zettel_path.name}")

            # Index the newly created zettel with normalized ID
            new_content = zettel_path.read_text()
            normalized_title = normalize_link(concept["title"])
            index_zettel(client, normalized_title, new_content, embedding_provider, embedding_model)
            zettels_created += 1

        if not args.dry_run:
            mark_processed(source_path)
            print(f"  Marked as processed.")
        sources_processed += 1

    print(f"\n=== Summary ===")
    print(f"Sources processed: {sources_processed}")
    print(f"Zettels created: {zettels_created}")
    print(f"Duplicates skipped: {duplicates_skipped}")


def cmd_sync_backlinks(args) -> None:
    """Sync backlinks command handler."""
    config = load_config()
    vault_root = Path(config["vault"]["root"])
    zettel_dir = vault_root / config["vault"]["zettel_dir"]

    print(f"Zettel dir: {zettel_dir}")
    sync_backlinks(zettel_dir, dry_run=args.dry_run)


def normalize_links(zettel_dir: Path, dry_run: bool = False) -> None:
    """Normalize all [[links]] in Tags lines to filename format."""
    zettels = list(zettel_dir.glob("*.md"))
    print(f"Scanning {len(zettels)} zettels...")

    if dry_run:
        print("*** DRY RUN MODE ***\n")

    files_to_update = 0
    links_normalized = 0
    duplicates_removed = 0

    print("Normalizing Tags:")
    for zettel_path in zettels:
        content = zettel_path.read_text()
        existing_tags = extract_tags(content)

        if not existing_tags:
            continue

        # Normalize all tags
        normalized_tags = []
        changes = []
        for tag in existing_tags:
            normalized = normalize_link(tag)
            if tag != normalized:
                changes.append((tag, normalized))
            if normalized not in normalized_tags:
                normalized_tags.append(normalized)
            else:
                duplicates_removed += 1

        # Check if anything changed
        if existing_tags != normalized_tags or len(existing_tags) != len(normalized_tags):
            files_to_update += 1
            links_normalized += len(changes)

            if changes or len(existing_tags) > len(normalized_tags):
                print(f"  {zettel_path.name}: {len(changes)} links normalized")
                for old, new in changes:
                    print(f"    [[{old}]] -> [[{new}]]")
                if len(existing_tags) > len(normalized_tags):
                    print(f"    ({len(existing_tags) - len(normalized_tags)} duplicates removed)")

            if not dry_run:
                # Rebuild Tags line
                new_tags_str = " ".join(f"[[{t}]]" for t in normalized_tags)
                updated = re.sub(r"^Tags:.*$", f"Tags: {new_tags_str}", content, flags=re.MULTILINE)
                zettel_path.write_text(updated)

    print(f"\n=== Summary ===")
    print(f"Files to update: {files_to_update}")
    print(f"Links normalized: {links_normalized}")
    print(f"Duplicates removed: {duplicates_removed}")

    if dry_run and files_to_update > 0:
        print("\nRun without --dry-run to apply changes.")


def cmd_normalize_links(args) -> None:
    """Normalize links command handler."""
    config = load_config()
    vault_root = Path(config["vault"]["root"])
    zettel_dir = vault_root / config["vault"]["zettel_dir"]

    print(f"Zettel dir: {zettel_dir}")
    normalize_links(zettel_dir, dry_run=args.dry_run)


def rename_files(zettel_dir: Path, dry_run: bool = False) -> None:
    """Rename zettel files to normalized format, merging duplicates."""
    zettels = list(zettel_dir.glob("*.md"))
    print(f"Scanning {len(zettels)} zettels...")

    if dry_run:
        print("*** DRY RUN MODE ***\n")

    # Group files by normalized name
    normalized_to_files: dict[str, list[Path]] = {}
    for z in zettels:
        norm_name = normalize_link(z.stem)
        if norm_name not in normalized_to_files:
            normalized_to_files[norm_name] = []
        normalized_to_files[norm_name].append(z)

    files_renamed = 0
    files_merged = 0
    files_deleted = 0

    print("Processing files:")
    for norm_name, files in normalized_to_files.items():
        target_path = zettel_dir / f"{norm_name}.md"

        if len(files) == 1:
            # Single file - just rename if needed
            source = files[0]
            if source.name != f"{norm_name}.md":
                print(f"  Rename: {source.name} -> {norm_name}.md")
                if not dry_run:
                    source.rename(target_path)
                files_renamed += 1
        else:
            # Multiple files - merge into one
            print(f"  Merge {len(files)} files -> {norm_name}.md:")
            for f in files:
                print(f"    - {f.name}")

            if not dry_run:
                # Collect all content
                merged_content = []
                for f in sorted(files, key=lambda x: x.stat().st_mtime):
                    content = f.read_text().strip()
                    if content:
                        merged_content.append(content)

                # Write merged content to target
                final_content = "\n\n---\n\n".join(merged_content)
                target_path.write_text(final_content)

                # Delete source files (except target if it existed)
                for f in files:
                    if f != target_path:
                        f.unlink()
                        files_deleted += 1

            files_merged += 1

    print(f"\n=== Summary ===")
    print(f"Files renamed: {files_renamed}")
    print(f"Merge operations: {files_merged}")
    print(f"Files deleted (merged): {files_deleted}")

    if dry_run:
        print("\nRun without --dry-run to apply changes.")


def cmd_rename_files(args) -> None:
    """Rename files command handler."""
    config = load_config()
    vault_root = Path(config["vault"]["root"])
    zettel_dir = vault_root / config["vault"]["zettel_dir"]

    print(f"Zettel dir: {zettel_dir}")
    rename_files(zettel_dir, dry_run=args.dry_run)


def rebuild_index(zettel_dir: Path, client, embedding_provider: str, embedding_model: str | None, dry_run: bool = False) -> None:
    """Rebuild ChromaDB index with normalized IDs from zettel filenames.

    When multiple files normalize to the same ID, their content is concatenated
    to create a combined embedding that represents all related notes.
    """
    from src.db import get_collection

    zettels = list(zettel_dir.glob("*.md"))
    print(f"Found {len(zettels)} zettels to index...")

    if dry_run:
        print("*** DRY RUN MODE ***\n")

    collection = get_collection(client, embedding_provider, embedding_model)

    # Get existing IDs to check for stale entries
    try:
        existing = collection.get(include=[])
        existing_ids = set(existing["ids"]) if existing["ids"] else set()
        print(f"Existing entries in index: {len(existing_ids)}")
    except Exception:
        existing_ids = set()

    # Group zettels by normalized ID to detect collisions
    normalized_to_files: dict[str, list[Path]] = {}
    for z in zettels:
        norm_id = normalize_link(z.stem)
        if norm_id not in normalized_to_files:
            normalized_to_files[norm_id] = []
        normalized_to_files[norm_id].append(z)

    # Report collisions
    collisions = {k: v for k, v in normalized_to_files.items() if len(v) > 1}
    if collisions:
        print(f"\nCollisions detected ({len(collisions)} IDs with multiple files):")
        for norm_id, files in collisions.items():
            print(f"  '{norm_id}' <- {[f.name for f in files]}")
        print("  Content will be concatenated for combined embedding.\n")

    # Build set of expected IDs
    expected_ids = set(normalized_to_files.keys())

    # Find stale IDs to delete
    stale_ids = existing_ids - expected_ids
    if stale_ids:
        print(f"Stale entries to remove: {len(stale_ids)}")
        for sid in sorted(stale_ids)[:10]:
            print(f"  - {sid}")
        if len(stale_ids) > 10:
            print(f"  ... and {len(stale_ids) - 10} more")

        if not dry_run:
            collection.delete(ids=list(stale_ids))
            print(f"Deleted {len(stale_ids)} stale entries.")

    # Index zettels, concatenating content for collisions
    print("\nIndexing zettels:")
    indexed = 0
    for norm_id, files in normalized_to_files.items():
        # Concatenate content from all files with this normalized ID
        combined_content = []
        for f in files:
            content = f.read_text().strip()
            if content:
                combined_content.append(content)

        if not combined_content:
            continue

        full_content = "\n\n---\n\n".join(combined_content)

        if dry_run:
            if norm_id not in existing_ids:
                print(f"  + {norm_id} (new)")
            if len(files) > 1:
                print(f"    (merged from {len(files)} files)")
            indexed += 1
        else:
            try:
                index_zettel(client, norm_id, full_content, embedding_provider, embedding_model, force=True)
                indexed += 1
            except Exception as e:
                print(f"  Error indexing {norm_id}: {e}")

    print(f"\n=== Summary ===")
    print(f"Stale entries removed: {len(stale_ids)}")
    print(f"Zettels indexed: {indexed}")
    print(f"Collisions (merged): {len(collisions)}")

    if dry_run:
        print("\nRun without --dry-run to apply changes.")


def cmd_rebuild_index(args) -> None:
    """Rebuild index command handler."""
    config = load_config()
    vault_root = Path(config["vault"]["root"])
    zettel_dir = vault_root / config["vault"]["zettel_dir"]

    db_path = str(vault_root / config["embeddings"]["db_path"])
    embedding_provider = config["embeddings"].get("provider", "openai")
    embedding_model = config["embeddings"].get("model", None)

    print(f"Zettel dir: {zettel_dir}")
    print(f"ChromaDB path: {db_path}")
    print(f"Embedding provider: {embedding_provider}")
    if embedding_model:
        print(f"Embedding model: {embedding_model}")

    client = get_client(db_path)
    rebuild_index(zettel_dir, client, embedding_provider, embedding_model, dry_run=args.dry_run)


def main() -> None:
    """Main CLI entry point with subcommands."""
    parser = argparse.ArgumentParser(
        description="Zettelkasten automation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Process command (default)
    process_parser = subparsers.add_parser(
        "process", help="Process source files and create zettels"
    )
    process_parser.add_argument(
        "--dry-run", action="store_true", help="Extract concepts but don't write files"
    )
    process_parser.add_argument(
        "--limit", type=int, default=0, help="Limit number of source files to process"
    )

    # Sync-backlinks command
    sync_parser = subparsers.add_parser(
        "sync-backlinks", help="Migrate See Also to Tags and ensure bidirectional backlinks"
    )
    sync_parser.add_argument(
        "--dry-run", action="store_true", help="Preview changes without writing"
    )

    # Normalize-links command
    normalize_parser = subparsers.add_parser(
        "normalize-links", help="Normalize all Tags links to filename format (lowercase-with-hyphens)"
    )
    normalize_parser.add_argument(
        "--dry-run", action="store_true", help="Preview changes without writing"
    )

    # Rebuild-index command
    rebuild_parser = subparsers.add_parser(
        "rebuild-index", help="Rebuild ChromaDB index with normalized IDs from zettel filenames"
    )
    rebuild_parser.add_argument(
        "--dry-run", action="store_true", help="Preview changes without writing"
    )

    # Rename-files command
    rename_parser = subparsers.add_parser(
        "rename-files", help="Rename zettel files to normalized format, merging duplicates"
    )
    rename_parser.add_argument(
        "--dry-run", action="store_true", help="Preview changes without writing"
    )

    # Handle backward compatibility: if first arg is a flag, assume 'process'
    if len(sys.argv) > 1 and sys.argv[1].startswith("-"):
        args = parser.parse_args(["process"] + sys.argv[1:])
    else:
        args = parser.parse_args()

    # Default to 'process' if no command given
    if args.command is None:
        args = parser.parse_args(["process"])

    if args.command == "process":
        cmd_process(args)
    elif args.command == "sync-backlinks":
        cmd_sync_backlinks(args)
    elif args.command == "normalize-links":
        cmd_normalize_links(args)
    elif args.command == "rebuild-index":
        cmd_rebuild_index(args)
    elif args.command == "rename-files":
        cmd_rename_files(args)


if __name__ == "__main__":
    main()
