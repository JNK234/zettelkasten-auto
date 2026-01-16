# ABOUTME: Main orchestration module for Zettelkasten automation.
# ABOUTME: Processes source files, extracts concepts via LLM, and creates linked zettels.

import argparse
import re
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
    """Yield .md files, skipping those with BOTH #processed AND #adult tags."""
    for path in sources_dir.glob("*.md"):
        content = path.read_text()
        has_processed = "#processed" in content
        has_adult = "#adult" in content
        if not (has_processed and has_adult):
            yield path


def sanitize_filename(title: str) -> str:
    """Convert title to safe filename."""
    sanitized = re.sub(r"[^\w\s-]", "", title)
    sanitized = re.sub(r"\s+", "-", sanitized.strip())
    return sanitized[:100].lower()


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
    """Write a zettel file using the loaded template."""
    filename = sanitize_filename(title) + ".md"
    output_path = output_dir / filename

    date_str = datetime.now().strftime("%d-%m-%Y")
    time_str = datetime.now().strftime("%H:%M")
    tags_str = " ".join(f"[[{tag}]]" for tag in tags) if tags else ""

    see_also = ""
    if similar:
        see_also = "# See Also\n\n" + "\n".join(f"- [[{s}]]" for s in similar)

    # Replace Obsidian Templater placeholders
    zettel_content = template
    zettel_content = zettel_content.replace("{{date}}", date_str)
    zettel_content = zettel_content.replace("{{time}}", time_str)
    zettel_content = zettel_content.replace("{{Title}}", title)
    zettel_content = zettel_content.replace("Tags:", f"Tags: {tags_str}" if tags_str else "Tags:")

    # Insert content after the title heading
    zettel_content = zettel_content.replace(f"# {title}\n", f"# {title}\n\n{content}\n")

    # Insert see_also before References
    zettel_content = zettel_content.replace("# References", f"{see_also}\n\n# References")

    # Add source reference
    zettel_content = zettel_content.replace("# References\n", f"# References\n\n1. [[{source}]]\n")

    output_path.write_text(zettel_content)
    return output_path


def mark_processed(source_path: Path) -> None:
    """Add #processed tag to source file."""
    content = source_path.read_text()
    if "#processed" not in content:
        source_path.write_text(content.rstrip() + "\n\n#processed\n")


def main() -> None:
    """Main orchestration: process sources, extract concepts, create zettels."""
    parser = argparse.ArgumentParser(description="Zettelkasten automation")
    parser.add_argument("--dry-run", action="store_true", help="Extract concepts but don't write files")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of source files to process")
    args = parser.parse_args()

    config = load_config()

    # Resolve paths from config structure
    vault_root = Path(config["vault"]["root"])
    zettel_dir = vault_root / config["vault"]["zettel_dir"]
    sources_dir = vault_root / config["vault"]["sources_dir"]
    model = config["llm"].get("model", "gpt-4o")

    template_path = vault_root / config["vault"]["template_path"]

    print(f"Vault root: {vault_root}")
    print(f"Sources dir: {sources_dir}")
    print(f"Zettel dir: {zettel_dir}")
    print(f"Template: {template_path}")
    print(f"Model: {model}")
    if args.dry_run:
        print("*** DRY RUN MODE - No files will be written ***")

    # Load template once
    template = load_template(template_path)

    # Initialize ChromaDB client with configured path
    db_path = str(vault_root / config["embeddings"]["db_path"])
    top_k = config["embeddings"].get("top_k", 5)
    max_distance = config["embeddings"].get("max_distance", 0.5)
    print(f"ChromaDB path: {db_path}")
    print(f"Similarity threshold: max_distance={max_distance}")
    client = get_client(db_path)

    # Index existing zettels
    existing_zettels = list(zettel_dir.glob("*.md"))
    print(f"Indexing {len(existing_zettels)} existing zettels...")
    for zettel_path in existing_zettels:
        zettel_content = zettel_path.read_text().strip()
        zettel_title = zettel_path.stem
        if not zettel_content:
            print(f"  Skipping empty file: {zettel_path.name}")
            continue
        try:
            index_zettel(client, zettel_title, zettel_content)
        except Exception as e:
            print(f"  Error indexing {zettel_path.name}: {e}")
    print("Indexing complete.")

    sources_processed = 0
    zettels_created = 0

    source_files = list(get_sources(sources_dir))
    if args.limit > 0:
        source_files = source_files[:args.limit]
    print(f"Found {len(source_files)} source files to process.")

    for source_path in source_files:
        print(f"\n--- Processing: {source_path.name} ---")
        content = source_path.read_text()
        source_name = source_path.stem

        print(f"  Extracting concepts via LLM...")
        concepts = extract_concepts(content, model)
        print(f"  Found {len(concepts)} concepts.")

        for concept in concepts:
            print(f"    [CONCEPT] {concept['title']}")
            print(f"      Tags: {concept.get('suggested_tags', [])}")

            # find_similar returns list of title strings
            similar_titles = find_similar(client, concept["content"], top_k, max_distance)
            if similar_titles:
                print(f"      Similar notes: {similar_titles[:3]}...")

            if args.dry_run:
                print(f"      --- Content ---")
                print(f"      {concept['content']}")
                print(f"      --- End Content ---")
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

            # Index the newly created zettel
            new_content = zettel_path.read_text()
            index_zettel(client, concept["title"], new_content)
            zettels_created += 1

        if not args.dry_run:
            mark_processed(source_path)
            print(f"  Marked as processed.")
        sources_processed += 1

    print(f"\n=== Summary ===")
    print(f"Sources processed: {sources_processed}")
    print(f"Zettels created: {zettels_created}")


if __name__ == "__main__":
    main()
