# ABOUTME: Main orchestration module for Zettelkasten automation.
# ABOUTME: Processes source files, extracts concepts via LLM, and creates linked zettels.

import re
from datetime import datetime
from pathlib import Path
from typing import Iterator

import yaml

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


def write_zettel(
    output_dir: Path, title: str, content: str, tags: list, similar: list, source: str
) -> Path:
    """Write a zettel file with the standard template format."""
    filename = sanitize_filename(title) + ".md"
    output_path = output_dir / filename

    timestamp = datetime.now().strftime("%d-%m-%Y %H:%M")
    tags_str = " ".join(f"[[{tag}]]" for tag in tags) if tags else ""

    see_also = ""
    if similar:
        see_also = "\n".join(f"- [[{s}]]" for s in similar)

    zettel_content = f"""Created: {timestamp}

Status: #baby

Tags: {tags_str}

# {title}

{content}

# See Also

{see_also}

# References

1. [[{source}]]
"""
    output_path.write_text(zettel_content)
    return output_path


def mark_processed(source_path: Path) -> None:
    """Add #processed tag to source file."""
    content = source_path.read_text()
    if "#processed" not in content:
        source_path.write_text(content.rstrip() + "\n\n#processed\n")


def main() -> None:
    """Main orchestration: process sources, extract concepts, create zettels."""
    config = load_config()

    # Resolve paths from config structure
    vault_root = Path(config["vault"]["root"])
    zettel_dir = vault_root / config["vault"]["zettel_dir"]
    sources_dir = vault_root / config["vault"]["sources_dir"]
    model = config["llm"].get("model", "gpt-4o")

    # Initialize ChromaDB client with configured path
    db_path = str(vault_root / config["embeddings"]["db_path"])
    top_k = config["embeddings"].get("top_k", 5)
    client = get_client(db_path)

    # Index existing zettels
    for zettel_path in zettel_dir.glob("*.md"):
        zettel_content = zettel_path.read_text()
        zettel_title = zettel_path.stem
        index_zettel(client, zettel_title, zettel_content)

    sources_processed = 0
    zettels_created = 0

    for source_path in get_sources(sources_dir):
        content = source_path.read_text()
        source_name = source_path.stem
        concepts = extract_concepts(content, model)

        for concept in concepts:
            # find_similar returns list of title strings
            similar_titles = find_similar(client, concept["content"], top_k)

            zettel_path = write_zettel(
                zettel_dir,
                concept["title"],
                concept["content"],
                concept.get("suggested_tags", []),
                similar_titles,
                source_name,
            )

            # Index the newly created zettel
            new_content = zettel_path.read_text()
            index_zettel(client, concept["title"], new_content)
            zettels_created += 1

        mark_processed(source_path)
        sources_processed += 1

    print(f"Sources processed: {sources_processed}")
    print(f"Zettels created: {zettels_created}")


if __name__ == "__main__":
    main()
