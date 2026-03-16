# ABOUTME: ChromaDB client and collection operations.
# ABOUTME: Handles indexing and similarity search for zettels.

import hashlib
import re
from typing import List

import chromadb

from src.db.embeddings import (
    EmbeddingProvider,
    get_default_embedding_model,
    get_embedding_function,
)


COLLECTION_PREFIX = "zettels"


def _normalize_collection_component(value: str) -> str:
    """Normalize a provider/model value for use in collection names."""
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized.lower() or "default"


def get_collection_name(
    provider: EmbeddingProvider = "openai",
    model_name: str | None = None,
) -> str:
    """Return a collection name scoped to the active embedding backend."""
    resolved_model = model_name or get_default_embedding_model(provider)
    return (
        f"{COLLECTION_PREFIX}__"
        f"{_normalize_collection_component(provider)}__"
        f"{_normalize_collection_component(resolved_model)}"
    )


def get_collection_metadata(
    provider: EmbeddingProvider = "openai",
    model_name: str | None = None,
) -> dict[str, str]:
    """Return metadata describing the active embedding backend."""
    resolved_model = model_name or get_default_embedding_model(provider)
    return {"provider": provider, "model": resolved_model}


def get_client(db_path: str) -> chromadb.PersistentClient:
    """Returns ChromaDB PersistentClient at the specified path."""
    return chromadb.PersistentClient(path=db_path)


def get_collection(
    client: chromadb.PersistentClient,
    provider: EmbeddingProvider = "openai",
    model_name: str | None = None,
    create: bool = True,
):
    """Gets or creates the zettels collection with configured embeddings."""
    embedding_fn = get_embedding_function(provider, model_name)
    collection_name = get_collection_name(provider, model_name)
    if create:
        collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_fn,
            metadata=get_collection_metadata(provider, model_name),
        )
    else:
        try:
            collection = client.get_collection(
                name=collection_name,
                embedding_function=embedding_fn,
            )
        except Exception:
            return None
    expected_metadata = get_collection_metadata(provider, model_name)
    existing_metadata = collection.metadata or {}
    if (
        existing_metadata.get("provider") != expected_metadata["provider"]
        or existing_metadata.get("model") != expected_metadata["model"]
    ):
        raise ValueError(
            "Embedding collection metadata mismatch for "
            f"{collection.name}: expected {expected_metadata}, found {existing_metadata}"
        )
    return collection


def _content_hash(content: str) -> str:
    """Generate SHA256 hash of content for change detection."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def needs_indexing(
    client: chromadb.PersistentClient,
    title: str,
    content: str,
    provider: EmbeddingProvider = "openai",
    model_name: str | None = None,
) -> bool:
    """Check if a zettel needs to be indexed (new or changed)."""
    collection = get_collection(client, provider, model_name)
    content_hash = _content_hash(content)

    # Try to get existing record
    try:
        result = collection.get(ids=[title], include=["metadatas"])
        if result and result["ids"] and result["metadatas"]:
            existing_hash = result["metadatas"][0].get("content_hash")
            return existing_hash != content_hash
    except Exception:
        pass

    return True  # New file or error, needs indexing


def index_zettel(
    client: chromadb.PersistentClient,
    title: str,
    content: str,
    provider: EmbeddingProvider = "openai",
    model_name: str | None = None,
    force: bool = False,
) -> bool:
    """Index a zettel if new or changed. Returns True if indexed, False if skipped."""
    collection = get_collection(client, provider, model_name)
    content_hash = _content_hash(content)

    # Skip if unchanged (unless forced)
    if not force:
        try:
            result = collection.get(ids=[title], include=["metadatas"])
            if result and result["ids"] and result["metadatas"]:
                existing_hash = result["metadatas"][0].get("content_hash")
                if existing_hash == content_hash:
                    return False  # Unchanged, skip
        except Exception:
            pass  # Proceed with indexing

    collection.upsert(
        ids=[title],
        documents=[content],
        metadatas=[{"title": title, "content_hash": content_hash}],
    )
    return True


def find_similar(
    client: chromadb.PersistentClient,
    content: str,
    top_k: int = 5,
    max_distance: float = 1.0,
    provider: EmbeddingProvider = "openai",
    model_name: str | None = None,
    create: bool = True,
) -> List[str]:
    """Query for similar notes, returning list of titles that meet threshold.

    Args:
        client: ChromaDB client
        content: Text to find similar notes for
        top_k: Maximum number of results to return
        max_distance: Maximum distance threshold (lower = more similar).
                      Cosine distance: 0 = identical, 2 = opposite.
                      Typical range: 0.3 (very similar) to 0.8 (loosely related)
        provider: Embedding provider to use
        model_name: Specific model name (optional)
    """
    collection = get_collection(client, provider, model_name, create=create)
    if collection is None:
        return []

    results = collection.query(
        query_texts=[content],
        n_results=top_k,
        include=["distances"],
    )

    # Filter by distance threshold and return titles
    if results and results["ids"] and results["ids"][0]:
        ids = results["ids"][0]
        distances = results["distances"][0]

        filtered = [
            title for title, dist in zip(ids, distances)
            if dist <= max_distance
        ]
        return filtered
    return []


def get_index_drift(
    client: chromadb.PersistentClient,
    entries: dict[str, str],
    provider: EmbeddingProvider = "openai",
    model_name: str | None = None,
    create: bool = True,
) -> dict[str, list[str]]:
    """Compare expected zettel entries with the active collection state."""
    collection = get_collection(client, provider, model_name, create=create)
    if collection is None:
        return {
            "missing_or_stale_ids": sorted(entries),
            "extra_ids": [],
        }
    expected_hashes = {entry_id: _content_hash(content) for entry_id, content in entries.items()}

    existing = collection.get(include=["metadatas"])
    existing_ids = existing["ids"] or []
    metadata_lookup = {
        entry_id: metadata or {}
        for entry_id, metadata in zip(existing_ids, existing.get("metadatas") or [])
    }

    missing_ids = sorted(
        entry_id
        for entry_id, content_hash in expected_hashes.items()
        if metadata_lookup.get(entry_id, {}).get("content_hash") != content_hash
    )
    extra_ids = sorted(set(existing_ids) - set(expected_hashes))

    return {
        "missing_or_stale_ids": missing_ids,
        "extra_ids": extra_ids,
    }


def delete_ids(
    client: chromadb.PersistentClient,
    ids: list[str],
    provider: EmbeddingProvider = "openai",
    model_name: str | None = None,
) -> None:
    """Delete IDs from the active collection."""
    if not ids:
        return
    collection = get_collection(client, provider, model_name)
    collection.delete(ids=ids)
