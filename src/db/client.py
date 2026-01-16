# ABOUTME: ChromaDB client and collection operations.
# ABOUTME: Handles indexing and similarity search for zettels.

import hashlib
from typing import List

import chromadb

from src.db.embeddings import get_embedding_function, EmbeddingProvider


COLLECTION_NAME = "zettels"


def get_client(db_path: str) -> chromadb.PersistentClient:
    """Returns ChromaDB PersistentClient at the specified path."""
    return chromadb.PersistentClient(path=db_path)


def get_collection(
    client: chromadb.PersistentClient,
    provider: EmbeddingProvider = "openai",
    model_name: str | None = None,
):
    """Gets or creates the zettels collection with configured embeddings."""
    embedding_fn = get_embedding_function(provider, model_name)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
    )


def _content_hash(content: str) -> str:
    """Generate SHA256 hash of content for change detection."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def index_zettel(
    client: chromadb.PersistentClient,
    title: str,
    content: str,
    provider: EmbeddingProvider = "openai",
    model_name: str | None = None,
) -> None:
    """Upsert a zettel with content hash in metadata for change detection."""
    collection = get_collection(client, provider, model_name)
    content_hash = _content_hash(content)

    collection.upsert(
        ids=[title],
        documents=[content],
        metadatas=[{"title": title, "content_hash": content_hash}],
    )


def find_similar(
    client: chromadb.PersistentClient,
    content: str,
    top_k: int = 5,
    max_distance: float = 1.0,
    provider: EmbeddingProvider = "openai",
    model_name: str | None = None,
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
    collection = get_collection(client, provider, model_name)

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
