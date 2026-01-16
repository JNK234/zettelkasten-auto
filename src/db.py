# ABOUTME: ChromaDB operations for zettel embeddings and similarity search
# ABOUTME: Uses OpenAI text-embedding-3-small for vector embeddings

import hashlib
import os
from typing import List

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction


COLLECTION_NAME = "zettels"


def get_client(db_path: str) -> chromadb.PersistentClient:
    """Returns ChromaDB PersistentClient at the specified path."""
    return chromadb.PersistentClient(path=db_path)


def get_collection(client: chromadb.PersistentClient):
    """Gets or creates the zettels collection with OpenAI embeddings."""
    api_key = os.environ.get("OPENAI_API_KEY")
    embedding_fn = OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name="text-embedding-3-small"
    )
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
    )


def _content_hash(content: str) -> str:
    """Generate SHA256 hash of content for change detection."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def index_zettel(client: chromadb.PersistentClient, title: str, content: str) -> None:
    """Upsert a zettel with content hash in metadata for change detection."""
    collection = get_collection(client)
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
) -> List[str]:
    """Query for similar notes, returning list of titles that meet threshold.

    Args:
        client: ChromaDB client
        content: Text to find similar notes for
        top_k: Maximum number of results to return
        max_distance: Maximum distance threshold (lower = more similar).
                      OpenAI embeddings use cosine distance where 0 = identical, 2 = opposite.
                      Typical useful range: 0.3 (very similar) to 0.8 (somewhat related)
    """
    collection = get_collection(client)

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
