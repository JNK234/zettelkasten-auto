# ABOUTME: ChromaDB operations for zettel embeddings and similarity search
# ABOUTME: Uses OpenAI text-embedding-3-small for vector embeddings

import hashlib
from typing import List

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction


COLLECTION_NAME = "zettels"


def get_client(db_path: str) -> chromadb.PersistentClient:
    """Returns ChromaDB PersistentClient at the specified path."""
    return chromadb.PersistentClient(path=db_path)


def get_collection(client: chromadb.PersistentClient):
    """Gets or creates the zettels collection with OpenAI embeddings."""
    embedding_fn = OpenAIEmbeddingFunction(model_name="text-embedding-3-small")
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
    client: chromadb.PersistentClient, content: str, top_k: int = 5
) -> List[str]:
    """Query for similar notes, returning list of titles."""
    collection = get_collection(client)

    results = collection.query(
        query_texts=[content],
        n_results=top_k,
    )

    # Extract titles from results
    if results and results["ids"] and results["ids"][0]:
        return results["ids"][0]
    return []
