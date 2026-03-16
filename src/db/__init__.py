# ABOUTME: Database module for zettel embeddings and similarity search.
# ABOUTME: Exports main functions for use by other modules.

from src.db.client import (
    delete_ids,
    find_similar,
    get_client,
    get_collection,
    get_collection_metadata,
    get_collection_name,
    get_index_drift,
    index_zettel,
    needs_indexing,
)
from src.db.embeddings import (
    EmbeddingProvider,
    get_default_embedding_model,
    get_embedding_function,
)

__all__ = [
    "delete_ids",
    "get_client",
    "get_collection",
    "get_collection_metadata",
    "get_collection_name",
    "get_index_drift",
    "index_zettel",
    "find_similar",
    "needs_indexing",
    "get_default_embedding_model",
    "get_embedding_function",
    "EmbeddingProvider",
]
