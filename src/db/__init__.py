# ABOUTME: Database module for zettel embeddings and similarity search.
# ABOUTME: Exports main functions for use by other modules.

from src.db.client import get_client, get_collection, index_zettel, find_similar
from src.db.embeddings import get_embedding_function, EmbeddingProvider

__all__ = [
    "get_client",
    "get_collection",
    "index_zettel",
    "find_similar",
    "get_embedding_function",
    "EmbeddingProvider",
]
