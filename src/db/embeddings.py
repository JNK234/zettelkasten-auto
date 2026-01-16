# ABOUTME: Embedding function factory for ChromaDB.
# ABOUTME: Supports OpenAI and SentenceTransformer backends.

import os
from typing import Literal

from chromadb.api.types import EmbeddingFunction


EmbeddingProvider = Literal["openai", "sentence-transformer"]


def get_embedding_function(
    provider: EmbeddingProvider = "openai",
    model_name: str | None = None,
) -> EmbeddingFunction:
    """Create an embedding function based on the configured provider.

    Args:
        provider: "openai" or "sentence-transformer"
        model_name: Model to use. Defaults:
            - openai: "text-embedding-3-small"
            - sentence-transformer: "all-MiniLM-L6-v2"

    Returns:
        ChromaDB-compatible embedding function
    """
    if provider == "openai":
        from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

        api_key = os.environ.get("OPENAI_API_KEY")
        model = model_name or "text-embedding-3-small"
        return OpenAIEmbeddingFunction(api_key=api_key, model_name=model)

    elif provider == "sentence-transformer":
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

        model = model_name or "all-MiniLM-L6-v2"
        return SentenceTransformerEmbeddingFunction(model_name=model)

    else:
        raise ValueError(f"Unknown embedding provider: {provider}")
