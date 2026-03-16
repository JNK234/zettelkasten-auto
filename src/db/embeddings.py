# ABOUTME: Embedding function factory for ChromaDB.
# ABOUTME: Supports OpenAI and SentenceTransformer backends.

import os
from typing import Literal

from chromadb.api.types import EmbeddingFunction


EmbeddingProvider = Literal["openai", "sentence-transformer"]
DEFAULT_EMBEDDING_MODELS: dict[EmbeddingProvider, str] = {
    "openai": "text-embedding-3-small",
    "sentence-transformer": "all-MiniLM-L6-v2",
}


def get_default_embedding_model(provider: EmbeddingProvider) -> str:
    """Return the default model for an embedding provider."""
    return DEFAULT_EMBEDDING_MODELS[provider]


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
        model = model_name or get_default_embedding_model("openai")
        return OpenAIEmbeddingFunction(api_key=api_key, model_name=model)

    elif provider == "sentence-transformer":
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

        model = model_name or get_default_embedding_model("sentence-transformer")
        return SentenceTransformerEmbeddingFunction(model_name=model)

    else:
        raise ValueError(f"Unknown embedding provider: {provider}")
