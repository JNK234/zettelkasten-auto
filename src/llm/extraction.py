# ABOUTME: Concept extraction logic using configurable LLM providers.
# ABOUTME: Wraps provider calls with prompt formatting.

from src.llm.providers import get_llm_provider, LLMProvider
from src.prompts import EXTRACTION_PROMPT, SYSTEM_CONTEXT


def extract_concepts(
    content: str,
    provider: LLMProvider = "openai",
    model: str | None = None,
    **kwargs,
) -> list[dict]:
    """Extract atomic Zettelkasten concepts from source text.

    Args:
        content: Source text to extract concepts from.
        provider: LLM provider to use ("openai", "ollama", "gemini")
        model: Model name (optional, uses provider defaults)
        **kwargs: Additional provider options (e.g., base_url for Ollama)

    Returns:
        List of dicts with keys: title, content, suggested_tags
    """
    llm = get_llm_provider(provider, model, **kwargs)
    prompt = EXTRACTION_PROMPT.format(content=content)
    return llm.extract(prompt, SYSTEM_CONTEXT)
