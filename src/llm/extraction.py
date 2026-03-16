# ABOUTME: Concept extraction logic using configurable LLM providers.
# ABOUTME: Wraps provider calls with prompt formatting and validates outputs.

from src.llm.providers import (
    InvalidLLMOutputError,
    LLMProvider,
    ProviderResponseError,
    get_llm_provider,
)
from src.prompts import EXTRACTION_PROMPT, SYSTEM_CONTEXT


REQUIRED_CONCEPT_FIELDS = {
    "title",
    "content",
    "suggested_tags",
    "concept_type",
    "abstraction_level",
}
VALID_CONCEPT_TYPES = {
    "mechanism",
    "pattern",
    "mental-model",
    "heuristic",
    "observation",
    "gotcha",
}
VALID_ABSTRACTION_LEVELS = {
    "concrete-example",
    "specific-technique",
    "general-principle",
    "meta-concept",
}


def _extract_concept_payload(raw_payload) -> list[dict]:
    """Normalize provider output to a concept list."""
    if isinstance(raw_payload, dict) and isinstance(raw_payload.get("concepts"), list):
        return raw_payload["concepts"]
    if isinstance(raw_payload, list):
        return raw_payload
    raise InvalidLLMOutputError("Provider response must be a JSON array or an object with a 'concepts' array.")


def _validate_concept(concept: object) -> dict | None:
    """Validate a single concept object and normalize string values."""
    if not isinstance(concept, dict):
        return None
    if not REQUIRED_CONCEPT_FIELDS.issubset(concept):
        return None

    title = concept.get("title")
    content = concept.get("content")
    suggested_tags = concept.get("suggested_tags")
    concept_type = concept.get("concept_type")
    abstraction_level = concept.get("abstraction_level")

    if not isinstance(title, str) or not title.strip():
        return None
    if not isinstance(content, str) or not content.strip():
        return None
    if not isinstance(suggested_tags, list) or not (2 <= len(suggested_tags) <= 3):
        return None
    if any(not isinstance(tag, str) or not tag.strip() for tag in suggested_tags):
        return None
    if concept_type not in VALID_CONCEPT_TYPES:
        return None
    if abstraction_level not in VALID_ABSTRACTION_LEVELS:
        return None

    return {
        "title": title.strip(),
        "content": content.strip(),
        "suggested_tags": [tag.strip() for tag in suggested_tags],
        "concept_type": concept_type,
        "abstraction_level": abstraction_level,
    }


def extract_concepts_with_diagnostics(
    content: str,
    provider: LLMProvider = "openai",
    model: str | None = None,
    **kwargs,
) -> tuple[list[dict], int]:
    """Extract concepts and report how many were discarded during validation."""
    llm = get_llm_provider(provider, model, **kwargs)
    prompt = EXTRACTION_PROMPT.format(content=content)
    raw_payload = llm.extract(prompt, SYSTEM_CONTEXT)
    concepts = _extract_concept_payload(raw_payload)

    valid_concepts: list[dict] = []
    invalid_count = 0
    for concept in concepts:
        validated = _validate_concept(concept)
        if validated is None:
            invalid_count += 1
            continue
        valid_concepts.append(validated)

    return valid_concepts, invalid_count


def extract_concepts(
    content: str,
    provider: LLMProvider = "openai",
    model: str | None = None,
    **kwargs,
) -> list[dict]:
    """Extract atomic Zettelkasten concepts from source text."""
    concepts, _ = extract_concepts_with_diagnostics(content, provider, model, **kwargs)
    return concepts
