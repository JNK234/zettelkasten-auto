# ABOUTME: LLM-based concept extraction for Zettelkasten notes.
# ABOUTME: Uses OpenAI structured outputs to extract atomic concepts from source text.

import os
import json
from openai import OpenAI


def extract_concepts(content: str, model: str = "gpt-4o") -> list[dict]:
    """Extract atomic Zettelkasten concepts from source text.

    Args:
        content: Source text to extract concepts from.
        model: OpenAI model to use for extraction.

    Returns:
        List of dicts with keys: title, content, suggested_tags
    """
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    prompt = f"""Given this source note, extract all atomic Zettelkasten concepts.

Each concept should be:
- Self-contained (understandable without the source)
- Atomic (one idea per concept)
- Linked (suggest 2-3 tags that group related concepts)

Source note:
---
{content}
---"""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "concept_extraction",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "concepts": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "title": {"type": "string"},
                                    "content": {"type": "string"},
                                    "suggested_tags": {
                                        "type": "array",
                                        "items": {"type": "string"}
                                    }
                                },
                                "required": ["title", "content", "suggested_tags"],
                                "additionalProperties": False
                            }
                        }
                    },
                    "required": ["concepts"],
                    "additionalProperties": False
                }
            }
        }
    )

    result = json.loads(response.choices[0].message.content)
    return result["concepts"]
