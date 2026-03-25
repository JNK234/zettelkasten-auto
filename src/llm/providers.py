# ABOUTME: LLM provider implementations for concept extraction.
# ABOUTME: Supports OpenAI, Ollama (local), Google Gemini, and Anthropic.

import json
import os
from abc import ABC, abstractmethod
from typing import Literal

import pydantic

LLMProvider = Literal["openai", "ollama", "gemini", "anthropic"]


class ProviderResponseError(Exception):
    """Raised when the upstream LLM provider request fails."""


class InvalidLLMOutputError(Exception):
    """Raised when an LLM provider returns invalid JSON output."""


class ConceptItem(pydantic.BaseModel):
    """Schema for a single extracted concept."""
    title: str
    content: str
    suggested_tags: list[str]
    concept_type: Literal[
        "mechanism", "pattern", "mental-model",
        "heuristic", "observation", "gotcha",
    ]
    abstraction_level: Literal[
        "concrete-example", "specific-technique",
        "general-principle", "meta-concept",
    ]


class ConceptExtraction(pydantic.BaseModel):
    """Schema for the full extraction response."""
    concepts: list[ConceptItem]


class BaseLLMProvider(ABC):
    """Base class for LLM providers."""

    @abstractmethod
    def extract(self, prompt: str, system_prompt: str):
        """Extract concepts from content using the LLM.

        Args:
            prompt: The formatted extraction prompt with content
            system_prompt: System context for the LLM

        Returns:
            Parsed JSON output from the provider
        """
        pass


class OpenAIProvider(BaseLLMProvider):
    """OpenAI API provider (GPT-4o, GPT-4o-mini, etc.)"""

    def __init__(self, model: str = "gpt-4o-mini"):
        from openai import OpenAI
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.model = model

    def extract(self, prompt: str, system_prompt: str):
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
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
                                                "items": {"type": "string"},
                                                "minItems": 2,
                                                "maxItems": 3,
                                            },
                                            "concept_type": {
                                                "type": "string",
                                                "enum": [
                                                    "mechanism",
                                                    "pattern",
                                                    "mental-model",
                                                    "heuristic",
                                                    "observation",
                                                    "gotcha",
                                                ],
                                            },
                                            "abstraction_level": {
                                                "type": "string",
                                                "enum": [
                                                    "concrete-example",
                                                    "specific-technique",
                                                    "general-principle",
                                                    "meta-concept",
                                                ],
                                            },
                                        },
                                        "required": [
                                            "title",
                                            "content",
                                            "suggested_tags",
                                            "concept_type",
                                            "abstraction_level",
                                        ],
                                        "additionalProperties": False,
                                    },
                                }
                            },
                            "required": ["concepts"],
                            "additionalProperties": False,
                        },
                    },
                },
            )
        except Exception as exc:
            raise ProviderResponseError(f"OpenAI request failed: {exc}") from exc

        try:
            return json.loads(response.choices[0].message.content)
        except json.JSONDecodeError as exc:
            raise InvalidLLMOutputError(f"OpenAI returned invalid JSON: {exc}") from exc


class OllamaProvider(BaseLLMProvider):
    """Ollama local LLM provider (Llama, Mistral, Qwen, etc.)"""

    def __init__(self, model: str = "llama3.1:8b", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url

    def extract(self, prompt: str, system_prompt: str):
        import requests

        full_prompt = (
            f"{system_prompt}\n\n{prompt}\n\n"
            "Respond with valid JSON only in this exact format:\n"
            "{\"concepts\": [{"
            "\"title\": \"...\", "
            "\"content\": \"...\", "
            "\"suggested_tags\": [\"Tag One\", \"Tag Two\"], "
            "\"concept_type\": \"mechanism\", "
            "\"abstraction_level\": \"general-principle\""
            "}]}"
        )

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": full_prompt,
                    "stream": False,
                    "format": "json",
                },
                timeout=120,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise ProviderResponseError(f"Ollama request failed: {exc}") from exc

        try:
            result_text = response.json()["response"]
            return json.loads(result_text)
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise InvalidLLMOutputError(f"Ollama returned invalid JSON: {exc}") from exc


class GeminiProvider(BaseLLMProvider):
    """Google Gemini API provider"""

    def __init__(self, model: str = "gemini-1.5-flash"):
        import google.generativeai as genai
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)

    def extract(self, prompt: str, system_prompt: str):
        full_prompt = (
            f"{system_prompt}\n\n{prompt}\n\n"
            "Respond with valid JSON only in this exact format:\n"
            "{\"concepts\": [{"
            "\"title\": \"...\", "
            "\"content\": \"...\", "
            "\"suggested_tags\": [\"Tag One\", \"Tag Two\"], "
            "\"concept_type\": \"mechanism\", "
            "\"abstraction_level\": \"general-principle\""
            "}]}"
        )

        try:
            response = self.model.generate_content(
                full_prompt,
                generation_config={
                    "response_mime_type": "application/json",
                }
            )
        except Exception as exc:
            raise ProviderResponseError(f"Gemini request failed: {exc}") from exc

        try:
            return json.loads(response.text)
        except json.JSONDecodeError as exc:
            raise InvalidLLMOutputError(f"Gemini returned invalid JSON: {exc}") from exc


class AnthropicProvider(BaseLLMProvider):
    """Anthropic API provider (Claude Sonnet, Haiku, Opus)"""

    def __init__(self, model: str = "claude-sonnet-4-5"):
        from anthropic import Anthropic
        self.client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self.model = model

    def extract(self, prompt: str, system_prompt: str):
        try:
            response = self.client.messages.parse(
                model=self.model,
                max_tokens=4096,
                system=system_prompt,
                output_format=ConceptExtraction,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as exc:
            raise ProviderResponseError(f"Anthropic request failed: {exc}") from exc

        if response.parsed_output is None:
            raise InvalidLLMOutputError("Anthropic returned no parsed output")

        return response.parsed_output.model_dump()


def get_llm_provider(
    provider: LLMProvider = "openai",
    model: str | None = None,
    **kwargs,
) -> BaseLLMProvider:
    """Factory function to create LLM provider instances.

    Args:
        provider: "openai", "ollama", "gemini", or "anthropic"
        model: Model name (optional, uses provider defaults)
        **kwargs: Additional provider-specific options (e.g., base_url for Ollama)

    Returns:
        Configured LLM provider instance
    """
    if provider == "openai":
        return OpenAIProvider(model=model or "gpt-4o-mini")

    elif provider == "ollama":
        return OllamaProvider(
            model=model or "llama3.1:8b",
            base_url=kwargs.get("base_url", "http://localhost:11434"),
        )

    elif provider == "gemini":
        return GeminiProvider(model=model or "gemini-1.5-flash")

    elif provider == "anthropic":
        return AnthropicProvider(model=model or "claude-sonnet-4-5")

    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
