"""LLM provider factory — resolves the correct provider from settings.llm_type."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.core.interfaces.llm import LLMProvider

if TYPE_CHECKING:
    from src.config.settings import Settings

_VALID_TYPES = ("ollama", "openai_compatible", "openai")


def create_llm(settings: "Settings") -> LLMProvider:
    """Instantiate the LLM provider declared in settings.llm_type.

    Supported values
    ----------------
    ``"ollama"``
        Native Ollama server (``/api/generate``).
    ``"openai_compatible"``
        Any OpenAI-compatible endpoint such as vLLM, LM Studio, or gpt-oss
        (``/v1/chat/completions``).
    ``"openai"``
        Official OpenAI API (requires ``openai_api_key`` in settings).
    """
    llm_type = settings.llm_type.lower()

    if llm_type == "ollama":
        from src.infrastructure.llms.ollama_llm import OllamaLLM

        return OllamaLLM(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )

    if llm_type == "openai_compatible":
        from src.infrastructure.llms.openai_compatible_llm import OpenAICompatibleLLM

        return OpenAICompatibleLLM(
            base_url=settings.openai_compatible_base_url,
            model=settings.openai_compatible_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )

    if llm_type == "openai":
        from src.infrastructure.llms.openai_llm import OpenAILLM

        return OpenAILLM(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            base_url=settings.openai_base_url,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )

    raise ValueError(
        f"Unknown llm_type {settings.llm_type!r}. "
        f"Valid options: {', '.join(repr(t) for t in _VALID_TYPES)}."
    )
