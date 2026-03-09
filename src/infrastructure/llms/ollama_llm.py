import httpx

from src.core.interfaces.llm import LLMProvider
from src.core.models import Chunk


class OllamaLLM(LLMProvider):
    """LLM provider backed by a native Ollama /api/generate endpoint."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3",
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    def generate(self, prompt: str, context: list[Chunk]) -> str:
        context_text = "\n\n---\n\n".join(c.content for c in context)
        system_prompt = (
            "You are a helpful assistant. Answer the user's question based ONLY on "
            "the provided context. If the context does not contain enough information "
            "to answer, say so clearly. Do not make up information.\n\n"
            f"Context:\n{context_text}"
        )
        payload = {
            "model": self._model,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
            "options": {
                "temperature": self._temperature,
                "num_predict": self._max_tokens,
            },
        }
        response = httpx.post(
            f"{self._base_url}/api/generate",
            json=payload,
            timeout=120.0,
        )
        response.raise_for_status()
        return response.json().get("response", "")
