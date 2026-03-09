import httpx

from src.core.interfaces.llm import LLMProvider
from src.core.models import Chunk


class OpenAICompatibleLLM(LLMProvider):
    """LLM provider for any OpenAI-compatible /v1/chat/completions endpoint.

    Works with vLLM, LM Studio, Ollama (OpenAI mode), gpt-oss, and similar
    self-hosted deployments. Pass api_key to enable Bearer token auth.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        model: str = "gpt-oss-20b",
        temperature: float = 0.1,
        max_tokens: int = 1024,
        api_key: str = "",
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._api_key = api_key

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
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }
        headers = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        response = httpx.post(
            f"{self._base_url}/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=120.0,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
