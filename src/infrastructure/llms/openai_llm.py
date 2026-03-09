from src.infrastructure.llms.openai_compatible_llm import OpenAICompatibleLLM


class OpenAILLM(OpenAICompatibleLLM):
    """LLM provider for the official OpenAI API (or Azure OpenAI).

    Extends OpenAICompatibleLLM with a required api_key and the default
    OpenAI base URL. Override base_url for Azure or other compatible endpoints.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        base_url: str = "https://api.openai.com/v1",
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> None:
        super().__init__(
            base_url=base_url,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
        )
