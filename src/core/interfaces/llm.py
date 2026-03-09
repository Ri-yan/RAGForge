from abc import ABC, abstractmethod

from src.core.models import Chunk


class LLMProvider(ABC):
    """Interface for interacting with a large language model."""

    @abstractmethod
    def generate(self, prompt: str, context: list[Chunk]) -> str:
        """Generate a response given a prompt and retrieved context chunks."""
