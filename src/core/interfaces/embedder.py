from abc import ABC, abstractmethod


class Embedder(ABC):
    """Interface for generating vector embeddings from text."""

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts."""

    @abstractmethod
    def embed_query(self, query: str) -> list[float]:
        """Generate embedding for a single query text."""
