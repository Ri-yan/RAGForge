from abc import ABC, abstractmethod

from src.core.models import Chunk


class VectorStore(ABC):
    """Interface for storing and retrieving vector embeddings."""

    @abstractmethod
    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Store chunks with their embeddings."""

    @abstractmethod
    def search(self, query_embedding: list[float], top_k: int = 5) -> list[Chunk]:
        """Retrieve the top-k most similar chunks for a query embedding."""

    @abstractmethod
    def search_with_filter(
        self,
        query_embedding: list[float],
        metadata_filter: dict,
        top_k: int = 5,
    ) -> list[Chunk]:
        """Retrieve top-k chunks matching both similarity and a metadata filter.

        The filter dict uses the same syntax as the underlying store implementation.
        Simple equality: ``{"doc_id": "abc123"}``.
        Compound: ``{"$and": [{"year": "2024"}, {"author": "alice"}]}``.
        """

    @abstractmethod
    def list_documents(self) -> list[dict]:
        """Return one summary entry per unique doc_id stored in the collection.

        Each entry contains at least:
        ``{"doc_id": str, "chunk_count": int, "metadata": dict}``
        where *metadata* is taken from the first chunk of that document.
        """

    @abstractmethod
    def delete_by_doc_id(self, doc_id: str) -> int:
        """Delete all chunks belonging to *doc_id*.  Returns the number of chunks removed."""

    @abstractmethod
    def delete_collection(self) -> None:
        """Delete the entire collection."""
