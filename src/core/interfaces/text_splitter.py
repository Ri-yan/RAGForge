from abc import ABC, abstractmethod

from src.core.models import Chunk, Document


class TextSplitter(ABC):
    """Interface for splitting documents into smaller chunks."""

    @abstractmethod
    def split(self, documents: list[Document]) -> list[Chunk]:
        """Split documents into chunks."""
