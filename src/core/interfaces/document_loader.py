from abc import ABC, abstractmethod
from pathlib import Path

from src.core.models import Document


class DocumentLoader(ABC):
    """Interface for loading documents from various sources."""

    @abstractmethod
    def load(self, file_path: Path) -> list[Document]:
        """Load documents from the given file path."""

    @abstractmethod
    def supported_extensions(self) -> list[str]:
        """Return list of supported file extensions (e.g., ['.pdf', '.txt'])."""
