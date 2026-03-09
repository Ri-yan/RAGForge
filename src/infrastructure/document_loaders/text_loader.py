from pathlib import Path

from src.core.interfaces.document_loader import DocumentLoader
from src.core.models import Document


class TextFileLoader(DocumentLoader):
    """Loads plain text files (.txt, .md)."""

    def load(self, file_path: Path) -> list[Document]:
        text = file_path.read_text(encoding="utf-8")
        return [
            Document(
                content=text,
                metadata={"source": str(file_path), "file_type": file_path.suffix},
            )
        ]

    def supported_extensions(self) -> list[str]:
        return [".txt", ".md"]
