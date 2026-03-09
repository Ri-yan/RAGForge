from pathlib import Path

from pypdf import PdfReader

from src.core.interfaces.document_loader import DocumentLoader
from src.core.models import Document


class PDFLoader(DocumentLoader):
    """Loads PDF files and extracts text page by page."""

    def load(self, file_path: Path) -> list[Document]:
        reader = PdfReader(str(file_path))
        documents: list[Document] = []
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                documents.append(
                    Document(
                        content=text,
                        metadata={
                            "source": str(file_path),
                            "file_type": ".pdf",
                            "page": page_num,
                        },
                    )
                )
        return documents

    def supported_extensions(self) -> list[str]:
        return [".pdf"]
