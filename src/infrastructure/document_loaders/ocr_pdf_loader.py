from pathlib import Path

import pytesseract
from pdf2image import convert_from_path
from pypdf import PdfReader

from src.core.interfaces.document_loader import DocumentLoader
from src.core.models import Document


class OCRPDFLoader(DocumentLoader):
    """Loads PDF files with OCR fallback for scanned/image-based pages.

    Tries text extraction first (fast). If a page yields little or no text,
    falls back to OCR via Tesseract.
    """

    MIN_TEXT_LENGTH = 50  # below this threshold, treat as scanned page

    def __init__(self, tesseract_lang: str = "eng") -> None:
        self._lang = tesseract_lang

    def load(self, file_path: Path) -> list[Document]:
        reader = PdfReader(str(file_path))
        documents: list[Document] = []

        for page_num, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()

            if len(text) < self.MIN_TEXT_LENGTH:
                text = self._ocr_page(file_path, page_num)

            if text.strip():
                documents.append(
                    Document(
                        content=text,
                        metadata={
                            "source": str(file_path),
                            "file_type": ".pdf",
                            "page": page_num,
                            "ocr": len((page.extract_text() or "").strip()) < self.MIN_TEXT_LENGTH,
                        },
                    )
                )
        return documents

    def _ocr_page(self, file_path: Path, page_num: int) -> str:
        images = convert_from_path(
            str(file_path),
            first_page=page_num,
            last_page=page_num,
        )
        if not images:
            return ""
        return pytesseract.image_to_string(images[0], lang=self._lang)

    def supported_extensions(self) -> list[str]:
        return [".pdf"]
