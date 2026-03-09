from pathlib import Path

import pytesseract
from PIL import Image

from src.core.interfaces.document_loader import DocumentLoader
from src.core.models import Document


class ImageLoader(DocumentLoader):
    """Loads image files and extracts text via Tesseract OCR."""

    def __init__(self, tesseract_lang: str = "eng") -> None:
        self._lang = tesseract_lang

    def load(self, file_path: Path) -> list[Document]:
        image = Image.open(file_path)
        text = pytesseract.image_to_string(image, lang=self._lang)
        if not text.strip():
            return []
        return [
            Document(
                content=text,
                metadata={
                    "source": str(file_path),
                    "file_type": file_path.suffix.lower(),
                    "ocr": True,
                },
            )
        ]

    def supported_extensions(self) -> list[str]:
        return [".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"]
