from pathlib import Path

from src.core.interfaces.document_loader import DocumentLoader
from src.infrastructure.document_loaders.image_loader import ImageLoader
from src.infrastructure.document_loaders.ocr_pdf_loader import OCRPDFLoader
from src.infrastructure.document_loaders.pdf_loader import PDFLoader
from src.infrastructure.document_loaders.text_loader import TextFileLoader


class DocumentLoaderFactory:
    """Factory that returns the appropriate DocumentLoader based on file extension.

    Open/Closed Principle: register new loaders without modifying existing code.
    """

    def __init__(self) -> None:
        self._loaders: dict[str, DocumentLoader] = {}

    def register_loader(self, loader: DocumentLoader) -> None:
        for ext in loader.supported_extensions():
            self._loaders[ext.lower()] = loader

    def get_loader(self, file_path: Path) -> DocumentLoader:
        ext = file_path.suffix.lower()
        loader = self._loaders.get(ext)
        if loader is None:
            raise ValueError(
                f"No loader registered for extension '{ext}'. "
                f"Supported: {list(self._loaders.keys())}"
            )
        return loader


def create_default_loader_factory(ocr_enabled: bool = True, tesseract_lang: str = "eng") -> DocumentLoaderFactory:
    """Create a factory pre-registered with all built-in loaders."""
    factory = DocumentLoaderFactory()
    factory.register_loader(TextFileLoader())

    if ocr_enabled:
        factory.register_loader(OCRPDFLoader(tesseract_lang=tesseract_lang))
        factory.register_loader(ImageLoader(tesseract_lang=tesseract_lang))
    else:
        factory.register_loader(PDFLoader())

    return factory
