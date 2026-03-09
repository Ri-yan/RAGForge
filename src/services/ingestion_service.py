import logging
import shutil
from pathlib import Path

from src.core.models import Document
from src.core.pipeline import RAGPipeline
from src.infrastructure.document_loaders.factory import DocumentLoaderFactory

logger = logging.getLogger(__name__)


class IngestionService:
    """Handles document upload, loading, and ingestion into the RAG pipeline."""

    def __init__(
        self,
        pipeline: RAGPipeline,
        loader_factory: DocumentLoaderFactory,
        upload_dir: str = "./data/uploads",
    ) -> None:
        self._pipeline = pipeline
        self._loader_factory = loader_factory
        self._upload_dir = Path(upload_dir)
        self._upload_dir.mkdir(parents=True, exist_ok=True)

    def ingest_file(self, file_path: Path) -> dict:
        """Load and ingest a single file. Returns ingestion stats including doc_id."""
        loader = self._loader_factory.get_loader(file_path)
        documents = loader.load(file_path)
        result = self._pipeline.ingest(documents)
        logger.info(
            "Ingested %s → %d documents, %d chunks (doc_id=%s)",
            file_path, len(documents), result["chunk_count"], result["doc_id"],
        )
        return {
            "file": str(file_path),
            "documents_loaded": len(documents),
            "chunks_created": result["chunk_count"],
            "doc_id": result["doc_id"],
        }

    def save_and_ingest(self, filename: str, content: bytes) -> dict:
        """Save uploaded bytes to disk, then ingest."""
        dest = self._upload_dir / filename
        dest.write_bytes(content)
        return self.ingest_file(dest)

    def ingest_directory(self, directory: Path) -> list[dict]:
        """Ingest all supported files in a directory."""
        results: list[dict] = []
        for file_path in sorted(directory.rglob("*")):
            if not file_path.is_file():
                continue
            try:
                result = self.ingest_file(file_path)
                results.append(result)
            except ValueError:
                logger.debug("Skipping unsupported file: %s", file_path)
        return results

    def clear_uploads(self) -> None:
        """Remove all uploaded files."""
        if self._upload_dir.exists():
            shutil.rmtree(self._upload_dir)
            self._upload_dir.mkdir(parents=True, exist_ok=True)
