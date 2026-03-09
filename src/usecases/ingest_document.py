"""UC2 – Upload a document and receive a stable doc_id for later querying."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.usecases.base import UseCase

if TYPE_CHECKING:
    from src.usecases.factory import UseCaseContext


@dataclass
class IngestDocumentInput:
    filename: str
    content: bytes
    collection_name: str | None = None  # None → uses the default collection


@dataclass
class IngestDocumentOutput:
    doc_id: str
    file: str
    documents_loaded: int
    chunks_created: int
    collection_name: str


class IngestDocumentUseCase(UseCase[IngestDocumentInput, IngestDocumentOutput]):
    """Save and ingest a document, returning its stable ``doc_id``.

    The ``doc_id`` can be used later with :class:`QueryByDocIdUseCase` (UC3)
    to restrict retrieval to this document.  Pass ``collection_name`` to store
    the document in a dedicated named collection (knowledge base).
    """

    def __init__(self, ctx: UseCaseContext) -> None:
        self._ctx = ctx

    def _execute(self, input_data: IngestDocumentInput) -> IngestDocumentOutput:
        # 1. Persist upload
        upload_path = self._ctx.upload_dir / input_data.filename
        upload_path.write_bytes(input_data.content)

        # 2. Load documents
        loader = self._ctx.loader_factory.get_loader(upload_path)
        documents = loader.load(upload_path)

        # 3. Pick or build the target pipeline (default or named collection)
        effective_collection = (
            input_data.collection_name or self._ctx.settings.chroma_collection_name
        )
        pipeline = self._ctx.pipeline_for(input_data.collection_name)

        # 4. Ingest
        result = pipeline.ingest(documents)

        return IngestDocumentOutput(
            doc_id=result["doc_id"],
            file=str(upload_path),
            documents_loaded=len(documents),
            chunks_created=result["chunk_count"],
            collection_name=effective_collection,
        )
