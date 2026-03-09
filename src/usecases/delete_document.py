"""UC7 – Delete a document by doc_id from the vector store and optionally from disk."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from src.usecases.base import UseCase

if TYPE_CHECKING:
    from src.usecases.factory import UseCaseContext

logger = logging.getLogger(__name__)


@dataclass
class DeleteDocumentInput:
    doc_id: str
    collection_name: str | None = None  # None → default collection
    delete_file: bool = True             # also remove the source file from disk


@dataclass
class DeleteDocumentOutput:
    doc_id: str
    collection_name: str
    chunks_deleted: int
    file_deleted: bool
    file_path: str | None


class DeleteDocumentUseCase(UseCase[DeleteDocumentInput, DeleteDocumentOutput]):
    """Remove all chunks for a doc_id from the vector store.

    When ``delete_file=True`` (the default), every file in the upload directory
    whose name matches the ``source`` metadata field of the deleted chunks is
    also removed from disk.  If no file path is recorded in metadata, or the
    file has already been removed, ``file_deleted`` will be ``False``.
    """

    def __init__(self, ctx: UseCaseContext) -> None:
        self._ctx = ctx

    def _execute(self, input_data: DeleteDocumentInput) -> DeleteDocumentOutput:
        pipeline = self._ctx.pipeline_for(input_data.collection_name)
        store = pipeline._vector_store
        effective_collection = (
            input_data.collection_name or self._ctx.settings.chroma_collection_name
        )

        # 1. Discover source file path before deletion (via list_documents metadata)
        file_path: str | None = None
        if input_data.delete_file:
            docs = store.list_documents()
            for doc in docs:
                if doc["doc_id"] == input_data.doc_id:
                    file_path = doc["metadata"].get("source")
                    break

        # 2. Delete chunks from vector store
        chunks_deleted = store.delete_by_doc_id(input_data.doc_id)
        logger.info(
            "Deleted %d chunks for doc_id=%s from collection=%s",
            chunks_deleted, input_data.doc_id, effective_collection,
        )

        # 3. Optionally remove file from disk
        file_deleted = False
        if input_data.delete_file and file_path:
            candidate = Path(file_path)
            # Also try relative to upload dir in case path is stored as basename
            if not candidate.exists():
                candidate = self._ctx.upload_dir / candidate.name
            if candidate.exists():
                candidate.unlink()
                file_deleted = True
                logger.info("Removed file from disk: %s", candidate)

        return DeleteDocumentOutput(
            doc_id=input_data.doc_id,
            collection_name=effective_collection,
            chunks_deleted=chunks_deleted,
            file_deleted=file_deleted,
            file_path=file_path,
        )
