"""UC6 – List all documents stored in a collection."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from src.usecases.base import UseCase

if TYPE_CHECKING:
    from src.usecases.factory import UseCaseContext


@dataclass
class ListDocumentsInput:
    collection_name: str | None = None  # None → default collection


@dataclass
class DocumentSummary:
    doc_id: str
    chunk_count: int
    metadata: dict = field(default_factory=dict)


@dataclass
class ListDocumentsOutput:
    collection_name: str
    documents: list[DocumentSummary]
    total_documents: int


class ListDocumentsUseCase(UseCase[ListDocumentsInput, ListDocumentsOutput]):
    """Return a summary of every document stored in a collection.

    Each entry contains the ``doc_id``, the number of chunks stored, and any
    extra metadata fields that were present on the first chunk (e.g. ``source``,
    ``author``, ``year``).
    """

    def __init__(self, ctx: UseCaseContext) -> None:
        self._ctx = ctx

    def _execute(self, input_data: ListDocumentsInput) -> ListDocumentsOutput:
        pipeline = self._ctx.pipeline_for(input_data.collection_name)
        records = pipeline._vector_store.list_documents()
        effective_collection = (
            input_data.collection_name or self._ctx.settings.chroma_collection_name
        )
        docs = [DocumentSummary(**r) for r in records]
        return ListDocumentsOutput(
            collection_name=effective_collection,
            documents=docs,
            total_documents=len(docs),
        )
