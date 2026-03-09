"""UC1 – Upload a document and immediately query ONLY that document."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from src.usecases.base import UseCase

if TYPE_CHECKING:
    from src.usecases.factory import UseCaseContext


@dataclass
class IngestAndQueryInput:
    filename: str
    content: bytes
    question: str
    top_k: int | None = None


@dataclass
class SourceItem:
    content: str
    metadata: dict


@dataclass
class IngestAndQueryOutput:
    doc_id: str
    question: str
    answer: str
    sources: list[SourceItem]
    documents_loaded: int
    chunks_created: int


class IngestAndQueryUseCase(UseCase[IngestAndQueryInput, IngestAndQueryOutput]):
    """Ingest a single document then immediately run a RAG query scoped to it.

    The query is automatically filtered to the just-ingested doc_id so the
    answer is grounded exclusively in the uploaded content.
    """

    def __init__(self, ctx: UseCaseContext) -> None:
        self._ctx = ctx

    def _execute(self, input_data: IngestAndQueryInput) -> IngestAndQueryOutput:
        # 1. Persist upload
        upload_path = self._ctx.upload_dir / input_data.filename
        upload_path.write_bytes(input_data.content)

        # 2. Load documents
        loader = self._ctx.loader_factory.get_loader(upload_path)
        documents = loader.load(upload_path)

        # 3. Ingest — pipeline stamps doc_id into every chunk's metadata
        ingest_result = self._ctx.pipeline.ingest(documents)
        doc_id = ingest_result["doc_id"]

        # 4. Query scoped to this doc only
        query_result = self._ctx.pipeline.query(
            question=input_data.question,
            metadata_filter={"doc_id": doc_id},
            top_k=input_data.top_k,
        )

        return IngestAndQueryOutput(
            doc_id=doc_id,
            question=query_result["question"],
            answer=query_result["answer"],
            sources=[SourceItem(**s) for s in query_result["sources"]],
            documents_loaded=len(documents),
            chunks_created=ingest_result["chunk_count"],
        )
