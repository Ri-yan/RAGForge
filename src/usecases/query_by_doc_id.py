"""UC3 – Ask a question scoped to a specific document identified by doc_id."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.usecases.base import UseCase

if TYPE_CHECKING:
    from src.usecases.factory import UseCaseContext


@dataclass
class QueryByDocIdInput:
    question: str
    doc_id: str
    collection_name: str | None = None  # None → searches default collection
    top_k: int | None = None


@dataclass
class SourceItem:
    content: str
    metadata: dict


@dataclass
class QueryByDocIdOutput:
    doc_id: str
    question: str
    answer: str
    sources: list[SourceItem]


class QueryByDocIdUseCase(UseCase[QueryByDocIdInput, QueryByDocIdOutput]):
    """Run a RAG query restricted to a single document.

    Retrieval is filtered by ``{"doc_id": <value>}`` so that only chunks
    belonging to the specified document are considered when answering.
    """

    def __init__(self, ctx: UseCaseContext) -> None:
        self._ctx = ctx

    def _execute(self, input_data: QueryByDocIdInput) -> QueryByDocIdOutput:
        pipeline = self._ctx.pipeline_for(input_data.collection_name)
        result = pipeline.query(
            question=input_data.question,
            metadata_filter={"doc_id": input_data.doc_id},
            top_k=input_data.top_k,
        )
        return QueryByDocIdOutput(
            doc_id=input_data.doc_id,
            question=result["question"],
            answer=result["answer"],
            sources=[SourceItem(**s) for s in result["sources"]],
        )
