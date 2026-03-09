"""UC4 – Ask a question against a named collection (knowledge base)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.usecases.base import UseCase

if TYPE_CHECKING:
    from src.usecases.factory import UseCaseContext


@dataclass
class QueryCollectionInput:
    question: str
    collection_name: str
    top_k: int | None = None


@dataclass
class SourceItem:
    content: str
    metadata: dict


@dataclass
class QueryCollectionOutput:
    collection_name: str
    question: str
    answer: str
    sources: list[SourceItem]


class QueryCollectionUseCase(UseCase[QueryCollectionInput, QueryCollectionOutput]):
    """Run a RAG query against a named ChromaDB collection.

    Each collection acts as an independent knowledge base.  Documents are
    ingested into a named collection via :class:`IngestDocumentUseCase` (UC2)
    by passing ``collection_name``.  The query here uses the exact same
    collection without any additional metadata filter.
    """

    def __init__(self, ctx: UseCaseContext) -> None:
        self._ctx = ctx

    def _execute(self, input_data: QueryCollectionInput) -> QueryCollectionOutput:
        pipeline = self._ctx.pipeline_for(input_data.collection_name)
        result = pipeline.query(
            question=input_data.question,
            top_k=input_data.top_k,
        )
        return QueryCollectionOutput(
            collection_name=input_data.collection_name,
            question=result["question"],
            answer=result["answer"],
            sources=[SourceItem(**s) for s in result["sources"]],
        )
