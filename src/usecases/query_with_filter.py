"""UC5 – Ask a question with an arbitrary metadata filter for targeted retrieval."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.usecases.base import UseCase

if TYPE_CHECKING:
    from src.usecases.factory import UseCaseContext


@dataclass
class QueryWithFilterInput:
    question: str
    metadata_filter: dict
    collection_name: str | None = None  # None → searches default collection
    top_k: int | None = None


@dataclass
class SourceItem:
    content: str
    metadata: dict


@dataclass
class QueryWithFilterOutput:
    metadata_filter: dict
    question: str
    answer: str
    sources: list[SourceItem]


class QueryWithFilterUseCase(UseCase[QueryWithFilterInput, QueryWithFilterOutput]):
    """Run a RAG query with an explicit metadata filter.

    The filter is passed directly to the vector store's ``search_with_filter``
    method.  Use it to pre-select documents based on any metadata stored at
    ingest time (e.g. ``{"author": "alice"}``, ``{"year": "2024"}``).

    Compound filters use ChromaDB's logical operators::

        {"$and": [{"author": "alice"}, {"year": "2024"}]}
    """

    def __init__(self, ctx: UseCaseContext) -> None:
        self._ctx = ctx

    def _execute(self, input_data: QueryWithFilterInput) -> QueryWithFilterOutput:
        pipeline = self._ctx.pipeline_for(input_data.collection_name)
        result = pipeline.query(
            question=input_data.question,
            metadata_filter=input_data.metadata_filter,
            top_k=input_data.top_k,
        )
        return QueryWithFilterOutput(
            metadata_filter=input_data.metadata_filter,
            question=result["question"],
            answer=result["answer"],
            sources=[SourceItem(**s) for s in result["sources"]],
        )
