from __future__ import annotations

import uuid

from src.core.interfaces import Embedder, LLMProvider, TextSplitter, VectorStore
from src.core.models import Chunk, Document


class RAGPipeline:
    """Orchestrates the full Retrieval-Augmented Generation pipeline.

    Follows the Dependency Inversion Principle — depends only on abstractions.
    """

    def __init__(
        self,
        text_splitter: TextSplitter,
        embedder: Embedder,
        vector_store: VectorStore,
        llm: LLMProvider,
        top_k: int = 5,
    ) -> None:
        self._text_splitter = text_splitter
        self._embedder = embedder
        self._vector_store = vector_store
        self._llm = llm
        self._top_k = top_k

    def ingest(self, documents: list[Document]) -> dict:
        """Ingest documents: split, embed, and store.

        Returns a dict with ``doc_id`` (stable identifier for this batch) and
        ``chunk_count`` (number of chunks stored).
        """
        doc_id = str(uuid.uuid4())
        chunks = self._text_splitter.split(documents)
        if not chunks:
            return {"doc_id": doc_id, "chunk_count": 0}
        for chunk in chunks:
            chunk.metadata["doc_id"] = doc_id
        embeddings = self._embedder.embed_texts([c.content for c in chunks])
        self._vector_store.add_chunks(chunks, embeddings)
        return {"doc_id": doc_id, "chunk_count": len(chunks)}

    def query(
        self,
        question: str,
        metadata_filter: dict | None = None,
        top_k: int | None = None,
    ) -> dict:
        """Run a RAG query: embed question, retrieve context, generate answer.

        Pass ``metadata_filter`` to restrict retrieval to specific docs/collections.
        Pass ``top_k`` to override the pipeline default.
        """
        effective_top_k = top_k if top_k is not None else self._top_k
        query_embedding = self._embedder.embed_query(question)
        if metadata_filter:
            relevant_chunks = self._vector_store.search_with_filter(
                query_embedding, metadata_filter, top_k=effective_top_k
            )
        else:
            relevant_chunks = self._vector_store.search(query_embedding, top_k=effective_top_k)
        answer = self._llm.generate(prompt=question, context=relevant_chunks)
        return {
            "question": question,
            "answer": answer,
            "sources": self._format_sources(relevant_chunks),
        }

    @staticmethod
    def _format_sources(chunks: list[Chunk]) -> list[dict]:
        return [
            {"content": c.content[:200], "metadata": c.metadata}
            for c in chunks
        ]
