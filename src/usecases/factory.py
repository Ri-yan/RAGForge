"""UseCaseFactory and UseCaseContext — wiring hub for all use cases."""
from __future__ import annotations

from pathlib import Path

from src.core.interfaces.embedder import Embedder
from src.core.interfaces.llm import LLMProvider
from src.core.interfaces.text_splitter import TextSplitter
from src.core.pipeline import RAGPipeline
from src.infrastructure.document_loaders.factory import DocumentLoaderFactory
from src.config.settings import Settings


class UseCaseContext:
    """Holds all shared dependencies for the use-case layer.

    Use :meth:`pipeline_for` to obtain a pipeline scoped to a named
    ChromaDB collection without creating any plumbing in individual use cases.
    """

    def __init__(
        self,
        pipeline: RAGPipeline,
        loader_factory: DocumentLoaderFactory,
        embedder: Embedder,
        llm: LLMProvider,
        text_splitter: TextSplitter,
        settings: Settings,
        upload_dir: Path,
    ) -> None:
        self.pipeline = pipeline
        self.loader_factory = loader_factory
        self.embedder = embedder
        self.llm = llm
        self.text_splitter = text_splitter
        self.settings = settings
        self.upload_dir = upload_dir
        self.metrics_enabled: bool = settings.metrics_enabled
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    def pipeline_for(self, collection_name: str | None) -> RAGPipeline:
        """Return the default pipeline or a new one scoped to *collection_name*."""
        if collection_name is None:
            return self.pipeline
        from src.infrastructure.vector_stores.factory import create_vector_store

        store = create_vector_store(self.settings, collection_name=collection_name)
        return RAGPipeline(
            text_splitter=self.text_splitter,
            embedder=self.embedder,
            vector_store=store,
            llm=self.llm,
            top_k=self.settings.retrieval_top_k,
        )


class UseCaseFactory:
    """Single entry point for instantiating any use case.

    All use cases share a :class:`UseCaseContext` so heavy components
    (embedder, LLM) are only constructed once.

    Example::

        factory = UseCaseFactory(ctx)
        result  = factory.ingest_and_query.execute(IngestAndQueryInput(...))
        doc_id  = factory.ingest_document.execute(IngestDocumentInput(...)).doc_id
    """

    def __init__(self, ctx: UseCaseContext) -> None:
        self._ctx = ctx

    def _make(self, use_case_cls):
        """Instantiate a use case and apply the metrics flag from context."""
        instance = use_case_cls(self._ctx)
        instance.metrics_enabled = self._ctx.metrics_enabled
        return instance

    # ── UC1 ──────────────────────────────────────────────────────────────────
    @property
    def ingest_and_query(self):
        from src.usecases.ingest_and_query import IngestAndQueryUseCase
        return self._make(IngestAndQueryUseCase)

    # ── UC2 ──────────────────────────────────────────────────────────────────
    @property
    def ingest_document(self):
        from src.usecases.ingest_document import IngestDocumentUseCase
        return self._make(IngestDocumentUseCase)

    # ── UC3 ──────────────────────────────────────────────────────────────────
    @property
    def query_by_doc_id(self):
        from src.usecases.query_by_doc_id import QueryByDocIdUseCase
        return self._make(QueryByDocIdUseCase)

    # ── UC4 ──────────────────────────────────────────────────────────────────
    @property
    def query_collection(self):
        from src.usecases.query_collection import QueryCollectionUseCase
        return self._make(QueryCollectionUseCase)

    # ── UC5 ──────────────────────────────────────────────────────────────────
    @property
    def query_with_filter(self):
        from src.usecases.query_with_filter import QueryWithFilterUseCase
        return self._make(QueryWithFilterUseCase)
    # ── UC6 ─────────────────────────────────────────────────────────────────────────────
    @property
    def list_documents(self):
        from src.usecases.list_documents import ListDocumentsUseCase
        return self._make(ListDocumentsUseCase)

    # ── UC7 ─────────────────────────────────────────────────────────────────────────────
    @property
    def delete_document(self):
        from src.usecases.delete_document import DeleteDocumentUseCase
        return self._make(DeleteDocumentUseCase)