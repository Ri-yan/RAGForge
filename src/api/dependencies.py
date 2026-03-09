"""FastAPI dependency injection — wires up all components via abstractions."""

from functools import lru_cache
from pathlib import Path

from src.config.settings import get_settings
from src.core.pipeline import RAGPipeline
from src.infrastructure.document_loaders.factory import create_default_loader_factory, DocumentLoaderFactory
from src.infrastructure.embedders.sentence_transformer_embedder import SentenceTransformerEmbedder
from src.core.interfaces.llm import LLMProvider
from src.core.interfaces.vector_store import VectorStore
from src.infrastructure.llms.factory import create_llm
from src.infrastructure.vector_stores.factory import create_vector_store
from src.infrastructure.text_splitters.recursive_splitter import RecursiveTextSplitter
from src.services.ingestion_service import IngestionService
from src.services.query_service import QueryService
from src.usecases.factory import UseCaseContext, UseCaseFactory


@lru_cache
def get_loader_factory() -> DocumentLoaderFactory:
    settings = get_settings()
    return create_default_loader_factory(
        ocr_enabled=settings.ocr_enabled,
        tesseract_lang=settings.tesseract_lang,
    )


@lru_cache
def get_embedder() -> SentenceTransformerEmbedder:
    settings = get_settings()
    return SentenceTransformerEmbedder(
        model_name=settings.embedding_model_name,
        device=settings.embedding_device,
    )


@lru_cache
def get_vector_store() -> VectorStore:
    return create_vector_store(get_settings())


@lru_cache
def get_text_splitter() -> RecursiveTextSplitter:
    settings = get_settings()
    return RecursiveTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )


@lru_cache
def get_llm() -> LLMProvider:
    return create_llm(get_settings())


@lru_cache
def get_pipeline() -> RAGPipeline:
    settings = get_settings()
    return RAGPipeline(
        text_splitter=get_text_splitter(),
        embedder=get_embedder(),
        vector_store=get_vector_store(),
        llm=get_llm(),
        top_k=settings.retrieval_top_k,
    )


def get_ingestion_service() -> IngestionService:
    settings = get_settings()
    return IngestionService(
        pipeline=get_pipeline(),
        loader_factory=get_loader_factory(),
        upload_dir=settings.upload_directory,
    )


def get_query_service() -> QueryService:
    return QueryService(pipeline=get_pipeline())


@lru_cache
def get_use_case_factory() -> UseCaseFactory:
    settings = get_settings()
    ctx = UseCaseContext(
        pipeline=get_pipeline(),
        loader_factory=get_loader_factory(),
        embedder=get_embedder(),
        llm=get_llm(),
        text_splitter=get_text_splitter(),
        settings=settings,
        upload_dir=Path(settings.upload_directory),
    )
    return UseCaseFactory(ctx)
