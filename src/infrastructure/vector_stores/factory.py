"""Vector store factory — resolves the correct provider from settings.vector_store_type."""
from __future__ import annotations

from typing import TYPE_CHECKING

from src.core.interfaces.vector_store import VectorStore

if TYPE_CHECKING:
    from src.config.settings import Settings

_VALID_TYPES = ("chroma", "milvus")


def create_vector_store(settings: "Settings", collection_name: str | None = None) -> VectorStore:
    """Instantiate the vector store declared in settings.vector_store_type.

    Parameters
    ----------
    settings:
        Application settings instance.
    collection_name:
        Override the default collection / database name from settings.
        Used by :meth:`UseCaseContext.pipeline_for` to create collection-scoped stores.

    Supported values for ``vector_store_type``
    ------------------------------------------
    ``"chroma"`` (default)
        ChromaDB persisted locally.  Requires no extra install.
    ``"milvus"``
        Milvus / Zilliz Cloud.  Requires ``pip install pymilvus>=2.4.0``.
    """
    store_type = settings.vector_store_type.lower()

    if store_type == "chroma":
        from src.infrastructure.vector_stores.chroma_store import ChromaVectorStore

        return ChromaVectorStore(
            persist_directory=settings.chroma_persist_directory,
            collection_name=collection_name or settings.chroma_collection_name,
        )

    if store_type == "milvus":
        from src.infrastructure.vector_stores.milvus_store import MilvusVectorStore

        return MilvusVectorStore(
            host=settings.milvus_host,
            port=settings.milvus_port,
            collection_name=collection_name or settings.milvus_collection_name,
            embedding_dim=settings.milvus_embedding_dim,
        )

    raise ValueError(
        f"Unknown vector_store_type {settings.vector_store_type!r}. "
        f"Valid options: {', '.join(repr(t) for t in _VALID_TYPES)}."
    )
