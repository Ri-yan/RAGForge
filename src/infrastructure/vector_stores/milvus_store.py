"""Milvus vector store implementation.

Install extras before use:
    pip install pymilvus>=2.4.0

Connection is established lazily on first use.  All public methods mirror the
:class:`~src.core.interfaces.vector_store.VectorStore` interface exactly, so
swapping from Chroma to Milvus requires only a settings change.
"""
from __future__ import annotations

import uuid
import logging
from typing import Any

from src.core.interfaces.vector_store import VectorStore
from src.core.models import Chunk

logger = logging.getLogger(__name__)

_PYMILVUS_AVAILABLE = False
try:
    from pymilvus import (  # type: ignore
        connections,
        Collection,
        CollectionSchema,
        FieldSchema,
        DataType,
        utility,
    )
    _PYMILVUS_AVAILABLE = True
except ImportError:
    pass  # Raises a clear error at instantiation time instead

_DEFAULT_DIM = 384  # matches all-MiniLM-L6-v2; overridden via constructor


class MilvusVectorStore(VectorStore):
    """Vector store backed by Milvus / Zilliz Cloud.

    Parameters
    ----------
    host:
        Milvus server host.
    port:
        Milvus server port (default 19530).
    collection_name:
        Name of the Milvus collection to use.
    embedding_dim:
        Dimensionality of the embedding vectors — must match the embedder.
    alias:
        Connection alias (useful when multiple connections are open).
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 19530,
        collection_name: str = "rag_collection",
        embedding_dim: int = _DEFAULT_DIM,
        alias: str = "default",
    ) -> None:
        if not _PYMILVUS_AVAILABLE:
            raise ImportError(
                "pymilvus is not installed. Run: pip install pymilvus>=2.4.0"
            )
        self._collection_name = collection_name
        self._dim = embedding_dim
        self._alias = alias

        connections.connect(alias=alias, host=host, port=str(port))
        self._collection = self._get_or_create_collection()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_or_create_collection(self) -> "Collection":
        if utility.has_collection(self._collection_name, using=self._alias):
            col = Collection(self._collection_name, using=self._alias)
            col.load()
            return col

        fields = [
            FieldSchema(name="id",       dtype=DataType.VARCHAR, is_primary=True, max_length=64),
            FieldSchema(name="content",  dtype=DataType.VARCHAR, max_length=65_535),
            FieldSchema(name="doc_id",   dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="source",   dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="vector",   dtype=DataType.FLOAT_VECTOR, dim=self._dim),
        ]
        schema = CollectionSchema(fields, description="RAG chunk store")
        col = Collection(self._collection_name, schema=schema, using=self._alias)
        col.create_index(
            field_name="vector",
            index_params={"metric_type": "COSINE", "index_type": "HNSW", "params": {"M": 16, "efConstruction": 200}},
        )
        col.load()
        logger.info("Created Milvus collection '%s'", self._collection_name)
        return col

    # ── VectorStore interface ─────────────────────────────────────────────────

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        rows: list[dict[str, Any]] = []
        for chunk, emb in zip(chunks, embeddings):
            rows.append({
                "id":      chunk.chunk_id or str(uuid.uuid4()),
                "content": chunk.content[:65_535],
                "doc_id":  str(chunk.metadata.get("doc_id", "")),
                "source":  str(chunk.metadata.get("source", "")),
                "vector":  emb,
            })
        self._collection.insert(rows)
        self._collection.flush()

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[Chunk]:
        results = self._collection.search(
            data=[query_embedding],
            anns_field="vector",
            param={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=top_k,
            output_fields=["content", "doc_id", "source"],
        )
        return self._hits_to_chunks(results[0])

    def search_with_filter(
        self,
        query_embedding: list[float],
        metadata_filter: dict,
        top_k: int = 5,
    ) -> list[Chunk]:
        expr = self._build_expr(metadata_filter)
        results = self._collection.search(
            data=[query_embedding],
            anns_field="vector",
            param={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=top_k,
            expr=expr,
            output_fields=["content", "doc_id", "source"],
        )
        return self._hits_to_chunks(results[0])

    def list_documents(self) -> list[dict]:
        results = self._collection.query(
            expr="doc_id != \"\"",
            output_fields=["id", "doc_id", "source"],
        )
        groups: dict[str, dict] = {}
        for row in results:
            doc_id = row.get("doc_id", "<unknown>") or "<unknown>"
            if doc_id not in groups:
                groups[doc_id] = {
                    "doc_id": doc_id,
                    "chunk_count": 0,
                    "metadata": {"source": row.get("source", "")},
                }
            groups[doc_id]["chunk_count"] += 1
        return list(groups.values())

    def delete_by_doc_id(self, doc_id: str) -> int:
        # Fetch matching IDs then delete by primary key
        rows = self._collection.query(
            expr=f'doc_id == "{doc_id}"',
            output_fields=["id"],
        )
        ids = [r["id"] for r in rows]
        if ids:
            self._collection.delete(expr=f'id in {ids}')
            self._collection.flush()
        return len(ids)

    def delete_collection(self) -> None:
        from pymilvus import utility  # type: ignore
        utility.drop_collection(self._collection_name, using=self._alias)
        logger.info("Dropped Milvus collection '%s'", self._collection_name)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _hits_to_chunks(hits: Any) -> list[Chunk]:
        chunks = []
        for hit in hits:
            entity = hit.entity
            chunks.append(Chunk(
                content=entity.get("content", ""),
                metadata={
                    "doc_id": entity.get("doc_id", ""),
                    "source": entity.get("source", ""),
                },
                chunk_id=hit.id,
            ))
        return chunks

    @staticmethod
    def _build_expr(metadata_filter: dict) -> str:
        """Convert a simple ``{key: value}`` dict to a Milvus boolean expression.

        Supports only flat equality for now.  The value must be a string.
        """
        parts = [f'{k} == "{v}"' for k, v in metadata_filter.items()]
        return " && ".join(parts)
