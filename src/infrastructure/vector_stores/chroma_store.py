import uuid

import chromadb

from src.core.interfaces.vector_store import VectorStore
from src.core.models import Chunk


class ChromaVectorStore(VectorStore):
    """Vector store backed by ChromaDB (open-source, local-first)."""

    def __init__(
        self,
        persist_directory: str = "./data/chroma_db",
        collection_name: str = "rag_collection",
    ) -> None:
        self._client = chromadb.PersistentClient(path=persist_directory)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        ids = [c.chunk_id or str(uuid.uuid4()) for c in chunks]
        documents = [c.content for c in chunks]
        metadatas = [c.metadata for c in chunks]
        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[Chunk]:
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas"],
        )
        chunks: list[Chunk] = []
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        ids = results.get("ids", [[]])[0]
        for doc, meta, chunk_id in zip(documents, metadatas, ids):
            chunks.append(Chunk(content=doc, metadata=meta or {}, chunk_id=chunk_id))
        return chunks

    def search_with_filter(
        self,
        query_embedding: list[float],
        metadata_filter: dict,
        top_k: int = 5,
    ) -> list[Chunk]:
        # ChromaDB raises if n_results > number of items matching the where filter,
        # so count first and cap accordingly.
        count_result = self._collection.get(where=metadata_filter, include=[])
        matching_ids = count_result.get("ids") or []
        if not matching_ids:
            return []
        effective_n = min(top_k, len(matching_ids))
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=effective_n,
            where=metadata_filter,
            include=["documents", "metadatas"],
        )
        chunks: list[Chunk] = []
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        ids = results.get("ids", [[]])[0]
        for doc, meta, chunk_id in zip(documents, metadatas, ids):
            chunks.append(Chunk(content=doc, metadata=meta or {}, chunk_id=chunk_id))
        return chunks

    def delete_collection(self) -> None:
        self._client.delete_collection(self._collection.name)

    def list_documents(self) -> list[dict]:
        """Return one summary entry per unique doc_id in the collection."""
        results = self._collection.get(include=["metadatas"])
        metadatas = results.get("metadatas") or []
        # Group chunks by doc_id
        groups: dict[str, dict] = {}
        for meta in metadatas:
            doc_id = meta.get("doc_id", "<unknown>") if meta else "<unknown>"
            if doc_id not in groups:
                groups[doc_id] = {"doc_id": doc_id, "chunk_count": 0, "metadata": {k: v for k, v in (meta or {}).items() if k != "doc_id"}}
            groups[doc_id]["chunk_count"] += 1
        return list(groups.values())

    def delete_by_doc_id(self, doc_id: str) -> int:
        """Delete all chunks for *doc_id*. Returns count of deleted chunks."""
        results = self._collection.get(
            where={"doc_id": doc_id},
            include=[],
        )
        ids = results.get("ids") or []
        if ids:
            self._collection.delete(ids=ids)
        return len(ids)
