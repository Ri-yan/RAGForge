"""Unit tests for the RAG pipeline."""

from src.core.interfaces.embedder import Embedder
from src.core.interfaces.llm import LLMProvider
from src.core.interfaces.text_splitter import TextSplitter
from src.core.interfaces.vector_store import VectorStore
from src.core.models import Chunk, Document
from src.core.pipeline import RAGPipeline


class FakeTextSplitter(TextSplitter):
    def split(self, documents: list[Document]) -> list[Chunk]:
        return [
            Chunk(content=d.content, metadata=d.metadata, chunk_id=str(i))
            for i, d in enumerate(documents)
        ]


class FakeEmbedder(Embedder):
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]

    def embed_query(self, query: str) -> list[float]:
        return [0.1, 0.2, 0.3]


class FakeVectorStore(VectorStore):
    def __init__(self) -> None:
        self.stored: list[Chunk] = []

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        self.stored.extend(chunks)

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[Chunk]:
        return self.stored[:top_k]

    def search_with_filter(
        self,
        query_embedding: list[float],
        metadata_filter: dict,
        top_k: int = 5,
    ) -> list[Chunk]:
        return [
            c for c in self.stored
            if all(c.metadata.get(k) == v for k, v in metadata_filter.items())
        ][:top_k]

    def list_documents(self) -> list[dict]:
        groups: dict[str, dict] = {}
        for c in self.stored:
            doc_id = c.metadata.get("doc_id", "<unknown>")
            if doc_id not in groups:
                groups[doc_id] = {"doc_id": doc_id, "chunk_count": 0, "metadata": {}}
            groups[doc_id]["chunk_count"] += 1
        return list(groups.values())

    def delete_by_doc_id(self, doc_id: str) -> int:
        before = len(self.stored)
        self.stored = [c for c in self.stored if c.metadata.get("doc_id") != doc_id]
        return before - len(self.stored)

    def delete_collection(self) -> None:
        self.stored.clear()


class FakeLLM(LLMProvider):
    def generate(self, prompt: str, context: list[Chunk]) -> str:
        return f"Answer based on {len(context)} chunks"


def test_ingest():
    store = FakeVectorStore()
    pipeline = RAGPipeline(
        text_splitter=FakeTextSplitter(),
        embedder=FakeEmbedder(),
        vector_store=store,
        llm=FakeLLM(),
    )
    docs = [Document(content="Hello world", metadata={"source": "test.txt"})]
    result = pipeline.ingest(docs)
    assert result["chunk_count"] == 1
    assert "doc_id" in result
    assert len(store.stored) == 1
    # doc_id should be stamped into chunk metadata
    assert store.stored[0].metadata["doc_id"] == result["doc_id"]


def test_query_with_filter():
    store = FakeVectorStore()
    store.stored = [
        Chunk(content="Relevant info", metadata={"source": "doc.txt", "doc_id": "abc"}, chunk_id="1"),
        Chunk(content="Other info",    metadata={"source": "other.txt", "doc_id": "xyz"}, chunk_id="2"),
    ]
    pipeline = RAGPipeline(
        text_splitter=FakeTextSplitter(),
        embedder=FakeEmbedder(),
        vector_store=store,
        llm=FakeLLM(),
    )
    result = pipeline.query("What is this?", metadata_filter={"doc_id": "abc"})
    assert result["question"] == "What is this?"
    assert "1 chunks" in result["answer"]
    assert len(result["sources"]) == 1
    assert result["sources"][0]["metadata"]["doc_id"] == "abc"
