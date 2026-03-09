"""Unit tests for the RecursiveTextSplitter."""

from src.core.models import Document
from src.infrastructure.text_splitters.recursive_splitter import RecursiveTextSplitter


def test_split_short_document():
    splitter = RecursiveTextSplitter(chunk_size=500, chunk_overlap=0)
    docs = [Document(content="Short text", metadata={"source": "test"})]
    chunks = splitter.split(docs)
    assert len(chunks) == 1
    assert chunks[0].content == "Short text"
    assert chunks[0].metadata["source"] == "test"


def test_split_long_document():
    splitter = RecursiveTextSplitter(chunk_size=50, chunk_overlap=0)
    text = "This is a sentence. " * 20
    docs = [Document(content=text)]
    chunks = splitter.split(docs)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.chunk_id is not None


def test_split_empty_document():
    splitter = RecursiveTextSplitter(chunk_size=500, chunk_overlap=0)
    docs = [Document(content="")]
    chunks = splitter.split(docs)
    assert len(chunks) == 0
