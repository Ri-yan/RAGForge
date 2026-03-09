import uuid

from src.core.interfaces.text_splitter import TextSplitter
from src.core.models import Chunk, Document


class RecursiveTextSplitter(TextSplitter):
    """Splits text recursively using a hierarchy of separators."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: list[str] | None = None,
    ) -> None:
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def split(self, documents: list[Document]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for doc in documents:
            text_chunks = self._split_text(doc.content)
            for text in text_chunks:
                chunks.append(
                    Chunk(
                        content=text,
                        metadata={**doc.metadata},
                        chunk_id=str(uuid.uuid4()),
                    )
                )
        return chunks

    def _split_text(self, text: str) -> list[str]:
        return self._recursive_split(text, self._separators)

    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        if len(text) <= self._chunk_size:
            return [text] if text.strip() else []

        separator = separators[0]
        remaining_separators = separators[1:]

        if separator == "":
            # Character-level split as last resort
            return self._hard_split(text)

        parts = text.split(separator)
        merged: list[str] = []
        current = ""

        for part in parts:
            candidate = f"{current}{separator}{part}" if current else part
            if len(candidate) <= self._chunk_size:
                current = candidate
            else:
                if current:
                    merged.append(current)
                if len(part) > self._chunk_size and remaining_separators:
                    merged.extend(self._recursive_split(part, remaining_separators))
                else:
                    current = part
                    continue
                current = ""

        if current.strip():
            merged.append(current)

        return self._apply_overlap(merged)

    def _hard_split(self, text: str) -> list[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = start + self._chunk_size
            chunks.append(text[start:end])
            start = end - self._chunk_overlap
        return chunks

    def _apply_overlap(self, chunks: list[str]) -> list[str]:
        if self._chunk_overlap <= 0 or len(chunks) <= 1:
            return chunks

        result: list[str] = []
        for i, chunk in enumerate(chunks):
            if i == 0:
                result.append(chunk)
            else:
                overlap_text = chunks[i - 1][-self._chunk_overlap :]
                result.append(overlap_text + chunk)
        return result
