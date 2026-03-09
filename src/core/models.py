from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Document:
    """Represents a single document with content and metadata."""

    content: str
    metadata: dict = field(default_factory=dict)


@dataclass
class Chunk:
    """Represents a chunk of a document after splitting."""

    content: str
    metadata: dict = field(default_factory=dict)
    chunk_id: str | None = None
