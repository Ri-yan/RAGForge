from pydantic import BaseModel


class SourceInfo(BaseModel):
    content: str
    metadata: dict


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceInfo]


class IngestFileResponse(BaseModel):
    file: str
    documents_loaded: int
    chunks_created: int
    doc_id: str


class IngestResponse(BaseModel):
    message: str
    results: list[IngestFileResponse]


class HealthResponse(BaseModel):
    status: str
    version: str


# ── Use-case response models ───────────────────────────────────────────────────

class IngestDocumentResponse(BaseModel):
    """UC2: upload a document and get back its stable doc_id."""
    doc_id: str
    file: str
    documents_loaded: int
    chunks_created: int
    collection_name: str


class IngestAndQueryResponse(BaseModel):
    """UC1: upload + immediate query against only the uploaded document."""
    doc_id: str
    question: str
    answer: str
    sources: list[SourceInfo]
    documents_loaded: int
    chunks_created: int


class QueryByDocIdResponse(BaseModel):
    """UC3: query scoped to a specific document by doc_id."""
    doc_id: str
    question: str
    answer: str
    sources: list[SourceInfo]


class QueryCollectionResponse(BaseModel):
    """UC4: query a named collection / knowledge base."""
    collection_name: str
    question: str
    answer: str
    sources: list[SourceInfo]


class QueryWithFilterResponse(BaseModel):
    """UC5: query with an arbitrary metadata filter for targeted retrieval."""
    metadata_filter: dict
    question: str
    answer: str
    sources: list[SourceInfo]


class DocumentSummary(BaseModel):
    """One document entry returned by UC6."""
    doc_id: str
    chunk_count: int
    metadata: dict


class ListDocumentsResponse(BaseModel):
    """UC6: list of documents in a collection."""
    collection_name: str
    total_documents: int
    documents: list[DocumentSummary]


class DeleteDocumentResponse(BaseModel):
    """UC7: result of deleting a document."""
    doc_id: str
    collection_name: str
    chunks_deleted: int
    file_deleted: bool
    file_path: str | None
