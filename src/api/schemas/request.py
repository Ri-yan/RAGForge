from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="The question to ask")


# ── Use-case request models ────────────────────────────────────────────────────

class QueryByDocIdRequest(BaseModel):
    """UC3: query a specific document by its doc_id."""
    question: str = Field(..., min_length=1, max_length=2000)
    doc_id: str = Field(..., description="doc_id returned at ingest time")
    collection_name: str | None = Field(None, description="Collection the document was ingested into. Must match the collection_name used at upload time.")
    top_k: int | None = Field(None, ge=1, le=50)


class QueryCollectionRequest(BaseModel):
    """UC4: query a named collection / knowledge base."""
    question: str = Field(..., min_length=1, max_length=2000)
    collection_name: str = Field(..., min_length=1)
    top_k: int | None = Field(None, ge=1, le=50)


class QueryWithFilterRequest(BaseModel):
    """UC5: query with arbitrary metadata filter."""
    question: str = Field(..., min_length=1, max_length=2000)
    metadata_filter: dict = Field(..., description="ChromaDB-compatible where clause, e.g. {\"author\": \"alice\"}")
    top_k: int | None = Field(None, ge=1, le=50)
