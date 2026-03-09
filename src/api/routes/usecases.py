"""API routes for the seven use-case scenarios."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, UploadFile

from src.api.dependencies import get_use_case_factory
from src.api.schemas.request import (
    QueryByDocIdRequest,
    QueryCollectionRequest,
    QueryWithFilterRequest,
)
from src.api.schemas.response import (
    IngestAndQueryResponse,
    IngestDocumentResponse,
    QueryByDocIdResponse,
    QueryCollectionResponse,
    QueryWithFilterResponse,
    ListDocumentsResponse,
    DeleteDocumentResponse,
    DocumentSummary,
    SourceInfo,
)
from src.usecases.factory import UseCaseFactory
from src.usecases.ingest_and_query import IngestAndQueryInput
from src.usecases.ingest_document import IngestDocumentInput
from src.usecases.query_by_doc_id import QueryByDocIdInput
from src.usecases.query_collection import QueryCollectionInput
from src.usecases.query_with_filter import QueryWithFilterInput
from src.usecases.list_documents import ListDocumentsInput
from src.usecases.delete_document import DeleteDocumentInput

router = APIRouter(prefix="/usecases", tags=["Use Cases"])

ALLOWED_EXTENSIONS = {".txt", ".md", ".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB


def _validate_upload(file: UploadFile, content: bytes) -> None:
    if not file.filename:
        raise HTTPException(status_code=400, detail="File must have a filename")
    ext = ("." + file.filename.rsplit(".", 1)[-1].lower()) if "." in file.filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File exceeds 50 MB limit")


# ── UC1: upload + immediate query ─────────────────────────────────────────────

@router.post(
    "/ingest-and-query",
    response_model=IngestAndQueryResponse,
    summary="UC1 – Upload a document and immediately query it",
    description=(
        "Ingests the uploaded file then runs the question **exclusively** "
        "against that document. The answer is grounded only in the uploaded content."
    ),
)
async def ingest_and_query(
    file: UploadFile,
    question: str,
    factory: UseCaseFactory = Depends(get_use_case_factory),
) -> IngestAndQueryResponse:
    content = await file.read()
    _validate_upload(file, content)
    output = factory.ingest_and_query.execute(
        IngestAndQueryInput(filename=file.filename, content=content, question=question)
    )
    return IngestAndQueryResponse(
        doc_id=output.doc_id,
        question=output.question,
        answer=output.answer,
        sources=[SourceInfo(**vars(s)) for s in output.sources],
        documents_loaded=output.documents_loaded,
        chunks_created=output.chunks_created,
    )


# ── UC2: upload → doc_id ──────────────────────────────────────────────────────

@router.post(
    "/upload",
    response_model=IngestDocumentResponse,
    summary="UC2 – Upload a document and get back a stable doc_id",
    description=(
        "Saves and ingests the file, returning a ``doc_id`` you can later pass "
        "to the *query/by-doc* endpoint (UC3).  Optionally ingest into a named "
        "collection (knowledge base) using the ``collection_name`` query parameter."
    ),
)
async def upload_document(
    file: UploadFile,
    collection_name: str | None = None,
    factory: UseCaseFactory = Depends(get_use_case_factory),
) -> IngestDocumentResponse:
    content = await file.read()
    _validate_upload(file, content)
    output = factory.ingest_document.execute(
        IngestDocumentInput(
            filename=file.filename,
            content=content,
            collection_name=collection_name,
        )
    )
    return IngestDocumentResponse(**vars(output))


# ── UC3: query by doc_id ──────────────────────────────────────────────────────

@router.post(
    "/query/by-doc",
    response_model=QueryByDocIdResponse,
    summary="UC3 – Query scoped to a specific document",
    description="Run retrieval filtered to a single document identified by its ``doc_id``.",
)
async def query_by_doc_id(
    body: QueryByDocIdRequest,
    factory: UseCaseFactory = Depends(get_use_case_factory),
) -> QueryByDocIdResponse:
    output = factory.query_by_doc_id.execute(
        QueryByDocIdInput(
            question=body.question,
            doc_id=body.doc_id,
            collection_name=body.collection_name,
            top_k=body.top_k,
        )
    )
    return QueryByDocIdResponse(
        doc_id=output.doc_id,
        question=output.question,
        answer=output.answer,
        sources=[SourceInfo(**vars(s)) for s in output.sources],
    )


# ── UC4: query a named collection ────────────────────────────────────────────

@router.post(
    "/query/collection",
    response_model=QueryCollectionResponse,
    summary="UC4 – Query a named collection / knowledge base",
    description=(
        "Searches within a dedicated ChromaDB collection.  Documents must have "
        "been ingested into the collection via *UC2* using the same "
        "``collection_name``."
    ),
)
async def query_collection(
    body: QueryCollectionRequest,
    factory: UseCaseFactory = Depends(get_use_case_factory),
) -> QueryCollectionResponse:
    output = factory.query_collection.execute(
        QueryCollectionInput(
            question=body.question,
            collection_name=body.collection_name,
            top_k=body.top_k,
        )
    )
    return QueryCollectionResponse(
        collection_name=output.collection_name,
        question=output.question,
        answer=output.answer,
        sources=[SourceInfo(**vars(s)) for s in output.sources],
    )


# ── UC5: query with metadata filter ──────────────────────────────────────────

@router.post(
    "/query/filtered",
    response_model=QueryWithFilterResponse,
    summary="UC5 – Query with metadata-driven document selection",
    description=(
        "Retrieves only chunks whose stored metadata matches the supplied filter, "
        "then generates an answer from that targeted context.  "
        "Simple equality: ``{\\\"author\\\": \\\"alice\\\"}``.  "
        "Compound: ``{\\\"$and\\\": [{\\\"year\\\": \\\"2024\\\"}, {\\\"dept\\\": \\\"eng\\\"}]}``."
    ),
)
async def query_with_filter(
    body: QueryWithFilterRequest,
    factory: UseCaseFactory = Depends(get_use_case_factory),
) -> QueryWithFilterResponse:
    output = factory.query_with_filter.execute(
        QueryWithFilterInput(
            question=body.question,
            metadata_filter=body.metadata_filter,
            top_k=body.top_k,
        )
    )
    return QueryWithFilterResponse(
        metadata_filter=output.metadata_filter,
        question=output.question,
        answer=output.answer,
        sources=[SourceInfo(**vars(s)) for s in output.sources],
    )


# ── UC6: list documents ───────────────────────────────────────────────────────

@router.get(
    "/documents",
    response_model=ListDocumentsResponse,
    summary="UC6 – List all documents in a collection",
    description=(
        "Returns one entry per unique ``doc_id`` with its chunk count and "
        "metadata fields (e.g. original filename, author, year).  "
        "Pass ``collection_name`` to inspect a specific knowledge base."
    ),
)
async def list_documents(
    collection_name: str | None = None,
    factory: UseCaseFactory = Depends(get_use_case_factory),
) -> ListDocumentsResponse:
    output = factory.list_documents.execute(
        ListDocumentsInput(collection_name=collection_name)
    )
    return ListDocumentsResponse(
        collection_name=output.collection_name,
        total_documents=output.total_documents,
        documents=[DocumentSummary(**vars(d)) for d in output.documents],
    )


# ── UC7: delete document ──────────────────────────────────────────────────────

@router.delete(
    "/documents/{doc_id}",
    response_model=DeleteDocumentResponse,
    summary="UC7 – Delete a document by doc_id",
    description=(
        "Removes all chunks for the given ``doc_id`` from the vector store.  "
        "When ``delete_file=true`` (default), the original uploaded file is also "
        "removed from the server's upload directory."
    ),
)
async def delete_document(
    doc_id: str,
    collection_name: str | None = None,
    delete_file: bool = True,
    factory: UseCaseFactory = Depends(get_use_case_factory),
) -> DeleteDocumentResponse:
    output = factory.delete_document.execute(
        DeleteDocumentInput(
            doc_id=doc_id,
            collection_name=collection_name,
            delete_file=delete_file,
        )
    )
    if output.chunks_deleted == 0:
        raise HTTPException(
            status_code=404,
            detail=f"No chunks found for doc_id '{doc_id}'. It may not exist or was already deleted.",
        )
    return DeleteDocumentResponse(**vars(output))
