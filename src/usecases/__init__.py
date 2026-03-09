"""Use-case layer — one class per user-facing scenario.

Available use cases
-------------------
IngestAndQueryUseCase      UC1 – upload a document and immediately query it
IngestDocumentUseCase      UC2 – upload a document, receive a stable doc_id
QueryByDocIdUseCase        UC3 – ask a question scoped to a specific doc_id
QueryCollectionUseCase     UC4 – ask a question against a named collection
QueryWithFilterUseCase     UC5 – ask a question with an arbitrary metadata filter
ListDocumentsUseCase       UC6 – list documents stored in a collection
DeleteDocumentUseCase      UC7 – delete a document from vector store and disk
"""

from src.usecases.ingest_and_query import IngestAndQueryInput, IngestAndQueryOutput, IngestAndQueryUseCase
from src.usecases.ingest_document import IngestDocumentInput, IngestDocumentOutput, IngestDocumentUseCase
from src.usecases.query_by_doc_id import QueryByDocIdInput, QueryByDocIdOutput, QueryByDocIdUseCase
from src.usecases.query_collection import QueryCollectionInput, QueryCollectionOutput, QueryCollectionUseCase
from src.usecases.query_with_filter import QueryWithFilterInput, QueryWithFilterOutput, QueryWithFilterUseCase
from src.usecases.list_documents import ListDocumentsInput, ListDocumentsOutput, ListDocumentsUseCase
from src.usecases.delete_document import DeleteDocumentInput, DeleteDocumentOutput, DeleteDocumentUseCase
from src.usecases.factory import UseCaseContext, UseCaseFactory

__all__ = [
    "IngestAndQueryInput", "IngestAndQueryOutput", "IngestAndQueryUseCase",
    "IngestDocumentInput", "IngestDocumentOutput", "IngestDocumentUseCase",
    "QueryByDocIdInput", "QueryByDocIdOutput", "QueryByDocIdUseCase",
    "QueryCollectionInput", "QueryCollectionOutput", "QueryCollectionUseCase",
    "QueryWithFilterInput", "QueryWithFilterOutput", "QueryWithFilterUseCase",
    "ListDocumentsInput", "ListDocumentsOutput", "ListDocumentsUseCase",
    "DeleteDocumentInput", "DeleteDocumentOutput", "DeleteDocumentUseCase",
    "UseCaseContext", "UseCaseFactory",
]
