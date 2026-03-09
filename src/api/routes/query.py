from fastapi import APIRouter, Depends

from src.api.dependencies import get_query_service
from src.api.schemas.request import QueryRequest
from src.api.schemas.response import QueryResponse
from src.services.query_service import QueryService

router = APIRouter(prefix="/query", tags=["Query"])


@router.post("/", response_model=QueryResponse)
async def query(
    body: QueryRequest,
    service: QueryService = Depends(get_query_service),
) -> QueryResponse:
    """Ask a question against the ingested documents."""
    result = service.ask(body.question)
    return QueryResponse(**result)
