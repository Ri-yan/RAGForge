from fastapi import APIRouter, Depends, HTTPException, UploadFile

from src.api.dependencies import get_ingestion_service
from src.api.schemas.response import IngestFileResponse, IngestResponse
from src.services.ingestion_service import IngestionService

router = APIRouter(prefix="/ingest", tags=["Ingestion"])

ALLOWED_EXTENSIONS = {".txt", ".md", ".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB


@router.post("/upload", response_model=IngestResponse)
async def upload_and_ingest(
    files: list[UploadFile],
    service: IngestionService = Depends(get_ingestion_service),
) -> IngestResponse:
    """Upload one or more files and ingest them into the RAG pipeline."""
    results: list[IngestFileResponse] = []

    for file in files:
        if not file.filename:
            raise HTTPException(status_code=400, detail="File must have a filename")

        ext = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type '{ext}'. Allowed: {ALLOWED_EXTENSIONS}",
            )

        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="File exceeds 50 MB limit")

        result = service.save_and_ingest(file.filename, content)
        results.append(IngestFileResponse(**result))

    return IngestResponse(
        message=f"Successfully ingested {len(results)} file(s)",
        results=results,
    )
