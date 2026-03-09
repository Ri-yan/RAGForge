import logging

from fastapi import FastAPI

from src.api.exception_handlers import register_exception_handlers
from src.api.routes import ingest, query, usecases, metrics
from src.api.schemas.response import HealthResponse
from src.config.settings import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)


def create_app() -> FastAPI:
    """Application factory — creates and configures the FastAPI app."""
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "A generic, production-grade Retrieval-Augmented Generation pipeline. "
            "Upload documents (PDF, TXT, images), ingest them into a vector store, "
            "and query them using natural language with LLM-powered answers."
        ),
        debug=settings.debug,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        openapi_tags=[
            {"name": "Health", "description": "Application health checks"},
            {"name": "Ingestion", "description": "Upload and ingest documents (PDF, TXT, images with OCR)"},
            {"name": "Query", "description": "Ask questions against ingested documents"},
            {"name": "Use Cases", "description": "Higher-level scenarios: ingest+query, doc-scoped query, collection query, metadata-filtered query"},
            {"name": "Metrics", "description": "Use-case performance stats (enabled via METRICS_ENABLED=true)"},
        ],
    )

    # Register global exception handlers
    register_exception_handlers(app)

    # Register routers
    app.include_router(ingest.router, prefix="/api/v1")
    app.include_router(query.router, prefix="/api/v1")
    app.include_router(usecases.router, prefix="/api/v1")
    app.include_router(metrics.router, prefix="/api/v1")

    # Configure metrics registry max capacity from settings
    if settings.metrics_enabled:
        from src.core.metrics import get_registry
        get_registry()._max_events = settings.metrics_max_events

    @app.get("/health", response_model=HealthResponse, tags=["Health"])
    async def health_check() -> HealthResponse:
        return HealthResponse(status="healthy", version=settings.app_version)

    return app


app = create_app()
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host="127.0.0.1", port=8001, reload=True)