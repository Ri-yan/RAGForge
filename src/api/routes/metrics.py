"""API routes for performance metrics."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from src.api.dependencies import get_settings
from src.core.metrics import get_registry

router = APIRouter(prefix="/metrics", tags=["Metrics"])


# ── Response models ───────────────────────────────────────────────────────────

class MetricEventOut(BaseModel):
    use_case: str
    status: str
    duration_ms: float
    timestamp: str
    error: str | None = None


class UseCaseSummaryOut(BaseModel):
    use_case: str
    call_count: int
    error_count: int
    avg_duration_ms: float
    min_duration_ms: float
    max_duration_ms: float
    last_called: str


class MetricsSummaryResponse(BaseModel):
    metrics_enabled: bool
    total_events: int
    use_cases: list[UseCaseSummaryOut]


class MetricsEventsResponse(BaseModel):
    metrics_enabled: bool
    total_events: int
    events: list[MetricEventOut]


class MetricsResetResponse(BaseModel):
    message: str
    events_cleared: int


# ── Endpoints ─────────────────────────────────────────────────────────────────

def _require_enabled() -> None:
    settings = get_settings()
    if not settings.metrics_enabled:
        raise HTTPException(
            status_code=403,
            detail="Metrics collection is disabled. Set METRICS_ENABLED=true in .env to enable.",
        )


@router.get(
    "/summary",
    response_model=MetricsSummaryResponse,
    summary="Aggregated performance stats per use case",
    description=(
        "Returns call count, error count, and timing statistics (avg/min/max ms) "
        "for every use case that has been invoked since the server started or the "
        "last reset.  Returns 403 when ``metrics_enabled=false``."
    ),
)
def get_metrics_summary() -> MetricsSummaryResponse:
    _require_enabled()
    registry = get_registry()
    events = registry.events()
    summary = registry.summary()
    return MetricsSummaryResponse(
        metrics_enabled=True,
        total_events=len(events),
        use_cases=[UseCaseSummaryOut(**vars(s)) for s in summary],
    )


@router.get(
    "/events",
    response_model=MetricsEventsResponse,
    summary="Raw metric event log",
    description=(
        "Returns every individual use-case execution record in chronological order.  "
        "Capped at ``metrics_max_events`` entries (oldest are dropped).  "
        "Returns 403 when ``metrics_enabled=false``."
    ),
)
def get_metrics_events() -> MetricsEventsResponse:
    _require_enabled()
    registry = get_registry()
    events = registry.events()
    return MetricsEventsResponse(
        metrics_enabled=True,
        total_events=len(events),
        events=[MetricEventOut(**vars(e)) for e in events],
    )


@router.delete(
    "/reset",
    response_model=MetricsResetResponse,
    summary="Clear all collected metrics",
    description="Deletes all stored events.  Useful between test runs.  Returns 403 when ``metrics_enabled=false``.",
)
def reset_metrics() -> MetricsResetResponse:
    _require_enabled()
    registry = get_registry()
    count = len(registry.events())
    registry.reset()
    return MetricsResetResponse(message="Metrics cleared.", events_cleared=count)
