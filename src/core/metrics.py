"""Thread-safe in-memory metrics registry for use-case performance tracking.

Every use case execution records:
- ``use_case``      — class name of the use case
- ``status``        — "success" | "error"
- ``duration_ms``   — wall-clock time in milliseconds
- ``timestamp``     — ISO-8601 UTC start time
- ``error``         — exception message if status == "error"

Usage
-----
The registry is a module-level singleton accessed via :func:`get_registry`.
:class:`~src.usecases.base.UseCase` records automatically when
``metrics_enabled=True``; you never call the registry directly.

To expose via API::

    from src.core.metrics import get_registry
    registry = get_registry()
    stats = registry.summary()   # aggregated per use-case
    events = registry.events()   # raw event list
    registry.reset()             # clear all data
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Sequence


@dataclass
class MetricEvent:
    use_case: str
    status: str           # "success" | "error"
    duration_ms: float
    timestamp: str        # ISO-8601 UTC
    error: str | None = None


@dataclass
class UseCaseSummary:
    use_case: str
    call_count: int
    error_count: int
    avg_duration_ms: float
    min_duration_ms: float
    max_duration_ms: float
    last_called: str


class MetricsRegistry:
    """Thread-safe in-memory store for metric events."""

    def __init__(self, max_events: int = 10_000) -> None:
        self._lock = threading.Lock()
        self._events: list[MetricEvent] = []
        self._max_events = max_events

    def record(self, event: MetricEvent) -> None:
        with self._lock:
            self._events.append(event)
            if len(self._events) > self._max_events:
                # Drop oldest entries when cap is exceeded
                self._events = self._events[-self._max_events:]

    def events(self) -> list[MetricEvent]:
        with self._lock:
            return list(self._events)

    def summary(self) -> list[UseCaseSummary]:
        with self._lock:
            groups: dict[str, list[MetricEvent]] = {}
            for e in self._events:
                groups.setdefault(e.use_case, []).append(e)

        result: list[UseCaseSummary] = []
        for uc_name, evts in groups.items():
            durations = [e.duration_ms for e in evts]
            errors = [e for e in evts if e.status == "error"]
            last = max(evts, key=lambda e: e.timestamp)
            result.append(UseCaseSummary(
                use_case=uc_name,
                call_count=len(evts),
                error_count=len(errors),
                avg_duration_ms=round(sum(durations) / len(durations), 3),
                min_duration_ms=round(min(durations), 3),
                max_duration_ms=round(max(durations), 3),
                last_called=last.timestamp,
            ))
        return sorted(result, key=lambda s: s.use_case)

    def reset(self) -> None:
        with self._lock:
            self._events.clear()


# Module-level singleton — lazy, reads max_events from settings on first access
_registry: MetricsRegistry | None = None


def get_registry() -> MetricsRegistry:
    """Return the global :class:`MetricsRegistry` singleton."""
    global _registry
    if _registry is None:
        try:
            from src.config.settings import get_settings
            _registry = MetricsRegistry(max_events=get_settings().metrics_max_events)
        except Exception:
            _registry = MetricsRegistry()
    return _registry
