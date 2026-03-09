from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Generic, TypeVar
import time

TInput = TypeVar("TInput")
TOutput = TypeVar("TOutput")


class UseCase(ABC, Generic[TInput, TOutput]):
    """Base class for all use cases.

    Each concrete use case implements a single ``execute()`` method that
    receives a typed input dataclass and returns a typed output dataclass.

    When ``metrics_enabled=True``, every call to :meth:`execute` is
    automatically timed and recorded in the global :class:`~src.core.metrics.MetricsRegistry`.
    No changes are needed in subclasses — new use cases inherit this behaviour
    for free.
    """

    metrics_enabled: bool = False  # overridden by UseCaseContext

    @abstractmethod
    def _execute(self, input_data: TInput) -> TOutput: ...

    def execute(self, input_data: TInput) -> TOutput:
        if not self.metrics_enabled:
            return self._execute(input_data)

        from src.core.metrics import MetricEvent, get_registry

        use_case_name = type(self).__name__
        start = time.perf_counter()
        timestamp = datetime.now(timezone.utc).isoformat()
        status = "success"
        error_msg: str | None = None
        try:
            result = self._execute(input_data)
            return result
        except Exception as exc:
            status = "error"
            error_msg = str(exc)
            raise
        finally:
            duration_ms = round((time.perf_counter() - start) * 1000, 3)
            get_registry().record(MetricEvent(
                use_case=use_case_name,
                status=status,
                duration_ms=duration_ms,
                timestamp=timestamp,
                error=error_msg,
            ))
