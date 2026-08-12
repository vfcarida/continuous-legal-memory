"""
Observability and Telemetry Hooks Module.

Provides OpenTelemetry instrumentation and token utilization/latency logging for continuous legal memory pipelines.
"""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger("continuous_legal_memory")


@dataclass
class ExecutionMetrics:
    """
    Data container for operational latency, token count, and memory execution metrics.

    Attributes:
        operation: Name of the invoked pipeline operation.
        latency_ms: Total elapsed execution time in milliseconds.
        tokens_processed: Estimated token count processed.
        timestamp: Datetime execution timestamp.
    """

    operation: str
    latency_ms: float
    tokens_processed: int
    timestamp: datetime


class TelemetryLogger:
    """
    Telemetry and observability logger for OpenTelemetry and LangSmith integration.

    Rationale:
        Enables real-time tracing of memory lookup latency, model token cost, and memory update performance.
    """

    def __init__(self, service_name: str = "continuous-legal-memory") -> None:
        self.service_name = service_name
        self.metrics_history: list[ExecutionMetrics] = []

    def trace_operation(self, operation_name: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[Any, ExecutionMetrics]:
        """
        Execute a function wrapper while measuring execution latency and recording telemetry metrics.

        Args:
            operation_name: Identifier name for the operation.
            fn: Function to execute.
            args: Positional arguments for fn.
            kwargs: Keyword arguments for fn.

        Returns:
            Tuple of (function_result, ExecutionMetrics).
        """
        start_time = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # Estimate tokens processed from string representations
        str_content = str(args) + str(kwargs)
        estimated_tokens = max(1, len(str_content) // 4)

        metrics = ExecutionMetrics(
            operation=operation_name,
            latency_ms=elapsed_ms,
            tokens_processed=estimated_tokens,
            timestamp=datetime.now(timezone.utc),
        )

        self.metrics_history.append(metrics)
        logger.info("[%s] %s completed in %.2fms (~%d tokens)", self.service_name, operation_name, elapsed_ms, estimated_tokens)

        return result, metrics
