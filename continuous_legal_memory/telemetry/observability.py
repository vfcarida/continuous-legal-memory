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


def export_prometheus_metrics(
    orchestrator: Any,
    telemetry_logger: TelemetryLogger | None = None,
    service_name: str = "continuous-legal-memory",
) -> str:
    """
    Format operational and cognitive memory statistics into standard Prometheus OpenMetrics text exposition.

    Args:
        orchestrator: Configured LegalMemoryOrchestrator instance.
        telemetry_logger: Optional TelemetryLogger with recorded historical traces.
        service_name: Identifier name for the service.

    Returns:
        Formatted Prometheus metrics string adhering to OpenMetrics text format.
    """
    lines: list[str] = [
        f"# HELP clm_service_info Metadata info about the {service_name} service.",
        "# TYPE clm_service_info gauge",
        f'clm_service_info{{service="{service_name}",engine_mode="{getattr(orchestrator, "engine_mode", "structured")}"}} 1',
    ]

    # Cognitive Tier Gauges
    num_episodic = len(orchestrator.episodic_memory.get_records()) if hasattr(orchestrator, "episodic_memory") else 0
    lines.extend([
        "# HELP clm_episodic_records_total Total number of stored immutable episodic records.",
        "# TYPE clm_episodic_records_total gauge",
        f"clm_episodic_records_total {num_episodic}",
    ])

    num_wm = len(orchestrator.working_memory.get_active_context()) if hasattr(orchestrator, "working_memory") else 0
    lines.extend([
        "# HELP clm_working_memory_active_items Active records in working memory sliding window.",
        "# TYPE clm_working_memory_active_items gauge",
        f"clm_working_memory_active_items {num_wm}",
    ])

    num_nodes = len(orchestrator.semantic_graph.nodes) if hasattr(orchestrator, "semantic_graph") else 0
    lines.extend([
        "# HELP clm_semantic_graph_nodes_total Total entity nodes in the semantic knowledge graph.",
        "# TYPE clm_semantic_graph_nodes_total gauge",
        f"clm_semantic_graph_nodes_total {num_nodes}",
    ])

    num_edges = len(orchestrator.semantic_graph.edges) if hasattr(orchestrator, "semantic_graph") else 0
    lines.extend([
        "# HELP clm_semantic_graph_edges_total Total dependency and precedence edges in semantic graph.",
        "# TYPE clm_semantic_graph_edges_total gauge",
        f"clm_semantic_graph_edges_total {num_edges}",
    ])

    chain_len = len(orchestrator.episodic_memory._hash_chain) if hasattr(orchestrator, "episodic_memory") and hasattr(orchestrator.episodic_memory, "_hash_chain") else 0
    lines.extend([
        "# HELP clm_hash_chain_length Current height of the cryptographic hash chain.",
        "# TYPE clm_hash_chain_length gauge",
        f"clm_hash_chain_length {chain_len}",
    ])

    # Telemetry execution metrics if logger present
    if telemetry_logger and telemetry_logger.metrics_history:
        op_counts: dict[str, int] = {}
        op_durations: dict[str, float] = {}
        for m in telemetry_logger.metrics_history:
            op_counts[m.operation] = op_counts.get(m.operation, 0) + 1
            op_durations[m.operation] = op_durations.get(m.operation, 0.0) + (m.latency_ms / 1000.0)

        lines.extend([
            "# HELP clm_operations_total Total count of executed memory pipeline operations.",
            "# TYPE clm_operations_total counter",
        ])
        for op, cnt in sorted(op_counts.items()):
            lines.append(f'clm_operations_total{{operation="{op}"}} {cnt}')

        lines.extend([
            "# HELP clm_operation_duration_seconds Total execution time in seconds per operation.",
            "# TYPE clm_operation_duration_seconds counter",
        ])
        for op, dur in sorted(op_durations.items()):
            lines.append(f'clm_operation_duration_seconds{{operation="{op}"}} {dur:.6f}')

    lines.append("")  # Trailing newline for Prometheus parser
    return "\n".join(lines)
