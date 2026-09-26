"""
Unit tests for Continuous Legal Memory Telemetry and Prometheus OpenMetrics exporter.
"""

from __future__ import annotations

import time

from continuous_legal_memory.adapters.mock_encoder import SemanticMockEncoder
from continuous_legal_memory.orchestrator import LegalMemoryOrchestrator
from continuous_legal_memory.telemetry.observability import (
    TelemetryLogger,
    export_prometheus_metrics,
)


def test_telemetry_logger_trace_operation() -> None:
    logger = TelemetryLogger(service_name="test-clm")

    def sample_op(x: int, y: int) -> int:
        time.sleep(0.005)
        return x + y

    res, metrics = logger.trace_operation("sample_add", sample_op, 10, 20)
    assert res == 30
    assert metrics.operation == "sample_add"
    assert metrics.latency_ms >= 1.0
    assert len(logger.metrics_history) == 1
    assert logger.metrics_history[0].operation == "sample_add"


def test_export_prometheus_metrics_formatting() -> None:
    encoder = SemanticMockEncoder()
    orch = LegalMemoryOrchestrator(
        encoder=encoder,
        engine_mode="structured",
        enable_telemetry=True,
    )

    orch.update_memory("Statute 101: Tax exemptions for R&D.", [1.0, 0.0])
    orch.predict("Inquiry on tax exemption")

    metrics_text = export_prometheus_metrics(orch, orch.telemetry, service_name="clm-test")

    assert "# HELP clm_service_info" in metrics_text
    assert 'clm_service_info{service="clm-test",engine_mode="structured"} 1' in metrics_text
    assert "# HELP clm_episodic_records_total" in metrics_text
    assert "clm_episodic_records_total 1" in metrics_text
    assert "# HELP clm_working_memory_active_items" in metrics_text
    assert "clm_working_memory_active_items 1" in metrics_text
    assert "# HELP clm_semantic_graph_nodes_total" in metrics_text
    assert "clm_semantic_graph_nodes_total 1" in metrics_text
    assert "# HELP clm_hash_chain_length" in metrics_text
    assert "clm_hash_chain_length 1" in metrics_text

    # Verify telemetry operation metrics were captured
    assert "# HELP clm_operations_total" in metrics_text
    assert 'clm_operations_total{operation="update_memory"} 1' in metrics_text
    assert 'clm_operations_total{operation="predict"} 1' in metrics_text
    assert "# HELP clm_operation_duration_seconds" in metrics_text
