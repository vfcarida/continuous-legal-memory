"""
Unit tests for TemporalStructuredRetriever and orchestrator engine strategy (ARCH-01, ARCH-02).
"""

import time

import pytest
import torch

from continuous_legal_memory.domain.interfaces import BaseEncoderPort
from continuous_legal_memory.domain.models import MemoryRecord
from continuous_legal_memory.experimental.neural_head import (
    ContinuousMemoryResearchHead,
    ParametricNeuralHeadWarning,
)
from continuous_legal_memory.orchestrator import LegalMemoryOrchestrator
from continuous_legal_memory.retrieval.temporal_structured import TemporalStructuredRetriever


def test_temporal_structured_retriever_basic_and_precedence() -> None:
    """Verify that TemporalStructuredRetriever correctly evaluates similarity, precedence, and snippets."""
    retriever = TemporalStructuredRetriever(temperature=0.05)

    q_embed = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    keys = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],  # Match
        [0.0, 1.0, 0.0, 0.0],  # Mismatch
    ])
    rule_importance = torch.tensor([1.0, 1.0])
    records = [
        MemoryRecord(record_id="r1", text="Statute 1: Active mandate", key_vector=keys[0:1], value_vector=torch.tensor([[1.0, 0.0]]), authority_rank=8),
        MemoryRecord(record_id="r2", text="Statute 2: Subordinate policy", key_vector=keys[1:2], value_vector=torch.tensor([[0.0, 1.0]]), authority_rank=2),
    ]

    scores, weights, snippets = retriever.retrieve(
        query_embed=q_embed,
        keys=keys,
        rule_importance=rule_importance,
        records=records,
    )

    assert scores.shape == (1, 2)
    assert weights.shape == (1, 2)
    assert weights[0, 0] > weights[0, 1]
    assert snippets == ["Statute 1: Active mandate"]


def test_orchestrator_engine_mode_validation(offline_encoder: BaseEncoderPort) -> None:
    """Verify engine_mode validation in LegalMemoryOrchestrator."""
    # Valid modes
    orch_struct = LegalMemoryOrchestrator(encoder=offline_encoder, engine_mode="structured")
    assert orch_struct.engine_mode == "structured"

    orch_hybrid = LegalMemoryOrchestrator(encoder=offline_encoder, engine_mode="hybrid")
    assert orch_hybrid.engine_mode == "hybrid"

    # Invalid mode raises ValueError
    with pytest.raises(ValueError, match="Unknown engine_mode"):
        LegalMemoryOrchestrator(encoder=offline_encoder, engine_mode="invalid_engine")


def test_orchestrator_structured_mode_sub_10ms_ingestion(offline_encoder: BaseEncoderPort) -> None:
    """
    Verify that structured engine mode bypasses neural backpropagation,
    delivering sub-10ms rule ingestion latency.
    """
    orch = LegalMemoryOrchestrator(encoder=offline_encoder, value_dim=2, engine_mode="structured")

    start_time = time.perf_counter()
    orch.update_memory(
        "Commercial Statute 404: Invoice audit requirements mandatory.",
        [0.0, 1.0],
        authority_rank=7,
    )
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    # Must execute in under 100ms even on constrained CI CPU, typically < 10ms
    assert elapsed_ms < 100.0, f"Structured ingestion took {elapsed_ms:.2f}ms, exceeding 100ms threshold."

    # Prediction operates accurately
    res = orch.predict("Invoice audit requirements")
    assert res.predicted_action_vector == [0.0, 1.0]
    assert res.most_relevant_rule == "Commercial Statute 404: Invoice audit requirements mandatory."
    assert res.fast_slow_gate == 0.0


def test_orchestrator_structured_supersedes_resolution(offline_encoder: BaseEncoderPort) -> None:
    """Verify that structured engine correctly arbitrates explicit supersedes relations."""
    orch = LegalMemoryOrchestrator(encoder=offline_encoder, value_dim=2, engine_mode="structured")

    # Ingest baseline rule
    orch.update_memory(
        "Rule 1: Physical delivery required.",
        [1.0, 0.0],
        authority_rank=5,
        metadata={"node_id": "rule_1"},
    )
    # Ingest superseding rule
    orch.update_memory(
        "Rule 2: Digital delivery accepted and physical delivery superseded.",
        [0.0, 1.0],
        authority_rank=5,
        metadata={"node_id": "rule_2", "supersedes": "rule_1"},
    )

    res = orch.predict("Delivery requirements")
    assert res.predicted_action_vector == pytest.approx([0.0, 1.0], abs=1e-5)
    assert res.most_relevant_rule == "Rule 2: Digital delivery accepted and physical delivery superseded."


def test_research_neural_head_warning() -> None:
    """Verify that ContinuousMemoryResearchHead emits ParametricNeuralHeadWarning upon instantiation."""
    with pytest.warns(ParametricNeuralHeadWarning, match="CLM-T08 NO-GO"):
        head = ContinuousMemoryResearchHead(embed_dim=64, value_dim=2)
        assert head.embed_dim == 64
        assert head.value_dim == 2
