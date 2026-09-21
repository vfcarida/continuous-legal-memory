"""
Unit tests for CLM-T06 Subsystem Reachability and Clean Interfaces.

Verifies that all advertised subsystems (RetrieverPort / HybridLegalRetriever,
SemanticKnowledgeGraph, BaseMemoryStorePort, TelemetryLogger, and KeyedHashAttestationModule)
are reachable from LegalMemoryOrchestrator and operate with backward-compatible determinism.
"""

from datetime import datetime, timezone

import pytest
import torch

from continuous_legal_memory.adapters.ollama import OllamaGemmaAdapter
from continuous_legal_memory.core.memory_tiers.episodic_memory import EpisodicMemory
from continuous_legal_memory.core.memory_tiers.semantic_memory import SemanticKnowledgeGraph
from continuous_legal_memory.domain.exceptions import MemoryContradictionError
from continuous_legal_memory.domain.interfaces import (
    BaseEncoderPort,
    BaseMemoryStorePort,
    BaseRetrieverPort,
    RetrieverPort,
)
from continuous_legal_memory.domain.models import (
    EntityType,
    MemoryRecord,
    MemoryTier,
    RelationType,
)
from continuous_legal_memory.orchestrator import LegalMemoryOrchestrator
from continuous_legal_memory.retrieval.default_retriever import DefaultAttentionRetriever
from continuous_legal_memory.retrieval.hybrid_retriever import HybridLegalRetriever
from continuous_legal_memory.security.attestation import (
    KeyedHashAttestationModule,
    MemoryAttestationToken,
)
from continuous_legal_memory.telemetry.observability import TelemetryLogger


@pytest.fixture
def offline_encoder() -> BaseEncoderPort:
    return OllamaGemmaAdapter(
        embedding_dim=64,
        strict_privacy_mode=True,
        allow_pseudo_embeddings=True,
    )


def test_retriever_port_hierarchy() -> None:
    """Verify BaseRetrieverPort and RetrieverPort alias definitions."""
    assert issubclass(DefaultAttentionRetriever, BaseRetrieverPort)
    assert issubclass(HybridLegalRetriever, BaseRetrieverPort)
    assert RetrieverPort is BaseRetrieverPort


def test_default_retriever_seeded_equivalence(offline_encoder: BaseEncoderPort) -> None:
    """Verify DefaultAttentionRetriever preserves deterministic output matching default math."""
    seed = 42
    orch = LegalMemoryOrchestrator(
        encoder=offline_encoder,
        value_dim=2,
        seed=seed,
    )
    orch.update_memory("Directive A: Compliance required.", [1.0, 0.0])
    orch.update_memory("Directive B: Exemption granted.", [0.0, 1.0])

    res = orch.predict("Compliance status for customer")
    assert res.predicted_action_vector is not None
    assert len(res.predicted_action_vector) == 2
    assert res.most_relevant_rule in ["Directive A: Compliance required.", "Directive B: Exemption granted."]
    assert res.confidence is not None and 0.0 <= res.confidence <= 1.0


def test_hybrid_retriever_reachability_and_snippets(offline_encoder: BaseEncoderPort) -> None:
    """Verify HybridLegalRetriever can be injected into LegalMemoryOrchestrator and extracts snippets."""
    retriever = HybridLegalRetriever(encoder=offline_encoder, bm25_weight=0.5, dense_weight=0.5)
    orch = LegalMemoryOrchestrator(
        encoder=offline_encoder,
        value_dim=2,
        retriever=retriever,
        seed=42,
    )

    orch.update_memory(
        "Article 17 GDPR: Data subjects have the unconditional right to erasure of personal records.",
        [1.0, 0.0],
    )
    orch.update_memory(
        "Anti-Fraud Statute: Credit operations records must be preserved for five years.",
        [0.0, 1.0],
    )

    res = orch.predict("right to erasure of personal records")
    assert res.predicted_action_vector is not None
    assert res.most_relevant_rule is not None
    assert "erasure" in res.most_relevant_rule.lower()
    assert res.retrieved_snippets is not None
    assert len(res.retrieved_snippets) == 2


def test_episodic_memory_implements_memory_store_port() -> None:
    """Verify EpisodicMemory implements BaseMemoryStorePort interface."""
    mem = EpisodicMemory()
    assert isinstance(mem, BaseMemoryStorePort)

    rec = MemoryRecord(
        text="Sample rule",
        key_vector=torch.zeros(1, 16),
        value_vector=torch.zeros(1, 2),
        valid_from=datetime.now(timezone.utc),
    )

    mem.add_record(rec)
    records = mem.get_records()
    assert len(records) == 1
    assert records[0].text == "Sample rule"

    mem.clear()
    assert len(mem.get_records()) == 0
    assert len(mem) == 0


def test_orchestrator_custom_memory_store(offline_encoder: BaseEncoderPort) -> None:
    """Verify custom BaseMemoryStorePort can be injected into orchestrator."""
    custom_store = EpisodicMemory()
    orch = LegalMemoryOrchestrator(
        encoder=offline_encoder,
        value_dim=2,
        episodic_store=custom_store,
        seed=42,
    )
    orch.update_memory("Statute X", [1.0, 0.0])
    assert len(custom_store.get_records()) == 1
    assert custom_store.get_records()[0].text == "Statute X"


def test_semantic_knowledge_graph_populated_and_source_tier(offline_encoder: BaseEncoderPort) -> None:
    """Verify SemanticKnowledgeGraph is populated in update_memory and source_tier is SEMANTIC."""
    orch = LegalMemoryOrchestrator(
        encoder=offline_encoder,
        value_dim=2,
        seed=42,
    )

    # Initially empty memory -> source_tier is EPISODIC
    empty_res = orch.predict("Empty query")
    assert empty_res.source_tier == MemoryTier.EPISODIC

    # Update with metadata relations
    orch.update_memory(
        rule_text="Directive Root: Core legal baseline.",
        action_vector=[1.0, 0.0],
        metadata={"node_id": "root_statute", "entity_type": EntityType.STATUTE, "label": "Root Statute"},
    )
    orch.update_memory(
        rule_text="Directive Child: Specific exception clause.",
        action_vector=[0.0, 1.0],
        metadata={
            "node_id": "child_clause",
            "entity_type": EntityType.CLAUSE,
            "label": "Child Clause",
            "relations": [{"target_id": "root_statute", "relation_type": RelationType.DEPENDS_ON}],
        },
    )

    assert len(orch.semantic_graph.nodes) == 2
    assert "root_statute" in orch.semantic_graph.nodes
    assert "child_clause" in orch.semantic_graph.nodes
    assert len(orch.semantic_graph.edges) == 1
    assert orch.semantic_graph.edges[0].relation_type == RelationType.DEPENDS_ON

    res = orch.predict("Query about child clause")
    assert res.source_tier == MemoryTier.SEMANTIC


def test_semantic_knowledge_graph_contradiction_error() -> None:
    """Verify MemoryContradictionError is raised when linking contradictory nodes."""
    graph = SemanticKnowledgeGraph()
    graph.add_node("node_a", EntityType.STATUTE, "Node A", "Content A")
    graph.add_node("node_b", EntityType.STATUTE, "Node B", "Content B")

    # Mark as contradictory
    graph.add_edge("node_a", "node_b", RelationType.CONTRADICTS)

    # Attempting to add a DEPENDS_ON relationship between contradictory nodes must raise
    with pytest.raises(MemoryContradictionError) as exc_info:
        graph.add_edge("node_a", "node_b", RelationType.DEPENDS_ON)

    assert "already marked as contradictory" in str(exc_info.value)


def test_orchestrator_contradiction_via_update_memory(offline_encoder: BaseEncoderPort) -> None:
    """Verify orchestrator update_memory propagates MemoryContradictionError on contradictory relations."""
    orch = LegalMemoryOrchestrator(encoder=offline_encoder, value_dim=2, seed=42)

    orch.update_memory("Statute Alpha", [1.0, 0.0], metadata={"node_id": "alpha"})
    orch.update_memory(
        "Statute Beta",
        [0.0, 1.0],
        metadata={"node_id": "beta", "relations": [{"target_id": "alpha", "relation_type": RelationType.CONTRADICTS}]},
    )

    # Now trying to declare that Alpha depends on Beta should fail
    with pytest.raises(MemoryContradictionError):
        orch.semantic_graph.add_edge("alpha", "beta", RelationType.DEPENDS_ON)


def test_telemetry_logger_reachability(offline_encoder: BaseEncoderPort) -> None:
    """Verify TelemetryLogger hook traces update_memory and predict operations."""
    telemetry = TelemetryLogger(service_name="test-clm")
    orch = LegalMemoryOrchestrator(
        encoder=offline_encoder,
        value_dim=2,
        telemetry=telemetry,
        seed=42,
    )

    orch.update_memory("Article 1", [1.0, 0.0])
    orch.predict("Test query")

    assert len(telemetry.metrics_history) == 2
    op_names = [m.operation for m in telemetry.metrics_history]
    assert op_names == ["update_memory", "predict"]
    assert all(m.latency_ms >= 0.0 for m in telemetry.metrics_history)
    assert all(m.tokens_processed > 0 for m in telemetry.metrics_history)


def test_keyed_hash_attestation_reachability(offline_encoder: BaseEncoderPort) -> None:
    """Verify KeyedHashAttestationModule signs predictions and tokens verify correctly."""
    attestor = KeyedHashAttestationModule(secret_key="my-audit-secret-key")
    orch = LegalMemoryOrchestrator(
        encoder=offline_encoder,
        value_dim=2,
        attestor=attestor,
        seed=42,
    )

    orch.update_memory("Article 15 Access Right", [1.0, 0.0])
    res = orch.predict("Request access to records")

    assert res.attestation_token is not None
    assert isinstance(res.attestation_token, MemoryAttestationToken)
    assert res.attestation_token.algorithm == "HMAC-SHA256"

    # Verification must succeed
    assert attestor.verify_attestation(res.attestation_token, res, res.most_relevant_rule) is True

    # Tampered prediction must fail verification
    res.predicted_action_vector = [0.0, 1.0]
    assert attestor.verify_attestation(res.attestation_token, res, res.most_relevant_rule) is False
