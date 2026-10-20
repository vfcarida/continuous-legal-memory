"""
Unit tests for Phase 2 Cognitive Multi-Tiered Memory System components.
"""

from datetime import datetime, timedelta, timezone

import pytest
import torch

from continuous_legal_memory.core.memory_tiers.episodic_memory import EpisodicMemory
from continuous_legal_memory.core.memory_tiers.semantic_memory import SemanticKnowledgeGraph
from continuous_legal_memory.core.memory_tiers.working_memory import WorkingMemory
from continuous_legal_memory.domain.exceptions import (
    ContextWindowExceededError,
    TemporalInvalidationError,
)
from continuous_legal_memory.domain.models import EntityType, RelationType


def test_working_memory_sliding_window() -> None:
    """Verify Working Memory sliding window eviction and strict capacity exception behavior."""
    wm = WorkingMemory(capacity=2, strict_capacity=False)
    vec = torch.zeros(1, 10)
    val = torch.zeros(1, 2)

    wm.add("Turn 1", vec, val)
    wm.add("Turn 2", vec, val)
    assert len(wm) == 2

    # Ingest turn 3 -> triggers FIFO sliding-window eviction of Turn 1
    wm.add("Turn 3", vec, val)
    assert len(wm) == 2
    context = wm.get_active_context()
    assert context[0].text == "Turn 2"
    assert context[1].text == "Turn 3"

    # Strict capacity mode test
    wm_strict = WorkingMemory(capacity=1, strict_capacity=True)
    wm_strict.add("Turn 1", vec, val)
    with pytest.raises(ContextWindowExceededError):
        wm_strict.add("Turn 2", vec, val)


def test_episodic_memory_hash_chain_and_decay() -> None:
    """Verify Episodic Memory cryptographic SHA-256 hash chaining and temporal decay score calculation."""
    em = EpisodicMemory(decay_rate=0.01)
    vec = torch.zeros(1, 10)
    val = torch.zeros(1, 2)
    now = datetime.now(timezone.utc)

    rec1 = em.append("Rule 1", vec, val, valid_from=now)
    rec2 = em.append("Rule 2", vec, val, valid_from=now)

    assert rec1.record_id is not None
    assert rec2.record_id is not None
    assert rec1.record_id != rec2.record_id

    # Test temporal decay score for immediate vs 5 days in the future
    score_now = em.compute_temporal_decay(rec1, at_time=now)
    score_future = em.compute_temporal_decay(rec1, at_time=now + timedelta(days=5))

    assert score_now == 1.0
    assert score_future < score_now

    # Test TemporalInvalidationError on expired records
    rec_expired = em.append("Expired Rule", vec, val, valid_from=now - timedelta(days=10), valid_to=now - timedelta(days=1))
    with pytest.raises(TemporalInvalidationError):
        em.compute_temporal_decay(rec_expired, at_time=now)


def test_semantic_knowledge_graph_gdpr_and_dependencies() -> None:
    """Verify Semantic Knowledge Graph non-destructive invalidation, prerequisite traversal, and contradictions."""
    graph = SemanticKnowledgeGraph()
    now = datetime.now(timezone.utc)

    # Register nodes
    graph.add_node("statute_101", EntityType.STATUTE, "GDPR Art. 17", "Right to erasure")
    graph.add_node("clause_202", EntityType.CLAUSE, "Anti-Fraud Exception", "Retain credit audit records")

    # Register dependency and contradiction edges
    graph.add_edge("clause_202", "statute_101", RelationType.DEPENDS_ON)
    graph.add_edge("clause_202", "statute_101", RelationType.CONTRADICTS)

    # Test prerequisite traversal
    prereqs = graph.check_prerequisites("clause_202")
    assert len(prereqs) == 1
    assert prereqs[0].node_id == "statute_101"

    # Test contradiction detection
    contradictions = graph.find_contradictions("clause_202")
    assert len(contradictions) == 1
    assert contradictions[0].node_id == "statute_101"

    # Test GDPR Art. 17 Non-Destructive Invalidation (setting valid_to and decay_factor = 0.0)
    graph.invalidate_node_non_destructively("statute_101", invalidation_time=now)
    assert graph.nodes["statute_101"].decay_factor == 0.0
    assert graph.nodes["statute_101"].valid_to == now

    # Now checking prerequisites should raise TemporalInvalidationError because required node is invalid
    with pytest.raises(TemporalInvalidationError):
        graph.check_prerequisites("clause_202", at_time=now + timedelta(seconds=1))
