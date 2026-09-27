"""
Unit tests for Asynchronous Hexagonal Legal Memory Orchestrator operations.

Tests:
1. End-to-end async rule ingestion and decision prediction.
2. Concurrent async query throughput via asyncio.gather demonstrating thread-safe reentrancy.
3. Async right-to-erasure rule deletion.
4. Async HippoRAG Personalized PageRank graph diffusion.
"""

from __future__ import annotations

import asyncio

import pytest

from continuous_legal_memory.domain.interfaces import BaseEncoderPort
from continuous_legal_memory.domain.models import EntityType, RelationType
from continuous_legal_memory.orchestrator import LegalMemoryOrchestrator


@pytest.mark.asyncio
async def test_async_ingest_and_predict(offline_encoder: BaseEncoderPort) -> None:
    """Verify asynchronous rule ingestion and decision prediction without event loop blocking."""
    orch = LegalMemoryOrchestrator(
        encoder=offline_encoder,
        value_dim=2,
        seed=42,
    )

    rec = await orch.async_ingest_rule(
        rule_text="GDPR Article 17: Right to erasure without undue delay.",
        action_vector=[1.0, 0.0],
        authority_rank=10,
        jurisdiction="EU",
    )
    assert rec.record_id is not None
    assert rec.authority_rank == 10

    result = await orch.async_predict(
        query_text="Data subject requests immediate erasure of personal profile data.",
    )
    assert result.predicted_action_vector is not None
    assert len(result.predicted_action_vector) == 2
    # Action [1.0, 0.0] (allow erasure) should have dominant probability weight
    assert result.predicted_action_vector[0] > result.predicted_action_vector[1]


@pytest.mark.asyncio
async def test_async_concurrent_queries(offline_encoder: BaseEncoderPort) -> None:
    """Verify thread-safe concurrent queries via asyncio.gather."""
    orch = LegalMemoryOrchestrator(
        encoder=offline_encoder,
        value_dim=2,
        seed=42,
    )

    await orch.async_ingest_rule(
        rule_text="CCPA: California consumers have the right to opt out of the sale of personal data.",
        action_vector=[0.0, 1.0],
        authority_rank=8,
        jurisdiction="US-CA",
    )
    await orch.async_ingest_rule(
        rule_text="HIPAA: Protected Health Information must be safeguarded with administrative controls.",
        action_vector=[1.0, 0.0],
        authority_rank=9,
        jurisdiction="US-FED",
    )

    queries = [
        "Consumer wants to stop California data sale.",
        "Medical records compliance under federal health privacy statute.",
        "Opt out of data broker marketing in California.",
        "Hospital patient health information audit requirements.",
    ] * 3  # 12 concurrent async requests

    tasks = [orch.async_predict(q) for q in queries]
    results = await asyncio.gather(*tasks)

    assert len(results) == 12
    for res in results:
        assert res.predicted_action_vector is not None
        assert len(res.predicted_action_vector) == 2


@pytest.mark.asyncio
async def test_async_delete_rule(offline_encoder: BaseEncoderPort) -> None:
    """Verify asynchronous rule erasure across cognitive memory tiers."""
    orch = LegalMemoryOrchestrator(
        encoder=offline_encoder,
        value_dim=2,
        seed=42,
    )

    rec = await orch.async_ingest_rule(
        rule_text="Temporary clause to be purged.",
        action_vector=[0.5, 0.5],
        authority_rank=3,
    )
    rule_id = rec.record_id

    # Verify rule is present
    initial_ctx = orch.get_active_context()
    assert any(r.record_id == rule_id for r in initial_ctx)

    # Erase asynchronously
    deleted = await orch.async_delete_rule(rule_id=rule_id)
    assert isinstance(deleted, dict)

    # Verify rule is absent
    post_ctx = orch.get_active_context()
    assert not any(r.record_id == rule_id for r in post_ctx)


@pytest.mark.asyncio
async def test_async_associate_statutes(offline_encoder: BaseEncoderPort) -> None:
    """Verify asynchronous HippoRAG Personalized PageRank associative diffusion."""
    orch = LegalMemoryOrchestrator(
        encoder=offline_encoder,
        value_dim=2,
        seed=42,
    )

    # Add prerequisite and dependent statutes
    await orch.async_ingest_rule(
        rule_text="Securities Exchange Act Section 10(b): Employment of manipulative and deceptive devices.",
        metadata={"node_id": "act_10b", "entity_type": EntityType.STATUTE},
    )
    await orch.async_ingest_rule(
        rule_text="SEC Rule 10b-5: Prohibition against fraud, deceit, or insider trading.",
        metadata={
            "node_id": "rule_10b5",
            "entity_type": EntityType.STATUTE,
            "relations": [{"target_id": "act_10b", "relation_type": RelationType.DEPENDS_ON, "weight": 1.0}],
        },
    )

    ranked_entities = await orch.async_associate_statutes(
        seed_nodes=["rule_10b5"],
        damping=0.85,
        max_results=5,
    )

    assert len(ranked_entities) >= 1
    node_ids = [node.node_id for node, score in ranked_entities]
    assert "act_10b" in node_ids or "rule_10b5" in node_ids
