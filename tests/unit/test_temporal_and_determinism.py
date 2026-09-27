"""
Unit tests for Temporal Validity Enforcement at inference and Seeded Determinism (CLM-T04).
"""

from datetime import datetime, timezone

from continuous_legal_memory.adapters.mock_encoder import SemanticMockEncoder
from continuous_legal_memory.orchestrator import LegalMemoryOrchestrator


def _create_offline_orchestrator(seed: int | None = None) -> LegalMemoryOrchestrator:
    """Helper creating an offline-safe orchestrator using the SemanticMockEncoder."""
    encoder = SemanticMockEncoder(embedding_dim=64, seed=seed or 42)
    return LegalMemoryOrchestrator(encoder=encoder, value_dim=2, seed=seed)


def test_temporal_filtering_at_inference_prevents_stale_recall() -> None:
    """
    Verify that temporally expired rules are strictly excluded from attention and prediction,
    achieving a stale-recall rate of zero even when a query semantically matches the expired rule.
    """
    orchestrator = _create_offline_orchestrator(seed=42)

    rule_a_expired = "Article 10: All biometric records must be purged immediately upon consent revocation."
    rule_b_valid = "Article 11: Anti-money laundering audit trails must be retained for 10 years."

    action_delete = [1.0, 0.0]
    action_retain = [0.0, 1.0]

    # Ingest Rule A with validity in the past (expired)
    orchestrator.update_memory(
        rule_text=rule_a_expired,
        action_vector=action_delete,
        valid_from=datetime(2020, 1, 1, tzinfo=timezone.utc),
        valid_to=datetime(2021, 1, 1, tzinfo=timezone.utc),
    )

    # Ingest Rule B valid indefinitely from 2022
    orchestrator.update_memory(
        rule_text=rule_b_valid,
        action_vector=action_retain,
        valid_from=datetime(2022, 1, 1, tzinfo=timezone.utc),
        valid_to=None,
    )

    # Query matching Rule A evaluated in 2023 (Rule A expired, Rule B active)
    query = "Purge biometric records following customer consent revocation."
    eval_time = datetime(2023, 1, 1, tzinfo=timezone.utc)
    result = orchestrator.predict(query, at_time=eval_time)

    # Acceptance criteria: stale rule A must never be recalled
    assert result.most_relevant_rule != rule_a_expired, "Expired rule was recalled!"
    assert result.most_relevant_rule == rule_b_valid
    assert len(result.attention_weights or []) == 1
    assert result.confidence == 1.0


def test_all_temporally_expired_rules_returns_empty_memory_result() -> None:
    """
    Verify that when all stored rules are temporally invalid at evaluation time,
    the orchestrator returns an empty-memory zeroed prediction result.
    """
    orchestrator = _create_offline_orchestrator(seed=42)

    orchestrator.update_memory(
        rule_text="Old Statute 1990: Temporary tax credit.",
        action_vector=[1.0, 0.0],
        valid_from=datetime(1990, 1, 1, tzinfo=timezone.utc),
        valid_to=datetime(1995, 1, 1, tzinfo=timezone.utc),
    )

    result = orchestrator.predict(
        query_text="Calculate tax credit under 1990 statute.",
        at_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )

    assert result.predicted_action_vector == [0.0, 0.0]
    assert result.most_relevant_rule is None
    assert result.confidence is None
    assert result.attention_weights is None
    assert result.fast_slow_gate is None


def test_seeded_orchestrator_initialization_determinism() -> None:
    """
    Verify that initializing two orchestrators with the same seed results in
    identical initial weights and deterministic inference outputs.
    """
    seed = 2026

    orch_1 = _create_offline_orchestrator(seed=seed)
    orch_2 = _create_offline_orchestrator(seed=seed)

    query = "Standard contract indemnity clause review."
    res_1 = orch_1.predict(query)
    res_2 = orch_2.predict(query)

    assert res_1.predicted_action_vector == res_2.predicted_action_vector
    assert res_1.fast_slow_gate == res_2.fast_slow_gate


def test_seeded_orchestrator_online_adaptation_determinism() -> None:
    """
    Verify that two orchestrators with the same seed undergoing the same rule ingestion
    produce identical post-adaptation decision vectors and attention distributions.
    """
    seed = 777

    orch_1 = _create_offline_orchestrator(seed=seed)
    orch_2 = _create_offline_orchestrator(seed=seed)

    rules = [
        ("Statute Alpha: Data retention mandatory for banking transactions.", [0.0, 1.0]),
        ("Statute Beta: Right to erasure applies to non-transactional metadata.", [1.0, 0.0]),
    ]

    for text, action in rules:
        orch_1.update_memory(text, action)
        orch_2.update_memory(text, action)

    query = "Client requests erasure of bank transaction logs."
    res_1 = orch_1.predict(query)
    res_2 = orch_2.predict(query)

    assert res_1.predicted_action_vector == res_2.predicted_action_vector
    assert res_1.attention_weights == res_2.attention_weights
    assert res_1.most_relevant_rule == res_2.most_relevant_rule
    assert res_1.confidence == res_2.confidence
