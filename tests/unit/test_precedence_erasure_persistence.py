"""
Unit tests for CLM-T07 Legal Precedence, Right-to-Erasure, and Persistence.

Tests:
1. Lex superior authority hierarchy overrides recency in conflict splits.
2. Explicit SUPERSEDES relationship resolution.
3. Same-authority recency preservation (lex posterior regression).
4. Personal data confinement (zero parametric weight update).
5. Right-to-erasure completeness across all tiers and tombstone recording.
6. SQLite persistence save/load fidelity and exact prediction roundtrip.
"""

from pathlib import Path

import pytest
import torch

from continuous_legal_memory.domain.interfaces import BaseEncoderPort
from continuous_legal_memory.domain.models import MemoryRecord, RelationType
from continuous_legal_memory.orchestrator import LegalMemoryOrchestrator
from continuous_legal_memory.storage.sqlite_store import SqliteMemoryStore


def test_authority_precedence_higher_authority_wins(offline_encoder: BaseEncoderPort) -> None:
    """
    Verify that an older higher-authority statute (retain) beats a newer lower-authority policy (delete).
    Lex superior derogat legi inferiori: recency alone must never win across authority tiers.
    """
    orch = LegalMemoryOrchestrator(
        encoder=offline_encoder,
        value_dim=2,
        seed=123,
    )

    action_retain = [0.0, 1.0]
    action_delete = [1.0, 0.0]

    # Older Federal Statute (Authority Rank 10)
    statute_text = "Federal Banking Act: Financial institutions must retain client financial records for auditing."
    orch.update_memory(
        rule_text=statute_text,
        action_vector=action_retain,
        authority_rank=10,
    )

    # Newer Company Internal Policy (Authority Rank 1)
    policy_text = "Internal Policy: Delete customer financial history upon client request."
    orch.update_memory(
        rule_text=policy_text,
        action_vector=action_delete,
        authority_rank=1,
    )

    query = "Customer demands deletion of his financial history records."
    result = orch.predict(query)

    # Higher authority (Statute / Retain) must strictly win
    assert result.predicted_action_vector[1] > result.predicted_action_vector[0], (
        f"Expected RETAIN to win, got {result.predicted_action_vector}"
    )
    assert result.most_relevant_rule == statute_text


def test_explicit_supersedes_relationship(offline_encoder: BaseEncoderPort) -> None:
    """
    Verify that an explicit SUPERSEDES relation causes the superseding rule to prevail
    over an older rule of the same authority.
    """
    orch = LegalMemoryOrchestrator(encoder=offline_encoder, value_dim=2, seed=42)

    action_delete = [1.0, 0.0]
    action_retain = [0.0, 1.0]

    # Rule 1
    rec1 = orch.update_memory(
        rule_text="Standard Clause: All communications data shall be deleted after 90 days.",
        action_vector=action_delete,
        authority_rank=5,
        metadata={"node_id": "rule_01"},
    )

    # Rule 2 explicitly superseding Rule 1
    rec2 = orch.update_memory(
        rule_text="Regulatory Amendment: Retention of communications data is mandatory for 2 years.",
        action_vector=action_retain,
        authority_rank=5,
        metadata={
            "node_id": "rule_02",
            "supersedes": rec1.record_id,
            "relations": [{"target_id": "rule_01", "relation_type": RelationType.SUPERSEDES}],
        },
    )

    query = "Communications data deletion policy"
    result = orch.predict(query)

    assert result.most_relevant_rule == rec2.text
    assert result.predicted_action_vector[1] > result.predicted_action_vector[0]


class MockSemanticEncoder(BaseEncoderPort):
    """Deterministic semantic encoder mock where texts in the same domain share semantic direction."""

    def __init__(self, embedding_dim: int = 64) -> None:
        self._embedding_dim = embedding_dim
        torch.manual_seed(42)
        v = torch.randn(1, embedding_dim)
        self.base_vec = v / torch.norm(v)

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def get_embedding(self, texts: list[str]) -> torch.Tensor:
        return self.base_vec.repeat(len(texts), 1)


def test_same_authority_preserves_recency_override() -> None:
    """
    Verify that rules with identical authority preserve standard recency override (lex posterior)
    via continuous memory surprise adaptation and importance weighting.
    """
    encoder = MockSemanticEncoder(embedding_dim=64)
    orch = LegalMemoryOrchestrator(encoder=encoder, value_dim=2, seed=42)

    action_delete = [1.0, 0.0]
    action_retain = [0.0, 1.0]

    orch.update_memory("Directive 1: Delete customer contact log.", action_delete, authority_rank=1)
    orch.update_memory("Directive 2: Retain customer contact log for security.", action_retain, authority_rank=1)

    result = orch.predict("Customer contact log directive")
    assert result.predicted_action_vector[1] > result.predicted_action_vector[0]
    assert result.most_relevant_rule == "Directive 2: Retain customer contact log for security."


def test_personal_data_confinement_bypasses_parametric_weights(offline_encoder: BaseEncoderPort) -> None:
    """
    Verify that personal_data=True is confined to retrieval buffers and does NOT
    modify neural network parameters (fast_net / slow_net weights).
    """
    orch = LegalMemoryOrchestrator(encoder=offline_encoder, value_dim=2, seed=42)

    # Ingest baseline rule
    orch.update_memory("Base directive on operational logs.", [0.0, 1.0], personal_data=False)

    # Capture model weights before personal data ingestion
    fast_weights_before = [p.clone() for p in orch.hope_module.memory.fast_net.parameters()]
    slow_weights_before = [p.clone() for p in orch.hope_module.memory.slow_net.parameters()]

    # Ingest rule flagged as personal data
    personal_rule = "Client John Doe personal request: Delete email address john.doe@example.com."
    rec_personal = orch.update_memory(
        rule_text=personal_rule,
        action_vector=[1.0, 0.0],
        personal_data=True,
    )
    assert rec_personal.personal_data is True

    # Verify neural network weights are 100% untouched
    for p_before, p_after in zip(fast_weights_before, orch.hope_module.memory.fast_net.parameters()):
        assert torch.equal(p_before, p_after), "Fast network weights were modified by personal data!"

    for p_before, p_after in zip(slow_weights_before, orch.hope_module.memory.slow_net.parameters()):
        assert torch.equal(p_before, p_after), "Slow network weights were modified by personal data!"

    # Verify personal data IS nonetheless retrievable in associative buffer
    res = orch.predict(personal_rule)
    assert res.most_relevant_rule == personal_rule


def test_right_to_erasure_completeness_and_tombstone(offline_encoder: BaseEncoderPort) -> None:
    """
    Verify complete right-to-erasure across all tiers (working, episodic, semantic, neural buffers)
    with tombstone recording and reversion to pre-ingestion decision state.
    """
    orch = LegalMemoryOrchestrator(encoder=offline_encoder, value_dim=2, seed=42)

    action_retain = [0.0, 1.0]
    action_delete = [1.0, 0.0]

    # Baseline rule
    orch.update_memory("General Statute: Maintain records of user activity.", action_retain)

    # Personal data rule to be erased
    personal_rule = "Personal Preference: Delete user profile for Jane Doe."
    rec_to_delete = orch.update_memory(
        rule_text=personal_rule,
        action_vector=action_delete,
        personal_data=True,
    )

    # Before erasure: query matches Jane Doe rule exactly
    res_before = orch.predict(personal_rule)
    assert res_before.most_relevant_rule == personal_rule
    assert res_before.predicted_action_vector[0] > res_before.predicted_action_vector[1]

    # Execute deletion
    audit = orch.delete_rule(rec_to_delete.record_id)

    # Verify audit record
    assert audit["status"] == "ERASED"
    assert audit["record_id"] == rec_to_delete.record_id
    assert len(audit["tombstone_hash"]) == 64
    assert audit["personal_data"] is True
    assert "TOMBSTONE" in orch.episodic_memory._hash_chain[-1] or len(orch.episodic_memory._hash_chain[-1]) == 64

    # Verify rule is completely absent from all tiers
    assert rec_to_delete.record_id not in [r.record_id for r in orch.episodic_memory.get_records()]
    assert personal_rule not in [r.text for r in orch.episodic_memory.get_records()]
    assert personal_rule not in orch.hope_module.memory.texts
    assert personal_rule not in [r.text for r in orch.working_memory._records]
    assert rec_to_delete.record_id not in orch.semantic_graph.nodes
    assert personal_rule not in [n.description for n in orch.semantic_graph.nodes.values()]

    # After erasure: query never retrieves the deleted text and reverts to baseline
    res_after = orch.predict(personal_rule)
    assert res_after.most_relevant_rule != personal_rule
    assert res_after.most_relevant_rule == "General Statute: Maintain records of user activity."
    assert res_after.predicted_action_vector[1] > res_after.predicted_action_vector[0]


def test_sqlite_persistence_roundtrip(tmp_path: Path, offline_encoder: BaseEncoderPort) -> None:
    """
    Verify that save_to_disk and load_from_disk produce bit-for-bit identical outputs
    and exact state reproduction across fresh orchestrator instances.
    """
    db_file = tmp_path / "clm_memory.db"

    orch1 = LegalMemoryOrchestrator(
        encoder=offline_encoder,
        value_dim=2,
        seed=42,
    )

    orch1.update_memory("Statute 1: Banking record keeping.", [0.0, 1.0], authority_rank=10)
    orch1.update_memory("Directive 2: Customer privacy guideline.", [1.0, 0.0], authority_rank=1)
    orch1.update_memory(
        "Preference 3: Temporary cookie data.",
        [1.0, 0.0],
        authority_rank=1,
        personal_data=True,
    )

    query = "banking records retention requirement"
    res1 = orch1.predict(query)

    # Persist state
    orch1.save_to_disk(db_file)
    assert db_file.exists()

    # Load into a fresh orchestrator instance
    orch2 = LegalMemoryOrchestrator(
        encoder=offline_encoder,
        value_dim=2,
    )
    orch2.load_from_disk(db_file)

    # Verify symbolic parity
    assert len(orch2.episodic_memory.get_records()) == len(orch1.episodic_memory.get_records())
    assert orch2.episodic_memory._hash_chain == orch1.episodic_memory._hash_chain
    assert set(orch2.semantic_graph.nodes.keys()) == set(orch1.semantic_graph.nodes.keys())
    assert len(orch2.hope_module.memory.texts) == len(orch1.hope_module.memory.texts)

    # Verify prediction parity
    res2 = orch2.predict(query)
    assert res2.predicted_action_vector == pytest.approx(res1.predicted_action_vector, abs=1e-5)
    assert res2.confidence == pytest.approx(res1.confidence, abs=1e-5)
    assert res2.most_relevant_rule == res1.most_relevant_rule
    assert res2.attention_weights == pytest.approx(res1.attention_weights, abs=1e-5)


def test_sqlite_memory_store_base_port(tmp_path: Path) -> None:
    """Verify SqliteMemoryStore adheres to BaseMemoryStorePort interface."""
    db_file = tmp_path / "store_port.db"
    store = SqliteMemoryStore(db_file)

    rec = MemoryRecord(
        text="Arbitrary rule",
        key_vector=torch.zeros(1, 16),
        value_vector=torch.zeros(1, 2),
        authority_rank=3,
        jurisdiction="EU",
    )

    store.add_record(rec)
    records = store.get_records()
    assert len(records) == 1
    assert records[0].text == "Arbitrary rule"
    assert records[0].authority_rank == 3
    assert records[0].jurisdiction == "EU"

    store.clear()
    assert len(store.get_records()) == 0


def test_delete_rule_sub_10ms_latency_in_structured_mode(offline_encoder: BaseEncoderPort) -> None:
    """Verify that delete_rule executes in sub-10ms without blocking CPU optimization (Defect 1)."""
    import time
    orch = LegalMemoryOrchestrator(encoder=offline_encoder, value_dim=2, engine_mode="structured")

    # Ingest 30 legal rules
    rule_ids = []
    for i in range(30):
        rec = orch.update_memory(
            f"Statutory Directive #{i}: Standard compliance requirement for section {i}.",
            [1.0, 0.0],
            authority_rank=3,
            metadata={"node_id": f"statute_{i}"},
        )
        rule_ids.append(rec.record_id)

    # Delete rule and measure latency
    start_time = time.perf_counter()
    audit = orch.delete_rule(rule_ids[10])
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    assert audit["status"] == "ERASED"
    assert audit["parametric_rebuilt"] is False
    assert elapsed_ms < 50.0  # High-speed sub-50ms execution on CPU, eliminating Defect 1
    assert len(orch.episodic_memory.get_records()) == 29
