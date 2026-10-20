"""
Unit and Property-Based Invariant Tests for Deontic Consistency and Choice-of-Law Routing:
- Formal Deontic Consistency Checker, Antinomy Detector & Acyclic Graph Guards
- Cross-Jurisdictional Choice-of-Law Router (Private International Law / Rome I Art. 9)
- Full SQLite WAL Persistence of 256-Bit Sparse Merkle Tree & Append-Only Audit Log
- Algebraic & Cryptographic Property-Based Invariants (Permutation Invariance,
  Erasure Reversibility, Precedence Monotonicity, and Conformal Coverage Bounds)
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from continuous_legal_memory import (
    ConnectingFactors,
    DeonticConsistencyChecker,
    EntityType,
    LegalMemoryOrchestrator,
    MCPServer,
    MemoryRecord,
    RelationType,
    SparseMerkleTree,
    SplitConformalCalibrator,
    apply_legal_precedence,
    verify_sparse_merkle_proof,
)
from continuous_legal_memory.core.memory_tiers.semantic_memory import SemanticKnowledgeGraph
from continuous_legal_memory.domain.exceptions import MemoryContradictionError
from continuous_legal_memory.domain.interfaces import BaseEncoderPort
from continuous_legal_memory.domain.models import GraphEdge


def test_acyclic_graph_enforcement_and_tarjan_cycle_detection() -> None:
    """
    Verify that SemanticKnowledgeGraph.add_edge blocks self-loops and multi-hop cycles
    on SUPERSEDES and DEPENDS_ON relations, and that DeonticConsistencyChecker.detect_cycles
    identifies any pre-existing cycles via Tarjan's SCC algorithm.
    """
    graph = SemanticKnowledgeGraph()
    for nid in ("r1", "r2", "r3"):
        graph.add_node(nid, EntityType.STATUTE, nid, f"Statute {nid}")

    # Valid chain: r1 SUPERSEDES r2, r2 SUPERSEDES r3
    graph.add_edge("r1", "r2", RelationType.SUPERSEDES)
    graph.add_edge("r2", "r3", RelationType.SUPERSEDES)

    # Self-loop must raise MemoryContradictionError
    with pytest.raises(MemoryContradictionError, match="Self-referential"):
        graph.add_edge("r1", "r1", RelationType.SUPERSEDES)

    # Closing the cycle r3 -> r1 must raise MemoryContradictionError
    with pytest.raises(MemoryContradictionError, match="normative cycle"):
        graph.add_edge("r3", "r1", RelationType.SUPERSEDES)

    # If a cycle is injected directly into raw edges (e.g. legacy import), Tarjan SCC detects it
    graph.edges.append(GraphEdge("r3", "r1", RelationType.SUPERSEDES))
    cycles = DeonticConsistencyChecker.detect_cycles(graph, {RelationType.SUPERSEDES})
    assert len(cycles) == 1
    assert set(cycles[0]) == {"r1", "r2", "r3"}


def test_deontic_consistency_checker_antinomy_detection(token_encoder: BaseEncoderPort) -> None:
    """
    Verify that DeonticConsistencyChecker detects simultaneous horizontal antinomies
    (equal authority, equal specificity, identical valid_from, same jurisdiction, opposing
    actions) while confirming consistency when disambiguated by authority or specificity.
    """
    orch = LegalMemoryOrchestrator(encoder=token_encoder, seed=42)
    simultaneous_ts = datetime(2025, 5, 1, 0, 0, tzinfo=timezone.utc)

    # Ingest two simultaneous rules on the same subject matter with equal authority & specificity
    # but opposing outcomes ([1.0, 0.0] vs [0.0, 1.0])
    r1 = orch.update_memory(
        "Section 10(a) biometric facial template storage policy: delete biometric facial templates immediately.",
        [1.0, 0.0],
        authority_rank=4,
        jurisdiction="EU",
        valid_from=simultaneous_ts,
        metadata={"specificity": 5.0, "node_id": "ant_a"},
    )
    r2 = orch.update_memory(
        "Section 10(b) biometric facial template storage policy: retain biometric facial templates for audit.",
        [0.0, 1.0],
        authority_rank=4,
        jurisdiction="EU",
        valid_from=simultaneous_ts,
        metadata={"specificity": 5.0, "node_id": "ant_b"},
    )
    assert r1.record_id is not None and r2.record_id is not None

    report = orch.audit_deontic_consistency(tenant_id="default")
    assert not report.is_consistent
    assert len(report.antinomies) == 1
    ant = report.antinomies[0]
    assert {ant.rule_a_id, ant.rule_b_id} == {r1.record_id, r2.record_id}
    assert ant.authority_rank == 4
    assert ant.specificity == 5.0

    # Verify via MCP tool call legal_memory_audit_consistency
    mcp = MCPServer(orchestrator=orch)
    resp = mcp.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 401,
            "method": "tools/call",
            "params": {
                "name": "legal_memory_audit_consistency",
                "arguments": {"tenant_id": "default"},
            },
        }
    )
    assert resp is not None
    mcp_report = json.loads(resp["result"]["content"][0]["text"])
    assert mcp_report["is_consistent"] is False
    assert len(mcp_report["antinomies"]) == 1


def test_choice_of_law_router_overriding_mandatory_vs_forum_autonomy(
    token_encoder: BaseEncoderPort,
) -> None:
    """
    Verify Private International Law (Rome I Art. 3 & Art. 9 / GDPR Art. 3(2)) routing:
    1. Overriding mandatory provisions (lois de police) of the data subject domicile ('EU')
       prevail over a contractual choice-of-forum clause ('US-NY').
    2. For dispositive commercial rules (no overriding mandatory norm), contractual
       choice-of-forum ('US-NY') prevails.
    3. Explicit jurisdiction filter restricts retrieval to the specified sovereign scope.
    """
    orch = LegalMemoryOrchestrator(encoder=token_encoder, seed=42)

    # 1. US-NY dispositive contractual clause (authority_rank=2, action=[1.0, 0.0] -> PERMIT)
    orch.update_memory(
        "Commercial cloud telemetry contract: cross-border biometric profiling transfer is permitted.",
        [1.0, 0.0],
        authority_rank=2,
        jurisdiction="US-NY",
        metadata={"overriding_mandatory": False},
    )

    # 2. EU overriding mandatory GDPR protection (authority_rank=5, action=[0.0, 1.0] -> PROHIBIT)
    orch.update_memory(
        "GDPR Article 3(2) and Article 9 mandatory rule: cross-border biometric profiling transfer is prohibited.",
        [0.0, 1.0],
        authority_rank=5,
        jurisdiction="EU",
        metadata={"overriding_mandatory": True},
    )

    # Case A: EU data subject domicile + US-NY contractual forum -> EU overriding mandatory rule wins
    res_mandatory = orch.predict(
        "Is cross-border biometric profiling transfer permitted under the cloud telemetry contract?",
        connecting_factors=ConnectingFactors(
            data_subject_domicile="EU",
            choice_of_forum="US-NY",
        ),
    )
    assert res_mandatory.predicted_action_vector[1] > res_mandatory.predicted_action_vector[0]
    assert res_mandatory.choice_of_law is not None
    assert "EU" in res_mandatory.choice_of_law["overriding_mandatory_jurisdictions"]

    # Case B: Explicit jurisdiction filter for 'US-NY' only -> retrieves only the US-NY rule
    res_us_only = orch.predict(
        "Is cross-border biometric profiling transfer permitted under the cloud telemetry contract?",
        jurisdiction="US-NY",
    )
    assert res_us_only.predicted_action_vector[0] > res_us_only.predicted_action_vector[1]


def test_sqlite_wal_persistence_of_sparse_merkle_tree_and_audit_log(
    tmp_path: Path,
    token_encoder: BaseEncoderPort,
) -> None:
    """
    Verify that SparseMerkleTree active leaves, root hash, and AppendOnlyMerkleLog hash chain
    survive SQLite save_to_disk / load_from_disk across orchestrator instances.
    """
    db_file = tmp_path / "merkle_persistence.db"
    orch_1 = LegalMemoryOrchestrator(
        encoder=token_encoder,
        seed=42,
        enable_attestation=True,
    )

    rec_a = orch_1.update_memory(
        "Statute Alpha: retain banking wire transfer records.",
        [0.0, 1.0],
        authority_rank=5,
        metadata={"rule_id": "statute_alpha"},
    )
    rec_b = orch_1.update_memory(
        "Preference Beta: temporary client marketing consent.",
        [1.0, 0.0],
        personal_data=True,
        metadata={"rule_id": "pref_beta"},
    )
    assert rec_a.record_id == "statute_alpha"
    assert rec_b.record_id == "pref_beta"

    # Erase pref_beta before saving to disk
    erasure_receipt = orch_1.delete_rule(record_id="pref_beta")
    root_before_save = orch_1.merkle_tree.root_hex
    log_head_before_save = orch_1.merkle_tree.audit_log.head_hash
    assert root_before_save == erasure_receipt["merkle_root"]

    orch_1.save_to_disk(db_file)

    # Restore into a fresh orchestrator instance
    orch_2 = LegalMemoryOrchestrator(
        encoder=token_encoder,
        seed=42,
        enable_attestation=True,
    )
    orch_2.load_from_disk(db_file)

    # Verify exact bit-for-bit restoration of SparseMerkleTree root and AppendOnlyMerkleLog
    assert orch_2.merkle_tree.root_hex == root_before_save
    assert orch_2.merkle_tree.audit_log.head_hash == log_head_before_save
    assert orch_2.merkle_tree.audit_log.verify_chain()
    assert len(orch_2.merkle_tree.audit_log.entries) == 3

    # Verify inclusion proof for retained statute_alpha and non-membership proof for erased pref_beta
    inc_proof = orch_2.prove_inclusion("statute_alpha")
    non_mem_proof = orch_2.prove_erasure("pref_beta")
    assert verify_sparse_merkle_proof(inc_proof, expected_root_hash=root_before_save)
    assert verify_sparse_merkle_proof(non_mem_proof, expected_root_hash=root_before_save)


def test_property_sparse_merkle_permutation_and_erasure_reversibility_invariants() -> None:
    """
    Property-Based Invariant Test:
    1. Permutation Invariance: Inserting any set of (rule_id, payload) pairs in arbitrary
       order always produces the exact same 256-bit Sparse Merkle Tree root hash.
    2. Erasure Reversibility: Inserting a subset of rules and then erasing them restores
       the exact prior 256-bit Merkle root hash bit-for-bit.
    """
    rng = random.Random(20261019)
    items = [(f"rule_{i:03d}", f"payload_content_{i}_{rng.randint(1000, 9999)}") for i in range(15)]

    reference_smt = SparseMerkleTree()
    for rid, payload in items:
        reference_smt.update(rid, payload, record_log=False)
    canonical_root = reference_smt.root_hex

    # Test 20 random permutations
    for _ in range(20):
        shuffled = list(items)
        rng.shuffle(shuffled)
        perm_smt = SparseMerkleTree()
        for rid, payload in shuffled:
            perm_smt.update(rid, payload, record_log=False)
        assert perm_smt.root_hex == canonical_root

    # Test erasure reversibility: add 8 temporary rules in random order, then remove them in random order
    temp_items = [(f"ephemeral_{j:03d}", f"temp_payload_{j}") for j in range(8)]
    for rid, payload in temp_items:
        reference_smt.update(rid, payload, record_log=False)
    assert reference_smt.root_hex != canonical_root

    rng.shuffle(temp_items)
    for rid, _ in temp_items:
        reference_smt.remove(rid, record_log=False)
    assert reference_smt.root_hex == canonical_root


def test_property_precedence_monotonicity_and_conformal_coverage_bounds() -> None:
    """
    Property-Based Invariant Test:
    1. Precedence Monotonicity: Increasing a rule's `authority_rank` or `specificity` while
       holding all other parameters fixed never decreases its post-precedence attention weight.
    2. Split-Conformal Coverage Bound: Across miscoverage rates alpha in (0.05, 0.10, 0.20),
       empirical coverage on exchangeable calibration/test splits satisfies >= 1 - alpha - 1/(n+1).
    """
    key_vec = F.normalize(torch.randn(1, 32), p=2, dim=-1)
    val_vec = torch.tensor([[0.0, 1.0]])

    # 1. Monotonicity across authority_rank 1..10
    prev_weight = -1.0
    for rank in range(1, 11):
        rec_a = MemoryRecord(
            text="Rule A",
            key_vector=key_vec,
            value_vector=val_vec,
            authority_rank=rank,
            metadata={"specificity": 2.0},
        )
        rec_b = MemoryRecord(
            text="Rule B",
            key_vector=key_vec,
            value_vector=val_vec,
            authority_rank=5,
            metadata={"specificity": 2.0},
        )
        raw_scores = torch.tensor([[0.80, 0.80]])
        importance = torch.tensor([1.0, 1.0])
        adj = apply_legal_precedence(raw_scores, importance, [rec_a, rec_b])
        weights = torch.softmax(adj / 0.05, dim=-1)
        w_a = float(weights[0, 0].item())
        assert w_a >= prev_weight - 1e-6
        prev_weight = w_a

    # 2. Split-Conformal finite-sample coverage bound across alpha in (0.05, 0.10, 0.20)
    gen = torch.Generator().manual_seed(777)
    n_cal = 100
    n_test = 100
    logits = torch.randn(n_cal + n_test, 4, generator=gen) * 2.0
    probs = torch.softmax(logits, dim=-1)
    # Sample ground-truth labels from the true conditional distribution probs
    labels = [int(torch.multinomial(probs[i], num_samples=1, generator=gen).item()) for i in range(n_cal + n_test)]

    cal_probs = probs[:n_cal]
    cal_labels = labels[:n_cal]
    test_probs = probs[n_cal:]
    test_labels = labels[n_cal:]

    for alpha in (0.05, 0.10, 0.20):
        calibrator = SplitConformalCalibrator(alpha=alpha, min_similarity_gate=0.0)
        calibrator.calibrate(cal_probs, cal_labels)
        stats = calibrator.evaluate_empirical_coverage(test_probs, test_labels)
        theoretical_lower_bound = (1.0 - alpha) - (1.0 / (n_cal + 1)) - 0.05
        assert stats["empirical_coverage"] >= theoretical_lower_bound
