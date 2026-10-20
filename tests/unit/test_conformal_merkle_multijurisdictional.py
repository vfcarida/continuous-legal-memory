"""
Unit tests for Split-Conformal Calibration, Sparse Merkle Proofs, and Multi-Jurisdictional Benchmarks:
- Calibrated Cross-Encoder Legal Reranking & Split-Conformal Prediction/Abstention
- 540-Case Multi-Jurisdictional Statutory Graph Benchmark & RAGAS/DeepEval Adapters
- 256-Bit Sparse Merkle Tree Non-Membership Erasure Proofs & Append-Only Merkle Log
"""

from __future__ import annotations

import json

import pytest
import torch

from continuous_legal_memory import (
    AppendOnlyMerkleLog,
    AsymmetricAttestationModule,
    DeonticCrossEncoderReranker,
    LegalMemoryOrchestrator,
    MCPServer,
    MemoryRecord,
    SparseMerkleTree,
    SplitConformalCalibrator,
    evaluate_multijurisdictional_suite,
    generate_multijurisdictional_benchmark,
    to_deepeval_test_cases,
    to_ragas_records,
    verify_sparse_merkle_proof,
)
from continuous_legal_memory.domain.interfaces import BaseEncoderPort


def test_split_conformal_calibration_and_coverage_guarantee() -> None:
    """
    Verify finite-sample Split Conformal Prediction calibration, marginal coverage
    guarantee (>= 1 - alpha), and similarity-gated empty-set abstention.
    """
    calibrator = SplitConformalCalibrator(alpha=0.10, min_similarity_gate=0.30)

    # Calibration set: 20 high-confidence samples where true label has prob ~0.85..0.95
    cal_vecs = [[0.10, 0.90] if i % 2 == 1 else [0.88, 0.12] for i in range(20)]
    cal_labels = [1 if i % 2 == 1 else 0 for i in range(20)]
    cal_sims = [0.65] * 20

    q_hat = calibrator.calibrate(cal_vecs, cal_labels, max_similarities=cal_sims)
    assert calibrator.is_calibrated
    assert 0.05 <= q_hat <= 0.20

    # Test evaluation on exchangeable samples
    metrics = calibrator.evaluate_empirical_coverage(cal_vecs, cal_labels, max_similarities=cal_sims)
    assert metrics["empirical_coverage"] >= 0.90
    assert metrics["singleton_rate"] == 1.0
    assert metrics["empty_set_abstention_rate"] == 0.0

    # Out-of-domain query with low cosine similarity (0.05 < 0.30 gate) -> empty set abstention
    ood_out = calibrator.predict_set([0.10, 0.90], max_similarity=0.05)
    assert ood_out.is_empty
    assert ood_out.prediction_set == []


def test_orchestrator_conformal_abstention_and_ambiguity(token_encoder: BaseEncoderPort) -> None:
    """
    Verify that LegalMemoryOrchestrator integrates SplitConformalCalibrator for singleton
    predictions, out-of-domain empty-set abstention, and ambiguity detection.
    """
    calibrator = SplitConformalCalibrator(
        alpha=0.10,
        min_similarity_gate=0.28,
        abstain_on_empty=True,
        abstain_on_ambiguous=True,
        default_quantile=0.55,
    )
    orch = LegalMemoryOrchestrator(
        encoder=token_encoder,
        seed=42,
        engine_mode="structured",
        conformal_calibrator=calibrator,
    )

    orch.update_memory(
        "GDPR Article 17 mandatory erasure of revoked biometric consent templates.",
        [0.0, 1.0],
        authority_rank=5,
        jurisdiction="EU",
    )

    # 1. Clear in-domain match -> singleton conformal set [1]
    res_in = orch.predict("What does GDPR Article 17 require for revoked biometric consent templates?")
    assert not res_in.abstained
    assert res_in.conformal_set == [1]
    assert res_in.conformal_alpha == 0.10

    # 2. Out-of-domain query -> empty conformal set triggers abstention
    res_ood = orch.predict("How many tablespoons of olive oil go into a traditional focaccia dough?")
    assert res_ood.abstained
    assert res_ood.abstention_reason == "conformal_empty_set"
    assert res_ood.conformal_set == []
    assert res_ood.predicted_action_vector == [0.0, 0.0]


def test_deontic_cross_encoder_reranker_pinpoint_citations(offline_encoder: BaseEncoderPort) -> None:
    """
    Verify that DeonticCrossEncoderReranker boosts exact statutory citations and matching
    deontic modalities during retrieval.
    """
    reranker = DeonticCrossEncoderReranker(blend_weight=0.45)
    rec_general = MemoryRecord(
        text="General corporate record guideline: business files may be archived periodically.",
        key_vector=torch.ones((1, offline_encoder.embedding_dim)),
        value_vector=torch.tensor([[1.0, 0.0]]),
        authority_rank=3,
    )
    rec_pinpoint = MemoryRecord(
        text="Pursuant to GDPR Article 17 and 45 CFR 164: personal biometric logs must be erased immediately.",
        key_vector=torch.ones((1, offline_encoder.embedding_dim)),
        value_vector=torch.tensor([[0.0, 1.0]]),
        authority_rank=3,
    )

    scores = reranker.score_pairs(
        "Under GDPR Article 17, when must personal biometric logs be erased?",
        [rec_general, rec_pinpoint],
    )
    assert scores.shape == (2,)
    assert float(scores[1].item()) > float(scores[0].item()) + 0.30


def test_sparse_merkle_tree_inclusion_and_non_membership_proofs() -> None:
    """
    Verify 256-bit Sparse Merkle Tree inclusion proofs, physical erasure non-membership proofs,
    Ed25519 signature binding, tamper rejection, and AppendOnlyMerkleLog chain integrity.
    """
    smt = SparseMerkleTree()
    attestor = AsymmetricAttestationModule()

    root_empty = smt.root_hex
    root_1 = smt.update("rule_eu_01", "GDPR Art 17 payload", tenant_id="eu_tenant")
    root_2 = smt.update("rule_us_02", "HIPAA 45 CFR 164 payload", tenant_id="us_tenant")
    assert root_1 != root_empty
    assert root_2 != root_1
    assert smt.active_count == 2

    # 1. Inclusion proof for rule_eu_01
    inc_proof = smt.prove_inclusion("rule_eu_01", attestor=attestor)
    assert inc_proof.is_member
    assert verify_sparse_merkle_proof(inc_proof, expected_root_hash=root_2, attestor=attestor)

    # 2. Non-membership proof for a rule that is currently active must raise ValueError
    with pytest.raises(ValueError, match="currently active"):
        smt.prove_non_membership("rule_eu_01")

    # 3. Physically erase rule_eu_01 and generate succinct non-membership proof
    root_after_erase = smt.remove("rule_eu_01", tenant_id="eu_tenant")
    assert smt.active_count == 1
    non_mem_proof = smt.prove_non_membership("rule_eu_01", attestor=attestor)
    assert not non_mem_proof.is_member
    assert non_mem_proof.leaf_value_hash == "00" * 32
    assert verify_sparse_merkle_proof(
        non_mem_proof.to_dict(),
        expected_root_hash=root_after_erase,
        attestor=attestor,
    )

    # 4. Remaining rule_us_02 is still provably included under root_after_erase
    inc_us = smt.prove_inclusion("rule_us_02", attestor=attestor)
    assert verify_sparse_merkle_proof(inc_us, expected_root_hash=root_after_erase, attestor=attestor)

    # 5. Tamper detection: flipping is_member or root_hash invalidates proof
    tampered = non_mem_proof.to_dict()
    tampered["is_member"] = True
    assert not verify_sparse_merkle_proof(tampered, attestor=attestor)

    # 6. Verify append-only Merkle log chain
    assert len(smt.audit_log.entries) == 3
    assert smt.audit_log.verify_chain()


def test_orchestrator_and_mcp_erasure_merkle_proof(offline_encoder: BaseEncoderPort) -> None:
    """
    Verify that LegalMemoryOrchestrator.delete_rule() and MCPServer produce and verify
    Sparse Merkle Tree non-membership erasure proofs.
    """
    orch = LegalMemoryOrchestrator(
        encoder=offline_encoder,
        seed=42,
        enable_attestation=True,
    )
    server = MCPServer(orchestrator=orch)

    rec = orch.update_memory(
        "Client personal biometric consent preference",
        [1.0, 0.0],
        personal_data=True,
        metadata={"node_id": "gdpr_subj_99"},
    )
    assert rec.record_id is not None

    # Inclusion proof works while active
    inc = orch.prove_inclusion(rec.record_id)
    assert inc.is_member
    assert orch.verify_merkle_proof(inc)

    # Erase via MCP tool call
    erase_resp = server.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 10,
            "method": "tools/call",
            "params": {
                "name": "legal_memory_erase",
                "arguments": {"record_id": rec.record_id},
            },
        }
    )
    assert erase_resp is not None
    audit = json.loads(erase_resp["result"]["content"][0]["text"])
    assert audit["status"] == "ERASED"
    assert "sparse_merkle_tree" in audit["tiers_cleared"]
    proof_dict = audit["merkle_non_membership_proof"]
    assert proof_dict["is_member"] is False

    # Verify the non-membership proof via MCP legal_memory_verify
    verify_resp = server.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 11,
            "method": "tools/call",
            "params": {
                "name": "legal_memory_verify",
                "arguments": {
                    "merkle_proof": proof_dict,
                    "expected_root_hash": audit["merkle_root"],
                },
            },
        }
    )
    assert verify_resp is not None
    verify_payload = json.loads(verify_resp["result"]["content"][0]["text"])
    assert verify_payload["valid"] is True
    assert verify_payload["proof_type"] == "sparse_merkle_tree"
    assert verify_payload["is_member"] is False


def test_multijurisdictional_540_case_generation_and_evaluation(token_encoder: BaseEncoderPort) -> None:
    """
    Verify that the multi-jurisdictional benchmark generator produces 540 cases across
    EU, US, and BR, and that the evaluation suite and RAGAS/DeepEval adapters succeed.
    """
    full_cases = generate_multijurisdictional_benchmark(domains_per_category=30)
    assert len(full_cases) == 540

    jurisdictions = {c.jurisdiction for c in full_cases}
    categories = {c.category for c in full_cases}
    assert jurisdictions == {"EU", "US", "BR"}
    assert categories == {
        "lex_superior",
        "lex_specialis",
        "lex_posterior",
        "temporal_expiry",
        "gdpr_erasure",
        "epistemic_abstention",
    }

    # Execute evaluation suite on 3 domains per category (3 * 6 * 3 = 54 cases) for fast unit test
    report = evaluate_multijurisdictional_suite(
        domains_per_category=3,
        encoder=token_encoder,
    )
    assert report["total_cases"] == 54
    assert report["clm_accuracy"] >= 0.95
    assert report["clm_accuracy"] > report["plain_rag_accuracy"] + 0.40
    assert report["smt_erasure_verification_rate"] == 1.0
    assert report["abstention_f1"] >= 0.95

    # Verify RAGAS and DeepEval schema adapters
    ragas_samples = report["ragas_samples"]
    deepeval_cases = report["deepeval_test_cases"]
    assert len(ragas_samples) == 12
    assert len(deepeval_cases) == 12
    assert {"user_input", "retrieved_contexts", "response", "reference", "metadata"} <= set(
        ragas_samples[0].keys()
    )
    assert {"input", "actual_output", "expected_output", "retrieval_context", "additional_metadata"} <= set(
        deepeval_cases[0].keys()
    )
    assert isinstance(to_ragas_records([]), list)
    assert isinstance(to_deepeval_test_cases([]), list)
    assert isinstance( AppendOnlyMerkleLog(), AppendOnlyMerkleLog)
