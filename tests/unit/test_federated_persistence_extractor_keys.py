"""
Unit tests for Federated Compliance Verification, Normative Graph Extraction, and Key Lifecycle:
1. Cross-Tenant Federated Normative Verification & Merkle State Diff (FederatedComplianceVerifier)
2. Full SQLite WAL Persistence for CryptographicKeyRing and SplitConformalCalibrator
3. Automated Normative Entity & Deontic Graph Extractor (NormativeGraphExtractor)
4. Key Lifecycle & Federated Verification across REST (/v1/keys/*, /v1/federated/verify),
   MCP (legal_memory_rotate_key, legal_memory_revoke_key, legal_memory_federated_verify),
   and CLI (clm key-list, clm key-rotate, clm key-revoke)
"""

from __future__ import annotations

import json
import threading
import urllib.request
from pathlib import Path

from continuous_legal_memory import (
    CryptographicKeyRing,
    EntityType,
    LegalMemoryOrchestrator,
    MCPServer,
    RelationType,
    SplitConformalCalibrator,
)
from continuous_legal_memory.adapters.mock_encoder import SemanticMockEncoder
from continuous_legal_memory.cli import main as cli_main
from continuous_legal_memory.server import create_server


def test_federated_merkle_state_diff_and_cross_party_alignment() -> None:
    """
    Verify MerkleStateDiff (added, removed, modified, unchanged rules) and
    CrossPartyAlignmentReport across sovereign tenant proofs.
    """
    orch_eu = LegalMemoryOrchestrator(encoder=SemanticMockEncoder(), seed=42)
    orch_br = LegalMemoryOrchestrator(encoder=SemanticMockEncoder(), seed=42)

    # Shared unchanged rule
    orch_eu.update_memory(
        "Shared Treaty Art. 1: Cross-border encryption is mandatory.",
        [1.0, 0.0],
        authority_rank=8,
        jurisdiction="EU",
        metadata={"record_id": "rule_shared"},
    )
    orch_br.update_memory(
        "Shared Treaty Art. 1: Cross-border encryption is mandatory.",
        [1.0, 0.0],
        authority_rank=8,
        jurisdiction="EU",
        metadata={"record_id": "rule_shared"},
    )

    # Rule modified in BR (different authority_rank)
    orch_eu.update_memory(
        "Data Retention Clause 5",
        [1.0, 0.0],
        authority_rank=5,
        metadata={"record_id": "rule_mod"},
    )
    orch_br.update_memory(
        "Data Retention Clause 5",
        [1.0, 0.0],
        authority_rank=7,
        metadata={"record_id": "rule_mod"},
    )

    # Rule only in EU (removed in BR) and rule only in BR (added in BR)
    orch_eu.update_memory(
        "EU Only Directive",
        [1.0, 0.0],
        authority_rank=6,
        metadata={"record_id": "rule_eu_only"},
    )
    orch_br.update_memory(
        "BR Only LGPD Provision",
        [1.0, 0.0],
        authority_rank=6,
        metadata={"record_id": "rule_br_only"},
    )

    bundle_br = orch_br.export_compliance_bundle()
    diff = orch_eu.diff_merkle_states(bundle_br)

    assert diff.roots_equal is False
    assert diff.bundle_b_valid is True
    assert diff.unchanged_rule_ids == ["rule_shared"]
    assert diff.modified_rule_ids == ["rule_mod"]
    assert diff.removed_rule_ids == ["rule_eu_only"]
    assert diff.added_rule_ids == ["rule_br_only"]

    # Cross-party zero-knowledge alignment check: aligned outcomes ([1.0, 0.0] and [1.0, 0.0])
    zk_eu = orch_eu.generate_compliance_proof("rule_shared", min_required_authority_rank=5)
    zk_br = orch_br.generate_compliance_proof("rule_br_only", min_required_authority_rank=5)

    aligned_report = orch_eu.verify_cross_party_alignment(
        proofs=[zk_eu.proof, zk_br.proof],
        min_authority_rank=5,
        min_action_alignment=0.5,
    )
    assert aligned_report.is_aligned is True
    assert aligned_report.parties_count == 2
    assert aligned_report.all_proofs_valid is True
    assert aligned_report.min_pairwise_action_alignment > 0.99
    assert len(aligned_report.conflicting_pairs) == 0

    # Conflicting party outcome ([0.0, 1.0] vs [1.0, 0.0])
    orch_conflict = LegalMemoryOrchestrator(encoder=SemanticMockEncoder(), seed=42)
    orch_conflict.update_memory(
        "Conflicting Processor Policy: Prohibition of encryption.",
        [0.0, 1.0],
        authority_rank=8,
        metadata={"record_id": "rule_conflict"},
    )
    zk_conflict = orch_conflict.generate_compliance_proof("rule_conflict", min_required_authority_rank=5)
    conflict_report = orch_eu.verify_cross_party_alignment(
        proofs=[zk_eu.proof, zk_conflict.proof],
        min_authority_rank=5,
        min_action_alignment=0.5,
    )
    assert conflict_report.is_aligned is False
    assert len(conflict_report.conflicting_pairs) == 1


def test_sqlite_persistence_of_keyring_and_conformal_calibrator(tmp_path: Path) -> None:
    """
    Verify that SqliteMemoryStore persists and restores CryptographicKeyRing
    (ACTIVE, RETIRED, REVOKED keys) and SplitConformalCalibrator quantile state.
    """
    db_path = tmp_path / "keyring_conformal.db"
    keyring = CryptographicKeyRing(initial_algorithm="Ed25519", initial_key_id="k1")
    calibrator = SplitConformalCalibrator(alpha=0.15, abstain_on_empty=True)
    q_hat = calibrator.calibrate(
        action_vectors=[[0.9, 0.1], [0.1, 0.9], [0.85, 0.15], [0.2, 0.8]],
        true_labels=[0, 1, 0, 1],
    )

    orch = LegalMemoryOrchestrator(
        encoder=SemanticMockEncoder(),
        attestor=keyring,
        conformal_calibrator=calibrator,
        seed=42,
    )
    orch.update_memory("Statute 1: Mandatory compliance.", [1.0, 0.0], authority_rank=7)

    pred_k1 = orch.predict("Mandatory compliance")
    tok_k1 = pred_k1.attestation_token
    assert tok_k1 is not None and tok_k1.key_id == "k1"

    # Rotate to k2 (k1 becomes RETIRED), then rotate to k3 and revoke k2
    orch.rotate_signing_key(new_key_id="k2")
    pred_k2 = orch.predict("Mandatory compliance")
    tok_k2 = pred_k2.attestation_token
    assert tok_k2 is not None and tok_k2.key_id == "k2"

    orch.rotate_signing_key(new_key_id="k3")
    orch.revoke_signing_key("k2")

    orch.save_to_disk(db_path)

    # Reload into a brand-new orchestrator
    reloaded = LegalMemoryOrchestrator(encoder=SemanticMockEncoder(), seed=42)
    reloaded.load_from_disk(db_path)

    assert reloaded.keyring.active_key_id == "k3"
    status_map = {k["key_id"]: k["status"] for k in reloaded.list_signing_keys()}
    assert status_map == {"k1": "RETIRED", "k2": "REVOKED", "k3": "ACTIVE"}

    # Retired k1 token must verify; revoked k2 token must be rejected
    assert reloaded.keyring.verify_attestation(tok_k1, pred_k1, retrieved_text=pred_k1.most_relevant_rule) is True
    assert reloaded.keyring.verify_attestation(tok_k2, pred_k2, retrieved_text=pred_k2.most_relevant_rule) is False

    # Conformal calibrator must be restored with exact quantile threshold
    assert reloaded.conformal_calibrator is not None
    assert reloaded.conformal_calibrator.is_calibrated is True
    assert abs(reloaded.conformal_calibrator.quantile_threshold - q_hat) < 1e-9
    assert abs(reloaded.conformal_calibrator.alpha - 0.15) < 1e-9


def test_normative_graph_extractor_entity_types_and_citation_edges() -> None:
    """
    Verify that NormativeGraphExtractor automatically classifies EntityType (OBLIGATION,
    PROHIBITION, PERMISSION) and links SUPERSEDES / DEPENDS_ON edges from text.
    """
    orch = LegalMemoryOrchestrator(encoder=SemanticMockEncoder(), seed=42)

    orch.update_memory(
        "Article 32 GDPR: Controllers shall implement encryption for personal data.",
        [1.0, 0.0],
        metadata={"node_id": "gdpr_32_base"},
    )
    orch.update_memory(
        "Pursuant to Article 32 GDPR, processors may use hardware security modules.",
        [1.0, 0.0],
        metadata={"node_id": "gdpr_32_perm"},
    )
    orch.update_memory(
        "Emergency Amendment that supersedes gdpr_32_base: Controllers must not use deprecated ciphers.",
        [0.0, 1.0],
        metadata={"node_id": "gdpr_32_amend"},
    )

    assert orch.semantic_graph.nodes["gdpr_32_base"].entity_type == EntityType.OBLIGATION
    assert orch.semantic_graph.nodes["gdpr_32_perm"].entity_type == EntityType.PERMISSION
    assert orch.semantic_graph.nodes["gdpr_32_amend"].entity_type == EntityType.PROHIBITION

    edge_tuples = {
        (e.source_id, e.target_id, e.relation_type)
        for e in orch.semantic_graph.edges
    }
    # Shared citation "Article 32 GDPR" links gdpr_32_perm -> gdpr_32_base via DEPENDS_ON
    assert ("gdpr_32_perm", "gdpr_32_base", RelationType.DEPENDS_ON) in edge_tuples
    # Textual clause "supersedes gdpr_32_base" links gdpr_32_amend -> gdpr_32_base via SUPERSEDES
    assert ("gdpr_32_amend", "gdpr_32_base", RelationType.SUPERSEDES) in edge_tuples


def test_rest_mcp_and_cli_key_and_federated_endpoints(tmp_path: Path) -> None:
    """
    Verify REST (/v1/keys, /v1/keys/rotate, /v1/keys/revoke, /v1/federated/verify),
    MCP (legal_memory_rotate_key, legal_memory_revoke_key, legal_memory_federated_verify),
    and CLI (key-list, key-rotate, key-revoke).
    """
    db_path = tmp_path / "cli_keys.db"
    assert cli_main([
        "ingest",
        "--text", "Controllers shall maintain audit logs.",
        "--action", "1.0", "0.0",
        "--db-path", str(db_path),
        "--mock-encoder",
    ]) == 0

    assert cli_main(["key-rotate", "--db-path", str(db_path), "--key-id", "cli-key-v2", "--mock-encoder"]) == 0
    assert cli_main(["key-list", "--db-path", str(db_path), "--mock-encoder"]) == 0
    assert cli_main(["key-revoke", "clm-key-v1", "--db-path", str(db_path), "--mock-encoder"]) == 0

    orch = LegalMemoryOrchestrator(encoder=SemanticMockEncoder())
    orch.load_from_disk(db_path)
    rec_id = orch.episodic_memory.get_records()[0].record_id
    assert rec_id is not None

    zk_bundle = orch.generate_compliance_proof(rec_id, min_required_authority_rank=1)

    # MCP tool calls
    mcp = MCPServer(orch)
    rot_resp = mcp.handle_request({
        "jsonrpc": "2.0",
        "id": 81,
        "method": "tools/call",
        "params": {"name": "legal_memory_rotate_key", "arguments": {"new_key_id": "mcp-key-v3"}},
    })
    assert rot_resp is not None
    assert json.loads(rot_resp["result"]["content"][0]["text"])["key_id"] == "mcp-key-v3"

    fed_resp = mcp.handle_request({
        "jsonrpc": "2.0",
        "id": 82,
        "method": "tools/call",
        "params": {
            "name": "legal_memory_federated_verify",
            "arguments": {"proofs": [zk_bundle.proof.to_dict()]},
        },
    })
    assert fed_resp is not None
    assert json.loads(fed_resp["result"]["content"][0]["text"])["is_aligned"] is True

    # REST endpoints
    server = create_server(orch, host="127.0.0.1", port=0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/keys", timeout=5) as resp:
            keys_payload = json.loads(resp.read().decode("utf-8"))
        assert keys_payload["active_key_id"] == "mcp-key-v3"

        req_fed = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/federated/verify",
            data=json.dumps({"proofs": [zk_bundle.proof.to_dict()]}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req_fed, timeout=5) as resp:
            http_fed = json.loads(resp.read().decode("utf-8"))
        assert http_fed["is_aligned"] is True
    finally:
        server.shutdown()
        server.server_close()
