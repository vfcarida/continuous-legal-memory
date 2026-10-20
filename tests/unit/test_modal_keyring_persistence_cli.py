"""
Unit tests for KD Modal Satisfiability, Cryptographic KeyRing Lifecycle, and SQLite Persistence:
1. Formal Deontic Modal Satisfiability (KD) & Pre-Flight Rule Ingestion Simulator
2. Cryptographic Key Lifecycle & Rotation Ring (CryptographicKeyRing: ACTIVE, RETIRED, REVOKED)
3. Full Bi-Temporal GraphNode.transaction_time Persistence in SqliteMemoryStore
4. CLI Subcommands (compliance-proof, export-bundle, restore-bundle, simulate-rule),
   REST (/v1/rules/simulate), MCP (legal_memory_simulate_rule), and Prometheus Merkle Gauges
"""

from __future__ import annotations

import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from continuous_legal_memory import (
    CryptographicKeyRing,
    DeonticModality,
    LegalMemoryOrchestrator,
    MCPServer,
    RelationType,
    check_kd_modal_conflict,
    infer_deontic_modality,
)
from continuous_legal_memory.adapters.mock_encoder import SemanticMockEncoder
from continuous_legal_memory.cli import main as cli_main
from continuous_legal_memory.server import create_server
from continuous_legal_memory.telemetry.observability import export_prometheus_metrics


def test_deontic_modal_kd_verifier_and_preflight_simulation() -> None:
    """
    Verify Standard Deontic Logic (KD) conflict detection (O(p) ^ F(p), F(p) ^ P(p)),
    defeasible shadowing detection, and non-mutating pre-flight simulation.
    """
    assert infer_deontic_modality("Processors shall encrypt all cross-border telemetry.") == DeonticModality.OBLIGATION
    assert infer_deontic_modality("Processors must not export unencrypted telemetry.") == DeonticModality.PROHIBITION
    assert infer_deontic_modality("Processors may archive anonymized logs.") == DeonticModality.PERMISSION

    has_of, code_of = check_kd_modal_conflict(DeonticModality.OBLIGATION, DeonticModality.PROHIBITION)
    assert has_of is True and code_of is not None and "KD_AXIOM_D" in code_of

    has_fp, code_fp = check_kd_modal_conflict(DeonticModality.PROHIBITION, DeonticModality.PERMISSION)
    assert has_fp is True and code_fp is not None and "KD_DUALITY" in code_fp

    has_op, code_op = check_kd_modal_conflict(DeonticModality.OBLIGATION, DeonticModality.PERMISSION)
    assert has_op is False and code_op is None

    orch = LegalMemoryOrchestrator(encoder=SemanticMockEncoder(), seed=42)
    t_same = datetime(2026, 1, 1, tzinfo=timezone.utc)
    orch.update_memory(
        rule_text="Article 32 GDPR: Controllers shall encrypt all biometric customer records.",
        action_vector=[1.0, 0.0],
        authority_rank=8,
        valid_from=t_same,
        metadata={"record_id": "gdpr_art32", "node_id": "gdpr_art32", "specificity": 0.5},
    )
    initial_count = len(orch.episodic_memory.get_records())
    initial_root = orch.merkle_tree.root_hex

    # 1. Pre-flight simulation of a horizontal antinomy (same authority_rank=8, same specificity=0.5, same valid_from)
    sim_conflict = orch.simulate_rule_ingestion(
        rule_text="Article 32 GDPR: Controllers must not encrypt biometric customer records.",
        action_vector=[0.0, 1.0],
        authority_rank=8,
        valid_from=t_same,
        metadata={"record_id": "cand_conflict", "specificity": 0.5},
    )
    assert sim_conflict.is_satisfiable is False
    assert sim_conflict.candidate_modality == "F"
    assert len(sim_conflict.unresolved_conflicts) == 1
    assert "KD_AXIOM_D" in sim_conflict.unresolved_conflicts[0].kd_violation

    # 2. Pre-flight simulation of a subordinate policy (authority_rank=2) shadowed by gdpr_art32 (rank=8)
    sim_shadowed = orch.simulate_rule_ingestion(
        rule_text="Article 32 GDPR: Controllers are prohibited from encrypting biometric customer records.",
        action_vector=[0.0, 1.0],
        authority_rank=2,
        valid_from=t_same,
        metadata={"record_id": "sub_policy", "specificity": 0.1},
    )
    assert sim_shadowed.is_satisfiable is True
    assert "gdpr_art32" in sim_shadowed.shadowed_by
    assert len(sim_shadowed.resolved_overrides) == 1
    assert "LEX_SUPERIOR" in sim_shadowed.resolved_overrides[0].governing_canon

    # 3. Pre-flight simulation of a self-cycle in proposed graph relations
    sim_cycle = orch.simulate_rule_ingestion(
        rule_text="Controllers may audit logs.",
        action_vector=[1.0, 0.0],
        authority_rank=5,
        metadata={
            "node_id": "self_cycle_node",
            "relations": [{"target_id": "self_cycle_node", "relation_type": RelationType.SUPERSEDES.value}],
        },
    )
    assert sim_cycle.is_satisfiable is False
    assert sim_cycle.would_create_cycle is True

    # Ensure zero state mutation occurred during simulations
    assert len(orch.episodic_memory.get_records()) == initial_count
    assert orch.merkle_tree.root_hex == initial_root


def test_cryptographic_keyring_rotation_and_revocation() -> None:
    """
    Verify that CryptographicKeyRing supports atomic key rotation (retiring old keys while
    keeping historical tokens verifiable) and immediate key revocation (rejecting compromised keys).
    """
    keyring = CryptographicKeyRing(initial_algorithm="Ed25519", initial_key_id="key-v1")
    orch = LegalMemoryOrchestrator(
        encoder=SemanticMockEncoder(),
        attestor=keyring,
        seed=42,
    )
    orch.update_memory(
        rule_text="Statute A: Mandatory retention of audit logs.",
        action_vector=[1.0, 0.0],
        authority_rank=7,
    )

    # 1. Sign prediction with initial active key (key-v1)
    pred_v1 = orch.predict("Mandatory retention of audit logs")
    tok_v1 = pred_v1.attestation_token
    assert tok_v1 is not None
    assert tok_v1.key_id == "key-v1"
    assert keyring.verify_attestation(tok_v1, pred_v1, retrieved_text=pred_v1.most_relevant_rule) is True

    # 2. Rotate to key-v2; key-v1 becomes RETIRED
    meta_v2 = orch.rotate_signing_key(new_key_id="key-v2")
    assert meta_v2.key_id == "key-v2"
    assert meta_v2.status == "ACTIVE"

    keys_list = {k["key_id"]: k["status"] for k in orch.list_signing_keys()}
    assert keys_list["key-v1"] == "RETIRED"
    assert keys_list["key-v2"] == "ACTIVE"

    # Historical token signed by retired key-v1 MUST still verify
    assert keyring.verify_attestation(tok_v1, pred_v1, retrieved_text=pred_v1.most_relevant_rule) is True

    # New prediction is signed by key-v2
    pred_v2 = orch.predict("Mandatory retention of audit logs")
    tok_v2 = pred_v2.attestation_token
    assert tok_v2 is not None
    assert tok_v2.key_id == "key-v2"
    assert keyring.verify_attestation(tok_v2, pred_v2, retrieved_text=pred_v2.most_relevant_rule) is True

    # 3. Revoke compromised key-v1; tokens signed by key-v1 MUST now be rejected
    revoked_meta = orch.revoke_signing_key("key-v1")
    assert revoked_meta.status == "REVOKED"
    assert keyring.verify_attestation(tok_v1, pred_v1, retrieved_text=pred_v1.most_relevant_rule) is False
    # Token signed by active key-v2 remains valid
    assert keyring.verify_attestation(tok_v2, pred_v2, retrieved_text=pred_v2.most_relevant_rule) is True


def test_bitemporal_graph_node_transaction_time_sqlite_persistence(tmp_path: Path) -> None:
    """
    Verify that SqliteMemoryStore persists and restores GraphNode.transaction_time
    bit-for-bit so post-reload associate_statutes(as_of_transaction_time=...) is exact.
    """
    db_file = tmp_path / "bitemporal_graph.db"
    orch = LegalMemoryOrchestrator(encoder=SemanticMockEncoder(), seed=42)

    tx_2025 = datetime(2025, 3, 15, 12, 0, tzinfo=timezone.utc)
    tx_2026 = datetime(2026, 8, 20, 15, 30, tzinfo=timezone.utc)

    orch.update_memory(
        rule_text="Base Statute 2025",
        action_vector=[1.0, 0.0],
        transaction_time=tx_2025,
        metadata={"node_id": "node_2025", "record_id": "node_2025"},
    )
    orch.update_memory(
        rule_text="Amendment Statute 2026",
        action_vector=[0.0, 1.0],
        transaction_time=tx_2026,
        metadata={
            "node_id": "node_2026",
            "record_id": "node_2026",
            "relations": [{"target_id": "node_2025", "relation_type": RelationType.DEPENDS_ON.value}],
        },
    )

    orch.save_to_disk(db_file)

    reloaded = LegalMemoryOrchestrator(encoder=SemanticMockEncoder(), seed=42)
    reloaded.load_from_disk(db_file)

    assert reloaded.semantic_graph.nodes["node_2025"].transaction_time == tx_2025
    assert reloaded.semantic_graph.nodes["node_2026"].transaction_time == tx_2026

    # Historical PageRank as of 2025 must exclude node_2026
    cutoff_2025 = datetime(2025, 12, 31, tzinfo=timezone.utc)
    assoc_2025 = {
        n.node_id: s
        for n, s in reloaded.associate_statutes(
            seed_nodes=["node_2025"],
            as_of_transaction_time=cutoff_2025,
        )
    }
    assert "node_2025" in assoc_2025
    assert "node_2026" not in assoc_2025


def test_rest_mcp_cli_and_prometheus_merkle_gauges(tmp_path: Path) -> None:
    """
    Verify REST /v1/rules/simulate, MCP legal_memory_simulate_rule, Prometheus Sparse Merkle
    gauges, and CLI subcommands (compliance-proof, export-bundle, restore-bundle, simulate-rule).
    """
    db_path = tmp_path / "cli_round7.db"
    bundle_path = tmp_path / "snapshot.clm.bundle.json"
    restored_db = tmp_path / "restored_round7.db"

    # 1. Ingest a rule via CLI
    rc_ingest = cli_main([
        "ingest",
        "--text", "Data controllers shall encrypt all biometric records.",
        "--action", "1.0", "0.0",
        "--authority-rank", "8",
        "--jurisdiction", "EU",
        "--db-path", str(db_path),
        "--mock-encoder",
    ])
    assert rc_ingest == 0

    orch = LegalMemoryOrchestrator(encoder=SemanticMockEncoder())
    orch.load_from_disk(db_path)
    rec_id = orch.episodic_memory.get_records()[0].record_id
    assert rec_id is not None

    # 2. Check Prometheus Sparse Merkle gauges
    prom_text = export_prometheus_metrics(orch)
    assert "clm_merkle_active_leaves 1" in prom_text
    assert "clm_merkle_audit_log_entries_total 1" in prom_text
    assert "clm_merkle_chain_verified 1" in prom_text

    # 3. Run CLI compliance-proof, simulate-rule, export-bundle, and restore-bundle
    assert cli_main([
        "compliance-proof",
        rec_id,
        "--db-path", str(db_path),
        "--min-authority-rank", "5",
        "--mock-encoder",
    ]) == 0

    assert cli_main([
        "simulate-rule",
        "--text", "Data controllers must not encrypt biometric records.",
        "--action", "0.0", "1.0",
        "--authority-rank", "8",
        "--db-path", str(db_path),
        "--mock-encoder",
    ]) == 0

    assert cli_main([
        "export-bundle",
        "--db-path", str(db_path),
        "--output", str(bundle_path),
        "--mock-encoder",
    ]) == 0
    assert bundle_path.exists()

    assert cli_main([
        "restore-bundle",
        "--bundle-file", str(bundle_path),
        "--db-path", str(restored_db),
        "--mock-encoder",
    ]) == 0
    assert restored_db.exists()

    # 4. Verify MCP legal_memory_simulate_rule and REST /v1/rules/simulate
    mcp = MCPServer(orch)
    mcp_resp = mcp.handle_request({
        "jsonrpc": "2.0",
        "id": 77,
        "method": "tools/call",
        "params": {
            "name": "legal_memory_simulate_rule",
            "arguments": {
                "rule_text": "Data controllers are prohibited from encrypting biometric records.",
                "action_vector": [0.0, 1.0],
                "authority_rank": 3,
            },
        },
    })
    assert mcp_resp is not None
    sim_data = json.loads(mcp_resp["result"]["content"][0]["text"])
    assert sim_data["is_satisfiable"] is True
    assert rec_id in sim_data["shadowed_by"]

    server = create_server(orch, host="127.0.0.1", port=0)
    port = server.server_address[1]
    import threading
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/rules/simulate",
            data=json.dumps({
                "rule_text": "Data controllers are prohibited from encrypting biometric records.",
                "action_vector": [0.0, 1.0],
                "authority_rank": 3,
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            http_sim = json.loads(resp.read().decode("utf-8"))
        assert http_sim["is_satisfiable"] is True
        assert rec_id in http_sim["shadowed_by"]
    finally:
        server.shutdown()
        server.server_close()
