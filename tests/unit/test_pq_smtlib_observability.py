"""
Unit tests for Post-Quantum Hybrid Attestation, SMT-LIB 2.6 Export, and Observability:
1. Post-Quantum Hybrid Digital Signature Scheme (`PQ-Hybrid-Ed25519-SHA256`) & SQLite WAL persistence
2. Formal SMT-LIB 2.6 Deontic Specification Exporter (`SMTLibDeonticExporter`) & `winning_canon`
3. Extended Prometheus Observability Metrics (keyring lifecycle, conformal calibration, abstentions)
4. REST API (`/v1/audit/smtlib`), MCP (`legal_memory_export_smtlib`, `legal_memory_rotate_key`), and CLI (`export-smtlib`, `key-rotate`)
"""

from __future__ import annotations

import asyncio
import json
import threading
import urllib.request
from pathlib import Path

from continuous_legal_memory.cli import main as cli_main
from continuous_legal_memory.domain.interfaces import BaseEncoderPort
from continuous_legal_memory.mcp import MCPServer
from continuous_legal_memory.orchestrator import LegalMemoryOrchestrator
from continuous_legal_memory.retrieval.conformal import SplitConformalCalibrator
from continuous_legal_memory.security.pq_hybrid import (
    PQ_HYBRID_ALGORITHM,
    PostQuantumHybridAttestationModule,
    pq_hybrid_sign,
    pq_hybrid_verify,
)
from continuous_legal_memory.server import create_server
from continuous_legal_memory.telemetry.observability import export_prometheus_metrics


def test_pq_hybrid_ed25519_wots_attestation_and_sqlite_persistence(
    offline_encoder: BaseEncoderPort,
    tmp_path: Path,
) -> None:
    """Verify PQ-Hybrid-Ed25519-SHA256 signing, dual verification, tamper rejection, and SQLite WAL persistence."""
    module = PostQuantumHybridAttestationModule(
        private_key="test-pq-seed-v1",
        key_id="pq-key-test-1",
    )
    assert module.algorithm == PQ_HYBRID_ALGORITHM
    assert len(bytes.fromhex(module.public_key_hex)) == 80

    payload = b'{"canonical":"legal-audit-bundle-v1.0.0"}'
    master_bytes = bytes.fromhex(module.private_key_hex)
    sig = pq_hybrid_sign(payload, master_bytes)
    assert ":" in sig
    assert pq_hybrid_verify(sig, payload, module.public_key_hex) is True
    assert pq_hybrid_verify(sig, b'{"canonical":"tampered"}', module.public_key_hex) is False

    # Tamper with classical Ed25519 half
    ed_part, wots_part = sig.split(":", 1)
    flipped_ed = ("0" if ed_part[0] != "0" else "1") + ed_part[1:]
    assert pq_hybrid_verify(f"{flipped_ed}:{wots_part}", payload, module.public_key_hex) is False

    # Tamper with WOTS+-SHA256 post-quantum half
    flipped_wots = wots_part[:-1] + ("0" if wots_part[-1] != "0" else "1")
    assert pq_hybrid_verify(f"{ed_part}:{flipped_wots}", payload, module.public_key_hex) is False

    # Verify KeyRing rotation to PQ-Hybrid-Ed25519-SHA256 and SQLite WAL persistence
    orch = LegalMemoryOrchestrator(encoder=offline_encoder, value_dim=2, engine_mode="structured")
    orch.update_memory(
        rule_text="Commercial delivery drones shall not operate at night in residential zones.",
        action_vector=[1.0, 0.0],
        authority_rank=8,
    )

    meta = orch.rotate_signing_key(
        new_key_id="pq-prod-2026",
        algorithm=PQ_HYBRID_ALGORITHM,
        key_material="pq-hybrid-production-seed-2026",
    )
    assert meta.key_id == "pq-prod-2026"
    assert meta.algorithm == PQ_HYBRID_ALGORITHM
    assert orch.keyring.active_key_id == "pq-prod-2026"

    pred = orch.predict("Are commercial delivery drones permitted to operate at night?")
    assert pred.attestation_token is not None
    assert pred.attestation_token.algorithm == PQ_HYBRID_ALGORITHM
    assert pred.attestation_token.key_id == "pq-prod-2026"
    assert ":" in pred.attestation_token.integrity_tag
    assert orch.keyring.verify_attestation(pred.attestation_token, pred) is True

    # Persist to SQLite WAL and reload into a fresh orchestrator
    db_file = str(tmp_path / "clm_pq_state.sqlite")
    orch.save_to_disk(db_file)

    reloaded = LegalMemoryOrchestrator(
        encoder=offline_encoder,
        value_dim=2,
        engine_mode="structured",
    )
    reloaded.load_from_disk(db_file)
    assert reloaded.keyring.active_key_id == "pq-prod-2026"
    assert reloaded.keyring.get_key_metadata("pq-prod-2026").algorithm == PQ_HYBRID_ALGORITHM
    assert reloaded.keyring.verify_attestation(pred.attestation_token, pred) is True


def test_smtlib_exporter_and_winning_canon_trace(
    offline_encoder: BaseEncoderPort,
) -> None:
    """Verify SMT-LIB 2.6 formal verification script generation and DefeasibleProofTrace.winning_canon."""
    orch = LegalMemoryOrchestrator(encoder=offline_encoder, value_dim=2, engine_mode="structured")
    orch.update_memory(
        rule_text="Commercial delivery drones are permitted to operate at night in residential zones.",
        action_vector=[0.0, 1.0],
        authority_rank=2,
        metadata={"record_id": "municipal_drone_permit", "specificity": 0.5},
    )
    orch.update_memory(
        rule_text="Commercial delivery drones are prohibited from operating at night in residential zones.",
        action_vector=[1.0, 0.0],
        authority_rank=9,
        metadata={"record_id": "federal_aviation_ban", "specificity": 0.8},
    )

    pred = orch.predict(
        "Commercial delivery drones operating at night in residential zones"
    )
    assert pred.proof_trace is not None
    assert pred.proof_trace["winning_canon"] == "LEX_SUPERIOR"
    assert pred.proof_trace["prevailing_record_id"] == "federal_aviation_ban"

    smt_bundle = orch.export_smtlib(
        query_text="Commercial delivery drones operating at night in residential zones"
    )
    assert smt_bundle.logic == "QF_UFLIA"
    assert smt_bundle.num_rules == 2
    assert smt_bundle.prevailing_record_id == "federal_aviation_ban"
    assert smt_bundle.winning_canon == "LEX_SUPERIOR"
    assert "(set-logic QF_UFLIA)" in smt_bundle.smtlib_script
    assert "(check-sat)" in smt_bundle.smtlib_script
    assert "(get-value ((prevails 0) (prevails 1)))" in smt_bundle.smtlib_script
    assert "federal_aviation_ban" in smt_bundle.smtlib_script
    assert "municipal_drone_permit" in smt_bundle.smtlib_script

    async_bundle = asyncio.run(
        orch.async_export_smtlib(
            query_text="Commercial delivery drones operating at night in residential zones"
        )
    )
    assert async_bundle.script_sha256 == smt_bundle.script_sha256


def test_extended_prometheus_observability_metrics(
    offline_encoder: BaseEncoderPort,
) -> None:
    """Verify keyring lifecycle, conformal calibration, and abstention counters in Prometheus metrics."""
    calibrator = SplitConformalCalibrator(alpha=0.10)
    calibrator.calibrate(
        action_vectors=[[1.0, 0.0], [0.95, 0.05], [0.9, 0.1]],
        true_labels=[0, 0, 0],
    )
    orch = LegalMemoryOrchestrator(
        encoder=offline_encoder,
        value_dim=2,
        engine_mode="structured",
        conformal_calibrator=calibrator,
    )
    orch.update_memory(
        rule_text="Data controllers must encrypt all financial records at rest.",
        action_vector=[1.0, 0.0],
        authority_rank=7,
    )

    initial_kid = orch.keyring.active_key_id
    orch.rotate_signing_key(new_key_id="ed-key-2", algorithm="Ed25519")
    orch.rotate_signing_key(new_key_id="pq-key-3", algorithm=PQ_HYBRID_ALGORITHM)
    orch.revoke_signing_key(initial_kid)

    # Trigger an epistemic abstention via strict similarity threshold
    orch.min_similarity_threshold = 0.999999
    abst_pred = orch.predict("Completely unrelated maritime salvage tariff regulation")
    assert abst_pred.abstained is True
    assert orch.abstained_predictions_count == 1
    orch.min_similarity_threshold = None

    metrics_text = export_prometheus_metrics(orch)
    assert 'clm_keyring_keys_total{status="ACTIVE"} 1' in metrics_text
    assert 'clm_keyring_keys_total{status="RETIRED"} 1' in metrics_text
    assert 'clm_keyring_keys_total{status="REVOKED"} 1' in metrics_text
    assert "clm_conformal_calibrated 1" in metrics_text
    assert "clm_conformal_q_hat " in metrics_text
    assert "clm_predictions_abstained_total 1" in metrics_text


def test_rest_mcp_and_cli_parity(
    offline_encoder: BaseEncoderPort,
    tmp_path: Path,
    capsys,
) -> None:
    """Verify REST /v1/audit/smtlib, MCP legal_memory_export_smtlib, and CLI export-smtlib + PQ key-rotate."""
    orch = LegalMemoryOrchestrator(encoder=offline_encoder, value_dim=2, engine_mode="structured")
    orch.update_memory(
        rule_text="Commercial delivery drones are prohibited from operating at night in residential zones.",
        action_vector=[1.0, 0.0],
        authority_rank=9,
        metadata={"record_id": "federal_aviation_ban"},
    )

    # 1. MCP Server verification
    mcp = MCPServer(orch)
    tool_names = {t["name"] for t in mcp.list_tools()}
    assert "legal_memory_export_smtlib" in tool_names

    rot_res = mcp.call_tool(
        "legal_memory_rotate_key",
        {"new_key_id": "mcp-pq-2026", "algorithm": PQ_HYBRID_ALGORITHM},
    )
    rot_payload = json.loads(rot_res["content"][0]["text"])
    assert rot_payload["algorithm"] == PQ_HYBRID_ALGORITHM

    smt_res = mcp.call_tool(
        "legal_memory_export_smtlib",
        {"query_text": "Commercial delivery drones operating at night"},
    )
    smt_payload = json.loads(smt_res["content"][0]["text"])
    assert smt_payload["logic"] == "QF_UFLIA"
    assert "(check-sat)" in smt_payload["smtlib_script"]

    # 2. REST API /v1/audit/smtlib verification
    server = create_server(orch, host="127.0.0.1", port=0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/audit/smtlib",
            data=json.dumps(
                {"query_text": "Commercial delivery drones operating at night"}
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            rest_data = json.loads(resp.read().decode("utf-8"))
        assert rest_data["logic"] == "QF_UFLIA"
        assert rest_data["prevailing_record_id"] == "federal_aviation_ban"
    finally:
        server.shutdown()
        server.server_close()

    # 3. CLI export-smtlib & key-rotate with PQ-Hybrid-Ed25519-SHA256
    db_file = str(tmp_path / "cli_r10.db")
    orch.save_to_disk(db_file)
    capsys.readouterr()

    rc_rot = cli_main(
        [
            "key-rotate",
            "--db-path",
            db_file,
            "--key-id",
            "cli-pq-key-99",
            "--algorithm",
            PQ_HYBRID_ALGORITHM,
            "--mock-encoder",
        ]
    )
    assert rc_rot == 0
    out_rot = json.loads(capsys.readouterr().out)
    assert out_rot["algorithm"] == PQ_HYBRID_ALGORITHM

    rc_smt = cli_main(
        [
            "export-smtlib",
            "--db-path",
            db_file,
            "--query",
            "Commercial delivery drones operating at night",
            "--mock-encoder",
        ]
    )
    assert rc_smt == 0
    out_smt = json.loads(capsys.readouterr().out)
    assert out_smt["logic"] == "QF_UFLIA"
    assert "(check-sat)" in out_smt["smtlib_script"]
