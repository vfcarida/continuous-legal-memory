"""
Unit and integration tests for:
1. RFC 3161-inspired Cryptographic Timestamp Authority (`TimestampToken`) and
   Decentralized Transparency Log Gossip Auditor (`TransparencyGossipAuditor`, `SplitViewAuditReport`).
2. Automated Statutory Diff & Downstream Amendment Impact Analyzer (`StatutoryAmendmentAnalyzer`,
   `ClauseDiffSpan`, `DownstreamObligationImpact`, `AmendmentImpactReport`).
3. OpenAPI 3.1.0 Specification (`GET /v1/openapi.json`) and REST, MCP, and CLI parity.
"""

from __future__ import annotations

import asyncio
import json
import threading
import urllib.request
from pathlib import Path

from continuous_legal_memory import (
    GossipCheckpoint,
    LegalMemoryOrchestrator,
    MCPServer,
    PostQuantumHybridAttestationModule,
    TimestampToken,
    TransparencyGossipAuditor,
)
from continuous_legal_memory.adapters.mock_encoder import SemanticMockEncoder
from continuous_legal_memory.cli import main as cli_main
from continuous_legal_memory.domain.models import EntityType, RelationType
from continuous_legal_memory.server import LegalMemoryHTTPServer, generate_openapi_spec


def test_rfc3161_timestamp_token_and_gossip_split_view_detection() -> None:
    """
    Verify RFC 3161 TimestampToken issuance, PQ-Hybrid verification, tamper rejection,
    and cross-witness split-view / equivocation detection.
    """
    pq_attestor = PostQuantumHybridAttestationModule(private_key=b"tsa-pq-seed-2026-000000000000000")
    orch = LegalMemoryOrchestrator(
        encoder=SemanticMockEncoder(embedding_dim=32),
        value_dim=2,
        attestor=pq_attestor,
    )

    r1 = orch.update_memory(
        rule_text="Article 12 EU AI Act: High-risk AI systems shall maintain automatic logs.",
        action_vector=[0.0, 1.0],
        authority_rank=9,
        jurisdiction="EU",
    )
    assert r1.record_id

    tst = orch.issue_timestamp_token(nonce="audit-nonce-001")
    assert tst.signature_algorithm == "PQ-Hybrid-Ed25519-SHA256"
    assert tst.sequence_number == 1
    assert tst.merkle_root == orch.merkle_tree.root_hex
    assert orch.verify_timestamp_token(tst, expected_merkle_root=orch.merkle_tree.root_hex, expected_nonce="audit-nonce-001")
    assert not orch.verify_timestamp_token(tst, expected_nonce="wrong-nonce")

    # Tampered merkle root in token dict must fail verification
    tampered_dict = tst.to_dict()
    tampered_dict["merkle_root"] = "f" * 64
    assert not orch.verify_timestamp_token(tampered_dict)

    # Consistent witness checkpoints
    cp_witness_1 = GossipCheckpoint(
        witness_id="eu_regulator_node_1",
        tenant_id="default",
        sequence_number=1,
        merkle_root=tst.merkle_root,
        log_head_hash=tst.log_head_hash,
        observed_at=tst.issued_at,
        timestamp_token=tst.to_dict(),
    )
    cp_witness_2 = GossipCheckpoint(
        witness_id="ch_finma_witness_2",
        tenant_id="default",
        sequence_number=1,
        merkle_root=tst.merkle_root,
        log_head_hash=tst.log_head_hash,
        observed_at=tst.issued_at,
        timestamp_token=tst.to_dict(),
    )
    clean_report = orch.audit_transparency_gossip([cp_witness_1, cp_witness_2])
    assert clean_report.consistent is True
    assert clean_report.split_view_detected is False
    assert clean_report.history_rollback_detected is False
    assert clean_report.invalid_timestamp_tokens == 0
    assert len(clean_report.anomalies) == 0

    # Inject a split-view equivocation checkpoint at sequence 1 and a rollback checkpoint at sequence 5
    equivocating_cp = GossipCheckpoint(
        witness_id="compromised_mirror",
        tenant_id="default",
        sequence_number=1,
        merkle_root="a" * 64,
        log_head_hash=tst.log_head_hash,
        observed_at=tst.issued_at,
    )
    rollback_cp = GossipCheckpoint(
        witness_id="future_witness",
        tenant_id="default",
        sequence_number=5,
        merkle_root="b" * 64,
        log_head_hash="c" * 64,
        observed_at=tst.issued_at,
    )
    attack_report = orch.audit_transparency_gossip([cp_witness_1, equivocating_cp, rollback_cp])
    assert attack_report.consistent is False
    assert attack_report.split_view_detected is True
    assert attack_report.history_rollback_detected is True
    anomaly_types = {a["type"] for a in attack_report.anomalies}
    assert "CROSS_WITNESS_SPLIT_VIEW" in anomaly_types
    assert "LOCAL_VS_WITNESS_EQUIVOCATION" in anomaly_types
    assert "HISTORY_ROLLBACK_OR_TRUNCATION" in anomaly_types

    # Verify standalone TransparencyGossipAuditor default HMAC fallback
    standalone_auditor = TransparencyGossipAuditor(secret_key="hmac-tsa-secret")
    hmac_tst = standalone_auditor.issue_timestamp_token(
        tenant_id="default",
        merkle_root="1" * 64,
        log_head_hash="2" * 64,
        sequence_number=1,
    )
    assert standalone_auditor.verify_timestamp_token(hmac_tst)


def test_statutory_amendment_diff_and_downstream_impact() -> None:
    """
    Verify clause-level statutory diffing and downstream obligation blast-radius detection
    across graph dependencies, shared statutory citations, and subordinate policies.
    """
    orch = LegalMemoryOrchestrator(
        encoder=SemanticMockEncoder(embedding_dim=32),
        value_dim=2,
    )

    # 1. Parent Statute (Rank 9, Obligation to retain records under Art. 30 GDPR)
    statute = orch.update_memory(
        rule_text=(
            "Article 30 GDPR: Controllers shall retain processing logs for 5 years. "
            "Records must include cross-border transfer safeguards."
        ),
        action_vector=[0.0, 1.0],
        authority_rank=9,
        jurisdiction="EU",
    )

    # 2. Subordinate corporate policy linked via shared citation + subject matter (Rank 3)
    policy_a = orch.update_memory(
        rule_text=(
            "Corporate Retention Policy (under Article 30 GDPR): Operations teams must retain "
            "processing logs for 5 years in cloud archives."
        ),
        action_vector=[0.0, 1.0],
        authority_rank=3,
        jurisdiction="EU",
    )

    # 3. Subordinate vendor contract clause linked via explicit DEPENDS_ON graph edge (Rank 2)
    contract_b = orch.update_memory(
        rule_text="Vendor SLA Clause 4: Sub-processor may retain customer logs indefinitely.",
        action_vector=[0.2, 0.8],
        authority_rank=2,
        jurisdiction="EU",
    )
    assert contract_b.record_id is not None
    assert statute.record_id is not None
    orch.semantic_graph.add_node(
        node_id=contract_b.record_id,
        entity_type=EntityType.CLAUSE,
        label="Vendor SLA Clause 4",
        description=contract_b.text,
        tenant_id="default",
    )
    orch.semantic_graph.add_edge(
        source_id=contract_b.record_id,
        target_id=statute.record_id,
        relation_type=RelationType.DEPENDS_ON,
        tenant_id="default",
    )

    # Proposed amendment reversing retention into a strict prohibition after 30 days
    amended_text = (
        "Article 30 GDPR: Controllers must not retain processing logs beyond 30 days; "
        "all logs shall be purged immediately. "
        "Records must include Article 46 GDPR transfer safeguards."
    )
    report = orch.analyze_statutory_amendment(
        target_rule_id=statute.record_id,
        amended_text=amended_text,
        amended_action_vector=[1.0, 0.0],
        amended_authority_rank=10,
    )

    assert report.target_rule_id == statute.record_id
    assert report.old_modality == "OBLIGATION"
    assert report.new_modality == "PROHIBITION"
    assert report.modality_reversed is True
    assert report.total_impacted_count >= 2
    assert report.critical_impact_count >= 2
    assert len(report.report_hash) == 64

    impacted_ids = {item.dependent_rule_id: item for item in report.impacted_obligations}
    assert policy_a.record_id in impacted_ids
    assert contract_b.record_id in impacted_ids
    assert impacted_ids[policy_a.record_id].requires_redrafting is True
    assert impacted_ids[policy_a.record_id].impact_severity == "CRITICAL_MODALITY_REVERSAL"
    assert impacted_ids[contract_b.record_id].requires_redrafting is True


def test_async_timestamp_gossip_and_amendment_api() -> None:
    """Verify async coroutine wrappers for timestamp issuance, gossip auditing, and amendment impact."""
    orch = LegalMemoryOrchestrator(
        encoder=SemanticMockEncoder(embedding_dim=32),
        value_dim=2,
    )
    rec = orch.update_memory(
        rule_text="Section 10: Institutions must report cyber incidents within 24 hours.",
        action_vector=[1.0, 0.0],
        authority_rank=8,
    )
    orch.update_memory(
        rule_text="Internal SOP: Security teams must report cyber incidents within 24 hours.",
        action_vector=[1.0, 0.0],
        authority_rank=2,
    )

    async def _run() -> None:
        tst = await orch.async_issue_timestamp_token(nonce="async-nonce-99")
        assert isinstance(tst, TimestampToken)
        assert orch.verify_timestamp_token(tst, expected_nonce="async-nonce-99")

        cp = GossipCheckpoint(
            witness_id="async_witness",
            tenant_id="default",
            sequence_number=tst.sequence_number,
            merkle_root=tst.merkle_root,
            log_head_hash=tst.log_head_hash,
            observed_at=tst.issued_at,
            timestamp_token=tst.to_dict(),
        )
        gossip_rep = await orch.async_audit_transparency_gossip([cp])
        assert gossip_rep.consistent is True

        impact_rep = await orch.async_analyze_statutory_amendment(
            target_rule_id=rec.record_id,
            amended_text="Section 10: Institutions are prohibited from disclosing cyber incidents externally.",
            amended_action_vector=[0.0, 1.0],
        )
        assert impact_rep.modality_reversed is True
        assert impact_rep.total_impacted_count >= 1

    asyncio.run(_run())


def test_openapi_rest_mcp_and_cli_parity(tmp_path: Path) -> None:
    """
    Verify OpenAPI 3.1.0 specification endpoint (`/v1/openapi.json`) and full
    REST, MCP, and CLI parity for timestamp tokens, gossip auditing, and amendment impact.
    """
    spec = generate_openapi_spec()
    assert spec["openapi"] == "3.1.0"
    assert "/v1/openapi.json" in spec["paths"]
    assert "/v1/merkle/timestamp" in spec["paths"]
    assert "/v1/merkle/gossip-audit" in spec["paths"]
    assert "/v1/rules/amendment-impact" in spec["paths"]

    orch = LegalMemoryOrchestrator(
        encoder=SemanticMockEncoder(embedding_dim=64),
        value_dim=2,
    )
    statute = orch.update_memory(
        rule_text="Art. 5 GDPR: Personal data must be retained only as necessary.",
        action_vector=[1.0, 0.0],
        authority_rank=9,
        jurisdiction="EU",
    )
    orch.update_memory(
        rule_text="Policy 1 (Art. 5 GDPR): Personal data must be retained for 12 months.",
        action_vector=[1.0, 0.0],
        authority_rank=3,
        jurisdiction="EU",
    )

    # 1. MCP Server parity
    mcp = MCPServer(orch)
    tools = {t["name"] for t in mcp.list_tools()}
    assert "legal_memory_issue_timestamp" in tools
    assert "legal_memory_gossip_audit" in tools
    assert "legal_memory_amendment_impact" in tools

    mcp_tst_res = mcp.call_tool("legal_memory_issue_timestamp", {"nonce": "mcp-nonce-1"})
    mcp_tst = json.loads(mcp_tst_res["content"][0]["text"])
    assert mcp_tst["nonce"] == "mcp-nonce-1"

    mcp_gos_res = mcp.call_tool(
        "legal_memory_gossip_audit",
        {
            "checkpoints": [
                {
                    "witness_id": "mcp_witness",
                    "tenant_id": "default",
                    "sequence_number": mcp_tst["sequence_number"],
                    "merkle_root": mcp_tst["merkle_root"],
                    "log_head_hash": mcp_tst["log_head_hash"],
                    "observed_at": mcp_tst["issued_at"],
                    "timestamp_token": mcp_tst,
                }
            ]
        },
    )
    mcp_gos = json.loads(mcp_gos_res["content"][0]["text"])
    assert mcp_gos["consistent"] is True

    mcp_amd_res = mcp.call_tool(
        "legal_memory_amendment_impact",
        {
            "target_rule_id": statute.record_id,
            "amended_text": "Art. 5 GDPR: Personal data shall not be retained under any circumstances.",
            "amended_action_vector": [0.0, 1.0],
        },
    )
    mcp_amd = json.loads(mcp_amd_res["content"][0]["text"])
    assert mcp_amd["modality_reversed"] is True
    assert mcp_amd["total_impacted_count"] >= 1

    # 2. REST API parity
    server = LegalMemoryHTTPServer(("127.0.0.1", 0), orch)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/openapi.json") as resp:
            openapi_data = json.loads(resp.read().decode("utf-8"))
            assert openapi_data["openapi"] == "3.1.0"

        req_tst = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/merkle/timestamp",
            data=json.dumps({"nonce": "http-nonce-1"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req_tst) as resp:
            http_tst = json.loads(resp.read().decode("utf-8"))
            assert http_tst["nonce"] == "http-nonce-1"

        req_gos = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/merkle/gossip-audit",
            data=json.dumps(
                {
                    "checkpoints": [
                        {
                            "witness_id": "http_witness",
                            "tenant_id": "default",
                            "sequence_number": http_tst["sequence_number"],
                            "merkle_root": http_tst["merkle_root"],
                            "log_head_hash": http_tst["log_head_hash"],
                            "observed_at": http_tst["issued_at"],
                            "timestamp_token": http_tst,
                        }
                    ]
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req_gos) as resp:
            http_gos = json.loads(resp.read().decode("utf-8"))
            assert http_gos["consistent"] is True

        req_amd = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/rules/amendment-impact",
            data=json.dumps(
                {
                    "target_rule_id": statute.record_id,
                    "amended_text": "Art. 5 GDPR: Personal data must not be retained.",
                    "amended_action_vector": [0.0, 1.0],
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req_amd) as resp:
            http_amd = json.loads(resp.read().decode("utf-8"))
            assert http_amd["modality_reversed"] is True
    finally:
        server.shutdown()
        server.server_close()

    # 3. CLI parity
    db_file = tmp_path / "cli_v12.db"
    orch.save_to_disk(str(db_file))

    rc_tst = cli_main(
        [
            "timestamp-root",
            "--db-path",
            str(db_file),
            "--nonce",
            "cli-nonce-1",
            "--mock-encoder",
        ]
    )
    assert rc_tst == 0

    cp_file = tmp_path / "checkpoints.json"
    cp_file.write_text(
        json.dumps(
            [
                {
                    "witness_id": "cli_witness",
                    "tenant_id": "default",
                    "sequence_number": mcp_tst["sequence_number"],
                    "merkle_root": mcp_tst["merkle_root"],
                    "log_head_hash": mcp_tst["log_head_hash"],
                    "observed_at": mcp_tst["issued_at"],
                }
            ]
        ),
        encoding="utf-8",
    )
    rc_gos = cli_main(
        [
            "gossip-audit",
            "--db-path",
            str(db_file),
            "--checkpoints-file",
            str(cp_file),
            "--mock-encoder",
        ]
    )
    assert rc_gos == 0

    rc_amd = cli_main(
        [
            "amendment-impact",
            statute.record_id,
            "--db-path",
            str(db_file),
            "--text",
            "Art. 5 GDPR: Personal data must not be retained.",
            "--action",
            "0.0",
            "1.0",
            "--mock-encoder",
        ]
    )
    assert rc_amd == 0
