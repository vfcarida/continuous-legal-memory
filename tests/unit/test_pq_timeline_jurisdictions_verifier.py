"""
Unit tests for End-to-End PQ-Hybrid Attestation, Temporal Horizon Simulation, and Standalone Verifier:
1. End-to-End Post-Quantum Hybrid Attestation (`PQ-Hybrid-Ed25519-SHA256`) across Selective-Disclosure,
   Portable Bundles, Federated Sync, TEE Quotes, and Sparse Merkle Proofs
2. Bi-Temporal Regulatory Horizon Simulator (`TemporalHorizonSimulator`)
3. Global Multi-Jurisdictional Choice-of-Law (`UK`, `SG`, `JP`, `CH`) & Multilingual Deontic Extraction
4. Client-Side Standalone Audit Verifier Exporter (`StandaloneAuditVerifierExporter`) + REST/MCP/CLI Parity
"""

from __future__ import annotations

import asyncio
import json
import threading
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from continuous_legal_memory.cli import main as cli_main
from continuous_legal_memory.domain.interfaces import BaseEncoderPort
from continuous_legal_memory.domain.models import EntityType
from continuous_legal_memory.mcp import MCPServer
from continuous_legal_memory.orchestrator import LegalMemoryOrchestrator
from continuous_legal_memory.reasoning.modal_verifier import (
    DeonticModality,
    infer_deontic_modality,
)
from continuous_legal_memory.reasoning.norm_extractor import NormativeGraphExtractor
from continuous_legal_memory.retrieval.choice_of_law import (
    SUPPORTED_JURISDICTION_REGIMES,
    ChoiceOfLawRouter,
    ConnectingFactors,
)
from continuous_legal_memory.retrieval.hybrid_lexical import extract_legal_citations
from continuous_legal_memory.security.pq_hybrid import PQ_HYBRID_ALGORITHM
from continuous_legal_memory.server import create_server


def test_end_to_end_pq_hybrid_across_zk_bundles_federated_tee_and_merkle(
    offline_encoder: BaseEncoderPort,
) -> None:
    """Verify PQ-Hybrid-Ed25519-SHA256 works end-to-end across all cryptographic subsystems."""
    orch = LegalMemoryOrchestrator(encoder=offline_encoder, value_dim=2, engine_mode="structured")
    orch.rotate_signing_key(
        new_key_id="pq-e2e-2026",
        algorithm=PQ_HYBRID_ALGORITHM,
        key_material="pq-e2e-master-seed-2026",
    )

    rec = orch.update_memory(
        rule_text="Financial institutions shall encrypt cross-border settlement messages.",
        action_vector=[1.0, 0.0],
        authority_rank=9,
        jurisdiction="CH",
        metadata={"record_id": "finma_encryption_rule"},
    )
    assert rec.record_id == "finma_encryption_rule"

    # 1. Sparse Merkle Inclusion Proof signed with PQ-Hybrid-Ed25519-SHA256
    smt_proof = orch.prove_inclusion("finma_encryption_rule")
    assert smt_proof.algorithm == PQ_HYBRID_ALGORITHM
    assert smt_proof.signature is not None and ":" in smt_proof.signature
    assert orch.verify_merkle_proof(smt_proof) is True

    # 2. Selective-Disclosure Compliance Proof signed with PQ-Hybrid-Ed25519-SHA256
    zk_bundle = orch.generate_compliance_proof(
        record_id="finma_encryption_rule",
        query="Must cross-border settlement messages be encrypted?",
        min_required_authority_rank=5,
    )
    assert zk_bundle.proof.algorithm == PQ_HYBRID_ALGORITHM
    assert ":" in zk_bundle.proof.signature
    assert orch.verify_compliance_proof(
        zk_bundle.proof,
        expected_merkle_root=orch.merkle_tree.root_hex,
        min_authority_rank=5,
        required_jurisdiction="CH",
    ) is True

    # 3. Signed Portable Compliance Bundle signed with PQ-Hybrid-Ed25519-SHA256
    comp_bundle = orch.export_compliance_bundle()
    assert comp_bundle.algorithm == PQ_HYBRID_ALGORITHM
    assert ":" in comp_bundle.signature
    assert orch.verify_compliance_bundle(comp_bundle) is True

    # 4. Federated Cross-Party Alignment with PQ-Hybrid proofs
    alignment = orch.verify_cross_party_alignment(
        proofs=[zk_bundle.proof],
        min_authority_rank=5,
        min_action_alignment=0.8,
    )
    assert alignment.is_aligned is True
    assert alignment.all_proofs_valid is True

    # 5. TEE Remote Attestation Quote automatically upgraded to PQ-Hybrid-Ed25519-SHA256
    tee_quote = orch.generate_tee_quote(nonce="pq-tee-nonce-77", platform="AWS_NITRO")
    assert tee_quote.signature_algorithm == PQ_HYBRID_ALGORITHM
    assert ":" in tee_quote.signature
    tee_ver = orch.verify_tee_quote(
        tee_quote,
        expected_nonce="pq-tee-nonce-77",
        expected_merkle_root=orch.merkle_tree.root_hex,
    )
    assert tee_ver.verified is True


def test_bitemporal_regulatory_horizon_simulator_tracks_modality_shifts(
    offline_encoder: BaseEncoderPort,
) -> None:
    """Verify TemporalHorizonSimulator sweeps across valid_from and valid_to epochs accurately."""
    orch = LegalMemoryOrchestrator(encoder=offline_encoder, value_dim=2, engine_mode="structured")

    # Epoch 1 (2025-01-01 to 2025-06-01): Transitional sandbox permission (Rank 4)
    orch.update_memory(
        rule_text="High-risk biometric AI systems are permitted in regulatory sandboxes.",
        action_vector=[0.0, 1.0],
        authority_rank=4,
        valid_from=datetime(2025, 1, 1, tzinfo=timezone.utc),
        valid_to=datetime(2025, 6, 1, tzinfo=timezone.utc),
        metadata={"record_id": "sandbox_permission_2025"},
    )
    # Epoch 2 (from 2025-03-01 onward): Statutory prohibition (Rank 9, overrides sandbox via Lex Superior)
    orch.update_memory(
        rule_text="Real-time remote biometric identification in public spaces is prohibited.",
        action_vector=[1.0, 0.0],
        authority_rank=9,
        valid_from=datetime(2025, 3, 1, tzinfo=timezone.utc),
        metadata={"record_id": "eu_ai_act_art5_ban"},
    )

    report = orch.simulate_temporal_horizon(
        query_text="Biometric AI systems in public spaces",
        start_time=datetime(2025, 1, 1, tzinfo=timezone.utc),
        end_time=datetime(2025, 12, 31, tzinfo=timezone.utc),
    )
    assert report.total_epochs >= 3
    assert report.modality_shifts_count >= 1
    assert len(report.regime_changes) >= 1

    # At start (2025-01-01), only sandbox_permission_2025 is active -> PERMISSION (P)
    first_epoch = report.epochs[0]
    assert first_epoch.prevailing_record_id == "sandbox_permission_2025"
    assert first_epoch.prevailing_modality == DeonticModality.PERMISSION.value

    # At 2025-03-01, eu_ai_act_art5_ban enters into force with Rank 9 -> PROHIBITION (F)
    march_epochs = [
        e for e in report.epochs if e.epoch_time.startswith("2025-03-01")
    ]
    assert len(march_epochs) == 1
    assert march_epochs[0].prevailing_record_id == "eu_ai_act_art5_ban"
    assert march_epochs[0].prevailing_modality == DeonticModality.PROHIBITION.value
    assert march_epochs[0].winning_canon == "LEX_SUPERIOR"

    async_report = asyncio.run(
        orch.async_simulate_temporal_horizon(
            query_text="Biometric AI systems in public spaces",
            start_time=datetime(2025, 1, 1, tzinfo=timezone.utc),
            end_time=datetime(2025, 12, 31, tzinfo=timezone.utc),
        )
    )
    assert async_report.total_epochs == report.total_epochs


def test_global_jurisdictions_uk_sg_jp_ch_and_multilingual_deontic_extraction() -> None:
    """Verify UK, SG, JP, CH choice-of-law regimes, pin-cites, and multilingual deontic regexes."""
    for code in ("EU", "US", "BR", "UK", "SG", "JP", "CH"):
        assert code in SUPPORTED_JURISDICTION_REGIMES

    router = ChoiceOfLawRouter()
    res_ch = router.resolve_factors(
        ConnectingFactors(
            data_subject_domicile="CH",
            place_of_performance="SG",
            choice_of_forum="UK",
        )
    )
    assert res_ch.primary_jurisdiction == "CH"
    assert "SR 235.1" in res_ch.rationale

    res_jp = router.resolve_factors(
        ConnectingFactors(
            data_subject_domicile="JP",
            choice_of_forum="US",
        )
    )
    assert "APPI" in res_jp.rationale

    # Verify statutory pin-cites for CH, SG, UK, JP
    cites_ch = extract_legal_citations("Swiss FADP SR 235.1 Artikel 18 and FINMA Circular")
    assert "ch_sr:235.1" in cites_ch
    assert "art:18" in cites_ch
    assert "inst:fadp_ch" in cites_ch
    assert "inst:finma" in cites_ch

    cites_sg_uk = extract_legal_citations("Singapore PDPA s. 24, MAS TRM, and UK GDPR s. 207")
    assert "sec:24" in cites_sg_uk
    assert "sec:207" in cites_sg_uk
    assert "inst:pdpa_sg" in cites_sg_uk
    assert "inst:mas_trm" in cites_sg_uk
    assert "inst:uk_gdpr" in cites_sg_uk

    # Verify multilingual deontic extraction (PT, FR, DE)
    assert infer_deontic_modality("É vedado o tratamento de dados sem consentimento.") == DeonticModality.PROHIBITION
    assert infer_deontic_modality("O controlador deverá notificar a autoridade.") == DeonticModality.OBLIGATION
    assert infer_deontic_modality("Le traitement est interdit sans base légale.") == DeonticModality.PROHIBITION
    assert infer_deontic_modality("Die Verarbeitung ist verboten.") == DeonticModality.PROHIBITION
    assert infer_deontic_modality("Der Verantwortliche muss die Daten schützen.") == DeonticModality.OBLIGATION
    assert infer_deontic_modality("Die Weitergabe ist zulässig.") == DeonticModality.PERMISSION

    assert NormativeGraphExtractor.infer_entity_type("É proibido compartilhar chaves privadas.") == EntityType.PROHIBITION


def test_standalone_audit_verifier_and_rest_mcp_cli_parity(
    offline_encoder: BaseEncoderPort,
    tmp_path: Path,
    capsys,
) -> None:
    """Verify StandaloneAuditVerifierExporter and REST, MCP, and CLI parity."""
    orch = LegalMemoryOrchestrator(encoder=offline_encoder, value_dim=2, engine_mode="structured")
    orch.update_memory(
        rule_text="Cross-border transfers out of Switzerland must comply with SR 235.1 Art. 16.",
        action_vector=[1.0, 0.0],
        authority_rank=8,
        jurisdiction="CH",
        valid_from=datetime(2025, 1, 1, tzinfo=timezone.utc),
        metadata={"record_id": "ch_fadp_art16"},
    )

    artifact = orch.export_standalone_verifier(query_text="Cross-border transfers out of Switzerland")
    assert artifact.tenant_id == "default"
    assert artifact.merkle_root == orch.merkle_tree.root_hex
    assert "<!DOCTYPE html>" in artifact.html_content
    assert "crypto.subtle.digest" in artifact.html_content
    assert "(set-logic QF_UFLIA)" in artifact.html_content
    assert len(artifact.html_sha256) == 64

    # 1. MCP Server verification
    mcp = MCPServer(orch)
    tool_names = {t["name"] for t in mcp.list_tools()}
    assert "legal_memory_simulate_timeline" in tool_names
    assert "legal_memory_export_verifier" in tool_names

    tl_mcp = mcp.call_tool(
        "legal_memory_simulate_timeline",
        {"query_text": "Cross-border transfers out of Switzerland"},
    )
    tl_data = json.loads(tl_mcp["content"][0]["text"])
    assert tl_data["total_epochs"] >= 1

    ver_mcp = mcp.call_tool(
        "legal_memory_export_verifier",
        {"query_text": "Cross-border transfers out of Switzerland"},
    )
    ver_data = json.loads(ver_mcp["content"][0]["text"])
    assert len(ver_data["html_sha256"]) == 64
    assert ver_data["merkle_root"] == artifact.merkle_root
    assert "(set-logic QF_UFLIA)" in ver_data["html_content"]

    # 2. REST API verification (/v1/rules/timeline and /v1/audit/standalone-verifier)
    server = create_server(orch, host="127.0.0.1", port=0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        req_tl = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/rules/timeline",
            data=json.dumps({"query_text": "Cross-border transfers"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req_tl, timeout=5) as resp:
            rest_tl = json.loads(resp.read().decode("utf-8"))
        assert rest_tl["total_epochs"] >= 1

        req_ver = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/audit/standalone-verifier",
            data=json.dumps({"query_text": "Cross-border transfers"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req_ver, timeout=5) as resp:
            rest_ver = json.loads(resp.read().decode("utf-8"))
        assert "<!DOCTYPE html>" in rest_ver["html_content"]
    finally:
        server.shutdown()
        server.server_close()

    # 3. CLI verification (simulate-timeline and export-verifier)
    db_file = str(tmp_path / "cli_r11.db")
    html_out = tmp_path / "verifier.html"
    orch.save_to_disk(db_file)
    capsys.readouterr()

    rc_tl = cli_main(
        [
            "simulate-timeline",
            "--db-path",
            db_file,
            "--query",
            "Cross-border transfers",
            "--mock-encoder",
        ]
    )
    assert rc_tl == 0
    out_tl = json.loads(capsys.readouterr().out)
    assert out_tl["total_epochs"] >= 1

    rc_ver = cli_main(
        [
            "export-verifier",
            "--db-path",
            db_file,
            "--output",
            str(html_out),
            "--query",
            "Cross-border transfers",
            "--mock-encoder",
        ]
    )
    assert rc_ver == 0
    out_ver = json.loads(capsys.readouterr().out)
    assert html_out.exists()
    assert out_ver["merkle_root"] == orch.merkle_tree.root_hex
