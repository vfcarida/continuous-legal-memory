"""
Unit tests for TEE Remote Attestation, Normative Drafting Co-Pilot, and Ecosystem Integrations:
1. Hardware/TEE Enclave Remote Attestation Quotes (`AWS_NITRO` & `INTEL_TDX`)
2. Formal Legislative & Contract Normative Drafting Co-Pilot (`NormativeDraftingCopilot`)
3. Upgraded Ecosystem Integrations (`langchain.py`, `llamaindex.py`, `letta.py`)
4. REST API (`/v1/rules/repair`, `/v1/tee/quote`, `/v1/tee/verify`), MCP tools, and CLI subcommands
"""

from __future__ import annotations

import json
import threading
import urllib.request
from pathlib import Path

from continuous_legal_memory.cli import main as cli_main
from continuous_legal_memory.domain.interfaces import BaseEncoderPort
from continuous_legal_memory.domain.models import EntityType, RelationType
from continuous_legal_memory.integrations.langchain import ContinuousLegalMemoryLangChain
from continuous_legal_memory.integrations.letta import (
    ContinuousLegalMemoryBlock,
    create_letta_tools,
)
from continuous_legal_memory.integrations.llamaindex import ContinuousLegalMemoryLlamaRetriever
from continuous_legal_memory.mcp import MCPServer
from continuous_legal_memory.orchestrator import LegalMemoryOrchestrator
from continuous_legal_memory.reasoning.drafting_copilot import DraftingRepairStrategy
from continuous_legal_memory.server import create_server


def test_tee_attestation_quote_nitro_and_tdx_verification(
    offline_encoder: BaseEncoderPort,
) -> None:
    """Verify AWS Nitro and Intel TDX remote attestation quote generation and tamper detection."""
    orch = LegalMemoryOrchestrator(encoder=offline_encoder, value_dim=2, engine_mode="structured")
    orch.update_memory(
        rule_text="Article 32 GDPR: Security of processing must include encryption.",
        action_vector=[1.0, 0.0],
        authority_rank=8,
        tenant_id="regulated_bank",
    )

    # 1. Generate and verify AWS Nitro Enclave quote
    nitro_quote = orch.generate_tee_quote(
        nonce="auditor-challenge-001",
        platform="AWS_NITRO",
        tenant_id="regulated_bank",
    )
    assert nitro_quote.platform == "AWS_NITRO"
    assert "PCR0" in nitro_quote.measurements
    assert "PCR1" in nitro_quote.measurements
    assert "PCR2" in nitro_quote.measurements
    assert nitro_quote.merkle_root == orch.merkle_tree.root_hex
    assert nitro_quote.active_key_id == orch.keyring.active_key_id

    report_ok = orch.verify_tee_quote(
        nitro_quote,
        expected_nonce="auditor-challenge-001",
        expected_merkle_root=orch.merkle_tree.root_hex,
        expected_measurements=nitro_quote.measurements,
    )
    assert report_ok.verified is True
    assert report_ok.signature_valid is True
    assert report_ok.user_data_valid is True
    assert report_ok.nonce_matched is True
    assert report_ok.merkle_root_matched is True
    assert report_ok.measurements_matched is True

    # 2. Generate and verify Intel TDX quote
    tdx_quote = orch.generate_tee_quote(
        nonce="auditor-challenge-002",
        platform="INTEL_TDX",
        tenant_id="regulated_bank",
    )
    assert tdx_quote.platform == "INTEL_TDX"
    assert "MRTD" in tdx_quote.measurements
    assert "RTMR0" in tdx_quote.measurements
    assert "RTMR1" in tdx_quote.measurements

    tdx_report = orch.verify_tee_quote(
        tdx_quote.to_dict(),
        expected_nonce="auditor-challenge-002",
        expected_merkle_root=orch.merkle_tree.root_hex,
    )
    assert tdx_report.verified is True

    # 3. Verify tamper detection on nonce, merkle_root, PCR0, and payload mutation
    wrong_nonce_report = orch.verify_tee_quote(
        nitro_quote,
        expected_nonce="wrong-nonce",
    )
    assert wrong_nonce_report.verified is False
    assert wrong_nonce_report.nonce_matched is False

    wrong_pcr_report = orch.verify_tee_quote(
        nitro_quote,
        expected_measurements={"PCR0": "00" * 48},
    )
    assert wrong_pcr_report.verified is False
    assert wrong_pcr_report.measurements_matched is False
    assert "PCR0" in wrong_pcr_report.mismatched_registers

    tampered_dict = nitro_quote.to_dict()
    tampered_dict["merkle_root"] = "ff" * 32
    tampered_report = orch.verify_tee_quote(tampered_dict)
    assert tampered_report.verified is False
    assert tampered_report.signature_valid is False
    assert tampered_report.user_data_valid is False


def test_normative_drafting_copilot_synthesizes_verified_repairs(
    offline_encoder: BaseEncoderPort,
) -> None:
    """Verify NormativeDraftingCopilot synthesizes KD-satisfiable repairs for conflicts, shadowing, and cycles."""
    orch = LegalMemoryOrchestrator(encoder=offline_encoder, value_dim=2, engine_mode="structured")

    orch.update_memory(
        rule_text="Data processors shall not retain customer personal data after contract termination.",
        action_vector=[1.0, 0.0],
        authority_rank=5,
        metadata={"record_id": "gdpr-retention-ban", "specificity": 1.0},
    )

    # 1. Conflicting horizontal antinomy (same authority_rank=5, opposite modality/action)
    copilot_report = orch.propose_norm_repairs(
        rule_text="Data processors shall retain customer personal data after contract termination.",
        action_vector=[0.0, 1.0],
        authority_rank=5,
        metadata={"specificity": 1.0},
    )
    assert copilot_report.needs_repair is True
    assert len(copilot_report.repair_options) == 3

    strategies = {opt.strategy for opt in copilot_report.repair_options}
    assert strategies == {
        DraftingRepairStrategy.EXPLICIT_SUPERSESSION.value,
        DraftingRepairStrategy.LEX_SPECIALIS_CARVEOUT.value,
        DraftingRepairStrategy.TEMPORAL_PARTITION.value,
    }
    for opt in copilot_report.repair_options:
        assert opt.post_repair_satisfiable is True
        assert len(opt.post_repair_shadowed_by) == 0
        assert "gdpr-retention-ban" in opt.target_conflict_ids

    assert copilot_report.recommended_option is not None
    assert copilot_report.recommended_option["post_repair_satisfiable"] is True

    # 2. Candidate shadowed by higher-authority statute (authority_rank=2 vs existing 5)
    shadow_report = orch.propose_norm_repairs(
        rule_text="Data processors may retain customer personal data after contract termination.",
        action_vector=[0.0, 1.0],
        authority_rank=2,
    )
    assert shadow_report.needs_repair is True
    assert shadow_report.recommended_option is not None
    assert shadow_report.recommended_option["repaired_authority_rank"] >= 5
    assert shadow_report.recommended_option["post_repair_satisfiable"] is True
    assert shadow_report.recommended_option["post_repair_shadowed_by"] == []

    # 3. Candidate introducing a DAG cycle
    orch.semantic_graph.add_node("node_a", EntityType.STATUTE, "Node A", "Rule A")
    orch.semantic_graph.add_node("node_b", EntityType.STATUTE, "Node B", "Rule B")
    orch.semantic_graph.add_edge("node_a", "node_b", RelationType.SUPERSEDES)

    cycle_report = orch.propose_norm_repairs(
        rule_text="Unrelated procedural notice period is 30 days.",
        action_vector=[1.0, 0.0],
        authority_rank=3,
        metadata={
            "node_id": "node_b",
            "relations": [{"relation_type": "supersedes", "target_id": "node_a"}],
        },
    )
    assert cycle_report.needs_repair is True
    assert any(
        o.strategy == DraftingRepairStrategy.DAG_CYCLE_PRUNING.value
        for o in cycle_report.repair_options
    )


def test_upgraded_ecosystem_integrations_langchain_llamaindex_letta(
    offline_encoder: BaseEncoderPort,
) -> None:
    """Verify upgraded LangChain, LlamaIndex, and Letta adapters expose proof traces, Merkle roots, and new tools."""
    orch = LegalMemoryOrchestrator(encoder=offline_encoder, value_dim=2, engine_mode="structured")
    rec = orch.update_memory(
        rule_text="Article 17 GDPR: Data subject shall have the right to erasure of personal data.",
        action_vector=[1.0, 0.0],
        authority_rank=9,
        tenant_id="eu_privacy",
    )

    # 1. LangChain with proof_trace and merkle_root enabled
    lc = ContinuousLegalMemoryLangChain(
        orchestrator=orch,
        tenant_id="eu_privacy",
        include_proof_trace=True,
        include_merkle_root=True,
    )
    lc_out = lc.load_memory_variables({"input": "Does the data subject have the right to erasure?"})
    ctx = lc_out["legal_context"]
    assert "Defeasible Winning Canon:" in ctx
    assert f"Merkle State Root: {orch.merkle_tree.root_hex}" in ctx
    last_pred = lc.get_last_prediction()
    assert last_pred is not None
    assert last_pred.merkle_root == orch.merkle_tree.root_hex

    # Verify LangChain epistemic abstention formatting when threshold is unattainable
    orch.min_similarity_threshold = 0.999999
    lc_abstain = lc.load_memory_variables({"input": "Completely unrelated maritime salvage tariff"})
    assert "[EPISTEMIC ABSTENTION]" in lc_abstain["legal_context"]
    orch.min_similarity_threshold = None

    # 2. LlamaIndex retriever with rich audit metadata
    retriever = ContinuousLegalMemoryLlamaRetriever(
        orchestrator=orch,
        tenant_id="eu_privacy",
    )
    nodes = retriever.retrieve("Right to erasure of personal data")
    assert len(nodes) == 1
    node_meta = nodes[0].metadata
    assert node_meta["merkle_root"] == orch.merkle_tree.root_hex
    assert node_meta["abstained"] is False
    assert isinstance(node_meta["proof_trace"], dict)

    # 3. Letta memory block and new agent tools (simulate, repair, erase)
    block = ContinuousLegalMemoryBlock(orchestrator=orch, tenant_id="eu_privacy")
    search_dict = block.search("Right to erasure of personal data")
    assert search_dict["merkle_root"] == orch.merkle_tree.root_hex
    assert isinstance(search_dict["proof_trace"], dict)

    tools = create_letta_tools(orchestrator=orch, tenant_id="eu_privacy")
    assert "legal_memory_simulate" in tools
    assert "legal_memory_repair" in tools
    assert "legal_memory_erase" in tools

    sim_str = tools["legal_memory_simulate"](
        "Data subject shall not have the right to erasure of personal data.",
        [0.0, 1.0],
        9,
    )
    assert "Satisfiable: False" in sim_str

    rep_str = tools["legal_memory_repair"](
        "Data subject shall not have the right to erasure of personal data.",
        [0.0, 1.0],
        9,
    )
    assert "Needs Repair: True" in rep_str

    erase_str = tools["legal_memory_erase"](str(rec.record_id))
    assert "Status: ERASED" in erase_str


def test_rest_mcp_cli_endpoints_and_commands(
    offline_encoder: BaseEncoderPort,
    tmp_path: Path,
) -> None:
    """Verify REST API, MCP tools, and CLI subcommands for TEE quotes and Drafting Co-Pilot."""
    orch = LegalMemoryOrchestrator(encoder=offline_encoder, value_dim=2, engine_mode="structured")
    orch.update_memory(
        rule_text="All wire transfers exceeding 10000 EUR shall be reported to the regulator.",
        action_vector=[1.0, 0.0],
        authority_rank=7,
        metadata={"record_id": "aml-wire-10k"},
    )

    # 1. MCP Server tools: legal_memory_propose_repairs & legal_memory_tee_quote
    mcp = MCPServer(orch)
    tool_names = {t["name"] for t in mcp.list_tools()}
    assert "legal_memory_propose_repairs" in tool_names
    assert "legal_memory_tee_quote" in tool_names

    mcp_rep = mcp.call_tool(
        "legal_memory_propose_repairs",
        {
            "rule_text": "Wire transfers exceeding 10000 EUR shall not be reported to the regulator.",
            "action_vector": [0.0, 1.0],
            "authority_rank": 7,
        },
    )
    mcp_rep_data = json.loads(mcp_rep["content"][0]["text"])
    assert mcp_rep_data["needs_repair"] is True
    assert len(mcp_rep_data["repair_options"]) == 3

    mcp_tee = mcp.call_tool(
        "legal_memory_tee_quote",
        {"nonce": "mcp-nonce-42", "platform": "INTEL_TDX"},
    )
    mcp_tee_data = json.loads(mcp_tee["content"][0]["text"])
    assert mcp_tee_data["platform"] == "INTEL_TDX"
    assert mcp_tee_data["nonce"] == "mcp-nonce-42"

    # 2. REST API endpoints: POST /v1/rules/repair, POST /v1/tee/quote, POST /v1/tee/verify
    server = create_server(orch, host="127.0.0.1", port=0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base_url = f"http://127.0.0.1:{port}"

        req_repair = urllib.request.Request(
            f"{base_url}/v1/rules/repair",
            data=json.dumps(
                {
                    "rule_text": "Wire transfers exceeding 10000 EUR shall not be reported to the regulator.",
                    "action_vector": [0.0, 1.0],
                    "authority_rank": 7,
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req_repair) as resp:
            repair_json = json.loads(resp.read().decode("utf-8"))
        assert repair_json["needs_repair"] is True
        assert repair_json["recommended_option"]["post_repair_satisfiable"] is True

        req_quote = urllib.request.Request(
            f"{base_url}/v1/tee/quote",
            data=json.dumps({"nonce": "rest-nonce-99", "platform": "AWS_NITRO"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req_quote) as resp:
            quote_json = json.loads(resp.read().decode("utf-8"))
        assert quote_json["platform"] == "AWS_NITRO"

        req_verify = urllib.request.Request(
            f"{base_url}/v1/tee/verify",
            data=json.dumps(
                {
                    "quote": quote_json,
                    "expected_nonce": "rest-nonce-99",
                    "expected_merkle_root": orch.merkle_tree.root_hex,
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req_verify) as resp:
            verify_json = json.loads(resp.read().decode("utf-8"))
        assert verify_json["verified"] is True
    finally:
        server.shutdown()
        server.server_close()

    # 3. CLI subcommands: clm repair-rule and clm tee-quote
    db_file = str(tmp_path / "test_cli.db")
    orch.save_to_disk(db_file)

    rc_rep = cli_main(
        [
            "repair-rule",
            "--db-path",
            db_file,
            "--text",
            "Wire transfers exceeding 10000 EUR shall not be reported to the regulator.",
            "--action",
            "0.0",
            "1.0",
            "--authority-rank",
            "7",
            "--mock-encoder",
        ]
    )
    assert rc_rep == 0

    rc_tee = cli_main(
        [
            "tee-quote",
            "--db-path",
            db_file,
            "--platform",
            "AWS_NITRO",
            "--nonce",
            "cli-nonce-1",
            "--mock-encoder",
        ]
    )
    assert rc_tee == 0
