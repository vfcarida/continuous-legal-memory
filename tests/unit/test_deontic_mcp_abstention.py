"""
Unit tests for Tri-Canon Deontic Precedence, Epistemic Abstention Guardrails,
Root Package Exports, and Model Context Protocol (MCP) JSON-RPC 2.0 Server.
"""

from __future__ import annotations

import io
import json
from datetime import datetime, timezone

import pytest
import torch

import continuous_legal_memory as clm
from continuous_legal_memory import (
    KeyedHashAttestationModule,
    LegalMemoryOrchestrator,
    MCPServer,
)
from continuous_legal_memory.domain.interfaces import BaseEncoderPort
from continuous_legal_memory.evaluation.harness import DeterministicTokenEncoder


def test_root_package_public_exports() -> None:
    """Verify that primary classes and symbols are exported at the package root."""
    assert hasattr(clm, "__all__")
    for symbol in (
        "LegalMemoryOrchestrator",
        "MCPServer",
        "MemoryRecord",
        "PredictionResult",
        "PrecedenceConfig",
        "apply_legal_precedence",
        "KeyedHashAttestationModule",
        "AsymmetricAttestationModule",
    ):
        assert symbol in clm.__all__
        assert getattr(clm, symbol) is not None


def test_structured_mode_deterministic_rule_importance(offline_encoder: BaseEncoderPort) -> None:
    """Verify that structured mode assigns deterministic unit importance without random slow_net noise."""
    orch_a = LegalMemoryOrchestrator(encoder=offline_encoder, seed=1, engine_mode="structured")
    orch_b = LegalMemoryOrchestrator(encoder=offline_encoder, seed=999, engine_mode="structured")

    for orch in (orch_a, orch_b):
        orch.update_memory("Statute A: Retain banking audit logs.", [0.0, 1.0], authority_rank=5)
        orch.update_memory("Statute B: Delete expired marketing cookies.", [1.0, 0.0], authority_rank=5)

    assert torch.allclose(orch_a.hope_module.memory.rule_importance, torch.tensor([1.0, 1.0]))
    assert torch.allclose(orch_b.hope_module.memory.rule_importance, torch.tensor([1.0, 1.0]))


def test_lex_specialis_overrides_general_rule_of_equal_authority() -> None:
    """
    Verify lex specialis derogat legi generali: within equal authority tiers,
    a more specific clause overrides a general provision.
    """
    encoder = DeterministicTokenEncoder(embedding_dim=64)
    orch = LegalMemoryOrchestrator(encoder=encoder, value_dim=2, engine_mode="structured")

    # General financial data retention rule (authority_rank=5, specificity=1.0)
    orch.update_memory(
        rule_text="Banking Data Regulation: Financial customer transaction records shall be deleted after 2 years.",
        action_vector=[1.0, 0.0],
        authority_rank=5,
        metadata={"specificity": 1.0},
    )

    # Specific wire transfer anti-money-laundering exception (authority_rank=5, specificity=3.0)
    specific_rule = (
        "Banking Data Regulation Wire Transfer Clause: Cross-border financial customer transaction "
        "records must be retained for 10 years."
    )
    orch.update_memory(
        rule_text=specific_rule,
        action_vector=[0.0, 1.0],
        authority_rank=5,
        metadata={"specificity": 3.0},
    )

    res = orch.predict("Retention of financial customer transaction records for cross-border transfers")
    assert res.most_relevant_rule == specific_rule
    assert res.predicted_action_vector[1] > res.predicted_action_vector[0]


def test_lex_posterior_chronological_valid_from_independent_of_ingestion_order() -> None:
    """
    Verify lex posterior derogat legi priori using explicit valid_from dates,
    even when the older statute is ingested into the database after the newer statute.
    """
    encoder = DeterministicTokenEncoder(embedding_dim=64)
    orch = LegalMemoryOrchestrator(encoder=encoder, value_dim=2, engine_mode="structured")

    newer_statute = "Retail Deposit Policy 2025: Destroy physical deposit slip copies immediately after digital imaging."
    older_statute = "Retail Deposit Policy 2021: Retain physical deposit slip copies for 180 days after digital imaging."

    # Ingest the 2025 statute first
    orch.update_memory(
        rule_text=newer_statute,
        action_vector=[1.0, 0.0],
        authority_rank=3,
        valid_from=datetime(2025, 1, 1, tzinfo=timezone.utc),
    )
    # Ingest the 2021 statute second (backfill)
    orch.update_memory(
        rule_text=older_statute,
        action_vector=[0.0, 1.0],
        authority_rank=3,
        valid_from=datetime(2021, 1, 1, tzinfo=timezone.utc),
    )

    res = orch.predict(
        "Retail policy on physical deposit slip copies after digital imaging",
        at_time=datetime(2025, 6, 1, tzinfo=timezone.utc),
    )
    assert res.most_relevant_rule == newer_statute
    assert res.predicted_action_vector[0] > res.predicted_action_vector[1]


def test_epistemic_abstention_below_similarity_threshold() -> None:
    """
    Verify EU AI Act Art. 14 epistemic abstention guardrail when query similarity
    falls below min_similarity_threshold.
    """
    encoder = DeterministicTokenEncoder(embedding_dim=64)
    orch = LegalMemoryOrchestrator(
        encoder=encoder,
        value_dim=2,
        engine_mode="structured",
        min_similarity_threshold=0.30,
    )

    orch.update_memory(
        rule_text="Maritime Admiralty Act: Vessel cargo manifests must be preserved for 7 years.",
        action_vector=[0.0, 1.0],
        authority_rank=8,
    )

    # Unrelated query has low cosine similarity (< 0.30) -> must abstain
    abstained_res = orch.predict("Deletion of employee cafeteria biometric lunch vouchers")
    assert abstained_res.abstained is True
    assert abstained_res.abstention_reason is not None
    assert "below epistemic threshold" in abstained_res.abstention_reason
    assert abstained_res.predicted_action_vector == [0.0, 0.0]
    assert abstained_res.most_relevant_rule is None
    assert abstained_res.max_similarity is not None and abstained_res.max_similarity < 0.30

    # Relevant query has high cosine similarity (>= 0.30) -> must not abstain
    confident_res = orch.predict("Vessel cargo manifests preservation period under Maritime Admiralty Act")
    assert confident_res.abstained is False
    assert confident_res.abstention_reason is None
    assert confident_res.predicted_action_vector == pytest.approx([0.0, 1.0], abs=1e-5)
    assert confident_res.max_similarity is not None and confident_res.max_similarity >= 0.30


def test_mcp_server_jsonrpc_lifecycle_tools_and_resources(offline_encoder: BaseEncoderPort) -> None:
    """Verify MCPServer JSON-RPC 2.0 protocol methods, tool calls, and resource reads."""
    attestor = KeyedHashAttestationModule(secret_key="mcp-test-secret")
    orch = LegalMemoryOrchestrator(encoder=offline_encoder, value_dim=2, attestor=attestor)
    mcp = MCPServer(orch)

    # 1. initialize
    init_resp = mcp.handle_request({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    assert init_resp is not None
    assert init_resp["result"]["protocolVersion"] == "2024-11-05"
    assert init_resp["result"]["serverInfo"]["name"] == "continuous-legal-memory"

    # 2. notification returns None
    notif_resp = mcp.handle_request({"jsonrpc": "2.0", "method": "notifications/initialized"})
    assert notif_resp is None

    # 3. tools/list
    tools_resp = mcp.handle_request({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    assert tools_resp is not None
    tool_names = {t["name"] for t in tools_resp["result"]["tools"]}
    assert {
        "legal_memory_ingest",
        "legal_memory_predict",
        "legal_memory_erase",
        "legal_memory_associate",
        "legal_memory_verify",
    }.issubset(tool_names)

    # 4. tools/call -> legal_memory_ingest
    ingest_resp = mcp.handle_request({
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "legal_memory_ingest",
            "arguments": {
                "rule_text": "Statute 99: Retain credit ledger records for 10 years.",
                "action_vector": [0.0, 1.0],
                "authority_rank": 9,
                "specificity": 2.0,
                "tenant_id": "tenant_mcp",
                "node_id": "node_statute_99",
            },
        },
    })
    assert ingest_resp is not None
    ingest_data = json.loads(ingest_resp["result"]["content"][0]["text"])
    assert ingest_data["status"] == "INGESTED"
    record_id = ingest_data["record_id"]

    # 5. tools/call -> legal_memory_predict
    pred_resp = mcp.handle_request({
        "jsonrpc": "2.0",
        "id": 4,
        "method": "tools/call",
        "params": {
            "name": "legal_memory_predict",
            "arguments": {
                "query_text": "Retain credit ledger records",
                "tenant_id": "tenant_mcp",
            },
        },
    })
    assert pred_resp is not None
    pred_data = json.loads(pred_resp["result"]["content"][0]["text"])
    assert pred_data["abstained"] is False
    assert pred_data["predicted_action_vector"] == pytest.approx([0.0, 1.0], abs=1e-4)
    assert pred_data["attestation_token"] is not None

    # 6. tools/call -> legal_memory_verify
    verify_resp = mcp.handle_request({
        "jsonrpc": "2.0",
        "id": 5,
        "method": "tools/call",
        "params": {
            "name": "legal_memory_verify",
            "arguments": {
                "token": pred_data["attestation_token"],
                "query": pred_data["query"],
                "predicted_action_vector": pred_data["predicted_action_vector"],
                "most_relevant_rule": pred_data["most_relevant_rule"],
            },
        },
    })
    assert verify_resp is not None
    verify_data = json.loads(verify_resp["result"]["content"][0]["text"])
    assert verify_data["valid"] is True

    # 7. resources/list & resources/read
    res_list = mcp.handle_request({"jsonrpc": "2.0", "id": 6, "method": "resources/list"})
    assert res_list is not None
    assert len(res_list["result"]["resources"]) == 3

    ctx_read = mcp.handle_request({
        "jsonrpc": "2.0",
        "id": 7,
        "method": "resources/read",
        "params": {"uri": "clm://context/tenant_mcp"},
    })
    assert ctx_read is not None
    ctx_payload = json.loads(ctx_read["result"]["contents"][0]["text"])
    assert len(ctx_payload["active_records"]) == 1

    # 8. tools/call -> legal_memory_erase
    erase_resp = mcp.handle_request({
        "jsonrpc": "2.0",
        "id": 8,
        "method": "tools/call",
        "params": {
            "name": "legal_memory_erase",
            "arguments": {"record_id": record_id, "tenant_id": "tenant_mcp"},
        },
    })
    assert erase_resp is not None
    erase_data = json.loads(erase_resp["result"]["content"][0]["text"])
    assert erase_data["status"] == "ERASED"

    # 9. stdio stream execution
    in_buf = io.StringIO('{"jsonrpc": "2.0", "id": 99, "method": "ping"}\n')
    out_buf = io.StringIO()
    mcp.serve_stdio(input_stream=in_buf, output_stream=out_buf)
    stdio_resp = json.loads(out_buf.getvalue().strip())
    assert stdio_resp["id"] == 99
    assert stdio_resp["result"] == {}
