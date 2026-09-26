"""
Unit tests for Continuous Legal Memory REST API HTTP server.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest

from continuous_legal_memory.adapters.mock_encoder import SemanticMockEncoder
from continuous_legal_memory.orchestrator import LegalMemoryOrchestrator
from continuous_legal_memory.security.attestation import AsymmetricAttestationModule
from continuous_legal_memory.server import create_server


def _make_request(
    url: str,
    method: str = "GET",
    data: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, Any] | str, dict[str, str]]:
    req_headers = headers.copy() if headers else {}
    body_bytes = None
    if data is not None:
        body_bytes = json.dumps(data).encode("utf-8")
        req_headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=body_bytes, headers=req_headers, method=method)
    try:
        with urllib.request.urlopen(req) as resp:
            status = resp.status
            content_type = resp.headers.get("Content-Type", "")
            raw_body = resp.read().decode("utf-8")
            resp_headers = dict(resp.headers)
            if "application/json" in content_type:
                return status, json.loads(raw_body), resp_headers
            return status, raw_body, resp_headers
    except urllib.error.HTTPError as err:
        content_type = err.headers.get("Content-Type", "")
        raw_body = err.read().decode("utf-8")
        resp_headers = dict(err.headers)
        if "application/json" in content_type:
            return err.code, json.loads(raw_body), resp_headers
        return err.code, raw_body, resp_headers


@pytest.fixture
def running_server() -> Generator[str, None, None]:
    encoder = SemanticMockEncoder()
    attestor = AsymmetricAttestationModule()
    orch = LegalMemoryOrchestrator(
        encoder=encoder,
        engine_mode="structured",
        enable_telemetry=True,
        attestor=attestor,
    )

    # Bind to ephemeral port (port 0 selects free OS port)
    server = create_server(orch, host="127.0.0.1", port=0)
    port = server.server_address[1]
    base_url = f"http://127.0.0.1:{port}"

    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    try:
        yield base_url
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=2.0)


def test_server_health_endpoint(running_server: str) -> None:
    status, body, _ = _make_request(f"{running_server}/health")
    assert status == 200
    assert isinstance(body, dict)
    assert body["status"] == "healthy"
    assert body["service"] == "continuous-legal-memory"
    assert body["version"] == "0.1.0"
    assert body["engine_mode"] == "structured"
    assert body["stored_records_total"] == 0


def test_server_metrics_endpoint(running_server: str) -> None:
    status, text, headers = _make_request(f"{running_server}/metrics")
    assert status == 200
    assert isinstance(text, str)
    assert "clm_service_info" in text
    assert "text/plain" in headers.get("Content-Type", "")


def test_server_options_cors(running_server: str) -> None:
    status, _, headers = _make_request(f"{running_server}/v1/rules", method="OPTIONS")
    assert status == 204
    assert headers.get("Access-Control-Allow-Origin") == "*"
    assert "POST" in headers.get("Access-Control-Allow-Methods", "")


def test_server_rule_lifecycle_and_predict(running_server: str) -> None:
    # 1. Ingest a rule via POST /v1/rules
    rule_payload = {
        "rule_text": "Statute Alpha: Financial transaction logs must be preserved for 5 years.",
        "action_vector": [0.0, 1.0],
        "authority_rank": 8,
        "jurisdiction": "FEDERAL",
        "tenant_id": "tenant_1",
    }
    status, body, _ = _make_request(
        f"{running_server}/v1/rules",
        method="POST",
        data=rule_payload,
        headers={"X-Tenant-ID": "tenant_1"},
    )
    assert status == 201
    assert isinstance(body, dict)
    assert body["status"] == "INGESTED"
    record_id = body["record_id"]
    assert record_id is not None
    assert body["tenant_id"] == "tenant_1"

    # 2. Query context via GET /v1/context
    status, ctx_body, _ = _make_request(
        f"{running_server}/v1/context?tenant_id=tenant_1",
        method="GET",
    )
    assert status == 200
    assert isinstance(ctx_body, dict)
    assert len(ctx_body["active_records"]) == 1

    # 3. Query prediction via POST /v1/predict
    query_payload = {
        "query_text": "How long must financial transaction logs be preserved?",
        "tenant_id": "tenant_1",
    }
    status, pred_body, _ = _make_request(
        f"{running_server}/v1/predict",
        method="POST",
        data=query_payload,
    )
    assert status == 200
    assert isinstance(pred_body, dict)
    assert pred_body["query"] == query_payload["query_text"]
    assert len(pred_body["predicted_action_vector"]) == 2
    assert pred_body["most_relevant_rule"] == rule_payload["rule_text"]
    assert "attestation_token" in pred_body
    tok = pred_body["attestation_token"]
    assert tok["algorithm"] == "Ed25519"
    assert tok["public_key"] is not None

    # 4. Verify attestation via POST /v1/attestation/verify
    verify_payload = {
        "token": tok,
        "result": {
            "query": pred_body["query"],
            "predicted_action_vector": pred_body["predicted_action_vector"],
            "most_relevant_rule": pred_body["most_relevant_rule"],
        },
        "retrieved_text": pred_body["most_relevant_rule"],
        "public_key": tok["public_key"],
    }
    status, ver_body, _ = _make_request(
        f"{running_server}/v1/attestation/verify",
        method="POST",
        data=verify_payload,
    )
    assert status == 200
    assert isinstance(ver_body, dict)
    assert ver_body["valid"] is True

    # 5. Delete rule via DELETE /v1/rules/{record_id}
    status, del_body, _ = _make_request(
        f"{running_server}/v1/rules/{record_id}",
        method="DELETE",
        headers={"X-Tenant-ID": "tenant_1"},
    )
    assert status == 200
    assert isinstance(del_body, dict)
    assert del_body["status"] == "ERASED"
    assert del_body["record_id"] == record_id

    # 6. Verify deleting nonexistent rule returns 404
    status, not_found_body, _ = _make_request(
        f"{running_server}/v1/rules/nonexistent_id",
        method="DELETE",
    )
    assert status == 404


def test_server_associate_and_persist(running_server: str, tmp_path: Path) -> None:
    # Ingest two related statutes
    r1 = {
        "rule_text": "Statute A: Prerequisite compliance guidelines.",
        "action_vector": [1.0, 0.0],
        "metadata": {"node_id": "statute_a", "label": "Statute A"},
        "tenant_id": "tenant_assoc",
    }
    r2 = {
        "rule_text": "Statute B: Dependent execution rules.",
        "action_vector": [0.0, 1.0],
        "metadata": {
            "node_id": "statute_b",
            "label": "Statute B",
            "relations": [{"target_id": "statute_a", "relation_type": "depends_on", "weight": 1.0}],
        },
        "tenant_id": "tenant_assoc",
    }
    _make_request(f"{running_server}/v1/rules", method="POST", data=r1)
    _make_request(f"{running_server}/v1/rules", method="POST", data=r2)

    # Test GET /v1/graph
    status, graph_body, _ = _make_request(f"{running_server}/v1/graph?tenant_id=tenant_assoc")
    assert status == 200
    assert isinstance(graph_body, dict)
    assert len(graph_body["nodes"]) == 2

    # Test POST /v1/associate
    assoc_payload = {
        "seed_nodes": ["statute_b"],
        "damping": 0.85,
        "tenant_id": "tenant_assoc",
    }
    status, assoc_body, _ = _make_request(f"{running_server}/v1/associate", method="POST", data=assoc_payload)
    assert status == 200
    assert isinstance(assoc_body, dict)
    assert assoc_body["total"] >= 1

    # Test POST /v1/persist and /v1/restore
    db_file = str(tmp_path / "server_test.db")
    status, persist_body, _ = _make_request(f"{running_server}/v1/persist", method="POST", data={"db_path": db_file})
    assert status == 200
    assert isinstance(persist_body, dict)
    assert persist_body["status"] == "SAVED"

    status, restore_body, _ = _make_request(f"{running_server}/v1/restore", method="POST", data={"db_path": db_file})
    assert status == 200
    assert isinstance(restore_body, dict)
    assert restore_body["status"] == "RESTORED"


def test_server_bad_request_validation(running_server: str) -> None:
    # Missing action_vector
    bad_rule = {"rule_text": "Incomplete rule"}
    status, body, _ = _make_request(f"{running_server}/v1/rules", method="POST", data=bad_rule)
    assert status == 400

    # Non-existent endpoint
    status, body, _ = _make_request(f"{running_server}/v1/unknown_route", method="GET")
    assert status == 404
