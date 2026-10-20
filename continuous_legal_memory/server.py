"""
REST API Microservice Server for Continuous Legal Memory.

Provides a lightweight, zero-dependency, multi-threaded JSON HTTP API and OpenMetrics
Prometheus telemetry exporter for on-premise, sidecar, and containerized deployments.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from continuous_legal_memory import __version__
from continuous_legal_memory.domain.exceptions import (
    ContinuousLegalMemoryError,
    InvalidMemoryVectorError,
)
from continuous_legal_memory.domain.models import PredictionResult
from continuous_legal_memory.mcp import MCPServer
from continuous_legal_memory.orchestrator import LegalMemoryOrchestrator
from continuous_legal_memory.retrieval.conformal import SplitConformalCalibrator
from continuous_legal_memory.security.attestation import (
    AsymmetricAttestationModule,
    KeyedHashAttestationModule,
    MemoryAttestationToken,
)
from continuous_legal_memory.telemetry.observability import (
    TelemetryLogger,
    export_prometheus_metrics,
)

logger = logging.getLogger(__name__)


def generate_openapi_spec() -> dict[str, Any]:
    """
    Generate an OpenAPI 3.1.0 specification describing all Continuous Legal Memory
    REST API endpoints, request schemas, and governance capabilities.
    """
    return {
        "openapi": "3.1.0",
        "info": {
            "title": "Continuous Legal Memory (CLM) REST & Governance API",
            "version": __version__,
            "description": (
                "Bi-temporal, deontic precedence-aware cognitive memory and cryptographic "
                "compliance verification framework for legal AI agents."
            ),
            "license": {"name": "MIT", "url": "https://opensource.org/licenses/MIT"},
        },
        "paths": {
            "/health": {"get": {"summary": "Liveness and memory tier status check"}},
            "/metrics": {"get": {"summary": "OpenMetrics / Prometheus telemetry exporter"}},
            "/v1/openapi.json": {"get": {"summary": "OpenAPI 3.1.0 specification"}},
            "/v1/context": {"get": {"summary": "Inspect active working memory context"}},
            "/v1/graph": {"get": {"summary": "Inspect Semantic Knowledge Graph nodes and edges"}},
            "/v1/rules": {"post": {"summary": "Ingest a legal rule into multi-tier cognitive memory"}},
            "/v1/rules/{record_id}": {"delete": {"summary": "Execute GDPR Art. 17 right-to-erasure"}},
            "/v1/rules/simulate": {"post": {"summary": "Pre-flight KD modal satisfiability simulation"}},
            "/v1/rules/repair": {"post": {"summary": "Synthesize conflict-free legislative/contract repairs"}},
            "/v1/rules/timeline": {"post": {"summary": "Simulate bi-temporal regulatory horizon transitions"}},
            "/v1/rules/amendment-impact": {
                "post": {"summary": "Clause-level statutory diff and downstream obligation blast radius"}
            },
            "/v1/predict": {"post": {"summary": "Evaluate legal query with Tri-Canon precedence and abstention"}},
            "/v1/associate": {"post": {"summary": "Run HippoRAG Personalized PageRank graph diffusion"}},
            "/v1/audit/consistency": {"post": {"summary": "Audit formal deontic consistency and DAG cycles"}},
            "/v1/audit/smtlib": {"post": {"summary": "Export formal SMT-LIB 2.6 specification for Z3/cvc5"}},
            "/v1/audit/standalone-verifier": {
                "post": {"summary": "Export zero-dependency HTML5 + WebCrypto standalone audit verifier"}
            },
            "/v1/merkle/root": {"get": {"summary": "Retrieve current or historical 256-bit Sparse Merkle root"}},
            "/v1/merkle/proof/{rule_id}": {"get": {"summary": "Generate Sparse Merkle inclusion/erasure proof"}},
            "/v1/merkle/verify": {"post": {"summary": "Verify a 256-bit Sparse Merkle proof"}},
            "/v1/merkle/timestamp": {"post": {"summary": "Issue RFC 3161 cryptographic TimestampToken"}},
            "/v1/merkle/gossip-audit": {
                "post": {"summary": "Audit cross-witness gossip checkpoints for split-view equivocation"}
            },
            "/v1/conformal/calibrate": {"post": {"summary": "Calibrate Split-Conformal prediction threshold"}},
            "/v1/compliance/prove": {"post": {"summary": "Generate zero-knowledge selective-disclosure proof"}},
            "/v1/compliance/verify": {"post": {"summary": "Verify zero-knowledge selective-disclosure proof"}},
            "/v1/bundle/export": {"post": {"summary": "Export signed portable compliance bundle"}},
            "/v1/bundle/restore": {"post": {"summary": "Verify and restore signed portable compliance bundle"}},
            "/v1/keys": {"get": {"summary": "List CryptographicKeyRing keys and statuses"}},
            "/v1/keys/rotate": {"post": {"summary": "Rotate active signing key (Ed25519 / PQ-Hybrid / HMAC)"}},
            "/v1/keys/revoke": {"post": {"summary": "Revoke compromised signing key"}},
            "/v1/federated/verify": {"post": {"summary": "Verify cross-party federated normative alignment"}},
            "/v1/tee/quote": {"post": {"summary": "Generate hardware-anchored TEE remote attestation quote"}},
            "/v1/tee/verify": {"post": {"summary": "Verify TEE remote attestation quote"}},
            "/v1/mcp": {"post": {"summary": "Model Context Protocol (MCP) JSON-RPC 2.0 endpoint"}},
        },
    }


class LegalMemoryHTTPRequestHandler(BaseHTTPRequestHandler):
    """
    HTTP Request Handler serving Continuous Legal Memory REST API endpoints.
    """

    server: LegalMemoryHTTPServer  # Type assertion for custom server attribute

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        logger.debug("%s - - [%s] %s", self.address_string(), self.log_date_time_string(), format % args)

    def _send_json(self, status_code: int, data: dict[str, Any] | list[Any]) -> None:
        payload = json.dumps(data, default=str).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Tenant-ID")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.end_headers()
        self.wfile.write(payload)

    def _send_text(self, status_code: int, text: str, content_type: str = "text/plain; charset=utf-8") -> None:
        payload = text.encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _read_json_body(self) -> dict[str, Any]:
        content_length_str = self.headers.get("Content-Length")
        if not content_length_str:
            return {}
        try:
            length = int(content_length_str)
            raw_data = self.rfile.read(length)
            return json.loads(raw_data.decode("utf-8")) if raw_data else {}
        except Exception as err:
            raise ValueError(f"Malformed JSON request body: {err}") from err

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Tenant-ID")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        parsed_url = urlparse(self.path)
        path = parsed_url.path.rstrip("/")
        query = parse_qs(parsed_url.query)

        try:
            if path in ("", "/health"):
                self._handle_health()
            elif path == "/metrics":
                self._handle_metrics()
            elif path == "/v1/context":
                tenant_id = self.headers.get("X-Tenant-ID") or query.get("tenant_id", [None])[0]
                self._handle_get_context(tenant_id)
            elif path == "/v1/graph":
                tenant_id = self.headers.get("X-Tenant-ID") or query.get("tenant_id", [None])[0]
                self._handle_get_graph(tenant_id)
            elif path == "/v1/merkle/root":
                tenant_id = self.headers.get("X-Tenant-ID") or query.get("tenant_id", [None])[0]
                as_of_tx = query.get("as_of_transaction_time", [None])[0]
                self._handle_get_merkle_root(tenant_id, as_of_tx)
            elif path.startswith("/v1/merkle/proof/"):
                rule_id = path[len("/v1/merkle/proof/") :]
                self._handle_get_merkle_proof(rule_id)
            elif path == "/v1/keys":
                self._handle_get_keys()
            elif path in ("/v1/openapi.json", "/openapi.json"):
                self._send_json(HTTPStatus.OK, generate_openapi_spec())
            else:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Endpoint '{path}' not found."})
        except Exception as exc:
            logger.exception("Error processing GET %s: %s", path, exc)
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

    def do_POST(self) -> None:  # noqa: N802
        parsed_url = urlparse(self.path)
        path = parsed_url.path.rstrip("/")

        try:
            body = self._read_json_body()
        except ValueError as val_err:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(val_err)})
            return

        try:
            if path == "/v1/rules":
                self._handle_post_rules(body)
            elif path == "/v1/rules/simulate":
                self._handle_post_rules_simulate(body)
            elif path == "/v1/rules/repair":
                self._handle_post_rules_repair(body)
            elif path == "/v1/predict":
                self._handle_post_predict(body)
            elif path == "/v1/associate":
                self._handle_post_associate(body)
            elif path == "/v1/audit/consistency":
                self._handle_post_audit_consistency(body)
            elif path == "/v1/audit/smtlib":
                self._handle_post_audit_smtlib(body)
            elif path == "/v1/audit/standalone-verifier":
                self._handle_post_audit_standalone_verifier(body)
            elif path == "/v1/rules/timeline":
                self._handle_post_rules_timeline(body)
            elif path == "/v1/merkle/verify":
                self._handle_post_merkle_verify(body)
            elif path == "/v1/conformal/calibrate":
                self._handle_post_conformal_calibrate(body)
            elif path == "/v1/attestation/verify":
                self._handle_post_verify_attestation(body)
            elif path == "/v1/persist":
                self._handle_post_persist(body)
            elif path == "/v1/restore":
                self._handle_post_restore(body)
            elif path == "/v1/compliance/prove":
                self._handle_post_compliance_prove(body)
            elif path == "/v1/compliance/verify":
                self._handle_post_compliance_verify(body)
            elif path == "/v1/bundle/export":
                self._handle_post_bundle_export(body)
            elif path == "/v1/bundle/restore":
                self._handle_post_bundle_restore(body)
            elif path == "/v1/keys/rotate":
                self._handle_post_keys_rotate(body)
            elif path == "/v1/keys/revoke":
                self._handle_post_keys_revoke(body)
            elif path == "/v1/federated/verify":
                self._handle_post_federated_verify(body)
            elif path == "/v1/tee/quote":
                self._handle_post_tee_quote(body)
            elif path == "/v1/tee/verify":
                self._handle_post_tee_verify(body)
            elif path == "/v1/merkle/timestamp":
                self._handle_post_merkle_timestamp(body)
            elif path == "/v1/merkle/gossip-audit":
                self._handle_post_merkle_gossip_audit(body)
            elif path == "/v1/rules/amendment-impact":
                self._handle_post_rules_amendment_impact(body)
            elif path == "/v1/mcp":
                self._handle_post_mcp(body)
            else:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Endpoint '{path}' not found."})
        except KeyError as exc:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
        except (InvalidMemoryVectorError, ValueError) as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except ContinuousLegalMemoryError as exc:
            self._send_json(HTTPStatus.UNPROCESSABLE_ENTITY, {"error": str(exc)})
        except Exception as exc:
            logger.exception("Error processing POST %s: %s", path, exc)
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

    def do_DELETE(self) -> None:  # noqa: N802
        parsed_url = urlparse(self.path)
        path = parsed_url.path.rstrip("/")
        query = parse_qs(parsed_url.query)

        if path.startswith("/v1/rules/"):
            record_id = path[len("/v1/rules/") :]
            tenant_id = self.headers.get("X-Tenant-ID") or query.get("tenant_id", [None])[0]
            try:
                self._handle_delete_rule(record_id, tenant_id)
            except KeyError as exc:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
            except Exception as exc:
                logger.exception("Error processing DELETE %s: %s", path, exc)
                self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})
        else:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Endpoint '{path}' not found."})

    # --- Endpoint Handlers ---

    def _handle_health(self) -> None:
        orch = self.server.orchestrator
        total_records = len(orch.episodic_memory.get_records()) if hasattr(orch, "episodic_memory") else 0
        total_nodes = len(orch.semantic_graph.nodes) if hasattr(orch, "semantic_graph") else 0

        self._send_json(HTTPStatus.OK, {
            "status": "healthy",
            "service": "continuous-legal-memory",
            "version": __version__,
            "engine_mode": getattr(orch, "engine_mode", "structured"),
            "value_dim": getattr(orch, "value_dim", 2),
            "stored_records_total": total_records,
            "semantic_nodes_total": total_nodes,
            "merkle_root": orch.merkle_tree.root_hex,
        })

    def _handle_metrics(self) -> None:
        orch = self.server.orchestrator
        logger_inst = getattr(self.server, "telemetry_logger", None)
        exposition = export_prometheus_metrics(orch, logger_inst, service_name="continuous-legal-memory")
        self._send_text(HTTPStatus.OK, exposition, content_type="text/plain; version=0.0.4; charset=utf-8")

    def _handle_get_context(self, tenant_id: str | None) -> None:
        orch = self.server.orchestrator
        context = orch.get_working_memory_context(tenant_id=tenant_id)
        items = [
            {
                "record_id": rec.record_id,
                "text": rec.text,
                "tenant_id": rec.tenant_id,
                "authority_rank": rec.authority_rank,
                "jurisdiction": rec.jurisdiction,
                "personal_data": rec.personal_data,
                "valid_from": rec.valid_from.isoformat() if rec.valid_from else None,
                "valid_to": rec.valid_to.isoformat() if rec.valid_to else None,
            }
            for rec in context
        ]
        self._send_json(HTTPStatus.OK, {"tenant_id": tenant_id or "default", "active_records": items})

    def _handle_get_graph(self, tenant_id: str | None) -> None:
        orch = self.server.orchestrator
        nodes = orch.get_semantic_graph_nodes(tenant_id=tenant_id)
        items = [
            {
                "node_id": node.node_id,
                "label": node.label,
                "entity_type": node.entity_type.value if hasattr(node.entity_type, "value") else str(node.entity_type),
                "description": node.description,
                "tenant_id": node.tenant_id,
            }
            for node in nodes
        ]
        self._send_json(HTTPStatus.OK, {"tenant_id": tenant_id or "default", "nodes": items})

    def _handle_get_merkle_root(self, tenant_id: str | None, as_of_tx: str | None) -> None:
        orch = self.server.orchestrator
        root_hex = orch.merkle_tree.get_root_as_of(as_of_tx, tenant_id=tenant_id)
        self._send_json(HTTPStatus.OK, {
            "merkle_root": root_hex,
            "as_of_transaction_time": as_of_tx,
            "tenant_id": tenant_id,
            "active_leaves": orch.merkle_tree.active_count,
            "audit_log_entries": len(orch.merkle_tree.audit_log.entries),
            "chain_verified": orch.merkle_tree.audit_log.verify_chain(),
        })

    def _handle_get_merkle_proof(self, rule_id: str) -> None:
        if not rule_id:
            raise ValueError("A non-empty rule_id is required.")
        orch = self.server.orchestrator
        proof = orch.merkle_tree.generate_proof(rule_id, attestor=orch.attestor)
        self._send_json(HTTPStatus.OK, proof.to_dict())

    def _handle_post_rules(self, body: dict[str, Any]) -> None:
        orch = self.server.orchestrator
        rule_text = body.get("rule_text")
        if not rule_text or not isinstance(rule_text, str):
            raise ValueError("Field 'rule_text' must be a non-empty string.")

        action_vector = body.get("action_vector")
        if action_vector is None or not isinstance(action_vector, list):
            raise ValueError(f"Field 'action_vector' must be a list of floats with length {orch.value_dim}.")

        valid_from = None
        if body.get("valid_from"):
            valid_from = datetime.fromisoformat(body["valid_from"])

        valid_to = None
        if body.get("valid_to"):
            valid_to = datetime.fromisoformat(body["valid_to"])

        transaction_time = None
        if body.get("transaction_time"):
            transaction_time = datetime.fromisoformat(body["transaction_time"])

        tenant_id = self.headers.get("X-Tenant-ID") or body.get("tenant_id", "default")
        authority_rank = int(body.get("authority_rank", 1))
        jurisdiction = body.get("jurisdiction")
        personal_data = bool(body.get("personal_data", False))
        metadata = dict(body.get("metadata", {}))
        if body.get("record_id") and "record_id" not in metadata:
            metadata["record_id"] = str(body["record_id"])

        rec = orch.update_memory(
            rule_text=rule_text,
            action_vector=action_vector,
            valid_from=valid_from,
            valid_to=valid_to,
            metadata=metadata,
            authority_rank=authority_rank,
            jurisdiction=jurisdiction,
            personal_data=personal_data,
            tenant_id=tenant_id,
            transaction_time=transaction_time,
        )

        self._send_json(HTTPStatus.CREATED, {
            "status": "INGESTED",
            "record_id": rec.record_id,
            "tenant_id": rec.tenant_id,
            "authority_rank": rec.authority_rank,
            "personal_data": rec.personal_data,
            "merkle_root": orch.merkle_tree.root_hex,
            "created_at": rec.valid_from.isoformat() if rec.valid_from else datetime.now(timezone.utc).isoformat(),
            "transaction_time": rec.transaction_time.isoformat() if rec.transaction_time else None,
        })

    def _handle_post_rules_simulate(self, body: dict[str, Any]) -> None:
        orch = self.server.orchestrator
        rule_text = body.get("rule_text")
        if not rule_text or not isinstance(rule_text, str):
            raise ValueError("Field 'rule_text' must be a non-empty string.")

        action_vector = body.get("action_vector")
        if action_vector is not None and not isinstance(action_vector, list):
            raise ValueError(f"Field 'action_vector' must be a list of floats with length {orch.value_dim}.")

        valid_from = datetime.fromisoformat(body["valid_from"]) if body.get("valid_from") else None
        valid_to = datetime.fromisoformat(body["valid_to"]) if body.get("valid_to") else None
        tenant_id = self.headers.get("X-Tenant-ID") or body.get("tenant_id", "default")
        authority_rank = int(body.get("authority_rank", 1))
        jurisdiction = body.get("jurisdiction")
        metadata = dict(body.get("metadata", {}))

        report = orch.simulate_rule_ingestion(
            rule_text=rule_text,
            action_vector=action_vector,
            valid_from=valid_from,
            valid_to=valid_to,
            metadata=metadata,
            authority_rank=authority_rank,
            jurisdiction=jurisdiction,
            tenant_id=tenant_id,
        )
        self._send_json(HTTPStatus.OK, report.to_dict())

    def _handle_post_rules_repair(self, body: dict[str, Any]) -> None:
        orch = self.server.orchestrator
        rule_text = body.get("rule_text")
        if not rule_text or not isinstance(rule_text, str):
            raise ValueError("Field 'rule_text' must be a non-empty string.")

        action_vector = body.get("action_vector")
        if action_vector is not None and not isinstance(action_vector, list):
            raise ValueError(f"Field 'action_vector' must be a list of floats with length {orch.value_dim}.")

        valid_from = datetime.fromisoformat(body["valid_from"]) if body.get("valid_from") else None
        valid_to = datetime.fromisoformat(body["valid_to"]) if body.get("valid_to") else None
        tenant_id = self.headers.get("X-Tenant-ID") or body.get("tenant_id", "default")
        authority_rank = int(body.get("authority_rank", 1))
        jurisdiction = body.get("jurisdiction")
        metadata = dict(body.get("metadata", {}))

        report = orch.propose_norm_repairs(
            rule_text=rule_text,
            action_vector=action_vector,
            valid_from=valid_from,
            valid_to=valid_to,
            metadata=metadata,
            authority_rank=authority_rank,
            jurisdiction=jurisdiction,
            tenant_id=tenant_id,
        )
        self._send_json(HTTPStatus.OK, report.to_dict())

    def _handle_post_predict(self, body: dict[str, Any]) -> None:
        orch = self.server.orchestrator
        query_text = body.get("query_text")
        if not query_text or not isinstance(query_text, str):
            raise ValueError("Field 'query_text' must be a non-empty string.")

        at_time = None
        if body.get("at_time"):
            at_time = datetime.fromisoformat(body["at_time"])

        as_of_tx = None
        if body.get("as_of_transaction_time"):
            as_of_tx = datetime.fromisoformat(body["as_of_transaction_time"])

        tenant_id = self.headers.get("X-Tenant-ID") or body.get("tenant_id")
        min_sim = (
            float(body["min_similarity_threshold"])
            if body.get("min_similarity_threshold") is not None
            else None
        )
        jurisdiction = body.get("jurisdiction")
        connecting_factors = body.get("connecting_factors")

        result = orch.predict(
            query_text=query_text,
            at_time=at_time,
            tenant_id=tenant_id,
            min_similarity_threshold=min_sim,
            jurisdiction=jurisdiction,
            connecting_factors=connecting_factors,
            as_of_transaction_time=as_of_tx,
        )

        response_data: dict[str, Any] = {
            "query": result.query,
            "predicted_action_vector": result.predicted_action_vector,
            "source_tier": result.source_tier.value if (result.source_tier is not None and hasattr(result.source_tier, "value")) else str(result.source_tier),
            "most_relevant_rule": result.most_relevant_rule,
            "confidence": result.confidence,
            "max_similarity": result.max_similarity,
            "abstained": result.abstained,
            "abstention_reason": result.abstention_reason,
            "attention_weights": result.attention_weights,
            "retrieved_snippets": result.retrieved_snippets,
            "tenant_id": result.tenant_id,
            "conformal_set": result.conformal_set,
            "conformal_threshold": result.conformal_threshold,
            "conformal_alpha": result.conformal_alpha,
            "choice_of_law": result.choice_of_law,
            "proof_trace": result.proof_trace,
            "merkle_root": result.merkle_root,
        }

        if result.attestation_token is not None:
            tok = result.attestation_token
            response_data["attestation_token"] = {
                "timestamp": tok.timestamp.isoformat(),
                "state_hash": tok.state_hash,
                "integrity_tag": tok.integrity_tag,
                "key_id": tok.key_id,
                "algorithm": tok.algorithm,
                "public_key": tok.public_key,
            }

        self._send_json(HTTPStatus.OK, response_data)

    def _handle_delete_rule(self, record_id: str, tenant_id: str | None) -> None:
        orch = self.server.orchestrator
        audit = orch.delete_rule(record_id=record_id, tenant_id=tenant_id)
        self._send_json(HTTPStatus.OK, audit)

    def _handle_post_associate(self, body: dict[str, Any]) -> None:
        orch = self.server.orchestrator
        seed_nodes = body.get("seed_nodes")
        if not seed_nodes:
            raise ValueError("Field 'seed_nodes' must be a non-empty list or mapping of node IDs.")

        damping = float(body.get("damping", 0.85))
        max_results = int(body.get("max_results", 10))
        tenant_id = self.headers.get("X-Tenant-ID") or body.get("tenant_id")

        at_time = None
        if body.get("at_time"):
            at_time = datetime.fromisoformat(body["at_time"])

        as_of_tx = None
        if body.get("as_of_transaction_time"):
            as_of_tx = datetime.fromisoformat(body["as_of_transaction_time"])

        deontic_polarity = bool(body.get("deontic_polarity", True))

        associated = orch.associate_statutes(
            seed_nodes=seed_nodes,
            damping=damping,
            max_results=max_results,
            tenant_id=tenant_id,
            at_time=at_time,
            as_of_transaction_time=as_of_tx,
            deontic_polarity=deontic_polarity,
        )

        res = [
            {
                "node_id": node.node_id,
                "label": node.label,
                "entity_type": node.entity_type.value if hasattr(node.entity_type, "value") else str(node.entity_type),
                "description": node.description,
                "score": score,
            }
            for node, score in associated
        ]
        self._send_json(HTTPStatus.OK, {"results": res, "total": len(res)})

    def _handle_post_audit_consistency(self, body: dict[str, Any]) -> None:
        orch = self.server.orchestrator
        tenant_id = self.headers.get("X-Tenant-ID") or body.get("tenant_id")
        at_time = datetime.fromisoformat(body["at_time"]) if body.get("at_time") else None
        register_edges = bool(body.get("register_contradiction_edges", True))
        report = orch.audit_deontic_consistency(
            tenant_id=tenant_id,
            at_time=at_time,
            register_contradiction_edges=register_edges,
        )
        self._send_json(HTTPStatus.OK, report.to_dict())

    def _handle_post_merkle_verify(self, body: dict[str, Any]) -> None:
        orch = self.server.orchestrator
        proof = body.get("proof")
        if not proof or not isinstance(proof, dict):
            raise ValueError("Field 'proof' is required and must be a SparseMerkleProof dictionary.")
        expected_root = body.get("expected_root_hash")
        is_valid = orch.verify_merkle_proof(proof, expected_root_hash=expected_root)
        self._send_json(HTTPStatus.OK, {
            "valid": is_valid,
            "rule_id": proof.get("rule_id"),
            "is_member": proof.get("is_member"),
            "root_hash": proof.get("root_hash"),
        })

    def _handle_post_conformal_calibrate(self, body: dict[str, Any]) -> None:
        orch = self.server.orchestrator
        pred_vecs = body.get("predicted_vectors")
        true_labels = body.get("true_labels")
        if not isinstance(pred_vecs, list) or not isinstance(true_labels, list) or not pred_vecs:
            raise ValueError("Fields 'predicted_vectors' and 'true_labels' must be non-empty lists.")
        alpha = float(body.get("alpha", 0.10))
        abstain_empty = bool(body.get("abstain_on_empty", True))
        abstain_ambig = bool(body.get("abstain_on_ambiguous", False))
        calibrator = SplitConformalCalibrator(
            alpha=alpha,
            abstain_on_empty=abstain_empty,
            abstain_on_ambiguous=abstain_ambig,
        )
        q_hat = calibrator.calibrate(pred_vecs, true_labels)
        orch.conformal_calibrator = calibrator
        self._send_json(HTTPStatus.OK, {
            "status": "CALIBRATED",
            "quantile_threshold": q_hat,
            "alpha": alpha,
            "calibration_size": len(true_labels),
        })

    def _handle_post_verify_attestation(self, body: dict[str, Any]) -> None:
        raw_tok = body.get("token")
        if not raw_tok or not isinstance(raw_tok, dict):
            raise ValueError("Field 'token' is required and must be an object.")

        raw_result = body.get("result")
        if not raw_result or not isinstance(raw_result, dict):
            raise ValueError("Field 'result' is required and must be an object.")

        query = raw_result.get("query", "")
        action_vec = raw_result.get("predicted_action_vector", [])
        most_relevant = raw_result.get("most_relevant_rule")
        retrieved_text = body.get("retrieved_text", most_relevant)

        pred_result = PredictionResult(
            query=query,
            predicted_action_vector=action_vec,
            most_relevant_rule=most_relevant,
        )

        ts = datetime.fromisoformat(raw_tok["timestamp"])
        tok = MemoryAttestationToken(
            timestamp=ts,
            state_hash=raw_tok["state_hash"],
            integrity_tag=raw_tok["integrity_tag"],
            key_id=raw_tok.get("key_id", "default"),
            algorithm=raw_tok.get("algorithm", "HMAC-SHA256"),
            public_key=raw_tok.get("public_key"),
        )

        public_key = body.get("public_key") or tok.public_key

        if tok.algorithm == "Ed25519":
            verifier = AsymmetricAttestationModule(public_key=public_key)
            is_valid = verifier.verify_attestation(tok, pred_result, retrieved_text=retrieved_text)
        else:
            orch = self.server.orchestrator
            if orch.attestor:
                is_valid = orch.attestor.verify_attestation(tok, pred_result, retrieved_text=retrieved_text)
            else:
                verifier_hmac = KeyedHashAttestationModule()
                is_valid = verifier_hmac.verify_attestation(tok, pred_result, retrieved_text=retrieved_text)

        self._send_json(HTTPStatus.OK, {
            "valid": is_valid,
            "algorithm": tok.algorithm,
            "state_hash": tok.state_hash,
            "verified_at": datetime.now(timezone.utc).isoformat(),
        })

    def _handle_post_persist(self, body: dict[str, Any]) -> None:
        db_path = body.get("db_path")
        if not db_path:
            raise ValueError("Field 'db_path' is required.")
        orch = self.server.orchestrator
        orch.save_to_disk(db_path)
        self._send_json(HTTPStatus.OK, {"status": "SAVED", "db_path": db_path})

    def _handle_post_restore(self, body: dict[str, Any]) -> None:
        db_path = body.get("db_path")
        if not db_path:
            raise ValueError("Field 'db_path' is required.")
        orch = self.server.orchestrator
        orch.load_from_disk(db_path)
        self._send_json(HTTPStatus.OK, {"status": "RESTORED", "db_path": db_path})

    def _handle_post_compliance_prove(self, body: dict[str, Any]) -> None:
        orch = self.server.orchestrator
        record_id = body.get("record_id")
        if not record_id or not isinstance(record_id, str):
            raise ValueError("Field 'record_id' must be a non-empty string.")
        query = str(body.get("query", ""))
        min_rank = int(body.get("min_required_authority_rank", 1))
        tenant_id = self.headers.get("X-Tenant-ID") or body.get("tenant_id")
        at_time = datetime.fromisoformat(body["at_time"]) if body.get("at_time") else None
        as_of_tx = (
            datetime.fromisoformat(body["as_of_transaction_time"])
            if body.get("as_of_transaction_time")
            else None
        )
        bundle = orch.generate_compliance_proof(
            record_id=record_id,
            query=query,
            min_required_authority_rank=min_rank,
            at_time=at_time,
            as_of_transaction_time=as_of_tx,
            tenant_id=tenant_id,
        )
        self._send_json(
            HTTPStatus.OK,
            {
                "proof": bundle.proof.to_dict(),
                "text_salt": bundle.text_salt,
                "query_salt": bundle.query_salt,
            },
        )

    def _handle_post_compliance_verify(self, body: dict[str, Any]) -> None:
        orch = self.server.orchestrator
        proof = body.get("proof")
        if not proof or not isinstance(proof, dict):
            raise ValueError("Field 'proof' is required and must be a dictionary.")
        valid = orch.verify_compliance_proof(
            proof=proof,
            expected_merkle_root=body.get("expected_merkle_root"),
            min_authority_rank=(
                int(body["min_authority_rank"])
                if body.get("min_authority_rank") is not None
                else None
            ),
            required_jurisdiction=body.get("required_jurisdiction"),
        )
        self._send_json(
            HTTPStatus.OK,
            {
                "valid": valid,
                "proof_id": proof.get("proof_id"),
                "record_id": proof.get("record_id"),
                "merkle_root": proof.get("merkle_root"),
            },
        )

    def _handle_post_bundle_export(self, body: dict[str, Any]) -> None:
        orch = self.server.orchestrator
        tenant_id = self.headers.get("X-Tenant-ID") or body.get("tenant_id", "default")
        redact = bool(body.get("redact_personal_data", False))
        bundle = orch.export_compliance_bundle(
            tenant_id=tenant_id,
            redact_personal_data=redact,
        )
        self._send_json(HTTPStatus.OK, bundle.to_dict())

    def _handle_post_bundle_restore(self, body: dict[str, Any]) -> None:
        orch = self.server.orchestrator
        bundle_data = body.get("bundle", body)
        restored = orch.restore_compliance_bundle(bundle_data)
        self._send_json(
            HTTPStatus.OK,
            {
                "status": "RESTORED",
                "tenant_id": restored.tenant_id,
                "merkle_root": restored.merkle_root,
                "records_restored": len(restored.records),
            },
        )

    def _handle_get_keys(self) -> None:
        orch = self.server.orchestrator
        self._send_json(
            HTTPStatus.OK,
            {
                "active_key_id": orch.keyring.active_key_id,
                "keys": orch.list_signing_keys(),
            },
        )

    def _handle_post_keys_rotate(self, body: dict[str, Any]) -> None:
        orch = self.server.orchestrator
        meta = orch.rotate_signing_key(
            new_key_id=body.get("new_key_id"),
            algorithm=body.get("algorithm"),
        )
        self._send_json(HTTPStatus.OK, meta.to_dict())

    def _handle_post_keys_revoke(self, body: dict[str, Any]) -> None:
        orch = self.server.orchestrator
        key_id = body.get("key_id")
        if not key_id or not isinstance(key_id, str):
            raise ValueError("Field 'key_id' must be a non-empty string.")
        meta = orch.revoke_signing_key(key_id)
        self._send_json(HTTPStatus.OK, meta.to_dict())

    def _handle_post_federated_verify(self, body: dict[str, Any]) -> None:
        orch = self.server.orchestrator
        proofs = body.get("proofs")
        if not isinstance(proofs, list) or not proofs:
            raise ValueError("Field 'proofs' must be a non-empty list of compliance proofs.")
        report = orch.verify_cross_party_alignment(
            proofs=proofs,
            min_authority_rank=int(body.get("min_authority_rank", 1)),
            min_action_alignment=float(body.get("min_action_alignment", 0.0)),
            require_temporal_validity=bool(body.get("require_temporal_validity", True)),
        )
        self._send_json(HTTPStatus.OK, report.to_dict())

    def _handle_post_tee_quote(self, body: dict[str, Any]) -> None:
        orch = self.server.orchestrator
        tenant_id = self.headers.get("X-Tenant-ID") or body.get("tenant_id", "default")
        quote = orch.generate_tee_quote(
            nonce=body.get("nonce"),
            platform=str(body.get("platform", "AWS_NITRO")),
            tenant_id=tenant_id,
        )
        self._send_json(HTTPStatus.OK, quote.to_dict())

    def _handle_post_tee_verify(self, body: dict[str, Any]) -> None:
        orch = self.server.orchestrator
        quote_data = body.get("quote")
        if not isinstance(quote_data, dict):
            raise ValueError("Field 'quote' must be a valid TEEAttestationQuote dictionary.")
        report = orch.verify_tee_quote(
            quote=quote_data,
            expected_nonce=body.get("expected_nonce"),
            expected_merkle_root=body.get("expected_merkle_root"),
            expected_measurements=body.get("expected_measurements"),
        )
        self._send_json(HTTPStatus.OK, report.to_dict())

    def _handle_post_audit_smtlib(self, body: dict[str, Any]) -> None:
        orch = self.server.orchestrator
        tenant_id = self.headers.get("X-Tenant-ID") or body.get("tenant_id", "default")
        at_time = datetime.fromisoformat(body["at_time"]) if body.get("at_time") else None
        bundle = orch.export_smtlib(
            query_text=body.get("query_text"),
            tenant_id=tenant_id,
            at_time=at_time,
        )
        self._send_json(HTTPStatus.OK, bundle.to_dict())

    def _handle_post_rules_timeline(self, body: dict[str, Any]) -> None:
        orch = self.server.orchestrator
        tenant_id = self.headers.get("X-Tenant-ID") or body.get("tenant_id", "default")
        start_time = datetime.fromisoformat(body["start_time"]) if body.get("start_time") else None
        end_time = datetime.fromisoformat(body["end_time"]) if body.get("end_time") else None
        report = orch.simulate_temporal_horizon(
            query_text=body.get("query_text"),
            start_time=start_time,
            end_time=end_time,
            tenant_id=tenant_id,
        )
        self._send_json(HTTPStatus.OK, report.to_dict())

    def _handle_post_audit_standalone_verifier(self, body: dict[str, Any]) -> None:
        orch = self.server.orchestrator
        tenant_id = self.headers.get("X-Tenant-ID") or body.get("tenant_id", "default")
        artifact = orch.export_standalone_verifier(
            tenant_id=tenant_id,
            redact_personal_data=bool(body.get("redact_personal_data", False)),
            query_text=body.get("query_text"),
        )
        self._send_json(HTTPStatus.OK, artifact.to_dict())

    def _handle_post_merkle_timestamp(self, body: dict[str, Any]) -> None:
        orch = self.server.orchestrator
        tenant_id = self.headers.get("X-Tenant-ID") or body.get("tenant_id", "default")
        token = orch.issue_timestamp_token(
            tenant_id=tenant_id,
            nonce=body.get("nonce"),
        )
        self._send_json(HTTPStatus.OK, token.to_dict())

    def _handle_post_merkle_gossip_audit(self, body: dict[str, Any]) -> None:
        orch = self.server.orchestrator
        tenant_id = self.headers.get("X-Tenant-ID") or body.get("tenant_id", "default")
        checkpoints = body.get("checkpoints")
        if not isinstance(checkpoints, list):
            raise ValueError("Field 'checkpoints' must be a list of GossipCheckpoint objects.")
        report = orch.audit_transparency_gossip(
            checkpoints=checkpoints,
            tenant_id=tenant_id,
        )
        self._send_json(HTTPStatus.OK, report.to_dict())

    def _handle_post_rules_amendment_impact(self, body: dict[str, Any]) -> None:
        orch = self.server.orchestrator
        tenant_id = self.headers.get("X-Tenant-ID") or body.get("tenant_id", "default")
        target_rule_id = body.get("target_rule_id")
        amended_text = body.get("amended_text")
        if not target_rule_id or not isinstance(target_rule_id, str):
            raise ValueError("Field 'target_rule_id' must be a non-empty string.")
        if not amended_text or not isinstance(amended_text, str):
            raise ValueError("Field 'amended_text' must be a non-empty string.")
        act_vec = (
            [float(x) for x in body["amended_action_vector"]]
            if body.get("amended_action_vector") is not None
            else None
        )
        new_rank = (
            int(body["amended_authority_rank"])
            if body.get("amended_authority_rank") is not None
            else None
        )
        report = orch.analyze_statutory_amendment(
            target_rule_id=target_rule_id,
            amended_text=amended_text,
            amended_action_vector=act_vec,
            amended_authority_rank=new_rank,
            tenant_id=tenant_id,
        )
        self._send_json(HTTPStatus.OK, report.to_dict())

    def _handle_post_mcp(self, body: dict[str, Any]) -> None:
        mcp_server = MCPServer(self.server.orchestrator)
        resp = mcp_server.handle_request(body)
        if resp is None:
            self.send_response(HTTPStatus.NO_CONTENT)
            self.end_headers()
            return
        self._send_json(HTTPStatus.OK, resp)


class LegalMemoryHTTPServer(ThreadingHTTPServer):
    """
    Multi-threaded HTTP Server binding LegalMemoryOrchestrator to incoming REST requests.
    """

    def __init__(
        self,
        server_address: tuple[str, int],
        orchestrator: LegalMemoryOrchestrator,
        telemetry_logger: TelemetryLogger | None = None,
    ) -> None:
        self.orchestrator = orchestrator
        self.telemetry_logger = telemetry_logger or getattr(orchestrator, "telemetry", None)
        super().__init__(server_address, LegalMemoryHTTPRequestHandler)


def create_server(
    orchestrator: LegalMemoryOrchestrator,
    host: str = "0.0.0.0",
    port: int = 8000,
    telemetry_logger: TelemetryLogger | None = None,
) -> LegalMemoryHTTPServer:
    """
    Factory function creating an unstarted LegalMemoryHTTPServer instance.
    """
    return LegalMemoryHTTPServer((host, port), orchestrator, telemetry_logger=telemetry_logger)


def run_server(
    orchestrator: LegalMemoryOrchestrator,
    host: str = "0.0.0.0",
    port: int = 8000,
) -> None:
    """
    Start the LegalMemory REST API server and block until interrupted.
    """
    server = create_server(orchestrator, host=host, port=port)
    logger.info("Serving Continuous Legal Memory REST API on http://%s:%d", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal; shutting down REST API server...")
    finally:
        server.server_close()
