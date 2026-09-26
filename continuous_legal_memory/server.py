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

from continuous_legal_memory.domain.exceptions import (
    ContinuousLegalMemoryError,
    InvalidMemoryVectorError,
)
from continuous_legal_memory.domain.models import PredictionResult
from continuous_legal_memory.orchestrator import LegalMemoryOrchestrator
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
            elif path == "/v1/predict":
                self._handle_post_predict(body)
            elif path == "/v1/associate":
                self._handle_post_associate(body)
            elif path == "/v1/attestation/verify":
                self._handle_post_verify_attestation(body)
            elif path == "/v1/persist":
                self._handle_post_persist(body)
            elif path == "/v1/restore":
                self._handle_post_restore(body)
            else:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Endpoint '{path}' not found."})
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
            "version": "0.1.0",
            "engine_mode": getattr(orch, "engine_mode", "structured"),
            "value_dim": getattr(orch, "value_dim", 2),
            "stored_records_total": total_records,
            "semantic_nodes_total": total_nodes,
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

        tenant_id = self.headers.get("X-Tenant-ID") or body.get("tenant_id", "default")
        authority_rank = int(body.get("authority_rank", 1))
        jurisdiction = body.get("jurisdiction")
        personal_data = bool(body.get("personal_data", False))
        metadata = body.get("metadata", {})

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
        )

        self._send_json(HTTPStatus.CREATED, {
            "status": "INGESTED",
            "record_id": rec.record_id,
            "tenant_id": rec.tenant_id,
            "authority_rank": rec.authority_rank,
            "personal_data": rec.personal_data,
            "created_at": rec.valid_from.isoformat() if rec.valid_from else datetime.now(timezone.utc).isoformat(),
        })

    def _handle_post_predict(self, body: dict[str, Any]) -> None:
        orch = self.server.orchestrator
        query_text = body.get("query_text")
        if not query_text or not isinstance(query_text, str):
            raise ValueError("Field 'query_text' must be a non-empty string.")

        at_time = None
        if body.get("at_time"):
            at_time = datetime.fromisoformat(body["at_time"])

        tenant_id = self.headers.get("X-Tenant-ID") or body.get("tenant_id")

        result = orch.predict(query_text=query_text, at_time=at_time, tenant_id=tenant_id)

        response_data: dict[str, Any] = {
            "query": result.query,
            "predicted_action_vector": result.predicted_action_vector,
            "source_tier": result.source_tier.value if (result.source_tier is not None and hasattr(result.source_tier, "value")) else str(result.source_tier),
            "most_relevant_rule": result.most_relevant_rule,
            "confidence": result.confidence,
            "attention_weights": result.attention_weights,
            "retrieved_snippets": result.retrieved_snippets,
            "tenant_id": result.tenant_id,
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

        associated = orch.associate_statutes(
            seed_nodes=seed_nodes,
            damping=damping,
            max_results=max_results,
            tenant_id=tenant_id,
            at_time=at_time,
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
