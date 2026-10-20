"""
Model Context Protocol (MCP) JSON-RPC 2.0 Server for Continuous Legal Memory.

Exposes deterministic legal memory tools (`legal_memory_ingest`, `legal_memory_predict`,
`legal_memory_erase`, `legal_memory_associate`, `legal_memory_verify`) and contextual
resources (`clm://context/{tenant_id}`, `clm://graph/{tenant_id}`, `clm://metrics`)
over standard JSON-RPC 2.0 (stdio and HTTP) with zero external protocol dependencies.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from typing import Any, TextIO

from continuous_legal_memory.domain.models import PredictionResult
from continuous_legal_memory.orchestrator import LegalMemoryOrchestrator
from continuous_legal_memory.security.attestation import (
    AsymmetricAttestationModule,
    KeyedHashAttestationModule,
    MemoryAttestationToken,
)
from continuous_legal_memory.telemetry.observability import export_prometheus_metrics

MCP_PROTOCOL_VERSION = "2024-11-05"


class MCPServer:
    """
    Zero-dependency Model Context Protocol (MCP) JSON-RPC 2.0 server for Continuous Legal Memory.

    Supports standard MCP lifecycle methods (`initialize`, `notifications/initialized`, `ping`),
    governed tool discovery/invocation (`tools/list`, `tools/call`), and tenant-scoped resource
    inspection (`resources/list`, `resources/read`).
    """

    def __init__(
        self,
        orchestrator: LegalMemoryOrchestrator,
        server_name: str = "continuous-legal-memory",
        server_version: str = "0.2.0",
    ) -> None:
        self.orchestrator = orchestrator
        self.server_name = server_name
        self.server_version = server_version

    def list_tools(self) -> list[dict[str, Any]]:
        """Return MCP tool definitions with JSON Schema input validation specs."""
        return [
            {
                "name": "legal_memory_ingest",
                "description": (
                    "Ingest a legal statute, regulatory directive, or client preference into "
                    "multi-tier cognitive memory with authority rank, specificity, and temporal bounds."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "rule_text": {"type": "string", "description": "Normative text of the legal rule."},
                        "action_vector": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "Target decision action vector (e.g. [0.0, 1.0]).",
                        },
                        "authority_rank": {
                            "type": "integer",
                            "default": 1,
                            "description": "Lex superior hierarchical authority tier (1-10).",
                        },
                        "specificity": {
                            "type": "number",
                            "default": 0.0,
                            "description": "Lex specialis specificity score within equal authority tiers.",
                        },
                        "tenant_id": {
                            "type": "string",
                            "default": "default",
                            "description": "Tenant namespace for multi-tenant isolation.",
                        },
                        "jurisdiction": {"type": "string", "description": "Optional jurisdictional code."},
                        "personal_data": {
                            "type": "boolean",
                            "default": False,
                            "description": "Flag for GDPR Art. 17 personal data confinement.",
                        },
                        "valid_from": {"type": "string", "description": "ISO 8601 start of validity."},
                        "valid_to": {"type": "string", "description": "ISO 8601 end of validity."},
                        "transaction_time": {"type": "string", "description": "Optional ISO 8601 transaction ingestion timestamp."},
                        "supersedes": {"type": "string", "description": "Optional record_id superseded by this rule."},
                    },
                    "required": ["rule_text", "action_vector"],
                },
            },
            {
                "name": "legal_memory_predict",
                "description": (
                    "Evaluate a legal query against active bi-temporal norms using tri-canon "
                    "precedence (lex superior, lex specialis, lex posterior) and optional epistemic abstention."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query_text": {"type": "string", "description": "Legal scenario or compliance query."},
                        "tenant_id": {"type": "string", "default": "default", "description": "Tenant namespace."},
                        "at_time": {"type": "string", "description": "Optional ISO 8601 valid-time evaluation timestamp."},
                        "as_of_transaction_time": {
                            "type": "string",
                            "description": "Optional ISO 8601 transaction-time cutoff for bi-temporal 'as-of' replay.",
                        },
                        "min_similarity_threshold": {
                            "type": "number",
                            "description": "Optional minimum cosine similarity threshold for epistemic abstention.",
                        },
                        "jurisdiction": {
                            "type": "string",
                            "description": "Optional jurisdiction filter.",
                        },
                        "connecting_factors": {
                            "type": "object",
                            "description": "Optional Private International Law connecting factors dictionary.",
                        },
                    },
                    "required": ["query_text"],
                },
            },
            {
                "name": "legal_memory_erase",
                "description": (
                    "Execute cryptographically attested GDPR Article 17 Right-to-Erasure across "
                    "Working, Episodic, Semantic Graph, and Neural Buffer memory tiers."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "record_id": {"type": "string", "description": "Unique record_id to permanently erase."},
                        "tenant_id": {"type": "string", "description": "Optional tenant authorization scope."},
                    },
                    "required": ["record_id"],
                },
            },
            {
                "name": "legal_memory_associate",
                "description": (
                    "Discover multi-hop related legal statutes and clauses via HippoRAG Personalized PageRank "
                    "over the tenant-scoped Semantic Knowledge Graph."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "seed_nodes": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of seed node IDs for Personalized PageRank.",
                        },
                        "damping": {"type": "number", "default": 0.85},
                        "max_results": {"type": "integer", "default": 10},
                        "tenant_id": {"type": "string", "default": "default"},
                    },
                    "required": ["seed_nodes"],
                },
            },
            {
                "name": "legal_memory_verify",
                "description": "Verify the cryptographic integrity of an HMAC-SHA256 or Ed25519 attestation token.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "token": {"type": "object", "description": "Serialized MemoryAttestationToken dictionary."},
                        "query": {"type": "string", "description": "Original query string."},
                        "predicted_action_vector": {
                            "type": "array",
                            "items": {"type": "number"},
                        },
                        "most_relevant_rule": {"type": "string"},
                        "public_key": {"type": "string", "description": "Optional hex-encoded Ed25519 public key."},
                    },
                    "required": ["token", "query", "predicted_action_vector"],
                },
            },
            {
                "name": "legal_memory_audit_consistency",
                "description": (
                    "Audit active legal memory records and semantic knowledge graph for deontic cycles "
                    "and unresolved horizontal antinomies."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "tenant_id": {"type": "string", "default": "default"},
                        "register_contradiction_edges": {"type": "boolean", "default": True},
                    },
                },
            },
            {
                "name": "legal_memory_compliance_prove",
                "description": (
                    "Generate a Zero-Knowledge / Selective-Disclosure Normative Compliance Proof "
                    "(EU AI Act Art. 12 & 78) without exposing confidential text or query in plaintext."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "record_id": {"type": "string"},
                        "query": {"type": "string", "default": ""},
                        "min_required_authority_rank": {"type": "integer", "default": 1},
                        "tenant_id": {"type": "string", "default": "default"},
                    },
                    "required": ["record_id"],
                },
            },
            {
                "name": "legal_memory_compliance_verify",
                "description": (
                    "Verify a Selective-Disclosure Normative Compliance Proof without accessing plaintext text."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "proof": {"type": "object"},
                        "expected_merkle_root": {"type": "string"},
                        "min_authority_rank": {"type": "integer"},
                        "required_jurisdiction": {"type": "string"},
                    },
                    "required": ["proof"],
                },
            },
            {
                "name": "legal_memory_bundle_export",
                "description": "Export a cryptographically signed .clm.bundle.json compliance snapshot.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "tenant_id": {"type": "string", "default": "default"},
                        "redact_personal_data": {"type": "boolean", "default": False},
                    },
                },
            },
            {
                "name": "legal_memory_simulate_rule",
                "description": (
                    "Execute a non-mutating Pre-Flight Standard Deontic Logic (KD) & Defeasible Precedence "
                    "simulation for a candidate rule prior to committing it to memory."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "rule_text": {"type": "string", "description": "Candidate legal rule text."},
                        "action_vector": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "Candidate target decision vector.",
                        },
                        "authority_rank": {"type": "integer", "default": 1},
                        "jurisdiction": {"type": "string"},
                        "tenant_id": {"type": "string", "default": "default"},
                        "metadata": {"type": "object"},
                    },
                    "required": ["rule_text"],
                },
            },
            {
                "name": "legal_memory_rotate_key",
                "description": (
                    "Atomically rotate the active cryptographic signing key in CryptographicKeyRing, "
                    "retiring the previous key while preserving historical verification."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "new_key_id": {"type": "string"},
                        "algorithm": {
                            "type": "string",
                            "enum": ["Ed25519", "HMAC-SHA256", "PQ-Hybrid-Ed25519-SHA256"],
                        },
                    },
                },
            },
            {
                "name": "legal_memory_revoke_key",
                "description": "Permanently revoke a compromised signing key ID in CryptographicKeyRing.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "key_id": {"type": "string"},
                    },
                    "required": ["key_id"],
                },
            },
            {
                "name": "legal_memory_federated_verify",
                "description": (
                    "Verify multi-party Selective-Disclosure Normative Compliance Proofs for cryptographic "
                    "validity, minimum authority rank, temporal validity, and pairwise action vector alignment."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "proofs": {"type": "array", "items": {"type": "object"}},
                        "min_authority_rank": {"type": "integer", "default": 1},
                        "min_action_alignment": {"type": "number", "default": 0.0},
                        "require_temporal_validity": {"type": "boolean", "default": True},
                    },
                    "required": ["proofs"],
                },
            },
            {
                "name": "legal_memory_propose_repairs",
                "description": (
                    "Synthesize and pre-verify legislative or contractual drafting repair proposals "
                    "(EXPLICIT_SUPERSESSION, LEX_SPECIALIS_CARVEOUT, TEMPORAL_PARTITION) for a candidate rule."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "rule_text": {"type": "string", "description": "Candidate legal rule text."},
                        "action_vector": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "Candidate target decision vector.",
                        },
                        "authority_rank": {"type": "integer", "default": 1},
                        "jurisdiction": {"type": "string"},
                        "tenant_id": {"type": "string", "default": "default"},
                        "metadata": {"type": "object"},
                    },
                    "required": ["rule_text"],
                },
            },
            {
                "name": "legal_memory_tee_quote",
                "description": (
                    "Generate a hardware-anchored TEE Remote Attestation Quote (AWS_NITRO or INTEL_TDX) "
                    "binding the current SparseMerkleTree root and active signing key to enclave registers."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "nonce": {"type": "string"},
                        "platform": {"type": "string", "enum": ["AWS_NITRO", "INTEL_TDX"], "default": "AWS_NITRO"},
                        "tenant_id": {"type": "string", "default": "default"},
                    },
                },
            },
            {
                "name": "legal_memory_export_smtlib",
                "description": (
                    "Export active tenant norms and optional query DefeasibleProofTrace as a formal "
                    "SMT-LIB 2.6 specification (QF_UFLIA) for Z3 / cvc5 theorem provers."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query_text": {"type": "string"},
                        "tenant_id": {"type": "string", "default": "default"},
                    },
                },
            },
            {
                "name": "legal_memory_simulate_timeline",
                "description": (
                    "Simulate bi-temporal regulatory horizon transitions (valid_from / valid_to) "
                    "across time to forecast deontic modality shifts and future antinomies."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query_text": {"type": "string"},
                        "start_time": {"type": "string"},
                        "end_time": {"type": "string"},
                        "tenant_id": {"type": "string", "default": "default"},
                    },
                },
            },
            {
                "name": "legal_memory_export_verifier",
                "description": (
                    "Export a self-contained, zero-dependency HTML5 + WebCrypto standalone audit "
                    "verifier embedding the tenant's signed compliance bundle and SMT-LIB specification."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query_text": {"type": "string"},
                        "redact_personal_data": {"type": "boolean", "default": False},
                        "tenant_id": {"type": "string", "default": "default"},
                    },
                },
            },
            {
                "name": "legal_memory_issue_timestamp",
                "description": (
                    "Issue an RFC 3161-inspired cryptographic TimestampToken binding the tenant's "
                    "current 256-bit Sparse Merkle root and append-only audit log head hash."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "nonce": {"type": "string"},
                        "tenant_id": {"type": "string", "default": "default"},
                    },
                },
            },
            {
                "name": "legal_memory_gossip_audit",
                "description": (
                    "Audit cross-witness transparency log checkpoints against the local append-only "
                    "Merkle log to detect split-view equivocation and history rollback attacks."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "checkpoints": {
                            "type": "array",
                            "items": {"type": "object"},
                        },
                        "tenant_id": {"type": "string", "default": "default"},
                    },
                    "required": ["checkpoints"],
                },
            },
            {
                "name": "legal_memory_amendment_impact",
                "description": (
                    "Analyze clause-level legislative amendment diffs for a target rule and trace "
                    "the downstream compliance blast radius across subordinate contracts and policies."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "target_rule_id": {"type": "string"},
                        "amended_text": {"type": "string"},
                        "amended_action_vector": {
                            "type": "array",
                            "items": {"type": "number"},
                        },
                        "amended_authority_rank": {"type": "integer"},
                        "tenant_id": {"type": "string", "default": "default"},
                    },
                    "required": ["target_rule_id", "amended_text"],
                },
            },
        ]

    def list_resources(self) -> list[dict[str, Any]]:
        """Return available MCP resources exposed by the legal memory server."""
        return [
            {
                "uri": "clm://context/default",
                "name": "Active Working Memory Context (default tenant)",
                "description": "Temporally valid working memory records for the default tenant.",
                "mimeType": "application/json",
            },
            {
                "uri": "clm://graph/default",
                "name": "Semantic Knowledge Graph (default tenant)",
                "description": "Legal entity nodes and deontic relationship edges in the semantic graph.",
                "mimeType": "application/json",
            },
            {
                "uri": "clm://metrics",
                "name": "Continuous Legal Memory Prometheus Metrics",
                "description": "Operational telemetry and memory state counters in OpenMetrics format.",
                "mimeType": "text/plain",
            },
        ]

    def read_resource(self, uri: str) -> list[dict[str, Any]]:
        """Read a resource by URI (`clm://context/{tenant}`, `clm://graph/{tenant}`, or `clm://metrics`)."""
        if uri.startswith("clm://context/"):
            tenant_id = uri[len("clm://context/") :] or "default"
            records = self.orchestrator.get_working_memory_context(tenant_id=tenant_id)
            payload = {
                "tenant_id": tenant_id,
                "active_records": [
                    {
                        "record_id": r.record_id,
                        "text": r.text,
                        "authority_rank": r.authority_rank,
                        "jurisdiction": r.jurisdiction,
                        "personal_data": r.personal_data,
                        "valid_from": r.valid_from.isoformat() if r.valid_from else None,
                        "valid_to": r.valid_to.isoformat() if r.valid_to else None,
                    }
                    for r in records
                ],
            }
            return [{"uri": uri, "mimeType": "application/json", "text": json.dumps(payload)}]

        if uri.startswith("clm://graph/"):
            tenant_id = uri[len("clm://graph/") :] or "default"
            nodes = self.orchestrator.get_semantic_graph_nodes(tenant_id=tenant_id)
            payload = {
                "tenant_id": tenant_id,
                "nodes": [
                    {
                        "node_id": n.node_id,
                        "label": n.label,
                        "entity_type": n.entity_type.value if hasattr(n.entity_type, "value") else str(n.entity_type),
                        "description": n.description,
                    }
                    for n in nodes
                ],
            }
            return [{"uri": uri, "mimeType": "application/json", "text": json.dumps(payload)}]

        if uri == "clm://metrics":
            metrics_text = export_prometheus_metrics(
                self.orchestrator,
                getattr(self.orchestrator, "telemetry", None),
                service_name=self.server_name,
            )
            return [{"uri": uri, "mimeType": "text/plain", "text": metrics_text}]

        raise ValueError(f"Unknown MCP resource URI: '{uri}'")

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        """Execute a governed MCP tool call against the underlying LegalMemoryOrchestrator."""
        args = arguments or {}

        if name == "legal_memory_ingest":
            valid_from = (
                datetime.fromisoformat(args["valid_from"].replace("Z", "+00:00"))
                if args.get("valid_from")
                else None
            )
            valid_to = (
                datetime.fromisoformat(args["valid_to"].replace("Z", "+00:00"))
                if args.get("valid_to")
                else None
            )
            tx_time = (
                datetime.fromisoformat(args["transaction_time"].replace("Z", "+00:00"))
                if args.get("transaction_time")
                else None
            )
            metadata: dict[str, Any] = dict(args.get("metadata") or {})
            if "specificity" in args:
                metadata["specificity"] = float(args["specificity"])
            if args.get("supersedes"):
                metadata["supersedes"] = str(args["supersedes"])
            if args.get("node_id"):
                metadata["node_id"] = str(args["node_id"])

            rec = self.orchestrator.update_memory(
                rule_text=str(args["rule_text"]),
                action_vector=[float(x) for x in args["action_vector"]],
                valid_from=valid_from,
                valid_to=valid_to,
                metadata=metadata,
                authority_rank=int(args.get("authority_rank", 1)),
                jurisdiction=args.get("jurisdiction"),
                personal_data=bool(args.get("personal_data", False)),
                tenant_id=str(args.get("tenant_id", "default")),
                transaction_time=tx_time,
            )
            result_obj = {
                "status": "INGESTED",
                "record_id": rec.record_id,
                "tenant_id": rec.tenant_id,
                "authority_rank": rec.authority_rank,
                "personal_data": rec.personal_data,
                "merkle_root": self.orchestrator.merkle_tree.root_hex,
            }
            return {"content": [{"type": "text", "text": json.dumps(result_obj)}], "isError": False}

        if name == "legal_memory_predict":
            at_time = (
                datetime.fromisoformat(args["at_time"].replace("Z", "+00:00"))
                if args.get("at_time")
                else None
            )
            as_of_tx = (
                datetime.fromisoformat(args["as_of_transaction_time"].replace("Z", "+00:00"))
                if args.get("as_of_transaction_time")
                else None
            )
            threshold = (
                float(args["min_similarity_threshold"])
                if args.get("min_similarity_threshold") is not None
                else None
            )
            res = self.orchestrator.predict(
                query_text=str(args["query_text"]),
                at_time=at_time,
                tenant_id=args.get("tenant_id"),
                min_similarity_threshold=threshold,
                jurisdiction=args.get("jurisdiction"),
                connecting_factors=args.get("connecting_factors"),
                as_of_transaction_time=as_of_tx,
            )
            payload: dict[str, Any] = {
                "query": res.query,
                "predicted_action_vector": res.predicted_action_vector,
                "most_relevant_rule": res.most_relevant_rule,
                "confidence": res.confidence,
                "max_similarity": res.max_similarity,
                "abstained": res.abstained,
                "abstention_reason": res.abstention_reason,
                "conformal_set": res.conformal_set,
                "conformal_threshold": res.conformal_threshold,
                "conformal_alpha": res.conformal_alpha,
                "choice_of_law": res.choice_of_law,
                "proof_trace": res.proof_trace,
                "merkle_root": res.merkle_root,
                "tenant_id": res.tenant_id,
            }
            if res.attestation_token is not None:
                tok = res.attestation_token
                payload["attestation_token"] = {
                    "timestamp": tok.timestamp.isoformat(),
                    "state_hash": tok.state_hash,
                    "integrity_tag": tok.integrity_tag,
                    "key_id": tok.key_id,
                    "algorithm": tok.algorithm,
                    "public_key": tok.public_key,
                }
            return {"content": [{"type": "text", "text": json.dumps(payload)}], "isError": False}

        if name == "legal_memory_erase":
            audit = self.orchestrator.delete_rule(
                record_id=str(args["record_id"]),
                tenant_id=args.get("tenant_id"),
            )
            return {"content": [{"type": "text", "text": json.dumps(audit)}], "isError": False}

        if name == "legal_memory_associate":
            assoc = self.orchestrator.associate_statutes(
                seed_nodes=args["seed_nodes"],
                damping=float(args.get("damping", 0.85)),
                max_results=int(args.get("max_results", 10)),
                tenant_id=args.get("tenant_id"),
            )
            items = [
                {
                    "node_id": node.node_id,
                    "label": node.label,
                    "entity_type": node.entity_type.value if hasattr(node.entity_type, "value") else str(node.entity_type),
                    "description": node.description,
                    "score": score,
                }
                for node, score in assoc
            ]
            return {
                "content": [{"type": "text", "text": json.dumps({"results": items, "total": len(items)})}],
                "isError": False,
            }

        if name == "legal_memory_verify":
            if "merkle_proof" in args:
                mp = args["merkle_proof"]
                valid_mp = self.orchestrator.verify_merkle_proof(
                    mp,
                    expected_root_hash=args.get("expected_root_hash"),
                )
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(
                                {
                                    "valid": valid_mp,
                                    "proof_type": "sparse_merkle_tree",
                                    "rule_id": mp.get("rule_id"),
                                    "is_member": mp.get("is_member"),
                                    "root_hash": mp.get("root_hash"),
                                    "verified_at": datetime.now(timezone.utc).isoformat(),
                                }
                            ),
                        }
                    ],
                    "isError": False,
                }

            raw_tok = args["token"]
            ts = datetime.fromisoformat(raw_tok["timestamp"].replace("Z", "+00:00"))
            tok = MemoryAttestationToken(
                timestamp=ts,
                state_hash=raw_tok["state_hash"],
                integrity_tag=raw_tok["integrity_tag"],
                key_id=raw_tok.get("key_id", "default"),
                algorithm=raw_tok.get("algorithm", "HMAC-SHA256"),
                public_key=raw_tok.get("public_key"),
            )
            pred = PredictionResult(
                query=str(args["query"]),
                predicted_action_vector=[float(x) for x in args["predicted_action_vector"]],
                most_relevant_rule=args.get("most_relevant_rule"),
            )
            public_key = args.get("public_key") or tok.public_key
            if tok.algorithm == "Ed25519":
                verifier = AsymmetricAttestationModule(public_key=public_key)
                valid = verifier.verify_attestation(tok, pred, retrieved_text=pred.most_relevant_rule)
            else:
                hmac_mod = self.orchestrator.attestor or KeyedHashAttestationModule()
                valid = hmac_mod.verify_attestation(tok, pred, retrieved_text=pred.most_relevant_rule)

            verify_out = {
                "valid": valid,
                "algorithm": tok.algorithm,
                "state_hash": tok.state_hash,
                "verified_at": datetime.now(timezone.utc).isoformat(),
            }
            return {"content": [{"type": "text", "text": json.dumps(verify_out)}], "isError": False}

        if name == "legal_memory_audit_consistency":
            report = self.orchestrator.audit_deontic_consistency(
                tenant_id=str(args.get("tenant_id", "default")),
                register_contradiction_edges=bool(args.get("register_contradiction_edges", True)),
            )
            return {
                "content": [{"type": "text", "text": json.dumps(report.to_dict())}],
                "isError": False,
            }

        if name == "legal_memory_compliance_prove":
            bundle = self.orchestrator.generate_compliance_proof(
                record_id=str(args["record_id"]),
                query=str(args.get("query", "")),
                min_required_authority_rank=int(args.get("min_required_authority_rank", 1)),
                tenant_id=args.get("tenant_id"),
            )
            out = {
                "proof": bundle.proof.to_dict(),
                "text_salt": bundle.text_salt,
                "query_salt": bundle.query_salt,
            }
            return {"content": [{"type": "text", "text": json.dumps(out)}], "isError": False}

        if name == "legal_memory_compliance_verify":
            valid = self.orchestrator.verify_compliance_proof(
                proof=args["proof"],
                expected_merkle_root=args.get("expected_merkle_root"),
                min_authority_rank=(
                    int(args["min_authority_rank"])
                    if args.get("min_authority_rank") is not None
                    else None
                ),
                required_jurisdiction=args.get("required_jurisdiction"),
            )
            return {
                "content": [{"type": "text", "text": json.dumps({"valid": valid})}],
                "isError": False,
            }

        if name == "legal_memory_bundle_export":
            signed_bundle = self.orchestrator.export_compliance_bundle(
                tenant_id=str(args.get("tenant_id", "default")),
                redact_personal_data=bool(args.get("redact_personal_data", False)),
            )
            return {
                "content": [{"type": "text", "text": json.dumps(signed_bundle.to_dict())}],
                "isError": False,
            }

        if name == "legal_memory_simulate_rule":
            act_vec = (
                [float(x) for x in args["action_vector"]]
                if args.get("action_vector") is not None
                else None
            )
            sim_report = self.orchestrator.simulate_rule_ingestion(
                rule_text=str(args["rule_text"]),
                action_vector=act_vec,
                metadata=dict(args.get("metadata") or {}),
                authority_rank=int(args.get("authority_rank", 1)),
                jurisdiction=args.get("jurisdiction"),
                tenant_id=str(args.get("tenant_id", "default")),
            )
            return {
                "content": [{"type": "text", "text": json.dumps(sim_report.to_dict())}],
                "isError": False,
            }

        if name == "legal_memory_rotate_key":
            meta = self.orchestrator.rotate_signing_key(
                new_key_id=args.get("new_key_id"),
                algorithm=args.get("algorithm"),
            )
            return {
                "content": [{"type": "text", "text": json.dumps(meta.to_dict())}],
                "isError": False,
            }

        if name == "legal_memory_revoke_key":
            meta_rev = self.orchestrator.revoke_signing_key(str(args["key_id"]))
            return {
                "content": [{"type": "text", "text": json.dumps(meta_rev.to_dict())}],
                "isError": False,
            }

        if name == "legal_memory_federated_verify":
            fed_report = self.orchestrator.verify_cross_party_alignment(
                proofs=list(args["proofs"]),
                min_authority_rank=int(args.get("min_authority_rank", 1)),
                min_action_alignment=float(args.get("min_action_alignment", 0.0)),
                require_temporal_validity=bool(args.get("require_temporal_validity", True)),
            )
            return {
                "content": [{"type": "text", "text": json.dumps(fed_report.to_dict())}],
                "isError": False,
            }

        if name == "legal_memory_propose_repairs":
            rep_action = [float(x) for x in args["action_vector"]] if "action_vector" in args else None
            rep_report = self.orchestrator.propose_norm_repairs(
                rule_text=str(args["rule_text"]),
                action_vector=rep_action,
                authority_rank=int(args.get("authority_rank", 1)),
                jurisdiction=args.get("jurisdiction"),
                metadata=dict(args.get("metadata") or {}),
                tenant_id=str(args.get("tenant_id", "default")),
            )
            return {
                "content": [{"type": "text", "text": json.dumps(rep_report.to_dict())}],
                "isError": False,
            }

        if name == "legal_memory_tee_quote":
            tee_quote = self.orchestrator.generate_tee_quote(
                nonce=args.get("nonce"),
                platform=str(args.get("platform", "AWS_NITRO")),
                tenant_id=str(args.get("tenant_id", "default")),
            )
            return {
                "content": [{"type": "text", "text": json.dumps(tee_quote.to_dict())}],
                "isError": False,
            }

        if name == "legal_memory_export_smtlib":
            smt_bundle = self.orchestrator.export_smtlib(
                query_text=args.get("query_text"),
                tenant_id=str(args.get("tenant_id", "default")),
            )
            return {
                "content": [{"type": "text", "text": json.dumps(smt_bundle.to_dict())}],
                "isError": False,
            }

        if name == "legal_memory_simulate_timeline":
            st_dt = datetime.fromisoformat(args["start_time"]) if args.get("start_time") else None
            et_dt = datetime.fromisoformat(args["end_time"]) if args.get("end_time") else None
            tl_report = self.orchestrator.simulate_temporal_horizon(
                query_text=args.get("query_text"),
                start_time=st_dt,
                end_time=et_dt,
                tenant_id=str(args.get("tenant_id", "default")),
            )
            return {
                "content": [{"type": "text", "text": json.dumps(tl_report.to_dict())}],
                "isError": False,
            }

        if name == "legal_memory_export_verifier":
            ver_art = self.orchestrator.export_standalone_verifier(
                tenant_id=str(args.get("tenant_id", "default")),
                redact_personal_data=bool(args.get("redact_personal_data", False)),
                query_text=args.get("query_text"),
            )
            return {
                "content": [{"type": "text", "text": json.dumps(ver_art.to_dict())}],
                "isError": False,
            }

        if name == "legal_memory_issue_timestamp":
            tst = self.orchestrator.issue_timestamp_token(
                tenant_id=str(args.get("tenant_id", "default")),
                nonce=args.get("nonce"),
            )
            return {
                "content": [{"type": "text", "text": json.dumps(tst.to_dict())}],
                "isError": False,
            }

        if name == "legal_memory_gossip_audit":
            gossip_rep = self.orchestrator.audit_transparency_gossip(
                checkpoints=list(args["checkpoints"]),
                tenant_id=str(args.get("tenant_id", "default")),
            )
            return {
                "content": [{"type": "text", "text": json.dumps(gossip_rep.to_dict())}],
                "isError": False,
            }

        if name == "legal_memory_amendment_impact":
            am_vec = (
                [float(x) for x in args["amended_action_vector"]]
                if args.get("amended_action_vector") is not None
                else None
            )
            am_rank = (
                int(args["amended_authority_rank"])
                if args.get("amended_authority_rank") is not None
                else None
            )
            am_rep = self.orchestrator.analyze_statutory_amendment(
                target_rule_id=str(args["target_rule_id"]),
                amended_text=str(args["amended_text"]),
                amended_action_vector=am_vec,
                amended_authority_rank=am_rank,
                tenant_id=str(args.get("tenant_id", "default")),
            )
            return {
                "content": [{"type": "text", "text": json.dumps(am_rep.to_dict())}],
                "isError": False,
            }

        raise ValueError(f"Unknown MCP tool: '{name}'")

    def handle_request(self, message: dict[str, Any]) -> dict[str, Any] | None:
        """
        Process a single JSON-RPC 2.0 request or notification dictionary.

        Returns:
            A JSON-RPC 2.0 response dictionary, or None for notifications without an `id`.
        """
        method = message.get("method")
        req_id = message.get("id")
        params = message.get("params") or {}

        # Notifications (no id field) do not return a response object
        if req_id is None and method and method.startswith("notifications/"):
            return None

        try:
            if method == "initialize":
                result: dict[str, Any] = {
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "capabilities": {
                        "tools": {"listChanged": False},
                        "resources": {"subscribe": False, "listChanged": False},
                    },
                    "serverInfo": {
                        "name": self.server_name,
                        "version": self.server_version,
                    },
                }
            elif method == "ping":
                result = {}
            elif method == "tools/list":
                result = {"tools": self.list_tools()}
            elif method == "tools/call":
                tool_name = params.get("name")
                if not tool_name:
                    raise ValueError("Missing required parameter 'name' for tools/call.")
                result = self.call_tool(str(tool_name), params.get("arguments"))
            elif method == "resources/list":
                result = {"resources": self.list_resources()}
            elif method == "resources/read":
                uri = params.get("uri")
                if not uri:
                    raise ValueError("Missing required parameter 'uri' for resources/read.")
                result = {"contents": self.read_resource(str(uri))}
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {
                        "code": -32601,
                        "message": f"Method not found: {method}",
                    },
                }

            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": result,
            }
        except Exception as exc:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {
                    "code": -32602 if isinstance(exc, (ValueError, KeyError)) else -32603,
                    "message": str(exc),
                },
            }

    def serve_stdio(
        self,
        input_stream: TextIO | None = None,
        output_stream: TextIO | None = None,
    ) -> None:
        """Run the MCP JSON-RPC 2.0 server over newline-delimited stdio."""
        in_stream = input_stream or sys.stdin
        out_stream = output_stream or sys.stdout

        for raw_line in in_stream:
            line = raw_line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError as err:
                err_resp = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": f"Parse error: {err}"},
                }
                out_stream.write(json.dumps(err_resp) + "\n")
                out_stream.flush()
                continue

            resp = self.handle_request(msg)
            if resp is not None:
                out_stream.write(json.dumps(resp) + "\n")
                out_stream.flush()
