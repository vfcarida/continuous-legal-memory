"""
Signed Portable Compliance Bundles & Distributed Vector/Graph Exporters.

Provides:
1. Cryptographically signed `.clm.bundle.json` snapshot export, verification, and restoration
   for air-gapped regulatory inspection (EU AI Act Art. 12 & GDPR Art. 20 / Art. 30).
2. Zero-copy structured exporters for distributed vector and graph backends:
   - Qdrant multi-vector point payloads (`to_qdrant_points`)
   - PostgreSQL `pgvector` parameterized row tuples (`to_pgvector_rows`)
   - Neo4j parameterized Cypher `MERGE` statements (`to_neo4j_cypher`)
"""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

import torch

from continuous_legal_memory.domain.models import (
    EntityType,
    GraphEdge,
    GraphNode,
    MemoryRecord,
    MemoryTier,
    RelationType,
)
from continuous_legal_memory.security import ed25519
from continuous_legal_memory.security.attestation import (
    AsymmetricAttestationModule,
    KeyedHashAttestationModule,
)
from continuous_legal_memory.security.merkle import (
    MerkleLogEntry,
    SparseMerkleTree,
    rule_id_to_key_int,
)
from continuous_legal_memory.security.pq_hybrid import (
    PQ_HYBRID_ALGORITHM,
    PostQuantumHybridAttestationModule,
    pq_hybrid_sign,
    pq_hybrid_verify,
)
from continuous_legal_memory.security.selective_disclosure import canonical_json_bytes


@dataclass(frozen=True)
class SignedComplianceBundle:
    """
    Immutable, cryptographically signed snapshot of a tenant's Continuous Legal Memory state.
    """

    bundle_version: str
    tenant_id: str
    exported_at: str
    merkle_root: str
    records: list[dict[str, Any]]
    graph_nodes: list[dict[str, Any]]
    graph_edges: list[dict[str, Any]]
    merkle_leaves: dict[str, str]
    audit_log: list[dict[str, Any]]
    algorithm: str
    signature: str
    public_key: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert bundle to a JSON-serializable dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialize bundle to a formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SignedComplianceBundle:
        """Reconstruct a `SignedComplianceBundle` from a dictionary."""
        return cls(
            bundle_version=str(data.get("bundle_version", "1.0")),
            tenant_id=str(data["tenant_id"]),
            exported_at=str(data["exported_at"]),
            merkle_root=str(data["merkle_root"]),
            records=list(data.get("records", [])),
            graph_nodes=list(data.get("graph_nodes", [])),
            graph_edges=list(data.get("graph_edges", [])),
            merkle_leaves=dict(data.get("merkle_leaves", {})),
            audit_log=list(data.get("audit_log", [])),
            algorithm=str(data.get("algorithm", "Ed25519")),
            signature=str(data["signature"]),
            public_key=data.get("public_key"),
        )

    @classmethod
    def from_json(cls, raw_json: str) -> SignedComplianceBundle:
        """Reconstruct a `SignedComplianceBundle` from a JSON string."""
        return cls.from_dict(json.loads(raw_json))


class PortableBundleManager:
    """
    Exports, cryptographically verifies, and restores `SignedComplianceBundle` archives.
    """

    def __init__(
        self,
        attestation_module: (
            AsymmetricAttestationModule
            | KeyedHashAttestationModule
            | PostQuantumHybridAttestationModule
            | None
        ) = None,
    ) -> None:
        self.attestation_module: (
            AsymmetricAttestationModule
            | KeyedHashAttestationModule
            | PostQuantumHybridAttestationModule
        ) = (
            attestation_module if attestation_module is not None else AsymmetricAttestationModule()
        )

    @staticmethod
    def _canonical_bundle_bytes(
        bundle_version: str,
        tenant_id: str,
        exported_at: str,
        merkle_root: str,
        records: list[dict[str, Any]],
        graph_nodes: list[dict[str, Any]],
        graph_edges: list[dict[str, Any]],
        merkle_leaves: dict[str, str],
        audit_log: list[dict[str, Any]],
        algorithm: str,
    ) -> bytes:
        payload = {
            "algorithm": algorithm,
            "audit_log": audit_log,
            "bundle_version": bundle_version,
            "exported_at": exported_at,
            "graph_edges": graph_edges,
            "graph_nodes": graph_nodes,
            "merkle_leaves": merkle_leaves,
            "merkle_root": merkle_root,
            "records": records,
            "tenant_id": tenant_id,
        }
        return canonical_json_bytes(payload)

    def export_bundle(
        self,
        records: list[MemoryRecord],
        graph_nodes: list[GraphNode],
        graph_edges: list[GraphEdge],
        merkle_tree: SparseMerkleTree,
        tenant_id: str = "default",
        redact_personal_data: bool = False,
    ) -> SignedComplianceBundle:
        """
        Export a cryptographically signed compliance bundle for `tenant_id`.
        """
        serialized_records: list[dict[str, Any]] = []
        for r in records:
            if r.tenant_id != tenant_id:
                continue
            text_val = "[REDACTED_PERSONAL_DATA]" if (redact_personal_data and r.personal_data) else r.text
            serialized_records.append(
                {
                    "record_id": r.record_id,
                    "text": text_val,
                    "key_vector": [float(x) for x in r.key_vector.detach().cpu().flatten().tolist()],
                    "value_vector": [float(x) for x in r.value_vector.detach().cpu().flatten().tolist()],
                    "importance_score": float(r.importance_score),
                    "tier": r.tier.value,
                    "valid_from": r.valid_from.isoformat(),
                    "valid_to": r.valid_to.isoformat() if r.valid_to is not None else None,
                    "transaction_time": r.transaction_time.isoformat(),
                    "metadata": dict(r.metadata),
                    "authority_rank": int(r.authority_rank),
                    "jurisdiction": r.jurisdiction,
                    "personal_data": bool(r.personal_data),
                    "tenant_id": r.tenant_id,
                }
            )

        serialized_nodes: list[dict[str, Any]] = []
        for n in graph_nodes:
            if n.tenant_id != tenant_id:
                continue
            serialized_nodes.append(
                {
                    "node_id": n.node_id,
                    "entity_type": n.entity_type.value,
                    "label": n.label,
                    "description": n.description,
                    "embedding": (
                        [float(x) for x in n.embedding.detach().cpu().flatten().tolist()]
                        if n.embedding is not None
                        else None
                    ),
                    "valid_from": n.valid_from.isoformat(),
                    "valid_to": n.valid_to.isoformat() if n.valid_to is not None else None,
                    "transaction_time": n.transaction_time.isoformat(),
                    "decay_factor": float(n.decay_factor),
                    "tenant_id": n.tenant_id,
                }
            )

        serialized_edges: list[dict[str, Any]] = []
        for e in graph_edges:
            if e.tenant_id != tenant_id:
                continue
            serialized_edges.append(
                {
                    "source_id": e.source_id,
                    "target_id": e.target_id,
                    "relation_type": e.relation_type.value,
                    "weight": float(e.weight),
                    "tenant_id": e.tenant_id,
                }
            )

        merkle_leaves = {rid: lhash.hex() for rid, lhash in merkle_tree._active_rules.items()}
        merkle_root = merkle_tree.root_hex
        audit_log = [entry.to_dict() for entry in merkle_tree.audit_log.entries]

        exported_at = datetime.now(timezone.utc).isoformat()
        bundle_version = "1.0"

        if isinstance(self.attestation_module, PostQuantumHybridAttestationModule):
            if self.attestation_module._private_key is None:
                raise PermissionError(
                    "Cannot sign bundle: PostQuantumHybridAttestationModule has no private key."
                )
            algorithm = PQ_HYBRID_ALGORITHM
            signable = self._canonical_bundle_bytes(
                bundle_version=bundle_version,
                tenant_id=tenant_id,
                exported_at=exported_at,
                merkle_root=merkle_root,
                records=serialized_records,
                graph_nodes=serialized_nodes,
                graph_edges=serialized_edges,
                merkle_leaves=merkle_leaves,
                audit_log=audit_log,
                algorithm=algorithm,
            )
            signature = pq_hybrid_sign(signable, self.attestation_module._private_key)
            public_key: str | None = self.attestation_module.public_key_hex
        elif isinstance(self.attestation_module, AsymmetricAttestationModule):
            if self.attestation_module._private_key is None:
                raise PermissionError("Cannot sign bundle: AsymmetricAttestationModule has no private key.")
            algorithm = "Ed25519"
            signable = self._canonical_bundle_bytes(
                bundle_version=bundle_version,
                tenant_id=tenant_id,
                exported_at=exported_at,
                merkle_root=merkle_root,
                records=serialized_records,
                graph_nodes=serialized_nodes,
                graph_edges=serialized_edges,
                merkle_leaves=merkle_leaves,
                audit_log=audit_log,
                algorithm=algorithm,
            )
            signature = ed25519.sign(signable, self.attestation_module._private_key).hex()
            public_key = self.attestation_module.public_key_hex
        elif isinstance(self.attestation_module, KeyedHashAttestationModule):
            algorithm = "HMAC-SHA256"
            signable = self._canonical_bundle_bytes(
                bundle_version=bundle_version,
                tenant_id=tenant_id,
                exported_at=exported_at,
                merkle_root=merkle_root,
                records=serialized_records,
                graph_nodes=serialized_nodes,
                graph_edges=serialized_edges,
                merkle_leaves=merkle_leaves,
                audit_log=audit_log,
                algorithm=algorithm,
            )
            signature = hmac.new(
                self.attestation_module._secret_key, signable, hashlib.sha256
            ).hexdigest()
            public_key = None
        else:
            raise TypeError(f"Unsupported attestation module: {type(self.attestation_module)}")

        return SignedComplianceBundle(
            bundle_version=bundle_version,
            tenant_id=tenant_id,
            exported_at=exported_at,
            merkle_root=merkle_root,
            records=serialized_records,
            graph_nodes=serialized_nodes,
            graph_edges=serialized_edges,
            merkle_leaves=merkle_leaves,
            audit_log=audit_log,
            algorithm=algorithm,
            signature=signature,
            public_key=public_key,
        )

    def verify_bundle(self, bundle: SignedComplianceBundle | dict[str, Any]) -> bool:
        """
        Verify both the cryptographic signature and the 256-bit Sparse Merkle root of `bundle`.
        """
        b = SignedComplianceBundle.from_dict(bundle) if isinstance(bundle, dict) else bundle
        signable = self._canonical_bundle_bytes(
            bundle_version=b.bundle_version,
            tenant_id=b.tenant_id,
            exported_at=b.exported_at,
            merkle_root=b.merkle_root,
            records=b.records,
            graph_nodes=b.graph_nodes,
            graph_edges=b.graph_edges,
            merkle_leaves=b.merkle_leaves,
            audit_log=b.audit_log,
            algorithm=b.algorithm,
        )

        if b.algorithm == PQ_HYBRID_ALGORITHM:
            if not b.public_key:
                return False
            if not pq_hybrid_verify(b.signature, signable, b.public_key):
                return False
        elif b.algorithm == "Ed25519":
            if not b.public_key:
                return False
            if not ed25519.verify(b.signature, signable, b.public_key):
                return False
        elif b.algorithm == "HMAC-SHA256":
            if not isinstance(self.attestation_module, KeyedHashAttestationModule):
                return False
            expected_mac = hmac.new(
                self.attestation_module._secret_key, signable, hashlib.sha256
            ).hexdigest()
            if not hmac.compare_digest(b.signature, expected_mac):
                return False
        else:
            return False

        # Reconstruct SparseMerkleTree from merkle_leaves and verify root & audit log chain
        reconstructed_smt = SparseMerkleTree()
        for rid, lhash_hex in b.merkle_leaves.items():
            try:
                leaf_bytes = bytes.fromhex(lhash_hex)
            except ValueError:
                return False
            reconstructed_smt._active_rules[rid] = leaf_bytes
            reconstructed_smt._set_leaf(rule_id_to_key_int(rid), leaf_bytes)

        if reconstructed_smt.root_hex != b.merkle_root:
            return False

        for item in b.audit_log:
            reconstructed_smt.audit_log._entries.append(
                MerkleLogEntry(
                    seq_no=int(item["seq_no"]),
                    operation=str(item["operation"]),
                    rule_id=str(item["rule_id"]),
                    smt_root_hash=str(item["smt_root_hash"]),
                    prev_entry_hash=str(item["prev_entry_hash"]),
                    entry_hash=str(item["entry_hash"]),
                    timestamp=str(item["timestamp"]),
                    tenant_id=str(item.get("tenant_id", b.tenant_id)),
                )
            )
        return reconstructed_smt.audit_log.verify_chain()

    def restore_bundle(
        self,
        bundle: SignedComplianceBundle | dict[str, Any],
    ) -> tuple[list[MemoryRecord], list[GraphNode], list[GraphEdge], SparseMerkleTree]:
        """
        Verify and deserialize a `SignedComplianceBundle` back into domain objects.

        Raises:
            ValueError: If cryptographic signature or Merkle root verification fails.
        """
        b = SignedComplianceBundle.from_dict(bundle) if isinstance(bundle, dict) else bundle
        if not self.verify_bundle(b):
            raise ValueError(
                "SignedComplianceBundle verification failed: signature or Merkle root mismatch."
            )

        restored_records: list[MemoryRecord] = []
        for r in b.records:
            restored_records.append(
                MemoryRecord(
                    record_id=r.get("record_id"),
                    text=str(r["text"]),
                    key_vector=torch.tensor(r["key_vector"], dtype=torch.float32).unsqueeze(0),
                    value_vector=torch.tensor(r["value_vector"], dtype=torch.float32).unsqueeze(0),
                    importance_score=float(r.get("importance_score", 1.0)),
                    tier=MemoryTier(r.get("tier", MemoryTier.EPISODIC.value)),
                    valid_from=datetime.fromisoformat(r["valid_from"]),
                    valid_to=(
                        datetime.fromisoformat(r["valid_to"])
                        if r.get("valid_to") is not None
                        else None
                    ),
                    transaction_time=datetime.fromisoformat(r["transaction_time"]),
                    metadata=dict(r.get("metadata", {})),
                    authority_rank=int(r.get("authority_rank", 1)),
                    jurisdiction=r.get("jurisdiction"),
                    personal_data=bool(r.get("personal_data", False)),
                    tenant_id=str(r.get("tenant_id", b.tenant_id)),
                )
            )

        restored_nodes: list[GraphNode] = []
        for n in b.graph_nodes:
            emb_raw = n.get("embedding")
            restored_nodes.append(
                GraphNode(
                    node_id=str(n["node_id"]),
                    entity_type=EntityType(n["entity_type"]),
                    label=str(n["label"]),
                    description=str(n["description"]),
                    embedding=(
                        torch.tensor(emb_raw, dtype=torch.float32) if emb_raw is not None else None
                    ),
                    valid_from=datetime.fromisoformat(n["valid_from"]),
                    valid_to=(
                        datetime.fromisoformat(n["valid_to"])
                        if n.get("valid_to") is not None
                        else None
                    ),
                    transaction_time=datetime.fromisoformat(n["transaction_time"]),
                    decay_factor=float(n.get("decay_factor", 1.0)),
                    tenant_id=str(n.get("tenant_id", b.tenant_id)),
                )
            )

        restored_edges: list[GraphEdge] = []
        for e in b.graph_edges:
            restored_edges.append(
                GraphEdge(
                    source_id=str(e["source_id"]),
                    target_id=str(e["target_id"]),
                    relation_type=RelationType(e["relation_type"]),
                    weight=float(e.get("weight", 1.0)),
                    tenant_id=str(e.get("tenant_id", b.tenant_id)),
                )
            )

        restored_smt = SparseMerkleTree()
        for rid, lhash_hex in b.merkle_leaves.items():
            leaf_bytes = bytes.fromhex(lhash_hex)
            restored_smt._active_rules[rid] = leaf_bytes
            restored_smt._set_leaf(rule_id_to_key_int(rid), leaf_bytes)

        for item in b.audit_log:
            restored_smt.audit_log._entries.append(
                MerkleLogEntry(
                    seq_no=int(item["seq_no"]),
                    operation=str(item["operation"]),
                    rule_id=str(item["rule_id"]),
                    smt_root_hash=str(item["smt_root_hash"]),
                    prev_entry_hash=str(item["prev_entry_hash"]),
                    entry_hash=str(item["entry_hash"]),
                    timestamp=str(item["timestamp"]),
                    tenant_id=str(item.get("tenant_id", b.tenant_id)),
                )
            )

        return restored_records, restored_nodes, restored_edges, restored_smt


class DistributedVectorGraphExporter:
    """
    Exports Continuous Legal Memory records and Semantic Knowledge Graph topology into
    production payloads for Qdrant, PostgreSQL `pgvector`, and Neo4j Cypher.
    """

    @staticmethod
    def to_qdrant_points(
        records: list[MemoryRecord],
        merkle_tree: SparseMerkleTree | None = None,
    ) -> list[dict[str, Any]]:
        """
        Format `MemoryRecord` items as Qdrant multi-vector point structures with bi-temporal payload filters.
        """
        points: list[dict[str, Any]] = []
        for idx, rec in enumerate(records):
            rid = rec.record_id or f"rec_{idx}"
            leaf_bytes = merkle_tree._active_rules.get(rid) if merkle_tree is not None else None
            leaf_hash = leaf_bytes.hex() if leaf_bytes is not None else None
            points.append(
                {
                    "id": rid,
                    "vector": {
                        "key_vector": [
                            float(x) for x in rec.key_vector.detach().cpu().flatten().tolist()
                        ],
                        "value_vector": [
                            float(x) for x in rec.value_vector.detach().cpu().flatten().tolist()
                        ],
                    },
                    "payload": {
                        "record_id": rid,
                        "text": rec.text,
                        "tenant_id": rec.tenant_id,
                        "tier": rec.tier.value,
                        "jurisdiction": rec.jurisdiction,
                        "authority_rank": int(rec.authority_rank),
                        "importance_score": float(rec.importance_score),
                        "valid_from": rec.valid_from.isoformat(),
                        "valid_to": rec.valid_to.isoformat() if rec.valid_to is not None else None,
                        "transaction_time": rec.transaction_time.isoformat(),
                        "personal_data": bool(rec.personal_data),
                        "metadata": dict(rec.metadata),
                        "merkle_leaf_hash": leaf_hash,
                    },
                }
            )
        return points

    @staticmethod
    def to_pgvector_rows(
        records: list[MemoryRecord],
        merkle_tree: SparseMerkleTree | None = None,
    ) -> list[dict[str, Any]]:
        """
        Format `MemoryRecord` items as PostgreSQL `pgvector` SQL-ready parameter dictionaries.
        """
        rows: list[dict[str, Any]] = []
        for idx, rec in enumerate(records):
            rid = rec.record_id or f"rec_{idx}"
            key_list = [float(x) for x in rec.key_vector.detach().cpu().flatten().tolist()]
            val_list = [float(x) for x in rec.value_vector.detach().cpu().flatten().tolist()]
            key_literal = "[" + ",".join(f"{v:.8f}" for v in key_list) + "]"
            val_literal = "[" + ",".join(f"{v:.8f}" for v in val_list) + "]"
            leaf_bytes = merkle_tree._active_rules.get(rid) if merkle_tree is not None else None
            leaf_hash = leaf_bytes.hex() if leaf_bytes is not None else None
            rows.append(
                {
                    "record_id": rid,
                    "tenant_id": rec.tenant_id,
                    "text": rec.text,
                    "key_embedding": key_literal,
                    "value_embedding": val_literal,
                    "tier": rec.tier.value,
                    "jurisdiction": rec.jurisdiction,
                    "authority_rank": int(rec.authority_rank),
                    "importance_score": float(rec.importance_score),
                    "valid_from": rec.valid_from.isoformat(),
                    "valid_to": rec.valid_to.isoformat() if rec.valid_to is not None else None,
                    "transaction_time": rec.transaction_time.isoformat(),
                    "personal_data": bool(rec.personal_data),
                    "metadata_json": json.dumps(rec.metadata, sort_keys=True),
                    "merkle_leaf_hash": leaf_hash,
                }
            )
        return rows

    @staticmethod
    def to_neo4j_cypher(
        nodes: list[GraphNode],
        edges: list[GraphEdge],
    ) -> list[dict[str, Any]]:
        """
        Export `GraphNode` and `GraphEdge` items as parameterized Neo4j Cypher `MERGE` queries.
        """
        statements: list[dict[str, Any]] = []
        for node in nodes:
            statements.append(
                {
                    "statement": (
                        "MERGE (n:LegalEntity {node_id: $node_id, tenant_id: $tenant_id}) "
                        "SET n.entity_type = $entity_type, n.label = $label, "
                        "n.description = $description, n.valid_from = $valid_from, "
                        "n.valid_to = $valid_to, n.transaction_time = $transaction_time, "
                        "n.decay_factor = $decay_factor"
                    ),
                    "parameters": {
                        "node_id": node.node_id,
                        "tenant_id": node.tenant_id,
                        "entity_type": node.entity_type.value,
                        "label": node.label,
                        "description": node.description,
                        "valid_from": node.valid_from.isoformat(),
                        "valid_to": (
                            node.valid_to.isoformat() if node.valid_to is not None else None
                        ),
                        "transaction_time": node.transaction_time.isoformat(),
                        "decay_factor": float(node.decay_factor),
                    },
                }
            )
        for edge in edges:
            rel_label = edge.relation_type.value.upper()
            statements.append(
                {
                    "statement": (
                        "MATCH (src:LegalEntity {node_id: $source_id, tenant_id: $tenant_id}) "
                        "MATCH (tgt:LegalEntity {node_id: $target_id, tenant_id: $tenant_id}) "
                        f"MERGE (src)-[r:{rel_label}]->(tgt) "
                        "SET r.weight = $weight, r.relation_type = $relation_type"
                    ),
                    "parameters": {
                        "source_id": edge.source_id,
                        "target_id": edge.target_id,
                        "tenant_id": edge.tenant_id,
                        "weight": float(e.weight) if (e := edge) else 1.0,
                        "relation_type": edge.relation_type.value,
                    },
                }
            )
        return statements
