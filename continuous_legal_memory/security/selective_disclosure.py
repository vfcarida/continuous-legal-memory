"""
Selective-Disclosure & Zero-Knowledge Normative Compliance Proofs.

Implements privacy-preserving compliance attestations aligned with EU AI Act Article 12
(Record-Keeping) and Article 78 (Confidentiality). Enables an organization to prove to an
external regulator or counterparty that a legal AI decision:
1. Relied on an active, non-revoked normative record included in the tenant's 256-bit Sparse Merkle Tree,
2. Satisfied minimum Kelsenian hierarchy authority rank and jurisdictional constraints, and
3. Was temporally valid on both valid-time (`at_time`) and transaction-time (`as_of_transaction_time`) axes,
WITHOUT disclosing the confidential commercial contract text or proprietary query in plaintext.

Cryptographic Primitives:
- Salted SHA-256 Pedersen-style hash commitments (`H(salt || utf8(plaintext))`) for selective unblinding.
- 256-bit Sparse Merkle Tree inclusion proofs (`SparseMerkleProof`) over record commitments.
- Canonical JSON Ed25519 (or HMAC-SHA256) digital signatures over disclosed normative predicates.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

from continuous_legal_memory.domain.models import MemoryRecord, PredictionResult
from continuous_legal_memory.security import ed25519
from continuous_legal_memory.security.attestation import (
    AsymmetricAttestationModule,
    KeyedHashAttestationModule,
)
from continuous_legal_memory.security.merkle import (
    SparseMerkleProof,
    SparseMerkleTree,
    verify_sparse_merkle_proof,
)
from continuous_legal_memory.security.pq_hybrid import (
    PQ_HYBRID_ALGORITHM,
    PostQuantumHybridAttestationModule,
    pq_hybrid_sign,
    pq_hybrid_verify,
)


def canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    """Serialize a dictionary to deterministic RFC-8785-style canonical JSON bytes."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def compute_salted_commitment(plaintext: str, salt_hex: str) -> str:
    """
    Compute a binding, hiding SHA-256 commitment over `plaintext` using a 256-bit random hex salt.

    Args:
        plaintext: Confidential text (e.g., proprietary contract clause or legal query).
        salt_hex: 64-character hex string representing 32 random bytes of salt.

    Returns:
        Hex-encoded SHA-256 commitment digest.
    """
    hasher = hashlib.sha256()
    hasher.update(bytes.fromhex(salt_hex))
    hasher.update(b":")
    hasher.update(plaintext.encode("utf-8"))
    return hasher.hexdigest()


@dataclass(frozen=True)
class SelectiveDisclosureComplianceProof:
    """
    Privacy-preserving normative compliance proof with blinded text commitments.

    Attributes:
        proof_id: Unique identifier of the compliance proof.
        record_id: Identifier of the governing MemoryRecord in the Sparse Merkle Tree.
        tenant_id: Multi-tenant namespace identifier.
        text_commitment: Salted SHA-256 commitment to `record.text` (`H(text_salt || record.text)`).
        query_commitment: Salted SHA-256 commitment to `query` (`H(query_salt || query)`).
        disclosed_claims: Publicly verifiable normative compliance predicates (jurisdiction,
                          authority_rank, temporal validity bounds, proof_trace_hash, etc.).
        merkle_proof: Serialized 256-bit SparseMerkleProof proving inclusion in `merkle_root`.
        merkle_root: 256-bit hex root of the tenant's Sparse Merkle Tree at the time of evaluation.
        signature: Hex-encoded Ed25519 or HMAC-SHA256 signature over the canonical proof body.
        public_key: Optional hex-encoded Ed25519 public key for third-party verification.
        algorithm: Signature algorithm (`"Ed25519"` or `"HMAC-SHA256"`).
        created_at: ISO-8601 UTC timestamp when the proof was generated.
    """

    proof_id: str
    record_id: str
    tenant_id: str
    text_commitment: str
    query_commitment: str
    disclosed_claims: dict[str, Any]
    merkle_proof: dict[str, Any]
    merkle_root: str
    signature: str
    public_key: str | None = None
    algorithm: str = "Ed25519"
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize the compliance proof to a JSON-compatible dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SelectiveDisclosureComplianceProof:
        """Reconstruct a `SelectiveDisclosureComplianceProof` from a dictionary."""
        return cls(
            proof_id=str(data["proof_id"]),
            record_id=str(data["record_id"]),
            tenant_id=str(data["tenant_id"]),
            text_commitment=str(data["text_commitment"]),
            query_commitment=str(data["query_commitment"]),
            disclosed_claims=dict(data["disclosed_claims"]),
            merkle_proof=dict(data["merkle_proof"]),
            merkle_root=str(data["merkle_root"]),
            signature=str(data["signature"]),
            public_key=data.get("public_key"),
            algorithm=str(data.get("algorithm", "Ed25519")),
            created_at=str(data.get("created_at", "")),
        )


@dataclass(frozen=True)
class SelectiveDisclosureBundle:
    """
    Holds the public `SelectiveDisclosureComplianceProof` alongside private unblinding salts.

    The `proof` can be shared freely with external auditors without leaking `record.text` or
    `query`. The `text_salt` and `query_salt` are retained by the data controller for optional
    selective unblinding under court or regulatory order.
    """

    proof: SelectiveDisclosureComplianceProof
    text_salt: str
    query_salt: str


class SelectiveDisclosureProver:
    """
    Generates and verifies Selective-Disclosure Normative Compliance Proofs.
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
    def _canonical_signable_bytes(
        proof_id: str,
        record_id: str,
        tenant_id: str,
        text_commitment: str,
        query_commitment: str,
        disclosed_claims: dict[str, Any],
        merkle_proof: dict[str, Any],
        merkle_root: str,
        algorithm: str,
        created_at: str,
    ) -> bytes:
        payload = {
            "algorithm": algorithm,
            "created_at": created_at,
            "disclosed_claims": disclosed_claims,
            "merkle_proof": merkle_proof,
            "merkle_root": merkle_root,
            "proof_id": proof_id,
            "query_commitment": query_commitment,
            "record_id": record_id,
            "tenant_id": tenant_id,
            "text_commitment": text_commitment,
        }
        return canonical_json_bytes(payload)

    def generate_proof(
        self,
        record: MemoryRecord,
        merkle_tree: SparseMerkleTree,
        query: str = "",
        prediction_result: PredictionResult | None = None,
        min_required_authority_rank: int = 1,
        at_time: datetime | None = None,
        as_of_transaction_time: datetime | None = None,
        text_salt: str | None = None,
        query_salt: str | None = None,
    ) -> SelectiveDisclosureBundle:
        """
        Generate a zero-knowledge / selective-disclosure compliance proof for `record`.
        """
        record_id = record.record_id or "unindexed"
        actual_text_salt = text_salt if text_salt is not None else secrets.token_hex(32)
        actual_query_salt = query_salt if query_salt is not None else secrets.token_hex(32)

        text_commitment = compute_salted_commitment(record.text, actual_text_salt)
        query_commitment = compute_salted_commitment(query, actual_query_salt)

        eval_time = at_time or datetime.now(timezone.utc)
        if eval_time.tzinfo is None:
            eval_time = eval_time.replace(tzinfo=timezone.utc)

        is_valid = record.is_temporally_valid(
            at_time=eval_time,
            as_of_transaction_time=as_of_transaction_time,
        )
        authority_satisfied = int(record.authority_rank) >= int(min_required_authority_rank)

        proof_trace_hash: str | None = None
        if prediction_result is not None and prediction_result.proof_trace is not None:
            proof_trace_hash = hashlib.sha256(
                canonical_json_bytes(prediction_result.proof_trace)
            ).hexdigest()

        disclosed_claims: dict[str, Any] = {
            "authority_rank": int(record.authority_rank),
            "authority_rank_satisfied": authority_satisfied,
            "min_required_authority_rank": int(min_required_authority_rank),
            "jurisdiction": record.jurisdiction,
            "tier": record.tier.value,
            "valid_from": record.valid_from.isoformat(),
            "valid_to": record.valid_to.isoformat() if record.valid_to is not None else None,
            "transaction_time": record.transaction_time.isoformat(),
            "evaluated_at_time": eval_time.isoformat(),
            "as_of_transaction_time": (
                as_of_transaction_time.isoformat() if as_of_transaction_time is not None else None
            ),
            "temporally_valid": is_valid,
            "personal_data": bool(record.personal_data),
            "predicted_action_vector": (
                list(prediction_result.predicted_action_vector)
                if prediction_result is not None
                else [float(x) for x in record.value_vector.detach().cpu().flatten().tolist()]
            ),
            "abstained": bool(prediction_result.abstained) if prediction_result is not None else False,
            "proof_trace_hash": proof_trace_hash,
        }

        smt_proof = merkle_tree.generate_proof(record_id, attestor=self.attestation_module)
        merkle_proof_dict = smt_proof.to_dict()
        merkle_root = smt_proof.root_hash
        created_at = datetime.now(timezone.utc).isoformat()
        proof_id = f"zkp_{hashlib.sha256(f'{record_id}:{text_commitment}:{created_at}'.encode()).hexdigest()[:16]}"

        if isinstance(self.attestation_module, PostQuantumHybridAttestationModule):
            if self.attestation_module._private_key is None:
                raise PermissionError(
                    "Cannot sign proof: PostQuantumHybridAttestationModule has no private key."
                )
            algorithm = PQ_HYBRID_ALGORITHM
            signable = self._canonical_signable_bytes(
                proof_id=proof_id,
                record_id=record_id,
                tenant_id=record.tenant_id,
                text_commitment=text_commitment,
                query_commitment=query_commitment,
                disclosed_claims=disclosed_claims,
                merkle_proof=merkle_proof_dict,
                merkle_root=merkle_root,
                algorithm=algorithm,
                created_at=created_at,
            )
            signature = pq_hybrid_sign(signable, self.attestation_module._private_key)
            public_key: str | None = self.attestation_module.public_key_hex
        elif isinstance(self.attestation_module, AsymmetricAttestationModule):
            if self.attestation_module._private_key is None:
                raise PermissionError("Cannot sign proof: AsymmetricAttestationModule has no private key.")
            algorithm = "Ed25519"
            signable = self._canonical_signable_bytes(
                proof_id=proof_id,
                record_id=record_id,
                tenant_id=record.tenant_id,
                text_commitment=text_commitment,
                query_commitment=query_commitment,
                disclosed_claims=disclosed_claims,
                merkle_proof=merkle_proof_dict,
                merkle_root=merkle_root,
                algorithm=algorithm,
                created_at=created_at,
            )
            signature = ed25519.sign(signable, self.attestation_module._private_key).hex()
            public_key = self.attestation_module.public_key_hex
        elif isinstance(self.attestation_module, KeyedHashAttestationModule):
            algorithm = "HMAC-SHA256"
            signable = self._canonical_signable_bytes(
                proof_id=proof_id,
                record_id=record_id,
                tenant_id=record.tenant_id,
                text_commitment=text_commitment,
                query_commitment=query_commitment,
                disclosed_claims=disclosed_claims,
                merkle_proof=merkle_proof_dict,
                merkle_root=merkle_root,
                algorithm=algorithm,
                created_at=created_at,
            )
            signature = hmac.new(
                self.attestation_module._secret_key, signable, hashlib.sha256
            ).hexdigest()
            public_key = None
        else:
            raise TypeError(f"Unsupported attestation module: {type(self.attestation_module)}")

        proof = SelectiveDisclosureComplianceProof(
            proof_id=proof_id,
            record_id=record_id,
            tenant_id=record.tenant_id,
            text_commitment=text_commitment,
            query_commitment=query_commitment,
            disclosed_claims=disclosed_claims,
            merkle_proof=merkle_proof_dict,
            merkle_root=merkle_root,
            signature=signature,
            public_key=public_key,
            algorithm=algorithm,
            created_at=created_at,
        )
        return SelectiveDisclosureBundle(
            proof=proof,
            text_salt=actual_text_salt,
            query_salt=actual_query_salt,
        )

    def verify_proof(
        self,
        proof: SelectiveDisclosureComplianceProof | dict[str, Any],
        expected_merkle_root: str | None = None,
        min_authority_rank: int | None = None,
        required_jurisdiction: str | None = None,
        require_temporal_validity: bool = True,
    ) -> bool:
        """
        Verify a `SelectiveDisclosureComplianceProof` without accessing plaintext text or query.
        """
        p = (
            SelectiveDisclosureComplianceProof.from_dict(proof)
            if isinstance(proof, dict)
            else proof
        )
        # 1. Verify cryptographic signature over canonical payload
        signable = self._canonical_signable_bytes(
            proof_id=p.proof_id,
            record_id=p.record_id,
            tenant_id=p.tenant_id,
            text_commitment=p.text_commitment,
            query_commitment=p.query_commitment,
            disclosed_claims=p.disclosed_claims,
            merkle_proof=p.merkle_proof,
            merkle_root=p.merkle_root,
            algorithm=p.algorithm,
            created_at=p.created_at,
        )

        if p.algorithm == PQ_HYBRID_ALGORITHM:
            if not p.public_key:
                return False
            if not pq_hybrid_verify(p.signature, signable, p.public_key):
                return False
        elif p.algorithm == "Ed25519":
            if not p.public_key:
                return False
            if not ed25519.verify(p.signature, signable, p.public_key):
                return False
        elif p.algorithm == "HMAC-SHA256":
            if not isinstance(self.attestation_module, KeyedHashAttestationModule):
                return False
            expected_mac = hmac.new(
                self.attestation_module._secret_key, signable, hashlib.sha256
            ).hexdigest()
            if not hmac.compare_digest(p.signature, expected_mac):
                return False
        else:
            return False

        # 2. Verify 256-bit Sparse Merkle inclusion proof
        if expected_merkle_root is not None and p.merkle_root != expected_merkle_root:
            return False

        try:
            smt_proof = SparseMerkleProof.from_dict(p.merkle_proof)
        except (KeyError, TypeError, ValueError):
            return False

        if not smt_proof.is_member or smt_proof.rule_id != p.record_id:
            return False
        if smt_proof.root_hash != p.merkle_root:
            return False
        if not verify_sparse_merkle_proof(
            smt_proof,
            expected_root_hash=p.merkle_root,
            attestor=self.attestation_module,
        ):
            return False

        # 3. Verify normative predicates
        claims = p.disclosed_claims
        if require_temporal_validity and not bool(claims.get("temporally_valid", False)):
            return False
        if not bool(claims.get("authority_rank_satisfied", False)):
            return False
        if min_authority_rank is not None and int(claims.get("authority_rank", 0)) < int(
            min_authority_rank
        ):
            return False
        return not (
            required_jurisdiction is not None
            and claims.get("jurisdiction") != required_jurisdiction
        )

    @staticmethod
    def verify_disclosed_text(
        proof: SelectiveDisclosureComplianceProof | dict[str, Any],
        disclosed_plaintext: str,
        text_salt: str,
    ) -> bool:
        """
        Verify a selectively unblinded record text against `proof.text_commitment`.
        """
        p = (
            SelectiveDisclosureComplianceProof.from_dict(proof)
            if isinstance(proof, dict)
            else proof
        )
        try:
            recomputed = compute_salted_commitment(disclosed_plaintext, text_salt)
        except ValueError:
            return False
        return hmac.compare_digest(recomputed, p.text_commitment)

    @staticmethod
    def verify_disclosed_query(
        proof: SelectiveDisclosureComplianceProof | dict[str, Any],
        disclosed_query: str,
        query_salt: str,
    ) -> bool:
        """
        Verify a selectively unblinded query string against `proof.query_commitment`.
        """
        p = (
            SelectiveDisclosureComplianceProof.from_dict(proof)
            if isinstance(proof, dict)
            else proof
        )
        try:
            recomputed = compute_salted_commitment(disclosed_query, query_salt)
        except ValueError:
            return False
        return hmac.compare_digest(recomputed, p.query_commitment)
