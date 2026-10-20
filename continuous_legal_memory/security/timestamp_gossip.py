"""
RFC 3161-Inspired Cryptographic Timestamp Authority (TSA) and
Decentralized Transparency Log Gossip Auditor (RFC 6962 / RFC 9162).

Provides verifiable timestamp tokens binding a tenant's 256-bit Sparse Merkle Tree
state root and append-only audit log head to a trusted clock reading, and detects
split-view / equivocation attacks across independent regulatory witness checkpoints.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from continuous_legal_memory.security import ed25519
from continuous_legal_memory.security.attestation import (
    AsymmetricAttestationModule,
    KeyedHashAttestationModule,
)
from continuous_legal_memory.security.keyring import CryptographicKeyRing
from continuous_legal_memory.security.merkle import AppendOnlyMerkleLog
from continuous_legal_memory.security.pq_hybrid import (
    PQ_HYBRID_ALGORITHM,
    PostQuantumHybridAttestationModule,
    pq_hybrid_sign,
    pq_hybrid_verify,
)


def _canonical_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


@dataclass(frozen=True)
class TimestampToken:
    """
    Cryptographic timestamp token inspired by RFC 3161 Time-Stamp Protocol (TSP).

    Binds a tenant's Sparse Merkle root and append-only Merkle log head hash
    to an authoritative UTC timestamp, serial number, policy OID, and anti-replay nonce.
    """

    token_id: str
    tenant_id: str
    merkle_root: str
    log_head_hash: str
    sequence_number: int
    issued_at: str
    nonce: str
    policy_oid: str
    tsa_authority_id: str
    message_imprint_sha256: str
    signature_algorithm: str
    signature_tag: str
    public_key_hex: str | None = None
    key_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "token_id": self.token_id,
            "tenant_id": self.tenant_id,
            "merkle_root": self.merkle_root,
            "log_head_hash": self.log_head_hash,
            "sequence_number": self.sequence_number,
            "issued_at": self.issued_at,
            "nonce": self.nonce,
            "policy_oid": self.policy_oid,
            "tsa_authority_id": self.tsa_authority_id,
            "message_imprint_sha256": self.message_imprint_sha256,
            "signature_algorithm": self.signature_algorithm,
            "signature_tag": self.signature_tag,
            "public_key_hex": self.public_key_hex,
            "key_id": self.key_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TimestampToken:
        return cls(
            token_id=str(data["token_id"]),
            tenant_id=str(data["tenant_id"]),
            merkle_root=str(data["merkle_root"]),
            log_head_hash=str(data["log_head_hash"]),
            sequence_number=int(data["sequence_number"]),
            issued_at=str(data["issued_at"]),
            nonce=str(data["nonce"]),
            policy_oid=str(data.get("policy_oid", "1.3.6.1.4.1.59291.1.1")),
            tsa_authority_id=str(data.get("tsa_authority_id", "CLM-TSA-PRIMARY")),
            message_imprint_sha256=str(data["message_imprint_sha256"]),
            signature_algorithm=str(data["signature_algorithm"]),
            signature_tag=str(data["signature_tag"]),
            public_key_hex=data.get("public_key_hex"),
            key_id=data.get("key_id"),
        )


@dataclass(frozen=True)
class GossipCheckpoint:
    """
    Signed transparency log checkpoint observed by an independent regulatory witness node.
    """

    witness_id: str
    tenant_id: str
    sequence_number: int
    merkle_root: str
    log_head_hash: str
    observed_at: str
    timestamp_token: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "witness_id": self.witness_id,
            "tenant_id": self.tenant_id,
            "sequence_number": self.sequence_number,
            "merkle_root": self.merkle_root,
            "log_head_hash": self.log_head_hash,
            "observed_at": self.observed_at,
            "timestamp_token": self.timestamp_token,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GossipCheckpoint:
        return cls(
            witness_id=str(data["witness_id"]),
            tenant_id=str(data["tenant_id"]),
            sequence_number=int(data["sequence_number"]),
            merkle_root=str(data["merkle_root"]),
            log_head_hash=str(data["log_head_hash"]),
            observed_at=str(data.get("observed_at", "")),
            timestamp_token=data.get("timestamp_token"),
        )


@dataclass(frozen=True)
class SplitViewAuditReport:
    """
    Result of auditing decentralized gossip checkpoints against the local append-only Merkle log.
    """

    consistent: bool
    tenant_id: str
    checked_checkpoints: int
    split_view_detected: bool
    history_rollback_detected: bool
    invalid_timestamp_tokens: int
    anomalies: list[dict[str, Any]] = field(default_factory=list)
    audit_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "consistent": self.consistent,
            "tenant_id": self.tenant_id,
            "checked_checkpoints": self.checked_checkpoints,
            "split_view_detected": self.split_view_detected,
            "history_rollback_detected": self.history_rollback_detected,
            "invalid_timestamp_tokens": self.invalid_timestamp_tokens,
            "anomalies": self.anomalies,
            "audit_hash": self.audit_hash,
        }


class TransparencyGossipAuditor:
    """
    Issues RFC 3161-inspired cryptographic timestamp tokens and audits cross-witness
    transparency log checkpoints for split-view equivocation and history rollback.
    """

    DEFAULT_POLICY_OID = "1.3.6.1.4.1.59291.1.1"

    def __init__(
        self,
        attestor: Any | None = None,
        secret_key: bytes | str | None = None,
        tsa_authority_id: str = "CLM-TSA-PRIMARY",
    ) -> None:
        if isinstance(attestor, CryptographicKeyRing):
            self.attestor: Any = attestor.active_attestor
        elif attestor is not None:
            self.attestor = attestor
        elif secret_key is not None:
            self.attestor = KeyedHashAttestationModule(secret_key=secret_key)
        else:
            self.attestor = PostQuantumHybridAttestationModule()
        self.tsa_authority_id = tsa_authority_id

    @staticmethod
    def _compute_imprint(
        tenant_id: str,
        merkle_root: str,
        log_head_hash: str,
        sequence_number: int,
        nonce: str,
        policy_oid: str,
    ) -> str:
        payload: dict[str, Any] = {
            "log_head_hash": log_head_hash,
            "merkle_root": merkle_root,
            "nonce": nonce,
            "policy_oid": policy_oid,
            "sequence_number": sequence_number,
            "tenant_id": tenant_id,
        }
        return hashlib.sha256(_canonical_bytes(payload)).hexdigest()

    @staticmethod
    def _signing_envelope(
        token_id: str,
        tenant_id: str,
        merkle_root: str,
        log_head_hash: str,
        sequence_number: int,
        issued_at: str,
        nonce: str,
        policy_oid: str,
        tsa_authority_id: str,
        message_imprint_sha256: str,
    ) -> bytes:
        payload: dict[str, Any] = {
            "issued_at": issued_at,
            "log_head_hash": log_head_hash,
            "merkle_root": merkle_root,
            "message_imprint_sha256": message_imprint_sha256,
            "nonce": nonce,
            "policy_oid": policy_oid,
            "sequence_number": sequence_number,
            "tenant_id": tenant_id,
            "token_id": token_id,
            "tsa_authority_id": tsa_authority_id,
        }
        return _canonical_bytes(payload)

    def issue_timestamp_token(
        self,
        tenant_id: str,
        merkle_root: str,
        log_head_hash: str,
        sequence_number: int,
        nonce: str | None = None,
        policy_oid: str = DEFAULT_POLICY_OID,
        issued_at: datetime | None = None,
    ) -> TimestampToken:
        """
        Issue a signed RFC 3161-style TimestampToken over a tenant's Merkle root and log head.
        """
        effective_nonce = nonce if nonce else secrets.token_hex(16)
        ts_str = (
            issued_at.astimezone(timezone.utc).isoformat()
            if issued_at is not None
            else datetime.now(timezone.utc).isoformat()
        )
        imprint = self._compute_imprint(
            tenant_id=tenant_id,
            merkle_root=merkle_root,
            log_head_hash=log_head_hash,
            sequence_number=sequence_number,
            nonce=effective_nonce,
            policy_oid=policy_oid,
        )
        token_id = f"tst_{imprint[:16]}"
        signable = self._signing_envelope(
            token_id=token_id,
            tenant_id=tenant_id,
            merkle_root=merkle_root,
            log_head_hash=log_head_hash,
            sequence_number=sequence_number,
            issued_at=ts_str,
            nonce=effective_nonce,
            policy_oid=policy_oid,
            tsa_authority_id=self.tsa_authority_id,
            message_imprint_sha256=imprint,
        )

        mod = self.attestor
        key_id = getattr(mod, "key_id", "default")
        if isinstance(mod, PostQuantumHybridAttestationModule):
            if mod._private_key is None:
                raise PermissionError("PostQuantumHybridAttestationModule has no private key.")
            alg = PQ_HYBRID_ALGORITHM
            sig_tag = pq_hybrid_sign(signable, mod._private_key)
            pub_hex: str | None = mod.public_key_hex
        elif isinstance(mod, AsymmetricAttestationModule):
            if mod._private_key is None:
                raise PermissionError("AsymmetricAttestationModule has no private key.")
            alg = "Ed25519"
            sig_tag = ed25519.sign(signable, mod._private_key).hex()
            pub_hex = mod.public_key_hex
        elif isinstance(mod, KeyedHashAttestationModule):
            alg = "HMAC-SHA256"
            sig_tag = hmac.new(mod._secret_key, signable, hashlib.sha256).hexdigest()
            pub_hex = None
        else:
            raise TypeError(f"Unsupported attestation module: {type(mod)}")

        return TimestampToken(
            token_id=token_id,
            tenant_id=tenant_id,
            merkle_root=merkle_root,
            log_head_hash=log_head_hash,
            sequence_number=sequence_number,
            issued_at=ts_str,
            nonce=effective_nonce,
            policy_oid=policy_oid,
            tsa_authority_id=self.tsa_authority_id,
            message_imprint_sha256=imprint,
            signature_algorithm=alg,
            signature_tag=sig_tag,
            public_key_hex=pub_hex,
            key_id=str(key_id),
        )

    def verify_timestamp_token(
        self,
        token: TimestampToken | dict[str, Any],
        expected_merkle_root: str | None = None,
        expected_nonce: str | None = None,
    ) -> bool:
        """
        Verify the message imprint, optional expected root/nonce, and cryptographic signature
        of a TimestampToken.
        """
        tst = TimestampToken.from_dict(token) if isinstance(token, dict) else token
        if expected_merkle_root is not None and tst.merkle_root != expected_merkle_root:
            return False
        if expected_nonce is not None and tst.nonce != expected_nonce:
            return False

        recomputed_imprint = self._compute_imprint(
            tenant_id=tst.tenant_id,
            merkle_root=tst.merkle_root,
            log_head_hash=tst.log_head_hash,
            sequence_number=tst.sequence_number,
            nonce=tst.nonce,
            policy_oid=tst.policy_oid,
        )
        if recomputed_imprint != tst.message_imprint_sha256:
            return False

        signable = self._signing_envelope(
            token_id=tst.token_id,
            tenant_id=tst.tenant_id,
            merkle_root=tst.merkle_root,
            log_head_hash=tst.log_head_hash,
            sequence_number=tst.sequence_number,
            issued_at=tst.issued_at,
            nonce=tst.nonce,
            policy_oid=tst.policy_oid,
            tsa_authority_id=tst.tsa_authority_id,
            message_imprint_sha256=tst.message_imprint_sha256,
        )

        if tst.signature_algorithm == PQ_HYBRID_ALGORITHM:
            pub = tst.public_key_hex or getattr(self.attestor, "public_key_hex", None)
            if not pub:
                return False
            return pq_hybrid_verify(tst.signature_tag, signable, pub)
        if tst.signature_algorithm == "Ed25519":
            pub = tst.public_key_hex or getattr(self.attestor, "public_key_hex", None)
            if not pub:
                return False
            try:
                return ed25519.verify(bytes.fromhex(tst.signature_tag), signable, bytes.fromhex(pub))
            except ValueError:
                return False
        if tst.signature_algorithm == "HMAC-SHA256" and isinstance(
            self.attestor, KeyedHashAttestationModule
        ):
            expected_mac = hmac.new(self.attestor._secret_key, signable, hashlib.sha256).hexdigest()
            return hmac.compare_digest(expected_mac, tst.signature_tag)
        return False

    def audit_gossip_checkpoints(
        self,
        tenant_id: str,
        local_log: AppendOnlyMerkleLog,
        checkpoints: list[GossipCheckpoint | dict[str, Any]],
    ) -> SplitViewAuditReport:
        """
        Audit a list of witness GossipCheckpoints against both each other and the local
        AppendOnlyMerkleLog to detect split-view equivocation and history rollback.
        """
        parsed: list[GossipCheckpoint] = [
            GossipCheckpoint.from_dict(cp) if isinstance(cp, dict) else cp for cp in checkpoints
        ]
        anomalies: list[dict[str, Any]] = []
        split_view = False
        rollback = False
        invalid_tokens = 0

        entries = local_log.entries
        local_seq_len = len(entries)
        local_by_seq: dict[int, tuple[str, str]] = {}
        for idx, entry in enumerate(entries, start=1):
            local_by_seq[idx] = (entry.smt_root_hash, entry.entry_hash)
        if local_seq_len == 0:
            local_by_seq[0] = ("0" * 64, "0" * 64)

        observed_by_seq: dict[int, tuple[str, str, str]] = {}

        for cp in parsed:
            if cp.tenant_id != tenant_id:
                anomalies.append(
                    {
                        "type": "TENANT_MISMATCH",
                        "witness_id": cp.witness_id,
                        "expected_tenant": tenant_id,
                        "actual_tenant": cp.tenant_id,
                    }
                )
                continue

            if cp.timestamp_token is not None and not self.verify_timestamp_token(
                cp.timestamp_token, expected_merkle_root=cp.merkle_root
            ):
                invalid_tokens += 1
                anomalies.append(
                    {
                        "type": "INVALID_TIMESTAMP_TOKEN",
                        "witness_id": cp.witness_id,
                        "sequence_number": cp.sequence_number,
                    }
                )

            seq = cp.sequence_number
            if seq in observed_by_seq:
                prev_witness, prev_root, prev_head = observed_by_seq[seq]
                if prev_root != cp.merkle_root or prev_head != cp.log_head_hash:
                    split_view = True
                    anomalies.append(
                        {
                            "type": "CROSS_WITNESS_SPLIT_VIEW",
                            "sequence_number": seq,
                            "witness_a": prev_witness,
                            "witness_b": cp.witness_id,
                            "root_a": prev_root,
                            "root_b": cp.merkle_root,
                        }
                    )
            else:
                observed_by_seq[seq] = (cp.witness_id, cp.merkle_root, cp.log_head_hash)

            if seq > local_seq_len:
                rollback = True
                anomalies.append(
                    {
                        "type": "HISTORY_ROLLBACK_OR_TRUNCATION",
                        "witness_id": cp.witness_id,
                        "witness_sequence": seq,
                        "local_sequence": local_seq_len,
                    }
                )
            elif seq in local_by_seq:
                local_root, local_head = local_by_seq[seq]
                if local_root != cp.merkle_root or local_head != cp.log_head_hash:
                    split_view = True
                    anomalies.append(
                        {
                            "type": "LOCAL_VS_WITNESS_EQUIVOCATION",
                            "witness_id": cp.witness_id,
                            "sequence_number": seq,
                            "local_merkle_root": local_root,
                            "witness_merkle_root": cp.merkle_root,
                            "local_log_head": local_head,
                            "witness_log_head": cp.log_head_hash,
                        }
                    )

        consistent = (
            not split_view
            and not rollback
            and invalid_tokens == 0
            and len(anomalies) == 0
            and local_log.verify_chain()
        )
        summary_payload: dict[str, Any] = {
            "anomalies": anomalies,
            "checked_checkpoints": len(parsed),
            "consistent": consistent,
            "history_rollback_detected": rollback,
            "invalid_timestamp_tokens": invalid_tokens,
            "split_view_detected": split_view,
            "tenant_id": tenant_id,
        }
        audit_hash = hashlib.sha256(_canonical_bytes(summary_payload)).hexdigest()
        return SplitViewAuditReport(
            consistent=consistent,
            tenant_id=tenant_id,
            checked_checkpoints=len(parsed),
            split_view_detected=split_view,
            history_rollback_detected=rollback,
            invalid_timestamp_tokens=invalid_tokens,
            anomalies=anomalies,
            audit_hash=audit_hash,
        )
