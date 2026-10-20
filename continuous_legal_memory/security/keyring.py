"""
Cryptographic Key Lifecycle & Rotation Ring (DORA / SOC 2 / EU AI Act Art. 12).

Manages active, retired, and revoked cryptographic signing keys (`Ed25519` and `HMAC-SHA256`)
so that periodic key rotation never invalidates historical `MemoryAttestationToken` audit
trails signed by retired keys, while compromised keys marked `REVOKED` are immediately
rejected across all verifiers.
"""

from __future__ import annotations

import secrets
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

from continuous_legal_memory.domain.models import PredictionResult
from continuous_legal_memory.security.attestation import (
    AsymmetricAttestationModule,
    KeyedHashAttestationModule,
    MemoryAttestationToken,
)
from continuous_legal_memory.security.pq_hybrid import (
    PQ_HYBRID_ALGORITHM,
    PostQuantumHybridAttestationModule,
)

AttestorType = (
    AsymmetricAttestationModule
    | KeyedHashAttestationModule
    | PostQuantumHybridAttestationModule
)


@dataclass
class KeyMetadata:
    """
    Lifecycle metadata for a managed cryptographic key in `CryptographicKeyRing`.
    """

    key_id: str
    algorithm: str
    status: str  # "ACTIVE" | "RETIRED" | "REVOKED"
    created_at: str
    retired_at: str | None = None
    revoked_at: str | None = None
    public_key_hex: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KeyMetadata:
        return cls(
            key_id=str(data["key_id"]),
            algorithm=str(data["algorithm"]),
            status=str(data["status"]),
            created_at=str(data["created_at"]),
            retired_at=data.get("retired_at"),
            revoked_at=data.get("revoked_at"),
            public_key_hex=data.get("public_key_hex"),
        )


class CryptographicKeyRing:
    """
    Multi-key cryptographic key ring supporting atomic key rotation, retired-key historical
    verification, and compromised-key revocation (KRL).
    """

    def __init__(
        self,
        initial_algorithm: str = "Ed25519",
        initial_key_id: str = "clm-key-v1",
        initial_key_material: bytes | str | None = None,
    ) -> None:
        self._attestors: dict[str, AttestorType] = {}
        self._metadata: dict[str, KeyMetadata] = {}
        self._active_key_id: str = ""
        self._register_new_active_key(
            key_id=initial_key_id,
            algorithm=initial_algorithm,
            key_material=initial_key_material,
        )

    def _register_new_active_key(
        self,
        key_id: str,
        algorithm: str,
        key_material: bytes | str | None = None,
    ) -> KeyMetadata:
        now_iso = datetime.now(timezone.utc).isoformat()
        if algorithm == "Ed25519":
            asym_mod = AsymmetricAttestationModule(
                private_key=key_material,
                key_id=key_id,
            )
            pub_hex: str | None = asym_mod.public_key_hex
            mod: AttestorType = asym_mod
        elif algorithm == "HMAC-SHA256":
            mod = KeyedHashAttestationModule(
                secret_key=key_material,
                key_id=key_id,
            )
            pub_hex = None
        elif algorithm == PQ_HYBRID_ALGORITHM:
            pq_mod = PostQuantumHybridAttestationModule(
                private_key=key_material,
                key_id=key_id,
            )
            pub_hex = pq_mod.public_key_hex
            mod = pq_mod
        else:
            raise ValueError(
                f"Unsupported key algorithm '{algorithm}'. Expected 'Ed25519', "
                f"'HMAC-SHA256', or '{PQ_HYBRID_ALGORITHM}'."
            )

        meta = KeyMetadata(
            key_id=key_id,
            algorithm=algorithm,
            status="ACTIVE",
            created_at=now_iso,
            public_key_hex=pub_hex,
        )
        self._attestors[key_id] = mod
        self._metadata[key_id] = meta
        self._active_key_id = key_id
        return meta

    @property
    def active_key_id(self) -> str:
        """Identifier of the currently active signing key."""
        return self._active_key_id

    @property
    def active_attestor(self) -> AttestorType:
        """Return the currently active attestation module."""
        return self._attestors[self._active_key_id]

    @property
    def public_key_hex(self) -> str | None:
        """Hex-encoded public key of the active key if Ed25519 or PQ-Hybrid, else None."""
        active = self.active_attestor
        if isinstance(active, (AsymmetricAttestationModule, PostQuantumHybridAttestationModule)):
            return active.public_key_hex
        return None

    @property
    def _private_key(self) -> bytes | None:
        """Expose active Ed25519 private key bytes for SparseMerkleTree compatibility."""
        active = self.active_attestor
        return getattr(active, "_private_key", None)

    @property
    def _secret_key(self) -> bytes | None:
        """Expose active HMAC secret key bytes for SparseMerkleTree compatibility."""
        active = self.active_attestor
        return getattr(active, "_secret_key", None)

    def rotate_key(
        self,
        new_key_id: str | None = None,
        algorithm: str | None = None,
        key_material: bytes | str | None = None,
    ) -> KeyMetadata:
        """
        Retire the current active key and promote a newly generated (or supplied) key to `ACTIVE`.

        Historical tokens signed by the retired key remain verifiable via `verify_attestation`.
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        if self._active_key_id in self._metadata:
            prev_meta = self._metadata[self._active_key_id]
            if prev_meta.status == "ACTIVE":
                prev_meta.status = "RETIRED"
                prev_meta.retired_at = now_iso

        eff_algo = algorithm or (
            self._metadata[self._active_key_id].algorithm
            if self._active_key_id in self._metadata
            else "Ed25519"
        )
        eff_key_id = new_key_id or f"clm-key-v{len(self._metadata) + 1}-{secrets.token_hex(3)}"
        if eff_key_id in self._metadata:
            raise ValueError(f"Key ID '{eff_key_id}' already exists in CryptographicKeyRing.")

        return self._register_new_active_key(
            key_id=eff_key_id,
            algorithm=eff_algo,
            key_material=key_material,
        )

    def revoke_key(self, key_id: str) -> KeyMetadata:
        """
        Permanently mark `key_id` as `REVOKED` (compromised).

        Any attestation token signed with a `REVOKED` key is rejected by `verify_attestation`.
        If `key_id` was the currently active key, automatically rotates to a new active key.
        """
        if key_id not in self._metadata:
            raise KeyError(f"Key ID '{key_id}' not found in CryptographicKeyRing.")
        now_iso = datetime.now(timezone.utc).isoformat()
        meta = self._metadata[key_id]
        meta.status = "REVOKED"
        meta.revoked_at = now_iso

        if key_id == self._active_key_id:
            self.rotate_key(algorithm=meta.algorithm)

        return meta

    def sign_attestation(
        self,
        result: PredictionResult,
        retrieved_text: str | None = None,
        timestamp: datetime | None = None,
    ) -> MemoryAttestationToken:
        """Sign a `PredictionResult` using the currently `ACTIVE` key in the ring."""
        return self.active_attestor.sign_attestation(
            result=result,
            retrieved_text=retrieved_text,
            timestamp=timestamp,
        )

    def verify_attestation(
        self,
        token: MemoryAttestationToken,
        result: PredictionResult,
        retrieved_text: str | None = None,
    ) -> bool:
        """
        Verify a `MemoryAttestationToken` against `ACTIVE` or `RETIRED` keys while rejecting `REVOKED` keys.
        """
        kid = token.key_id
        if kid in self._metadata:
            meta = self._metadata[kid]
            if meta.status == "REVOKED":
                return False
            attestor = self._attestors[kid]
            return attestor.verify_attestation(token, result, retrieved_text=retrieved_text)

        # Fallback for external Ed25519 or PQ-Hybrid tokens carrying an embedded public key not in the KRL
        if token.algorithm in ("Ed25519", PQ_HYBRID_ALGORITHM) and token.public_key:
            for m in self._metadata.values():
                if m.status == "REVOKED" and m.public_key_hex == token.public_key:
                    return False
            if token.algorithm == PQ_HYBRID_ALGORITHM:
                ext_pq = PostQuantumHybridAttestationModule(public_key=token.public_key, key_id=kid)
                return ext_pq.verify_attestation(token, result, retrieved_text=retrieved_text)
            ext_verifier = AsymmetricAttestationModule(public_key=token.public_key, key_id=kid)
            return ext_verifier.verify_attestation(token, result, retrieved_text=retrieved_text)

        return False

    def get_key_metadata(self, key_id: str) -> KeyMetadata:
        """Return `KeyMetadata` for `key_id`, raising `KeyError` if not registered."""
        if key_id not in self._metadata:
            raise KeyError(f"Key ID '{key_id}' not found in CryptographicKeyRing.")
        return self._metadata[key_id]

    def list_keys(self) -> list[dict[str, Any]]:
        """Return serialized lifecycle metadata for all keys in the ring."""
        return [m.to_dict() for m in self._metadata.values()]

    def to_dict(self) -> dict[str, Any]:
        """Serialize complete CryptographicKeyRing state (metadata and key material) for SQLite persistence."""
        entries: list[dict[str, Any]] = []
        for kid, meta in self._metadata.items():
            att = self._attestors[kid]
            mat_hex: str | None = None
            if (
                isinstance(att, (AsymmetricAttestationModule, PostQuantumHybridAttestationModule))
                and att._private_key is not None
            ):
                mat_hex = att._private_key.hex()
            elif isinstance(att, KeyedHashAttestationModule) and att._secret_key is not None:
                mat_hex = att._secret_key.hex()
            entries.append(
                {
                    "metadata": meta.to_dict(),
                    "key_material_hex": mat_hex,
                }
            )
        return {
            "active_key_id": self._active_key_id,
            "entries": entries,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CryptographicKeyRing:
        """Reconstruct a `CryptographicKeyRing` from a serialized dictionary."""
        ring = cls.__new__(cls)
        ring._attestors = {}
        ring._metadata = {}
        ring._active_key_id = str(data.get("active_key_id", ""))
        for item in data.get("entries", []):
            meta = KeyMetadata.from_dict(item["metadata"])
            mat_hex = item.get("key_material_hex")
            raw_mat = bytes.fromhex(mat_hex) if mat_hex else None
            if meta.algorithm == "Ed25519":
                mod: AttestorType = AsymmetricAttestationModule(
                    private_key=raw_mat,
                    public_key=meta.public_key_hex if raw_mat is None else None,
                    key_id=meta.key_id,
                )
            elif meta.algorithm == PQ_HYBRID_ALGORITHM:
                mod = PostQuantumHybridAttestationModule(
                    private_key=raw_mat,
                    public_key=meta.public_key_hex if raw_mat is None else None,
                    key_id=meta.key_id,
                )
            else:
                mod = KeyedHashAttestationModule(
                    secret_key=raw_mat,
                    key_id=meta.key_id,
                )
            ring._attestors[meta.key_id] = mod
            ring._metadata[meta.key_id] = meta
        return ring

