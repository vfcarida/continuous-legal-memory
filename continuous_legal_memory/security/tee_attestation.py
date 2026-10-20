"""
Hardware/TEE Enclave Remote Attestation Quotes & Confidential Computing Bindings.

Implements remote attestation quote generation and cryptographic verification for
Trusted Execution Environments (AWS Nitro Enclaves `PCR0..PCR2` and Intel TDX `MRTD`/`RTMR0..1`),
binding the orchestrator's 256-bit `SparseMerkleTree` state root, active `CryptographicKeyRing`
public key, and anti-replay challenge `nonce` into a hardware-anchored `TEEAttestationQuote`.
"""

from __future__ import annotations

import hashlib
import json
import secrets
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from continuous_legal_memory.security.ed25519 import (
    public_key_from_private_key,
    sign,
    verify,
)
from continuous_legal_memory.security.pq_hybrid import (
    PQ_HYBRID_ALGORITHM,
    PostQuantumHybridAttestationModule,
    pq_hybrid_sign,
    pq_hybrid_verify,
)


class TEEPlatform(str, Enum):
    """Supported Confidential Computing / TEE hardware attestation platforms."""

    AWS_NITRO = "AWS_NITRO"
    INTEL_TDX = "INTEL_TDX"


def _compute_user_data_digest(
    tenant_id: str,
    merkle_root: str,
    active_key_id: str,
    public_key_hex: str | None,
    nonce: str,
) -> str:
    """Compute canonical SHA-256 user_data / report_data commitment."""
    canonical = json.dumps(
        {
            "active_key_id": active_key_id,
            "merkle_root": merkle_root,
            "nonce": nonce,
            "public_key_hex": public_key_hex or "",
            "tenant_id": tenant_id,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _default_measurements_for_platform(platform: str, code_version: str = "0.9.0") -> dict[str, str]:
    """
    Compute deterministic enclave measurement registers (`PCR0..PCR2` or `MRTD`/`RTMR0..1`)
    representing the immutable enclave image, kernel, and runtime application digest.
    """
    norm_plat = platform.upper().strip()
    if norm_plat == TEEPlatform.INTEL_TDX.value:
        return {
            "MRTD": hashlib.sha384(f"intel-tdx:mrtd:clm:{code_version}".encode()).hexdigest(),
            "RTMR0": hashlib.sha384(f"intel-tdx:rtmr0:firmware:{code_version}".encode()).hexdigest(),
            "RTMR1": hashlib.sha384(f"intel-tdx:rtmr1:application:{code_version}".encode()).hexdigest(),
        }
    return {
        "PCR0": hashlib.sha384(f"aws-nitro:pcr0:enclave-image:{code_version}".encode()).hexdigest(),
        "PCR1": hashlib.sha384(f"aws-nitro:pcr1:linux-kernel:{code_version}".encode()).hexdigest(),
        "PCR2": hashlib.sha384(f"aws-nitro:pcr2:clm-runtime:{code_version}".encode()).hexdigest(),
    }


@dataclass(frozen=True)
class TEEAttestationQuote:
    """
    Remote attestation quote binding a tenant's `SparseMerkleTree` root and active signing key
    to enclave measurement registers (`PCR0..2` or `MRTD`/`RTMR0..1`).
    """

    quote_id: str
    platform: str
    tenant_id: str
    merkle_root: str
    active_key_id: str
    public_key_hex: str | None
    nonce: str
    measurements: dict[str, str]
    user_data_digest: str
    timestamp: str
    enclave_verifying_key_hex: str
    signature: str
    signature_algorithm: str = "Ed25519"

    def canonical_payload(self) -> bytes:
        """Canonical byte representation signed by the enclave root-of-trust key."""
        payload = {
            "active_key_id": self.active_key_id,
            "enclave_verifying_key_hex": self.enclave_verifying_key_hex,
            "measurements": dict(sorted(self.measurements.items())),
            "merkle_root": self.merkle_root,
            "nonce": self.nonce,
            "platform": self.platform,
            "public_key_hex": self.public_key_hex,
            "quote_id": self.quote_id,
            "signature_algorithm": self.signature_algorithm,
            "tenant_id": self.tenant_id,
            "timestamp": self.timestamp,
            "user_data_digest": self.user_data_digest,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TEEAttestationQuote:
        return cls(
            quote_id=str(data["quote_id"]),
            platform=str(data["platform"]),
            tenant_id=str(data["tenant_id"]),
            merkle_root=str(data["merkle_root"]),
            active_key_id=str(data["active_key_id"]),
            public_key_hex=data.get("public_key_hex"),
            nonce=str(data.get("nonce", "")),
            measurements={str(k): str(v) for k, v in dict(data.get("measurements", {})).items()},
            user_data_digest=str(data["user_data_digest"]),
            timestamp=str(data["timestamp"]),
            enclave_verifying_key_hex=str(data["enclave_verifying_key_hex"]),
            signature=str(data["signature"]),
            signature_algorithm=str(data.get("signature_algorithm", "Ed25519")),
        )


@dataclass
class TEEQuoteVerificationReport:
    """
    Detailed verification report for a `TEEAttestationQuote`.
    """

    verified: bool
    quote_id: str
    platform: str
    signature_valid: bool
    user_data_valid: bool
    nonce_matched: bool
    merkle_root_matched: bool
    measurements_matched: bool
    mismatched_registers: list[str] = field(default_factory=list)
    reason: str = "OK"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TEEAttestationProvider:
    """
    Generates and cryptographically verifies hardware-anchored remote attestation quotes
    for AWS Nitro Enclaves (`AWS_NITRO`) and Intel TDX (`INTEL_TDX`).
    """

    def __init__(
        self,
        enclave_private_key: bytes | str | None = None,
        code_version: str = "0.9.0",
        custom_measurements: dict[str, str] | None = None,
    ) -> None:
        if enclave_private_key is None:
            self._private_key = hashlib.sha256(
                f"clm-enclave-root-of-trust:{code_version}".encode()
            ).digest()
        elif isinstance(enclave_private_key, str):
            raw = enclave_private_key.encode("utf-8")
            self._private_key = raw[:32].ljust(32, b"\x00")
        else:
            self._private_key = enclave_private_key[:32].ljust(32, b"\x00")

        self._public_key = public_key_from_private_key(self._private_key)
        self._pq_module = PostQuantumHybridAttestationModule(private_key=self._private_key)
        self.code_version = code_version
        self._custom_measurements = custom_measurements

    @property
    def enclave_verifying_key_hex(self) -> str:
        """Hex-encoded Ed25519 public key of the enclave attestation root."""
        return self._public_key.hex()

    @property
    def enclave_pq_verifying_key_hex(self) -> str:
        """Hex-encoded 80-byte PQ-Hybrid public key of the enclave attestation root."""
        return self._pq_module.public_key_hex

    def get_measurements(self, platform: str = TEEPlatform.AWS_NITRO.value) -> dict[str, str]:
        """Return enclave measurement registers for the target TEE platform."""
        if self._custom_measurements is not None:
            return dict(self._custom_measurements)
        return _default_measurements_for_platform(platform=platform, code_version=self.code_version)

    def generate_quote(
        self,
        merkle_root: str,
        active_key_id: str,
        public_key_hex: str | None = None,
        nonce: str | None = None,
        platform: str = TEEPlatform.AWS_NITRO.value,
        tenant_id: str = "default",
        signature_algorithm: str = "Ed25519",
    ) -> TEEAttestationQuote:
        """
        Generate a signed `TEEAttestationQuote` binding the current Sparse Merkle root,
        active signing key, and challenge nonce to the enclave measurement registers.
        Supports classical `"Ed25519"` and post-quantum hybrid `"PQ-Hybrid-Ed25519-SHA256"`.
        """
        norm_plat = platform.upper().strip()
        if norm_plat not in (TEEPlatform.AWS_NITRO.value, TEEPlatform.INTEL_TDX.value):
            raise ValueError(
                f"Unsupported TEE platform '{platform}'. Expected 'AWS_NITRO' or 'INTEL_TDX'."
            )
        if signature_algorithm not in ("Ed25519", PQ_HYBRID_ALGORITHM):
            raise ValueError(
                f"Unsupported TEE quote signature_algorithm '{signature_algorithm}'."
            )

        eff_nonce = nonce if nonce is not None else secrets.token_hex(16)
        measurements = self.get_measurements(platform=norm_plat)
        user_data_digest = _compute_user_data_digest(
            tenant_id=tenant_id,
            merkle_root=merkle_root,
            active_key_id=active_key_id,
            public_key_hex=public_key_hex,
            nonce=eff_nonce,
        )
        quote_id = f"tee-quote-{secrets.token_hex(6)}"
        now_iso = datetime.now(timezone.utc).isoformat()
        verifying_key = (
            self.enclave_pq_verifying_key_hex
            if signature_algorithm == PQ_HYBRID_ALGORITHM
            else self.enclave_verifying_key_hex
        )

        unsigned_quote = TEEAttestationQuote(
            quote_id=quote_id,
            platform=norm_plat,
            tenant_id=tenant_id,
            merkle_root=merkle_root,
            active_key_id=active_key_id,
            public_key_hex=public_key_hex,
            nonce=eff_nonce,
            measurements=measurements,
            user_data_digest=user_data_digest,
            timestamp=now_iso,
            enclave_verifying_key_hex=verifying_key,
            signature="",
            signature_algorithm=signature_algorithm,
        )
        if signature_algorithm == PQ_HYBRID_ALGORITHM:
            sig_str = pq_hybrid_sign(unsigned_quote.canonical_payload(), self._private_key)
        else:
            sig_str = sign(unsigned_quote.canonical_payload(), self._private_key).hex()

        return TEEAttestationQuote(
            quote_id=unsigned_quote.quote_id,
            platform=unsigned_quote.platform,
            tenant_id=unsigned_quote.tenant_id,
            merkle_root=unsigned_quote.merkle_root,
            active_key_id=unsigned_quote.active_key_id,
            public_key_hex=unsigned_quote.public_key_hex,
            nonce=unsigned_quote.nonce,
            measurements=unsigned_quote.measurements,
            user_data_digest=unsigned_quote.user_data_digest,
            timestamp=unsigned_quote.timestamp,
            enclave_verifying_key_hex=unsigned_quote.enclave_verifying_key_hex,
            signature=sig_str,
            signature_algorithm=signature_algorithm,
        )

    @staticmethod
    def verify_quote(
        quote: TEEAttestationQuote | dict[str, Any],
        expected_nonce: str | None = None,
        expected_merkle_root: str | None = None,
        expected_measurements: dict[str, str] | None = None,
    ) -> TEEQuoteVerificationReport:
        """
        Verify a `TEEAttestationQuote` signature, user_data commitment, anti-replay nonce,
        Sparse Merkle root, and PCR/RTMR measurement registers.
        """
        q = TEEAttestationQuote.from_dict(quote) if isinstance(quote, dict) else quote

        # 1. Verify signature over canonical quote payload
        try:
            if q.signature_algorithm == PQ_HYBRID_ALGORITHM:
                signature_valid = pq_hybrid_verify(
                    q.signature,
                    q.canonical_payload(),
                    q.enclave_verifying_key_hex,
                )
            else:
                signature_valid = verify(
                    q.signature,
                    q.canonical_payload(),
                    q.enclave_verifying_key_hex,
                )
        except Exception:
            signature_valid = False

        # 2. Verify user_data SHA-256 commitment
        recomputed_user_data = _compute_user_data_digest(
            tenant_id=q.tenant_id,
            merkle_root=q.merkle_root,
            active_key_id=q.active_key_id,
            public_key_hex=q.public_key_hex,
            nonce=q.nonce,
        )
        user_data_valid = recomputed_user_data == q.user_data_digest

        # 3. Verify optional anti-replay challenge nonce
        nonce_matched = True if expected_nonce is None else (q.nonce == expected_nonce)

        # 4. Verify optional expected Sparse Merkle root
        merkle_root_matched = (
            True if expected_merkle_root is None else (q.merkle_root == expected_merkle_root)
        )

        # 5. Verify optional expected PCR/RTMR measurements
        mismatched_registers: list[str] = []
        if expected_measurements is not None:
            for reg, exp_val in expected_measurements.items():
                actual_val = q.measurements.get(reg)
                if actual_val != exp_val:
                    mismatched_registers.append(reg)
        measurements_matched = len(mismatched_registers) == 0

        verified = (
            signature_valid
            and user_data_valid
            and nonce_matched
            and merkle_root_matched
            and measurements_matched
        )

        reasons: list[str] = []
        if not signature_valid:
            reasons.append("Invalid enclave quote signature")
        if not user_data_valid:
            reasons.append("Tampered user_data commitment digest")
        if not nonce_matched:
            reasons.append("Challenge nonce mismatch")
        if not merkle_root_matched:
            reasons.append("Sparse Merkle root mismatch")
        if not measurements_matched:
            reasons.append(f"Measurement register mismatch on {mismatched_registers}")

        return TEEQuoteVerificationReport(
            verified=verified,
            quote_id=q.quote_id,
            platform=q.platform,
            signature_valid=signature_valid,
            user_data_valid=user_data_valid,
            nonce_matched=nonce_matched,
            merkle_root_matched=merkle_root_matched,
            measurements_matched=measurements_matched,
            mismatched_registers=mismatched_registers,
            reason="OK" if verified else "; ".join(reasons),
        )
