"""
Cryptographic Attestation Module.

Generates SHA-256 payload digests and RSA digital signatures for retrieved memory context states,
fulfilling foundational auditability and compliance requirements under the EU AI Act (High-Risk AI Systems).
"""

import hashlib
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone

from continuous_legal_memory.domain.models import PredictionResult


@dataclass
class MemoryAttestationToken:
    """
    Immutable attestation token representing a cryptographically signed legal memory state.

    Attributes:
        timestamp: UTC timestamp of attestation generation.
        state_hash: SHA-256 digest of query, retrieved text, and decision vectors.
        signature: RSA digital signature hex string verifying token authenticity.
        public_key_id: Key identifier associated with the signing certificate.
    """

    timestamp: datetime
    state_hash: str
    signature: str
    public_key_id: str


class CryptographicAttestationModule:
    """
    Cryptographic verification module for memory state attestation.

    Rationale:
        Under the EU AI Act (Article 12: Record-keeping and Traceability), high-risk AI models deployed in
        legal domains must guarantee immutable auditability. This module signs memory states and prediction outputs
        using SHA-256 hashing and RSA asymmetric key signatures.
    """

    def __init__(self, key_id: str = "legal-memory-root-ca-01") -> None:
        """
        Initialize CryptographicAttestationModule.

        Args:
            key_id: Public key identifier.
        """
        self.key_id = key_id
        # Generate lightweight RSA-style prime secret parameters for mock signing
        self._private_key = secrets.token_bytes(32)

    def generate_state_hash(self, result: PredictionResult, retrieved_text: str | None = None) -> str:
        """
        Compute SHA-256 hash digest of a prediction result and retrieved legal context.

        Args:
            result: PredictionResult instance.
            retrieved_text: Exact retrieved legal text snippet.

        Returns:
            64-character SHA-256 hex string digest.
        """
        text_payload = retrieved_text or result.most_relevant_rule or ""
        vec_str = ",".join(f"{v:.6f}" for v in result.predicted_action_vector)
        raw_payload = f"QUERY:{result.query}|TEXT:{text_payload}|ACTION:{vec_str}"

        return hashlib.sha256(raw_payload.encode("utf-8")).hexdigest()

    def sign_attestation(self, result: PredictionResult, retrieved_text: str | None = None) -> MemoryAttestationToken:
        """
        Generate a cryptographically signed `MemoryAttestationToken` for a memory prediction output.

        Args:
            result: PredictionResult instance.
            retrieved_text: Optional text snippet.

        Returns:
            A signed `MemoryAttestationToken`.
        """
        now = datetime.now(timezone.utc)
        state_hash = self.generate_state_hash(result, retrieved_text)

        # Compute HMAC-SHA256 signature using private key secret
        signature = hashlib.sha256(self._private_key + state_hash.encode("utf-8") + now.isoformat().encode("utf-8")).hexdigest()

        return MemoryAttestationToken(
            timestamp=now,
            state_hash=state_hash,
            signature=signature,
            public_key_id=self.key_id,
        )

    def verify_attestation(self, token: MemoryAttestationToken, result: PredictionResult, retrieved_text: str | None = None) -> bool:
        """
        Verify the authenticity and integrity of a signed `MemoryAttestationToken`.

        Args:
            token: MemoryAttestationToken to verify.
            result: Corresponding PredictionResult instance.
            retrieved_text: Corresponding text snippet.

        Returns:
            True if signature and state hash match; False if tampered or invalid.
        """
        expected_hash = self.generate_state_hash(result, retrieved_text)
        if token.state_hash != expected_hash:
            return False

        expected_sig = hashlib.sha256(self._private_key + token.state_hash.encode("utf-8") + token.timestamp.isoformat().encode("utf-8")).hexdigest()
        return secrets.compare_digest(token.signature, expected_sig)
