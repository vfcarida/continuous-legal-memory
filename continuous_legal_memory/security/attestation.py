"""
Keyed-Hash Attestation Module.

Generates deterministic SHA-256 payload digests and HMAC-SHA256 integrity tags for
retrieved memory context states, providing verifiable tamper-detection for technical
auditability and record-keeping controls.

Trust Model & Scope:
    This module implements a symmetric, single-trust-domain integrity tag (HMAC-SHA256).
    Verification requires possession of the shared secret key. When configured with a durable
    secret key (via secret_key parameter or CLM_ATTESTATION_SECRET environment variable),
    attestation tokens can be verified across process instances and service restarts.
    If no key is configured, an ephemeral in-process key is generated with single-session scope.

    Asymmetric multi-party zero-trust signatures (e.g., Ed25519 or RSA-PSS with public-key
    distribution) require an external cryptographic provider (e.g., cryptography library)
    and are scoped as future work.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import secrets
from datetime import datetime, timezone
from typing import Any

from continuous_legal_memory.domain.models import PredictionResult

logger = logging.getLogger(__name__)


def canonicalize_payload(result: PredictionResult, retrieved_text: str | None = None) -> bytes:
    """
    Deterministically canonicalize a PredictionResult and retrieved context into bytes.

    Ensures stable field ordering, normalized unicode, and fixed float precision to guarantee
    identical byte representations for identical semantic inputs across platforms and versions.

    Args:
        result: PredictionResult instance.
        retrieved_text: Optional retrieved legal text snippet.

    Returns:
        UTF-8 encoded canonical byte string.
    """
    text_payload = (retrieved_text if retrieved_text is not None else result.most_relevant_rule) or ""
    # Fixed-precision float formatting (6 decimal places) for numerical determinism
    vec_str = ",".join(f"{float(v):.6f}" for v in result.predicted_action_vector)
    canonical_str = (
        f"ACTION:{vec_str}\n"
        f"QUERY:{result.query.strip()}\n"
        f"TEXT:{text_payload.strip()}"
    )
    return canonical_str.encode()


class MemoryAttestationToken:
    """
    Immutable attestation token representing a cryptographically verified legal memory state.

    Attributes:
        timestamp: UTC timestamp of attestation generation.
        state_hash: SHA-256 digest of canonical query, retrieved text, and decision vector.
        integrity_tag: HMAC-SHA256 hex string verifying token authenticity and state integrity.
        key_id: Key identifier associated with the signing secret.
        algorithm: Integrity algorithm identifier ('HMAC-SHA256').
    """

    def __init__(
        self,
        timestamp: datetime,
        state_hash: str,
        integrity_tag: str | None = None,
        key_id: str = "legal-memory-default-key",
        algorithm: str = "HMAC-SHA256",
        signature: str | None = None,
        public_key_id: str | None = None,
    ) -> None:
        self.timestamp = timestamp
        self.state_hash = state_hash
        # Support both new integrity_tag and legacy signature argument
        self.integrity_tag = integrity_tag or signature or ""
        # Support both new key_id and legacy public_key_id argument
        self.key_id = key_id if key_id != "legal-memory-default-key" or not public_key_id else (public_key_id or key_id)
        self.algorithm = algorithm

    @property
    def signature(self) -> str:
        """Backward-compatibility alias for integrity_tag."""
        return self.integrity_tag

    @property
    def public_key_id(self) -> str:
        """Backward-compatibility alias for key_id."""
        return self.key_id

    def __repr__(self) -> str:
        return (
            f"MemoryAttestationToken(timestamp={self.timestamp.isoformat()!r}, "
            f"state_hash={self.state_hash[:16]}..., "
            f"integrity_tag={self.integrity_tag[:16]}..., "
            f"key_id={self.key_id!r}, algorithm={self.algorithm!r})"
        )

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, MemoryAttestationToken):
            return False
        return (
            self.timestamp == other.timestamp
            and self.state_hash == other.state_hash
            and self.integrity_tag == other.integrity_tag
            and self.key_id == other.key_id
            and self.algorithm == other.algorithm
        )


class KeyedHashAttestationModule:
    """
    Keyed-hash verification module for memory state attestation.

    Rationale:
        Under high-risk AI record-keeping and traceability requirements, legal AI systems
        benefit from verifiable, tamper-evident audit logs. This module produces SHA-256 state
        hashes and HMAC-SHA256 integrity tags over retrieved contexts and decision outputs.

    Trust Model:
        Symmetric HMAC scheme. Verification requires the shared secret. To verify tokens across
        process restarts or between authorized services, provide a persistent secret key via
        the `secret_key` parameter or the `CLM_ATTESTATION_SECRET` environment variable.
    """

    def __init__(
        self,
        secret_key: bytes | str | None = None,
        key_id: str = "legal-memory-default-key",
        env_var: str = "CLM_ATTESTATION_SECRET",
    ) -> None:
        """
        Initialize KeyedHashAttestationModule.

        Args:
            secret_key: Symmetric secret key (bytes or str). If None, checks `env_var`.
                        If both are absent, generates an ephemeral 32-byte secret (scoped to single process).
            key_id: Identifier label associated with the secret key.
            env_var: Name of environment variable containing the fallback shared secret.
        """
        self.key_id = key_id
        self.is_ephemeral = False

        if secret_key is not None:
            self._secret_key = secret_key.encode() if isinstance(secret_key, str) else secret_key
        elif os.environ.get(env_var):
            self._secret_key = os.environ[env_var].encode()
        else:
            self._secret_key = secrets.token_bytes(32)
            self.is_ephemeral = True
            logger.debug(
                "No secret_key provided or found in %s; generated an ephemeral key. "
                "Tokens will not verify across process restarts.",
                env_var,
            )

    def generate_state_hash(self, result: PredictionResult, retrieved_text: str | None = None) -> str:
        """
        Compute SHA-256 hash digest of a canonical prediction result and retrieved legal context.

        Args:
            result: PredictionResult instance.
            retrieved_text: Exact retrieved legal text snippet.

        Returns:
            64-character SHA-256 hex string digest.
        """
        payload_bytes = canonicalize_payload(result, retrieved_text)
        return hashlib.sha256(payload_bytes).hexdigest()

    def sign_attestation(
        self,
        result: PredictionResult,
        retrieved_text: str | None = None,
        timestamp: datetime | None = None,
    ) -> MemoryAttestationToken:
        """
        Generate a cryptographically verified `MemoryAttestationToken` for a memory prediction output.

        Args:
            result: PredictionResult instance.
            retrieved_text: Optional text snippet.
            timestamp: Optional generation timestamp (defaults to current UTC time).

        Returns:
            A populated `MemoryAttestationToken` containing state hash and HMAC-SHA256 integrity tag.
        """
        now = timestamp or datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)

        state_hash = self.generate_state_hash(result, retrieved_text)

        # Canonical message for HMAC: state_hash || ISO 8601 UTC timestamp
        msg = f"{state_hash}|{now.isoformat()}".encode()
        tag = hmac.new(self._secret_key, msg, hashlib.sha256).hexdigest()

        return MemoryAttestationToken(
            timestamp=now,
            state_hash=state_hash,
            integrity_tag=tag,
            key_id=self.key_id,
            algorithm="HMAC-SHA256",
        )

    def verify_attestation(
        self,
        token: MemoryAttestationToken,
        result: PredictionResult,
        retrieved_text: str | None = None,
    ) -> bool:
        """
        Verify the authenticity and integrity of a `MemoryAttestationToken`.

        Args:
            token: MemoryAttestationToken to verify.
            result: Corresponding PredictionResult instance.
            retrieved_text: Corresponding text snippet.

        Returns:
            True if integrity tag and state hash match; False if tampered, invalid, or key mismatch.
        """
        # 1. Verify that the state hash matches the current canonical payload
        expected_hash = self.generate_state_hash(result, retrieved_text)
        if not hmac.compare_digest(token.state_hash, expected_hash):
            return False

        # 2. Verify HMAC integrity tag against secret key and token timestamp
        ts = token.timestamp
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        msg = f"{token.state_hash}|{ts.isoformat()}".encode()
        expected_tag = hmac.new(self._secret_key, msg, hashlib.sha256).hexdigest()

        return hmac.compare_digest(token.integrity_tag, expected_tag)


# Backward-compatibility alias
CryptographicAttestationModule = KeyedHashAttestationModule
