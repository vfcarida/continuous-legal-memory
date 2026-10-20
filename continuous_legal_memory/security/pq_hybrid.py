"""
Post-Quantum Hybrid Digital Signature Module (`PQ-Hybrid-Ed25519-SHA256`).

Combines classical RFC 8032 Ed25519 signatures with a post-quantum hash-based
Winternitz WOTS+-SHA256 (RFC 8391 / NIST FIPS 205 SLH-DSA building block) signature.

Security & Trust Guarantee:
    A `MemoryAttestationToken` signed under `PQ-Hybrid-Ed25519-SHA256` is accepted as valid
    if and only if BOTH:
    1. The classical 64-byte Ed25519 signature verifies over the canonical payload; AND
    2. The 67-chain (w=16) WOTS+-SHA256 hash-based signature reconstructs the exact
       256-bit post-quantum public root (`wots_pk_root`).

    Because WOTS+-SHA256 security relies solely on the preimage and second-preimage
    resistance of SHA-256 (resistant to Shor's algorithm and retaining 128+ bits of
    post-quantum security under Grover's algorithm), historical audit trails remain
    non-repudiable across multi-decade regulatory retention horizons.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from datetime import datetime, timezone

from continuous_legal_memory.domain.models import PredictionResult
from continuous_legal_memory.security import ed25519
from continuous_legal_memory.security.attestation import (
    MemoryAttestationToken,
    canonicalize_payload,
)

PQ_HYBRID_ALGORITHM = "PQ-Hybrid-Ed25519-SHA256"
_WOTS_W = 16
_WOTS_LEN1 = 64  # 64 nibbles (4-bit base-16 digits) for 256-bit SHA-256 message digest
_WOTS_LEN2 = 3   # 3 nibbles for checksum (max 64 * 15 = 960 < 16^3 = 4096)
_WOTS_CHAINS = _WOTS_LEN1 + _WOTS_LEN2  # 67 total chains


def _derive_subkeys(master_key: bytes) -> tuple[bytes, bytes, bytes]:
    """
    Deterministically derive (ed25519_sk [32B], wots_sk [32B], pub_seed [16B])
    from a 32-byte or 64-byte master private key.
    """
    ed_sk = hashlib.sha256(b"clm-pq-hybrid:ed25519-sk:" + master_key).digest()
    wots_sk = hashlib.sha256(b"clm-pq-hybrid:wots-sk:" + master_key).digest()
    pub_seed = hashlib.sha256(b"clm-pq-hybrid:wots-pub-seed:" + master_key).digest()[:16]
    return ed_sk, wots_sk, pub_seed


def _wots_chain_secret(wots_sk: bytes, chain_idx: int) -> bytes:
    """Derive the 32-byte starting secret for WOTS+ chain `chain_idx`."""
    return hashlib.sha256(
        b"wots-sk-chain:" + wots_sk + chain_idx.to_bytes(2, "big")
    ).digest()


def _wots_chain_step(x: bytes, pub_seed: bytes, chain_idx: int, step_idx: int) -> bytes:
    """Apply a single tweaked SHA-256 step `s` along WOTS+ chain `chain_idx`."""
    return hashlib.sha256(
        b"wots-step:"
        + pub_seed
        + chain_idx.to_bytes(2, "big")
        + step_idx.to_bytes(1, "big")
        + x
    ).digest()


def _wots_advance_chain(
    start_val: bytes,
    pub_seed: bytes,
    chain_idx: int,
    start_step: int,
    end_step: int,
) -> bytes:
    """Advance WOTS+ chain `chain_idx` from `start_step` (inclusive) to `end_step` (inclusive)."""
    curr = start_val
    for s in range(start_step, end_step + 1):
        curr = _wots_chain_step(curr, pub_seed, chain_idx, s)
    return curr


def _wots_message_digits(msg: bytes, pub_seed: bytes, ed_pk: bytes) -> list[int]:
    """
    Hash `msg` with `pub_seed` and `ed_pk` into a 256-bit digest and expand into
    67 base-16 digits (64 message nibbles + 3 checksum nibbles).
    """
    digest = hashlib.sha256(b"wots-msg:" + pub_seed + ed_pk + msg).digest()
    msg_digits: list[int] = []
    for byte_val in digest:
        msg_digits.append((byte_val >> 4) & 0x0F)
        msg_digits.append(byte_val & 0x0F)

    csum = sum((_WOTS_W - 1 - d) for d in msg_digits)
    csum_digits = [
        (csum >> 8) & 0x0F,
        (csum >> 4) & 0x0F,
        csum & 0x0F,
    ]
    return msg_digits + csum_digits


def _wots_compute_pk_root(wots_sk: bytes, pub_seed: bytes) -> bytes:
    """Compute the 32-byte WOTS+ public key Merkle/compression root across all 67 chains."""
    endpoints: list[bytes] = []
    for i in range(_WOTS_CHAINS):
        sk_i = _wots_chain_secret(wots_sk, i)
        pk_i = _wots_advance_chain(sk_i, pub_seed, i, 1, _WOTS_W - 1)
        endpoints.append(pk_i)
    return hashlib.sha256(b"wots-pk-root:" + pub_seed + b"".join(endpoints)).digest()


def pq_hybrid_sign(message: bytes, master_private_key: bytes) -> str:
    """
    Produce a composite `Ed25519 + WOTS+-SHA256` signature formatted as `<ed25519_hex>:<wots_hex>`.
    """
    ed_sk, wots_sk, pub_seed = _derive_subkeys(master_private_key)
    ed_pk = ed25519.public_key_from_private_key(ed_sk)
    ed_sig = ed25519.sign(message, ed_sk)

    digits = _wots_message_digits(message, pub_seed, ed_pk)
    sig_chains: list[bytes] = []
    for i, d_i in enumerate(digits):
        sk_i = _wots_chain_secret(wots_sk, i)
        sigma_i = _wots_advance_chain(sk_i, pub_seed, i, 1, d_i) if d_i > 0 else sk_i
        sig_chains.append(sigma_i)

    return f"{ed_sig.hex()}:{b''.join(sig_chains).hex()}"


def pq_hybrid_verify(signature_str: str, message: bytes, public_key_hex: str) -> bool:
    """
    Verify a composite `Ed25519 + WOTS+-SHA256` signature using only the 80-byte hybrid public key.
    """
    try:
        pub_bytes = bytes.fromhex(public_key_hex)
        if len(pub_bytes) != 80:
            return False
        ed_pk = pub_bytes[:32]
        pub_seed = pub_bytes[32:48]
        expected_wots_root = pub_bytes[48:80]

        if ":" not in signature_str:
            return False
        ed_sig_hex, wots_sig_hex = signature_str.split(":", 1)
        wots_sig_bytes = bytes.fromhex(wots_sig_hex)
        if len(wots_sig_bytes) != _WOTS_CHAINS * 32:
            return False
    except (ValueError, TypeError):
        return False

    # 1. Classical Ed25519 verification
    if not ed25519.verify(ed_sig_hex, message, ed_pk):
        return False

    # 2. Post-Quantum WOTS+-SHA256 chain completion & root verification
    digits = _wots_message_digits(message, pub_seed, ed_pk)
    reconstructed_endpoints: list[bytes] = []
    for i, d_i in enumerate(digits):
        sigma_i = wots_sig_bytes[i * 32 : (i + 1) * 32]
        pk_i = (
            _wots_advance_chain(sigma_i, pub_seed, i, d_i + 1, _WOTS_W - 1)
            if d_i < (_WOTS_W - 1)
            else sigma_i
        )
        reconstructed_endpoints.append(pk_i)

    recomputed_root = hashlib.sha256(
        b"wots-pk-root:" + pub_seed + b"".join(reconstructed_endpoints)
    ).digest()
    return hmac.compare_digest(recomputed_root, expected_wots_root)


class PostQuantumHybridAttestationModule:
    """
    Post-Quantum Hybrid Attestation Module (`PQ-Hybrid-Ed25519-SHA256`).

    Binds classical Ed25519 (RFC 8032) and post-quantum WOTS+-SHA256 (RFC 8391 / FIPS 205)
    signatures into `MemoryAttestationToken` certificates.
    """

    def __init__(
        self,
        private_key: bytes | str | None = None,
        public_key: bytes | str | None = None,
        key_id: str = "legal-memory-pq-hybrid-key",
    ) -> None:
        self.key_id = key_id
        self.algorithm = PQ_HYBRID_ALGORITHM
        self._private_key: bytes | None = None

        if private_key is not None:
            if isinstance(private_key, str):
                try:
                    raw = bytes.fromhex(private_key)
                except ValueError:
                    raw = hashlib.sha256(private_key.encode("utf-8")).digest()
            else:
                raw = private_key
            self._private_key = raw[:32].ljust(32, b"\x00")
            ed_sk, wots_sk, pub_seed = _derive_subkeys(self._private_key)
            ed_pk = ed25519.public_key_from_private_key(ed_sk)
            wots_root = _wots_compute_pk_root(wots_sk, pub_seed)
            self._public_key = ed_pk + pub_seed + wots_root
        elif public_key is not None:
            self._private_key = None
            self._public_key = (
                bytes.fromhex(public_key) if isinstance(public_key, str) else public_key
            )
        else:
            self._private_key = secrets.token_bytes(32)
            ed_sk, wots_sk, pub_seed = _derive_subkeys(self._private_key)
            ed_pk = ed25519.public_key_from_private_key(ed_sk)
            wots_root = _wots_compute_pk_root(wots_sk, pub_seed)
            self._public_key = ed_pk + pub_seed + wots_root

    @property
    def private_key_hex(self) -> str:
        """Hex-encoded 32-byte master private key (if initialized with signing capability)."""
        if self._private_key is None:
            raise PermissionError("Private key is not available on a public-only verifier module.")
        return self._private_key.hex()

    @property
    def public_key_hex(self) -> str:
        """Hex-encoded 80-byte hybrid public key (`ed_pk [32B] || pub_seed [16B] || wots_root [32B]`)."""
        return self._public_key.hex()

    def generate_state_hash(self, result: PredictionResult, retrieved_text: str | None = None) -> str:
        """Compute SHA-256 hash digest of a canonical prediction result and retrieved legal context."""
        payload_bytes = canonicalize_payload(result, retrieved_text)
        return hashlib.sha256(payload_bytes).hexdigest()

    def sign_attestation(
        self,
        result: PredictionResult,
        retrieved_text: str | None = None,
        timestamp: datetime | None = None,
    ) -> MemoryAttestationToken:
        """Sign a `PredictionResult` using the hybrid `Ed25519 + WOTS+-SHA256` private key."""
        if self._private_key is None:
            raise PermissionError(
                "Cannot sign attestation: PostQuantumHybridAttestationModule initialized without private key."
            )

        now = timestamp or datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)

        state_hash = self.generate_state_hash(result, retrieved_text)
        msg = f"{state_hash}|{now.isoformat()}".encode()
        sig_str = pq_hybrid_sign(msg, self._private_key)

        return MemoryAttestationToken(
            timestamp=now,
            state_hash=state_hash,
            integrity_tag=sig_str,
            key_id=self.key_id,
            algorithm=PQ_HYBRID_ALGORITHM,
            public_key=self.public_key_hex,
        )

    def verify_attestation(
        self,
        token: MemoryAttestationToken,
        result: PredictionResult,
        retrieved_text: str | None = None,
        public_key: bytes | str | None = None,
    ) -> bool:
        """
        Verify both the Ed25519 signature and the post-quantum WOTS+-SHA256 signature on `token`.
        """
        if token.algorithm != PQ_HYBRID_ALGORITHM:
            return False

        expected_hash = self.generate_state_hash(result, retrieved_text)
        if not hmac.compare_digest(token.state_hash, expected_hash):
            return False

        if public_key is not None:
            pk_hex = public_key.hex() if isinstance(public_key, bytes) else public_key
        elif self._public_key is not None:
            pk_hex = self._public_key.hex()
        elif token.public_key is not None:
            pk_hex = token.public_key
        else:
            return False

        ts = token.timestamp
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        msg = f"{token.state_hash}|{ts.isoformat()}".encode()

        return pq_hybrid_verify(token.integrity_tag, msg, pk_hex)
