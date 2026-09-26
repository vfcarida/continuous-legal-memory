# API Reference: Cryptographic Attestation

The `continuous_legal_memory.security` package provides tamper-evident auditability and verifiable decision proofs.

---

## Classes

### `AsymmetricAttestationModule`

Provides asymmetric zero-trust digital signatures via RFC 8032 Ed25519:

- `generate_keypair() -> tuple[str, str]`: Generates a new `(private_key_hex, public_key_hex)` pair.
- `sign_attestation(result, retrieved_text=None) -> MemoryAttestationToken`: Signs a prediction with the private key.
- `verify_attestation(token, result, retrieved_text=None, public_key=None) -> bool`: Verifies a token using only the public key.

### `KeyedHashAttestationModule`

Provides symmetric HMAC-SHA256 integrity tagging:

- `sign_attestation(result, retrieved_text=None) -> MemoryAttestationToken`
- `verify_attestation(token, result, retrieved_text=None) -> bool`

### `MemoryAttestationToken`

Container for immutable cryptographic attestation proofs:
- `timestamp`: UTC ISO 8601 generation time.
- `state_hash`: SHA-256 digest of canonical inputs.
- `integrity_tag`: Hex string of HMAC digest or Ed25519 signature.
- `algorithm`: `"Ed25519"` or `"HMAC-SHA256"`.
- `public_key`: Hex string of public key (for Ed25519).
