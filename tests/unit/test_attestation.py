"""
Unit tests for Keyed-Hash Attestation, Canonicalization, and Key Management (CLM-T05).
"""

from datetime import datetime, timezone

from continuous_legal_memory.domain.models import PredictionResult
from continuous_legal_memory.security.attestation import (
    AsymmetricAttestationModule,
    CryptographicAttestationModule,
    KeyedHashAttestationModule,
    canonicalize_payload,
)


def _make_prediction(
    query: str = "Client requested data deletion",
    vector: list[float] | None = None,
    rule: str = "Article 17 GDPR: Right to erasure",
) -> PredictionResult:
    """Helper creating standard test prediction result."""
    return PredictionResult(
        query=query,
        predicted_action_vector=vector or [1.0, 0.0],
        most_relevant_rule=rule,
    )


def test_tamper_detection_on_action_vector() -> None:
    """Verify that tampering with the predicted decision vector causes verification to fail."""
    module = KeyedHashAttestationModule(secret_key="test-secret-key")
    pred = _make_prediction(vector=[1.0, 0.0])

    token = module.sign_attestation(pred)
    assert module.verify_attestation(token, pred) is True

    # Tampered prediction vector
    tampered = _make_prediction(vector=[0.0, 1.0])
    assert module.verify_attestation(token, tampered) is False


def test_tamper_detection_on_query_and_text() -> None:
    """Verify that altering query string or retrieved text snippet invalidates the token."""
    module = KeyedHashAttestationModule(secret_key="test-secret-key")
    pred = _make_prediction(query="Original Query", rule="Rule Text Alpha")

    token = module.sign_attestation(pred, retrieved_text="Rule Text Alpha")
    assert module.verify_attestation(token, pred, retrieved_text="Rule Text Alpha") is True

    # Tampered query
    tampered_query = _make_prediction(query="Manipulated Query", rule="Rule Text Alpha")
    assert module.verify_attestation(token, tampered_query, retrieved_text="Rule Text Alpha") is False

    # Tampered snippet
    assert module.verify_attestation(token, pred, retrieved_text="Tampered Text Beta") is False


def test_canonical_payload_stability() -> None:
    """Verify that identical inputs produce bitwise identical canonical payloads and SHA-256 hashes."""
    pred1 = _make_prediction(query="Test query with trailing space  ", vector=[1.0000001, 0.0])
    pred2 = _make_prediction(query="Test query with trailing space", vector=[1.0, 0.0])

    # Stripping whitespace should produce identical canonical bytes
    p1_bytes = canonicalize_payload(pred1, retrieved_text="  Article 1   ")
    p2_bytes = canonicalize_payload(pred2, retrieved_text="Article 1")
    assert p1_bytes == p2_bytes

    module = KeyedHashAttestationModule(secret_key="fixed-key")
    h1 = module.generate_state_hash(pred1, retrieved_text="  Article 1   ")
    h2 = module.generate_state_hash(pred2, retrieved_text="Article 1")
    assert h1 == h2


def test_cross_instance_verification_with_configured_secret() -> None:
    """
    Verify that an attestation token generated in one process/instance can be
    verified in a completely separate instance when the shared secret key is provided.
    """
    secret = b"durable-enterprise-secret-key-32b!"

    # Instance 1 signs the token
    signer = KeyedHashAttestationModule(secret_key=secret, key_id="key-v1")
    pred = _make_prediction()
    token = signer.sign_attestation(pred)

    # Separate Instance 2 verifies the token
    verifier = KeyedHashAttestationModule(secret_key=secret, key_id="key-v1")
    assert verifier.verify_attestation(token, pred) is True


def test_cross_instance_verification_via_environment_variable(monkeypatch) -> None:
    """Verify that key management via CLM_ATTESTATION_SECRET works seamlessly across instances."""
    monkeypatch.setenv("CLM_ATTESTATION_SECRET", "super-secret-audit-key-2026")

    signer = KeyedHashAttestationModule(key_id="env-key-01")
    assert signer.is_ephemeral is False

    pred = _make_prediction()
    token = signer.sign_attestation(pred)

    verifier = KeyedHashAttestationModule(key_id="env-key-01")
    assert verifier.is_ephemeral is False
    assert verifier.verify_attestation(token, pred) is True


def test_verification_fails_with_different_key() -> None:
    """Verify that verification fails if the verifier does not possess the correct signing secret."""
    signer = KeyedHashAttestationModule(secret_key="secret-key-A")
    pred = _make_prediction()
    token = signer.sign_attestation(pred)

    wrong_verifier = KeyedHashAttestationModule(secret_key="secret-key-B")
    assert wrong_verifier.verify_attestation(token, pred) is False


def test_backward_compatibility_aliases() -> None:
    """Verify that legacy CryptographicAttestationModule and token.signature properties work as expected."""
    module = CryptographicAttestationModule(key_id="legacy-id")
    pred = _make_prediction()

    fixed_time = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    token = module.sign_attestation(pred, timestamp=fixed_time)

    # Check that legacy property aliases point to primary fields
    assert token.signature == token.integrity_tag
    assert token.public_key_id == token.key_id
    assert token.algorithm == "HMAC-SHA256"

    # Verify works via legacy class name
    assert module.verify_attestation(token, pred) is True


def test_asymmetric_ed25519_keypair_generation_and_signing() -> None:
    """Verify Ed25519 keypair generation, token signing, and public-key-only verification."""
    priv_hex, pub_hex = AsymmetricAttestationModule.generate_keypair()
    assert len(priv_hex) == 64  # 32 bytes hex
    assert len(pub_hex) == 64  # 32 bytes hex

    signer = AsymmetricAttestationModule(private_key=priv_hex, key_id="court-audit-key")
    pred = _make_prediction(query="GDPR erasure request", vector=[1.0, 0.0])

    token = signer.sign_attestation(pred)
    assert token.algorithm == "Ed25519"
    assert token.public_key == pub_hex
    assert len(token.integrity_tag) == 128  # 64 bytes hex

    # Verification using the signer instance
    assert signer.verify_attestation(token, pred) is True


def test_asymmetric_ed25519_verification_only_client() -> None:
    """
    Verify that an auditor or regulator can verify attestation tokens using ONLY
    the public key, with zero knowledge of the private signing key.
    """
    priv_hex, pub_hex = AsymmetricAttestationModule.generate_keypair()
    signer = AsymmetricAttestationModule(private_key=priv_hex)
    pred = _make_prediction()
    token = signer.sign_attestation(pred)

    # Auditor only possesses public key
    auditor = AsymmetricAttestationModule(public_key=pub_hex)
    assert auditor._private_key is None

    # Signing must fail for verification-only auditor
    import pytest
    with pytest.raises(PermissionError, match="verification-only mode"):
        auditor.sign_attestation(pred)

    # But verification must succeed
    assert auditor.verify_attestation(token, pred) is True

    # Tampered prediction vector fails
    tampered_pred = _make_prediction(vector=[0.0, 1.0])
    assert auditor.verify_attestation(token, tampered_pred) is False


def test_asymmetric_ed25519_tamper_detection() -> None:
    """Verify that tampering with query, retrieved snippet, or signature invalidates token."""
    priv_hex, pub_hex = AsymmetricAttestationModule.generate_keypair()
    signer = AsymmetricAttestationModule(private_key=priv_hex)
    pred = _make_prediction(query="Original Legal Query", rule="Article 5 Directive")

    token = signer.sign_attestation(pred, retrieved_text="Article 5 Directive")
    assert signer.verify_attestation(token, pred, retrieved_text="Article 5 Directive") is True

    # Tampered query
    tampered_q = _make_prediction(query="Altered Legal Query", rule="Article 5 Directive")
    assert signer.verify_attestation(token, tampered_q, retrieved_text="Article 5 Directive") is False

    # Tampered snippet
    assert signer.verify_attestation(token, pred, retrieved_text="Article 9 Contradiction") is False

    # Corrupted signature byte
    corrupted_tag = "00" + token.integrity_tag[2:]
    token.integrity_tag = corrupted_tag
    assert signer.verify_attestation(token, pred, retrieved_text="Article 5 Directive") is False


def test_asymmetric_ed25519_wrong_public_key_rejection() -> None:
    """Verify that verification fails when checked against an unrelated public key."""
    priv_hex_a, _ = AsymmetricAttestationModule.generate_keypair()
    _, pub_hex_b = AsymmetricAttestationModule.generate_keypair()

    signer = AsymmetricAttestationModule(private_key=priv_hex_a)
    pred = _make_prediction()
    token = signer.sign_attestation(pred)

    auditor_wrong = AsymmetricAttestationModule(public_key=pub_hex_b)
    assert auditor_wrong.verify_attestation(token, pred) is False


def test_asymmetric_ed25519_env_var_configuration(monkeypatch) -> None:
    """Verify that CLM_ATTESTATION_PRIVATE_KEY environment variable is automatically loaded."""
    priv_hex, pub_hex = AsymmetricAttestationModule.generate_keypair()
    monkeypatch.setenv("CLM_ATTESTATION_PRIVATE_KEY", priv_hex)

    module = AsymmetricAttestationModule()
    assert module.is_ephemeral is False
    assert module.public_key_hex == pub_hex

    pred = _make_prediction()
    token = module.sign_attestation(pred)
    assert module.verify_attestation(token, pred) is True


def test_keyed_hash_module_can_verify_ed25519_token() -> None:
    """Verify cross-module interoperability: KeyedHash module verifies Ed25519 token via public key."""
    priv_hex, _ = AsymmetricAttestationModule.generate_keypair()
    signer = AsymmetricAttestationModule(private_key=priv_hex)
    pred = _make_prediction()
    token = signer.sign_attestation(pred)

    symmetric_module = KeyedHashAttestationModule(secret_key="some-hmac-secret")
    assert symmetric_module.verify_attestation(token, pred) is True

