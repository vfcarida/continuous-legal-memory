"""
Unit tests for Keyed-Hash Attestation, Canonicalization, and Key Management (CLM-T05).
"""

from datetime import datetime, timezone

from continuous_legal_memory.domain.models import PredictionResult
from continuous_legal_memory.security.attestation import (
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
