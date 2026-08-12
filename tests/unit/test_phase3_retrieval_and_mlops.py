"""
Unit tests for Phase 3 Precision Retrieval, Cryptographic Attestation, Agentic Evaluation, and Observability.
"""

import torch

from continuous_legal_memory.adapters.ollama import OllamaGemmaAdapter
from continuous_legal_memory.domain.models import MemoryRecord, PredictionResult
from continuous_legal_memory.evaluation.agentic_eval import AgenticLegalEvaluator
from continuous_legal_memory.retrieval.hybrid_retriever import BM25Okapi, HybridLegalRetriever
from continuous_legal_memory.security.attestation import CryptographicAttestationModule
from continuous_legal_memory.telemetry.observability import TelemetryLogger


def test_bm25_okapi_scoring() -> None:
    """Verify BM25 keyword matching scores."""
    corpus = [
        "Article 1 states that data subjects have the right to erasure.",
        "Article 2 states that credit history must be retained for anti-fraud auditing.",
    ]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores("erasure right")

    assert scores[0] > scores[1]
    assert scores[0] > 0.0


def test_hybrid_legal_retriever() -> None:
    """Verify hybrid retrieval with BM25, dense vector similarities, and character snippet extraction."""
    encoder = OllamaGemmaAdapter(embedding_dim=128, strict_privacy_mode=True)
    retriever = HybridLegalRetriever(encoder=encoder)

    text_1 = "Article 1: All personal data must be erased upon customer request within 15 business days."
    text_2 = "Article 2: Credit operations history active in last 5 years is exempt from deletion for auditing."

    key_1 = encoder.get_embedding([text_1])
    key_2 = encoder.get_embedding([text_2])
    val = torch.zeros(1, 2)

    rec1 = MemoryRecord(text=text_1, key_vector=key_1, value_vector=val)
    rec2 = MemoryRecord(text=text_2, key_vector=key_2, value_vector=val)

    results = retriever.retrieve_with_snippets("erased customer request", [rec1, rec2], top_k=2)
    assert len(results) == 2

    top_rec, score, bounds, snippet = results[0]
    assert top_rec.text == text_1
    assert score > 0.0
    assert isinstance(bounds, tuple)
    assert len(snippet) > 0


def test_cryptographic_attestation_module() -> None:
    """Verify SHA-256 hash generation, attestation signing, and signature verification."""
    attestation_mod = CryptographicAttestationModule(key_id="test-key-01")
    pred = PredictionResult(
        query="Delete customer data",
        predicted_action_vector=[1.0, 0.0],
        most_relevant_rule="Article 1: Erasure right",
    )

    token = attestation_mod.sign_attestation(pred, retrieved_text="Article 1: Erasure right")
    assert token.state_hash is not None
    assert token.signature is not None

    # Verification must succeed for authentic prediction result
    is_valid = attestation_mod.verify_attestation(token, pred, retrieved_text="Article 1: Erasure right")
    assert is_valid is True

    # Verification must fail if prediction result is tampered
    tampered_pred = PredictionResult(
        query="Delete customer data",
        predicted_action_vector=[0.0, 1.0],  # Altered decision vector
        most_relevant_rule="Article 1: Erasure right",
    )
    is_valid_tampered = attestation_mod.verify_attestation(token, tampered_pred, retrieved_text="Article 1: Erasure right")
    assert is_valid_tampered is False


def test_agentic_evaluator_metrics() -> None:
    """Verify Context Precision, Plan Adherence, and Cross-Session Recall metric functions."""
    evaluator = AgenticLegalEvaluator()

    # Test Context Precision
    cp = evaluator.evaluate_context_precision(
        query="Right to erasure",
        retrieved_snippet="Article 1 states that customers have the right to erasure.",
        ground_truth="Customers have the right to erasure of personal data.",
    )
    assert cp > 0.5

    # Test Plan Adherence
    executed = ["1. Ingest document", "2. Check contradictions", "3. Predict decision"]
    expected = ["1. Ingest document", "2. Check contradictions", "3. Predict decision"]
    pa = evaluator.evaluate_plan_adherence(executed, expected)
    assert pa == 1.0

    # Test Cross-Session Recall
    csr = evaluator.evaluate_cross_session_recall([1.0, 0.0], [1.0, 0.0])
    assert csr == 1.0


def test_telemetry_logger() -> None:
    """Verify TelemetryLogger tracing function execution and latency tracking."""
    logger = TelemetryLogger(service_name="test-service")

    def mock_operation(x: int) -> int:
        return x * 2

    res, metrics = logger.trace_operation("mock_op", mock_operation, 5)
    assert res == 10
    assert metrics.latency_ms >= 0.0
    assert metrics.tokens_processed > 0
