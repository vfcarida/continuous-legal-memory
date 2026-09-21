"""
Unit tests for CLM-T08 Evaluation Harness and Synthetic Benchmark.

Validates:
1. Benchmark dataset parsing and schema validation.
2. DeterministicTokenEncoder properties (norm, determinism, keyword overlap sensitivity).
3. All 5 baselines execution and valid bounded action vector output.
4. Metric computation (override, authority, backward transfer, stale recall).
5. Adversarial evaluations (poisoning ASR and multi-tenant leakage).
6. Go/No-Go pre-registered decision logic.
"""

from pathlib import Path

import pytest
import torch

from continuous_legal_memory.evaluation.harness import (
    BenchmarkDataset,
    BoundedHistoryBaseline,
    DeterministicTokenEncoder,
    EvaluationHarness,
    HybridMemoryBaseline,
    NoMemoryBaseline,
    PlainRAGBaseline,
    TemporalStructuredBaseline,
)

FIXTURE_PATH = Path("tests/fixtures/synthetic_legal_benchmark.jsonl")


@pytest.fixture
def dataset() -> BenchmarkDataset:
    assert FIXTURE_PATH.exists(), f"Benchmark fixture missing at {FIXTURE_PATH}"
    return BenchmarkDataset.load_from_jsonl(FIXTURE_PATH)


@pytest.fixture
def token_encoder() -> DeterministicTokenEncoder:
    return DeterministicTokenEncoder(embedding_dim=64)


def test_dataset_loading_and_schema(dataset: BenchmarkDataset) -> None:
    """Verify synthetic dataset metadata, rule schema, and query properties."""
    assert dataset.metadata["dataset_name"] == "SyntheticLegalBenchmark-v1"
    assert "NOT LEGALLY VALIDATED" in dataset.metadata["description"]

    assert len(dataset.rules) == 60
    assert len(dataset.queries) == 40

    # Verify rule fields
    for r in dataset.rules:
        assert "rule_id" in r
        assert "text" in r and len(r["text"]) > 0
        assert "action" in r and len(r["action"]) == 2
        assert "authority_rank" in r and 1 <= r["authority_rank"] <= 10
        assert "valid_from" in r

    # Verify query fields
    scenarios = set()
    for q in dataset.queries:
        assert "query_id" in q
        assert "query_text" in q and len(q["query_text"]) > 0
        assert "ground_truth_action" in q and len(q["ground_truth_action"]) == 2
        assert "scenario_type" in q
        scenarios.add(q["scenario_type"])

    expected_scenarios = {
        "authority_hierarchy",
        "override_conflict",
        "explicit_supersedes",
        "temporal_validity",
        "standard_recall",
        "right_to_erasure",
        "tenant_isolation",
        "poisoning_probe",
    }
    assert expected_scenarios.issubset(scenarios)


def test_deterministic_token_encoder(token_encoder: DeterministicTokenEncoder) -> None:
    """Verify token encoder produces normalized, deterministic, keyword-sensitive embeddings."""
    texts = [
        "Federal Banking Statute: Retain financial transaction ledgers.",
        "Internal Policy: Customer loan ledgers deletion protocol.",
        "Unrelated environmental forest preservation guideline.",
    ]
    emb = token_encoder.get_embedding(texts)
    assert emb.shape == (3, 64)

    # Unit norm check
    norms = torch.norm(emb, p=2, dim=-1)
    for n in norms:
        assert float(n.item()) == pytest.approx(1.0, abs=1e-4)

    # Determinism across fresh instance
    encoder2 = DeterministicTokenEncoder(embedding_dim=64)
    emb2 = encoder2.get_embedding(texts)
    assert torch.equal(emb, emb2)

    # Keyword overlap check: texts 0 and 1 share banking/transaction/ledger terms; text 2 is unrelated
    sim_related = float(torch.dot(emb[0], emb[1]).item())
    sim_unrelated = float(torch.dot(emb[0], emb[2]).item())
    assert sim_related > sim_unrelated, f"Expected {sim_related} > {sim_unrelated}"


def test_all_baselines_execution(dataset: BenchmarkDataset, token_encoder: DeterministicTokenEncoder) -> None:
    """Verify all 5 baselines initialize, fit rules, and predict valid action vectors."""
    rules_subset = dataset.rules[:10]
    query = "Customer financial transaction record retention."

    baselines = [
        NoMemoryBaseline(),
        BoundedHistoryBaseline(capacity=3),
        PlainRAGBaseline(),
        TemporalStructuredBaseline(),
        HybridMemoryBaseline(seed=42),
    ]

    for b in baselines:
        b.fit(rules_subset, token_encoder)
        pred = b.predict(query)
        assert isinstance(pred, list)
        assert len(pred) == 2
        assert all(isinstance(x, float) for x in pred)


def test_evaluation_harness_single_run(dataset: BenchmarkDataset, token_encoder: DeterministicTokenEncoder) -> None:
    """Verify harness computes bounded metric values across test queries."""
    harness = EvaluationHarness(dataset, token_encoder)
    normal_rules = [r for r in dataset.rules if r.get("jurisdiction") != "ATTACK"]
    test_queries = [q for q in dataset.queries if q["scenario_type"] not in ("tenant_isolation", "poisoning_probe")]

    model = TemporalStructuredBaseline()
    metrics = harness.evaluate_model(model, normal_rules, test_queries)

    assert 0.0 <= metrics.overall_accuracy <= 1.0
    assert 0.0 <= metrics.override_accuracy <= 1.0
    assert 0.0 <= metrics.authority_correctness <= 1.0
    assert -1.0 <= metrics.forgetting_bwt <= 1.0
    assert 0.0 <= metrics.stale_recall_rate <= 1.0


def test_adversarial_evaluations(dataset: BenchmarkDataset, token_encoder: DeterministicTokenEncoder) -> None:
    """Verify adversarial poisoning and tenant leakage evaluations."""
    harness = EvaluationHarness(dataset, token_encoder)
    normal_rules = [r for r in dataset.rules if r.get("jurisdiction") != "ATTACK"]
    poison_rules = [r for r in dataset.rules if r.get("jurisdiction") == "ATTACK"]
    poison_queries = [q for q in dataset.queries if q["scenario_type"] == "poisoning_probe"]
    tenant_queries = [q for q in dataset.queries if q["scenario_type"] == "tenant_isolation"]

    # Poisoning ASR on PlainRAG vs TemporalStructured
    asr_rag = harness.evaluate_adversarial_poisoning(
        lambda: PlainRAGBaseline(), normal_rules, poison_rules, poison_queries
    )
    assert 0.0 <= asr_rag <= 1.0

    asr_structured = harness.evaluate_adversarial_poisoning(
        lambda: TemporalStructuredBaseline(), normal_rules, poison_rules, poison_queries
    )
    assert 0.0 <= asr_structured <= 1.0

    # Tenant leakage evaluation
    leakage = harness.evaluate_tenant_isolation(
        lambda: PlainRAGBaseline(), normal_rules, tenant_queries
    )
    assert 0.0 <= leakage <= 1.0


def test_full_benchmark_smoke(dataset: BenchmarkDataset, token_encoder: DeterministicTokenEncoder) -> None:
    """Smoke test running full benchmark over 2 seeds and verifying result dictionary."""
    harness = EvaluationHarness(dataset, token_encoder)
    results = harness.run_full_benchmark(num_seeds=2)

    assert "baselines" in results
    assert "NoMemory" in results["baselines"]
    assert "HybridMemory" in results["baselines"]
    assert "TemporalStructured" in results["baselines"]
    assert "ablations" in results
    assert "adversarial" in results
    assert "go_no_go" in results
    assert results["go_no_go"]["decision"] in ("GO", "NO-GO")

    md_report = harness.format_markdown_report(results)
    assert "# Synthetic Legal Memory Benchmark Report" in md_report
    assert "Pre-Registered Go/No-Go Recommendation" in md_report
