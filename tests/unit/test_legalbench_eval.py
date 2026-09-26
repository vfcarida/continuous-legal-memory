"""
Unit tests for LegalBench-RAG evaluation benchmark suite.
"""

from continuous_legal_memory.domain.interfaces import BaseEncoderPort
from continuous_legal_memory.evaluation.legalbench_eval import (
    LegalBenchEvaluator,
    LegalBenchMetrics,
    LegalBenchRAGDataset,
)


def test_legalbench_dataset_integrity() -> None:
    """Verify LegalBench-RAG dataset clauses, character-level snippets, and query mappings."""
    dataset = LegalBenchRAGDataset()

    assert len(dataset.clauses) >= 5
    assert len(dataset.queries) >= 5

    clause_ids = {c.clause_id for c in dataset.clauses}
    for q in dataset.queries:
        assert q.target_clause_id in clause_ids
        assert len(q.query_text) > 10
        assert len(q.expected_action_vector) == 2

    for c in dataset.clauses:
        assert len(c.text) > 30
        assert len(c.ground_truth_snippets) > 0
        for s in c.ground_truth_snippets:
            assert s in c.text


def test_legalbench_evaluator_execution(offline_encoder: BaseEncoderPort) -> None:
    """Verify LegalBenchEvaluator execution, metrics calculation, and serialization."""
    evaluator = LegalBenchEvaluator(encoder=offline_encoder, tenant_id="test_legalbench_run")
    metrics = evaluator.run_evaluation()

    assert isinstance(metrics, LegalBenchMetrics)
    assert metrics.num_queries == len(evaluator.dataset.queries)
    assert 0.0 <= metrics.precision_at_1 <= 1.0
    assert 0.0 <= metrics.mean_reciprocal_rank <= 1.0
    assert 0.0 <= metrics.snippet_precision <= 1.0
    assert 0.0 <= metrics.snippet_recall <= 1.0
    assert -1.0 <= metrics.action_vector_cosine_sim <= 1.0

    # Ensure to_dict converts to JSON-serializable structure
    data_dict = metrics.to_dict()
    assert "precision_at_1" in data_dict
    assert "mean_reciprocal_rank" in data_dict
    assert "snippet_precision" in data_dict
    assert "snippet_recall" in data_dict
    assert data_dict["num_queries"] == len(evaluator.dataset.queries)


def test_legalbench_markdown_report_generation(offline_encoder: BaseEncoderPort) -> None:
    """Verify Markdown report generation with summary table and analysis."""
    evaluator = LegalBenchEvaluator(encoder=offline_encoder)
    metrics = LegalBenchMetrics(
        precision_at_1=1.0,
        mean_reciprocal_rank=1.0,
        snippet_precision=0.85,
        snippet_recall=0.95,
        action_vector_cosine_sim=0.92,
        num_queries=5,
    )
    report = evaluator.generate_markdown_report(metrics)

    assert "# LegalBench-RAG Evaluation Report" in report
    assert "Precision@1 (Governing Rule)" in report
    assert "Character Snippet Precision" in report
    assert "Character Snippet Recall" in report
    assert "Lex Superior Authority Hierarchy" in report


def test_legalbench_character_overlap_and_math() -> None:
    """Verify character overlap calculation and cosine similarity edge cases."""
    # Exact match
    prec, rec = LegalBenchEvaluator._calculate_character_overlap(
        retrieved_snippets=["liability shall not exceed $1,000,000"],
        ground_truth_spans=["liability shall not exceed $1,000,000"],
    )
    assert prec == 1.0
    assert rec == 1.0

    # Empty retrieved
    prec_empty, rec_empty = LegalBenchEvaluator._calculate_character_overlap(
        retrieved_snippets=[],
        ground_truth_spans=["governing law"],
    )
    assert prec_empty == 0.0
    assert rec_empty == 0.0

    # Empty ground truth
    prec_no_gt, rec_no_gt = LegalBenchEvaluator._calculate_character_overlap(
        retrieved_snippets=["some text"],
        ground_truth_spans=[],
    )
    assert prec_no_gt == 1.0
    assert rec_no_gt == 1.0

    # Cosine similarity
    sim_identical = LegalBenchEvaluator._calculate_cosine_similarity([1.0, 0.0], [1.0, 0.0])
    assert abs(sim_identical - 1.0) < 1e-5

    sim_orthogonal = LegalBenchEvaluator._calculate_cosine_similarity([1.0, 0.0], [0.0, 1.0])
    assert abs(sim_orthogonal - 0.0) < 1e-5

    sim_empty = LegalBenchEvaluator._calculate_cosine_similarity([], [])
    assert sim_empty == 0.0
