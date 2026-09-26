"""
Evaluation package.
"""

from continuous_legal_memory.evaluation.agentic_eval import AgenticLegalEvaluator
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
from continuous_legal_memory.evaluation.legalbench_eval import (
    LegalBenchClause,
    LegalBenchEvaluator,
    LegalBenchMetrics,
    LegalBenchQuery,
    LegalBenchRAGDataset,
)

__all__ = [
    "AgenticLegalEvaluator",
    "BenchmarkDataset",
    "BoundedHistoryBaseline",
    "DeterministicTokenEncoder",
    "EvaluationHarness",
    "HybridMemoryBaseline",
    "NoMemoryBaseline",
    "PlainRAGBaseline",
    "TemporalStructuredBaseline",
    "LegalBenchClause",
    "LegalBenchEvaluator",
    "LegalBenchMetrics",
    "LegalBenchQuery",
    "LegalBenchRAGDataset",
]
