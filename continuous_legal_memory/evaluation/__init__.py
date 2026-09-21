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
]
