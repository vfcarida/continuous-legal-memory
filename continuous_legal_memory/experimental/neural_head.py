"""
Experimental Parametric Neural Memory Head (Research Module).

RESEARCH NOTICE & EMPIRICAL DISCLAIMER (CLM-T08 NO-GO):
    The online-adapted parametric neural head (`fast_net`/`slow_net`) is maintained strictly
    for academic replication, ablation studies, and comparative research benchmarks.

    As documented in `docs/evaluation_report.md`:
    Empirical evaluations across 20 random seeds against a 100-item legal conflict benchmark
    demonstrated that deterministic temporal structured retrieval (85.3% accuracy) strictly
    outperformed the hybrid parametric head (76.0% accuracy), while completely eliminating
    catastrophic forgetting (+0.110 BWT) and parameter drift.

    For production legal applications, use `engine_mode='structured'` (the default in `LegalMemoryOrchestrator`).
"""

from __future__ import annotations

import warnings

from continuous_legal_memory.core.continuous_memory import ContinuousMemory


class ParametricNeuralHeadWarning(UserWarning):
    """Warning emitted when instantiating the experimental online parametric neural head."""


class ContinuousMemoryResearchHead(ContinuousMemory):
    """
    Experimental dual-timescale online parametric neural memory head (`fast_net` / `slow_net`).

    Maintained for research and benchmark replication. Emits `ParametricNeuralHeadWarning`
    upon instantiation to alert developers of production limitations.
    """

    def __init__(self, embed_dim: int, value_dim: int = 2, hidden_dim: int = 64) -> None:
        warnings.warn(
            "ContinuousMemoryResearchHead is an experimental research component (CLM-T08 NO-GO). "
            "For production deployments, use deterministic structured retrieval (engine_mode='structured').",
            category=ParametricNeuralHeadWarning,
            stacklevel=2,
        )
        super().__init__(embed_dim, value_dim, hidden_dim)
