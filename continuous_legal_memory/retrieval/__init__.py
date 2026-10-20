"""
Retrieval, Precedence, Choice-of-Law, and Conformal Calibration Subsystem.
"""

from continuous_legal_memory.domain.interfaces import (
    BaseRerankerPort,
    BaseRetrieverPort,
    RetrieverPort,
)
from continuous_legal_memory.retrieval.choice_of_law import (
    SUPPORTED_JURISDICTION_REGIMES,
    ChoiceOfLawResolution,
    ChoiceOfLawRouter,
    ConnectingFactors,
)
from continuous_legal_memory.retrieval.conformal import (
    ConformalPredictionOutput,
    SplitConformalCalibrator,
)
from continuous_legal_memory.retrieval.default_retriever import DefaultAttentionRetriever
from continuous_legal_memory.retrieval.hybrid_lexical import (
    HybridLexicalCitationScorer,
    compute_lexical_citation_scores,
    extract_legal_citations,
)
from continuous_legal_memory.retrieval.hybrid_retriever import BM25Okapi, HybridLegalRetriever
from continuous_legal_memory.retrieval.precedence import PrecedenceConfig, apply_legal_precedence
from continuous_legal_memory.retrieval.reranker import (
    DeonticCrossEncoderReranker,
    HuggingFaceCrossEncoderReranker,
)
from continuous_legal_memory.retrieval.temporal_structured import TemporalStructuredRetriever

__all__ = [
    "BM25Okapi",
    "BaseRerankerPort",
    "BaseRetrieverPort",
    "ChoiceOfLawResolution",
    "ChoiceOfLawRouter",
    "ConformalPredictionOutput",
    "ConnectingFactors",
    "DefaultAttentionRetriever",
    "DeonticCrossEncoderReranker",
    "HuggingFaceCrossEncoderReranker",
    "HybridLegalRetriever",
    "HybridLexicalCitationScorer",
    "PrecedenceConfig",
    "RetrieverPort",
    "SUPPORTED_JURISDICTION_REGIMES",
    "SplitConformalCalibrator",
    "TemporalStructuredRetriever",
    "apply_legal_precedence",
    "compute_lexical_citation_scores",
    "extract_legal_citations",
]
