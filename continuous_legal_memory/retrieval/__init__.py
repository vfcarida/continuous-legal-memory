"""
Retrieval package.
"""

from continuous_legal_memory.domain.interfaces import BaseRetrieverPort, RetrieverPort
from continuous_legal_memory.retrieval.default_retriever import DefaultAttentionRetriever
from continuous_legal_memory.retrieval.hybrid_retriever import BM25Okapi, HybridLegalRetriever
from continuous_legal_memory.retrieval.precedence import PrecedenceConfig, apply_legal_precedence
from continuous_legal_memory.retrieval.temporal_structured import TemporalStructuredRetriever

__all__ = [
    "BaseRetrieverPort",
    "RetrieverPort",
    "DefaultAttentionRetriever",
    "HybridLegalRetriever",
    "TemporalStructuredRetriever",
    "PrecedenceConfig",
    "apply_legal_precedence",
    "BM25Okapi",
]

