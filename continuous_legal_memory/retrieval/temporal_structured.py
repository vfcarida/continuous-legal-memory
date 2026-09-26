"""
Temporal Structured Precedence Retriever.

Production-grade default retrieval engine enforcing bi-temporal validity filtering,
hierarchical statutory authority tiers (lex superior), explicit supersession suppression,
and recency arbitration (lex posterior) without non-deterministic neural drift.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from continuous_legal_memory.domain.interfaces import BaseRetrieverPort
from continuous_legal_memory.domain.models import MemoryRecord
from continuous_legal_memory.retrieval.precedence import apply_legal_precedence


class TemporalStructuredRetriever(BaseRetrieverPort):
    """
    Deterministic, Temporal Structured Precedence Retriever.

    Rationale:
        Evaluated as the superior production engine (CLM-T08 NO-GO decision):
        Outperforms parametric neural adaptation (+9.3% accuracy) while guaranteeing
        zero catastrophic backward transfer and deterministic sub-5ms latency.
    """

    def __init__(self, temperature: float = 0.05) -> None:
        self.temperature = temperature

    def retrieve(
        self,
        query_embed: torch.Tensor,
        keys: torch.Tensor,
        rule_importance: torch.Tensor,
        temperature: float = 0.05,
        query_text: str | None = None,
        records: list[MemoryRecord] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, list[str] | None]:
        """
        Compute cosine similarity, apply legal precedence hierarchy, and return attention weights.

        Args:
            query_embed: 2D Tensor of shape (batch_size, embed_dim).
            keys: 2D Tensor of shape (num_keys, embed_dim).
            rule_importance: Tensor of shape (num_keys,) containing importance weights.
            temperature: Softmax scaling temperature.
            query_text: Optional query text string.
            records: Optional list of MemoryRecord objects for hierarchy and supersession.

        Returns:
            Tuple of (adjusted_scores, attention_weights, snippets).
        """
        _ = query_text
        temp = temperature or self.temperature

        query_norm = F.normalize(query_embed, p=2, dim=-1)
        keys_norm = F.normalize(keys, p=2, dim=-1)
        scores = torch.matmul(query_norm, keys_norm.T)

        adjusted_scores = apply_legal_precedence(scores, rule_importance, records)
        attention_weights = F.softmax(adjusted_scores / temp, dim=-1)

        snippets: list[str] | None = None
        if records:
            top_idx = int(torch.argmax(attention_weights, dim=-1).item())
            if top_idx < len(records):
                snippets = [records[top_idx].text]

        return adjusted_scores, attention_weights, snippets
