"""
Default Attention Retriever.

Standard retrieval implementation using normalized cosine similarity scaled by surprise rule importance
and scaled-dot-product-style temperature softmax.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from continuous_legal_memory.domain.interfaces import BaseRetrieverPort
from continuous_legal_memory.domain.models import MemoryRecord
from continuous_legal_memory.retrieval.precedence import apply_legal_precedence


class DefaultAttentionRetriever(BaseRetrieverPort):
    """
    Default Associative Memory Retriever.

    Rationale:
        Preserves the standard associative memory retrieval mechanism:
        Computes cosine similarities between query embeddings and stored memory keys,
        scaled by surprise rule importance coefficients, hierarchical legal authority ranks,
        and temperature hyperparameter.
    """

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
        Compute normalized cosine similarity and temperature-scaled softmax attention weights.

        Args:
            query_embed: 2D PyTorch Tensor of shape (batch_size, embed_dim).
            keys: 2D PyTorch Tensor of shape (num_keys, embed_dim).
            rule_importance: 1D or 2D Tensor containing rule importance weights.
            temperature: Softmax scaling temperature for attention sharpness.
            query_text: Optional query text (unused in default vector-only retrieval).
            records: Optional list of MemoryRecord objects for authority and supersession resolution.

        Returns:
            Tuple containing:
            - scores: 2D Tensor of shape (batch_size, num_keys) unscaled cosine similarities.
            - attention_weights: 2D Tensor of shape (batch_size, num_keys) softmax distribution.
            - snippets: None (default retriever does not extract character snippets).
        """
        _ = query_text
        query_norm = F.normalize(query_embed, p=2, dim=-1)
        keys_norm = F.normalize(keys, p=2, dim=-1)
        scores = torch.matmul(query_norm, keys_norm.T)  # Shape: (batch, num_valid_keys)

        scaled_scores = apply_legal_precedence(scores, rule_importance, records)
        attention_weights = F.softmax(scaled_scores / temperature, dim=-1)

        return scores, attention_weights, None

