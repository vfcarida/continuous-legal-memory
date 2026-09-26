"""
Hierarchical Optimization & Retrieval Layer (Hope Module).

Combines direct associative episodic retrieval with a non-linear active neural prediction routing mechanism
to deliver explainable, high-precision legal memory querying.
"""


import torch
import torch.nn as nn

from continuous_legal_memory.core.continuous_memory import ContinuousMemory
from continuous_legal_memory.domain.interfaces import BaseRetrieverPort
from continuous_legal_memory.domain.models import MemoryRecord
from continuous_legal_memory.retrieval.default_retriever import DefaultAttentionRetriever


class HopeModule(nn.Module):
    """
    Hope Module attention and dynamic routing layer.

    Rationale:
        Pure vector retrieval can fail under complex legal precedence rules, while pure neural generation can hallucinate.
        The Hope Module synthesizes both paradigms:
        1. Softmax Attention Retrieval: Computes similarities between query embeddings and stored memory keys via
           pluggable BaseRetrieverPort, scaled by surprise rule importance coefficients and temperature hyperparameter.
        2. Dynamic Fast-Slow Gating: Measures query similarity against recent instructions. High similarity routes
           activation toward the fast network (short-term adaptation), whereas familiar or general queries rely on slow weights.
        3. Explainability Output: Returns attention weights mapping precisely to stored textual statutes for auditability.
    """

    def __init__(
        self,
        embed_dim: int,
        value_dim: int = 2,
        temperature: float = 0.05,
        hidden_dim: int = 64,
        retriever: BaseRetrieverPort | None = None,
        engine_mode: str = "structured",
    ) -> None:
        """
        Initialize the HopeModule.

        Args:
            embed_dim: Key vector embedding dimension.
            value_dim: Action decision vector output dimension.
            temperature: Softmax scaling temperature for attention sharpness.
            hidden_dim: Hidden dimension for internal memory MLPs.
            retriever: Optional BaseRetrieverPort instance. Defaults to TemporalStructuredRetriever
                       when engine_mode='structured', or DefaultAttentionRetriever when 'hybrid'.
            engine_mode: Execution mode ('structured' [default] or 'hybrid').
        """
        super().__init__()
        self.memory = ContinuousMemory(embed_dim, value_dim, hidden_dim)
        self.temperature = temperature
        self.engine_mode = engine_mode.lower()

        if retriever is not None:
            self.retriever: BaseRetrieverPort = retriever
        elif self.engine_mode == "structured":
            from continuous_legal_memory.retrieval.temporal_structured import (
                TemporalStructuredRetriever,
            )

            self.retriever = TemporalStructuredRetriever(temperature=temperature)
        else:
            self.retriever = DefaultAttentionRetriever()

        self.last_snippets: list[str] | None = None


    def forward(
        self,
        query_embed: torch.Tensor,
        valid_indices: list[int] | None = None,
        query_text: str | None = None,
        records: list[MemoryRecord] | None = None,
        return_snippets: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, float | None] | tuple[torch.Tensor, torch.Tensor | None, float | None, list[str] | None]:
        """
        Perform forward query evaluation over active memory networks and key buffers.

        Args:
            query_embed: 2D PyTorch Tensor of shape (batch_size, embed_dim).
            valid_indices: Optional list of integer indices restricting retrieval to temporally valid records.
            query_text: Optional query text string for hybrid keyword matching.
            records: Optional list of MemoryRecord objects corresponding to memory slots.
            return_snippets: Whether to include extracted snippets in returned tuple.

        Returns:
            Tuple containing:
            - Retrieved action vector tensor of shape (batch_size, value_dim).
            - Softmax attention weight tensor of shape (batch_size, num_evaluated_keys) or None if memory is empty.
            - Dynamic gating ratio scalar float (0.0 to 1.0) indicating fast vs. slow network balance, or None.
            - (Optional if return_snippets=True) list of extracted text snippet strings.
        """
        num_keys = self.memory.keys.size(0)
        device = query_embed.device

        if num_keys == 0 or (valid_indices is not None and len(valid_indices) == 0):
            self.last_snippets = None
            empty_val = torch.zeros(query_embed.size(0), self.memory.value_dim, device=device)
            if return_snippets:
                return empty_val, None, None, None
            return empty_val, None, None

        # Ensure active networks are set to evaluation mode
        self.memory.fast_net.eval()
        self.memory.slow_net.eval()

        with torch.no_grad():
            v_fast = self.memory.fast_net(query_embed)
            v_slow = self.memory.slow_net(query_embed)

        keys = self.memory.keys if valid_indices is None else self.memory.keys[valid_indices]
        values = self.memory.values if valid_indices is None else self.memory.values[valid_indices]
        rule_importance = (
            self.memory.rule_importance
            if valid_indices is None
            else self.memory.rule_importance[valid_indices]
        )

        # Delegate retrieval to pluggable BaseRetrieverPort
        scores, attention_weights, snippets = self.retriever.retrieve(
            query_embed=query_embed,
            keys=keys,
            rule_importance=rule_importance,
            temperature=self.temperature,
            query_text=query_text,
            records=records,
        )
        self.last_snippets = snippets

        # Direct attention-weighted retrieval from episodic action buffers
        v_retrieved = torch.matmul(attention_weights, values)

        # Dynamic Gating: Evaluate maximum similarity to determine fast vs slow routing
        max_sim = torch.max(scores, dim=-1)[0].unsqueeze(-1)
        gate = torch.clamp((max_sim - 0.2) / 0.6, min=0.0, max=1.0)

        # Combine fast short-term and slow long-term neural predictions
        v_net = gate * v_fast + (1.0 - gate) * v_slow

        # Check if authority hierarchy is present across candidate records
        has_authority_hierarchy = False
        if records:
            ranks = [getattr(r, "authority_rank", 1) for r in records]
            has_authority_hierarchy = max(ranks) > min(ranks)

        if self.engine_mode == "structured" or has_authority_hierarchy:
            # Deterministic statutory authority governance: structured precedence governs decision
            retrieved_values = v_retrieved
            gate_ratio = 0.0
        else:
            # Final non-linear synthesis of direct episodic retrieval and active neural memory prediction
            retrieved_values = 0.4 * v_retrieved + 0.6 * v_net
            gate_ratio = gate.squeeze().item() if gate.numel() == 1 else gate.mean().item()

        if return_snippets:
            return retrieved_values, attention_weights, gate_ratio, snippets

        return retrieved_values, attention_weights, gate_ratio


