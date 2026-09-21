"""
Continuum Memory System (CMS) Core Module.

Implements fast-slow dual-timescale associative neural adaptation, surprise-driven priority scaling,
and experience replay regularization to prevent catastrophic forgetting during online legal rule ingestion.
Note: The base text encoder remains frozen (zero-backpropagation on the transformer). The small memory
head MLP (fast_net) is adapted online via local gradient descent with experience replay and EMA
consolidation into slow_net, combined with attention retrieval over an external key/value buffer.
"""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from continuous_legal_memory.core.memory_mlp import MemoryMLP
from continuous_legal_memory.domain.exceptions import InvalidMemoryVectorError


class ContinuousMemory(nn.Module):
    """
    Continuum Memory System (CMS) dual-timescale neural memory core.

    Rationale:
        Traditional RAG buffers rely on static vector databases without parameter adaptation.
        CMS introduces two coupled networks operating on frozen encoder representations:
        1. Fast Network (`fast_net`): Rapidly adapts a lightweight memory MLP via inner-loop local gradient descent
           to immediately integrate overriding legal rules or recent case precedents.
        2. Slow Network (`slow_net`): Consolidates long-term knowledge via outer-loop Exponential Moving Average (EMA),
           preserving foundational legal principles and guarding against catastrophic forgetting.
    """

    def __init__(self, embed_dim: int, value_dim: int = 2, hidden_dim: int = 64) -> None:
        """
        Initialize the ContinuousMemory system.

        Args:
            embed_dim: Dimension of key embeddings produced by the encoder.
            value_dim: Dimension of target action/decision output vectors.
            hidden_dim: Number of hidden units in active memory MLPs.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.value_dim = value_dim
        self.hidden_dim = hidden_dim

        # Dual-timescale active memory modules
        self.fast_net = MemoryMLP(embed_dim, hidden_dim, value_dim)
        self.slow_net = MemoryMLP(embed_dim, hidden_dim, value_dim)

        # Synchronize slow network to match initial fast network weights
        self.slow_net.load_state_dict(self.fast_net.state_dict())

        # Memory buffers to store raw key/value pairs and importance coefficients
        self.register_buffer("keys", torch.empty(0, embed_dim))
        self.register_buffer("values", torch.empty(0, value_dim))
        self.register_buffer("rule_importance", torch.empty(0))
        self.texts: list[str] = []

        # Surprise tracking momentum buffer for dynamic learning rate adjustments
        self.register_buffer("surprise_momentum", torch.zeros(1))

    def add_memory(
        self,
        key_vector: torch.Tensor,
        value_vector: torch.Tensor,
        text: str,
        skip_parametric: bool = False,
        authority_rank: int = 1,
    ) -> None:
        """
        Dynamically ingest a new legal directive into active memory.

        Algorithmic Workflow:
            1. Input validation enforcing shape and numerical integrity.
            2. Compute prediction surprise via the consolidated slow network BEFORE buffer insertion.
            3. Update surprise momentum tracker and calculate rule importance scaling (contradictory rules get up to 4x weight).
            4. If skip_parametric is True (e.g. personal data), append to retrieval buffers and return immediately.
            5. Execute Inner Fast-Loop Optimization: Run gradient descent on `fast_net` balancing new rule adaptation,
               weighted experience replay over previously stored rules, and proximal regularization towards slow weights.
            6. Execute Outer Slow-Loop Consolidation: Update `slow_net` parameters using EMA.

        Args:
            key_vector: Key embedding tensor of shape (1, embed_dim).
            value_vector: Action value target tensor of shape (1, value_dim).
            text: Human-readable legal text string describing the rule.
            skip_parametric: If True, confines rule to retrieval buffer and bypasses neural weight updates.
            authority_rank: Hierarchical authority rank of the rule.

        Raises:
            InvalidMemoryVectorError: If input text is empty or tensor dimensions do not match expected bounds.
        """
        _ = authority_rank
        if not isinstance(text, str) or len(text.strip()) == 0:
            raise InvalidMemoryVectorError("Rule text must be a non-empty string.")

        if key_vector.ndim != 2 or key_vector.size(1) != self.embed_dim:
            raise InvalidMemoryVectorError(
                f"Key vector must be 2D with last dimension equal to {self.embed_dim}, got shape {tuple(key_vector.shape)}.",
                payload={"key_vector_shape": list(key_vector.shape), "expected_dim": self.embed_dim},
            )

        if value_vector.ndim != 2 or value_vector.size(1) != self.value_dim:
            raise InvalidMemoryVectorError(
                f"Value vector must be 2D with last dimension equal to {self.value_dim}, got shape {tuple(value_vector.shape)}.",
                payload={"value_vector_shape": list(value_vector.shape), "expected_dim": self.value_dim},
            )

        # Set evaluation mode to evaluate surprise signal
        self.fast_net.eval()
        self.slow_net.eval()

        surprise = 0.0
        if self.keys.size(0) > 0:
            with torch.no_grad():
                pred_slow = self.slow_net(key_vector)
                surprise = F.mse_loss(pred_slow, value_vector).item()
                # Exponential moving average update for surprise momentum tracking
                self.surprise_momentum = 0.9 * self.surprise_momentum + 0.1 * surprise

        # Append new tensors to memory buffers
        self.keys = torch.cat([self.keys, key_vector], dim=0)
        self.values = torch.cat([self.values, value_vector], dim=0)
        self.texts.append(text)

        # Rule importance weighting: surprise amplifies memory replay weight up to 4x
        importance_val = 1.0 + 3.0 * surprise
        importance_tensor = torch.tensor([importance_val], device=key_vector.device)
        self.rule_importance = torch.cat([self.rule_importance, importance_tensor], dim=0)

        # If flagged to skip parametric absorption (e.g. personal data), stop here
        if skip_parametric:
            return

        # Dynamically scale inner-loop learning rate and epoch count based on surprise score
        base_lr = 0.01
        lr = base_lr * (1.0 + min(2.0, surprise * 1.5))
        epochs = int(35 * (1.0 + min(1.0, surprise)))

        optimizer = torch.optim.Adam(self.fast_net.parameters(), lr=lr, weight_decay=1e-4)

        self.fast_net.train()
        num_stored = self.keys.size(0)

        # Inner Fast Loop Optimization Step
        for _ in range(epochs):
            optimizer.zero_grad()

            # 1. Immediate adaptation term for new rule
            pred_new = self.fast_net(key_vector)
            loss_new = F.mse_loss(pred_new, value_vector) * self.rule_importance[-1]

            # 2. Weighted experience replay over past legal directives
            loss_replay = torch.tensor(0.0, device=key_vector.device)
            if num_stored > 1:
                prev_keys = self.keys[:-1]
                prev_values = self.values[:-1]
                pred_prev = self.fast_net(prev_keys)
                raw_errors = (pred_prev - prev_values) ** 2
                weighted_errors = raw_errors * self.rule_importance[:-1].unsqueeze(-1)
                loss_replay = weighted_errors.mean()

            # 3. Proximal regularization to consolidated slow network weights
            loss_reg = torch.tensor(0.0, device=key_vector.device)
            for p_fast, p_slow in zip(self.fast_net.parameters(), self.slow_net.parameters()):
                loss_reg += F.mse_loss(p_fast, p_slow)

            # Combined multi-objective optimization loss
            total_loss = loss_new + 0.7 * loss_replay + 0.15 * loss_reg
            total_loss.backward()
            optimizer.step()

        self.fast_net.eval()

        # Outer Slow Loop Consolidation Step (EMA weight transfer)
        tau = 0.15  # Consolidation rate
        with torch.no_grad():
            for p_fast, p_slow in zip(self.fast_net.parameters(), self.slow_net.parameters()):
                p_slow.copy_(p_slow + tau * (p_fast - p_slow))

    def delete_buffer_index(self, index: int) -> None:
        """
        Remove a memory entry at the specified index from the retrieval buffers.

        Args:
            index: Integer buffer index to purge.

        Raises:
            IndexError: If index is out of bounds.
        """
        if index < 0 or index >= self.keys.size(0):
            raise IndexError(f"Buffer index {index} out of range [0, {self.keys.size(0)}).")

        self.keys = torch.cat([self.keys[:index], self.keys[index + 1:]], dim=0)
        self.values = torch.cat([self.values[:index], self.values[index + 1:]], dim=0)
        self.rule_importance = torch.cat([self.rule_importance[:index], self.rule_importance[index + 1:]], dim=0)
        self.texts.pop(index)

    def rebuild_from_records(self, records: list[Any], seed: int | None = None) -> None:
        """
        Reconstruct neural memory buffers and retrain parametric weights from a clean list of records.

        Args:
            records: List of MemoryRecord instances to consolidate.
            seed: Optional random seed for reproducible network reinitialization.
        """
        if seed is not None:
            torch.manual_seed(seed)

        self.fast_net = MemoryMLP(self.embed_dim, self.hidden_dim, self.value_dim)
        self.slow_net = MemoryMLP(self.embed_dim, self.hidden_dim, self.value_dim)
        self.slow_net.load_state_dict(self.fast_net.state_dict())

        self.keys = torch.empty(0, self.embed_dim, device=self.keys.device)
        self.values = torch.empty(0, self.value_dim, device=self.values.device)
        self.rule_importance = torch.empty(0, device=self.rule_importance.device)
        self.texts = []
        self.surprise_momentum = torch.zeros(1, device=self.surprise_momentum.device)

        for rec in records:
            self.add_memory(
                rec.key_vector,
                rec.value_vector,
                rec.text,
                skip_parametric=rec.personal_data,
                authority_rank=rec.authority_rank,
            )

