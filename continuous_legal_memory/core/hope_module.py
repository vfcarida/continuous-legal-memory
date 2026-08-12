"""
Hierarchical Optimization & Retrieval Layer (Hope Module).

Combines direct associative episodic retrieval with a non-linear active neural prediction routing mechanism
to deliver explainable, high-precision legal memory querying.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

from continuous_legal_memory.core.continuous_memory import ContinuousMemory


class HopeModule(nn.Module):
    """
    Hope Module attention and dynamic routing layer.

    Rationale:
        Pure vector retrieval can fail under complex legal precedence rules, while pure neural generation can hallucinate.
        The Hope Module synthesizes both paradigms:
        1. Softmax Attention Retrieval: Computes cosine similarities between query embeddings and stored memory keys,
           scaled by surprise rule importance coefficients and temperature hyperparameter.
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
    ) -> None:
        """
        Initialize the HopeModule.

        Args:
            embed_dim: Key vector embedding dimension.
            value_dim: Action decision vector output dimension.
            temperature: Softmax scaling temperature for attention sharpness.
            hidden_dim: Hidden dimension for internal memory MLPs.
        """
        super().__init__()
        self.memory = ContinuousMemory(embed_dim, value_dim, hidden_dim)
        self.temperature = temperature

    def forward(self, query_embed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None, float | None]:
        """
        Perform forward query evaluation over active memory networks and key buffers.

        Args:
            query_embed: 2D PyTorch Tensor of shape (batch_size, embed_dim).

        Returns:
            Tuple containing:
            - Retrieved action vector tensor of shape (batch_size, value_dim).
            - Softmax attention weight tensor of shape (batch_size, num_stored_keys) or None if memory is empty.
            - Dynamic gating ratio scalar float (0.0 to 1.0) indicating fast vs. slow network balance, or None.
        """
        num_keys = self.memory.keys.size(0)
        device = query_embed.device

        if num_keys == 0:
            return torch.zeros(query_embed.size(0), self.memory.value_dim, device=device), None, None

        # Ensure active networks are set to evaluation mode
        self.memory.fast_net.eval()
        self.memory.slow_net.eval()

        with torch.no_grad():
            v_fast = self.memory.fast_net(query_embed)
            v_slow = self.memory.slow_net(query_embed)

        # Calculate normalized semantic cosine similarities
        query_norm = F.normalize(query_embed, p=2, dim=-1)
        keys_norm = F.normalize(self.memory.keys, p=2, dim=-1)
        scores = torch.matmul(query_norm, keys_norm.T)  # Shape: (batch, num_keys)

        # Scale semantic scores by rule importance coefficients to resolve legal precedence
        scaled_scores = scores * self.memory.rule_importance

        # Compute Softmax attention distribution for explainable retrieval
        attention_weights = F.softmax(scaled_scores / self.temperature, dim=-1)

        # Direct attention-weighted retrieval from episodic action buffers
        v_retrieved = torch.matmul(attention_weights, self.memory.values)

        # Dynamic Gating: Evaluate maximum similarity to determine fast vs slow routing
        max_sim = torch.max(scores, dim=-1)[0].unsqueeze(-1)
        gate = torch.clamp((max_sim - 0.2) / 0.6, min=0.0, max=1.0)

        # Combine fast short-term and slow long-term neural predictions
        v_net = gate * v_fast + (1.0 - gate) * v_slow

        # Final non-linear synthesis of direct episodic retrieval and active neural memory prediction
        retrieved_values = 0.4 * v_retrieved + 0.6 * v_net
        gate_ratio = gate.squeeze().item() if gate.numel() == 1 else gate.mean().item()

        return retrieved_values, attention_weights, gate_ratio
