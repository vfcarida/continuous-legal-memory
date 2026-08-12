"""
Active Memory Neural Network Component.

This module provides the multi-layer perceptron (MLP) mapping high-dimensional key embeddings
to low-dimensional decision vectors within the Continuum Memory System (CMS).
"""

import torch
import torch.nn as nn


class MemoryMLP(nn.Module):
    """
    Active Memory Network mapping key embeddings to target action decision vectors.

    Rationale:
        Employs LayerNorm and GELU non-linear activations to ensure stable parameter updates
        during fast-loop online gradient steps at inference time. LayerNorm prevents internal
        activation drift when learning conflicting legal directives incrementally.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 2) -> None:
        """
        Initialize the MemoryMLP architecture.

        Args:
            input_dim: Vector dimension of the incoming key embedding.
            hidden_dim: Number of hidden units in the intermediate linear layer.
            output_dim: Dimension of target action decision vector.
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform forward pass mapping key embedding vector to decision vector.

        Args:
            x: PyTorch Tensor of shape (batch_size, input_dim).

        Returns:
            PyTorch Tensor of shape (batch_size, output_dim).
        """
        return self.net(x)
