"""
Hexagonal Application Service (Legal Memory Orchestrator).

This module serves as the primary entry point and orchestrator for the Continuous Legal Memory engine,
seamlessly combining abstract encoder ports, neural memory modules, and domain exceptions.
"""

import torch

from continuous_legal_memory.adapters.encoders import HuggingFaceEncoderAdapter
from continuous_legal_memory.core.hope_module import HopeModule
from continuous_legal_memory.domain.exceptions import (
    EncoderInferenceError,
    InvalidMemoryVectorError,
)
from continuous_legal_memory.domain.interfaces import BaseEncoderPort
from continuous_legal_memory.domain.models import PredictionResult


class LegalMemoryOrchestrator:
    """
    Main Hexagonal Application Orchestrator for Continuous Legal Memory operations.

    Rationale:
        Decouples user-facing agent workflows from internal PyTorch operations and specific model providers.
        Accepts any concrete implementation of `BaseEncoderPort` via dependency injection, guaranteeing clean
        architecture adherence and seamless offline/edge compatibility.
    """

    def __init__(
        self,
        encoder: BaseEncoderPort | None = None,
        value_dim: int = 2,
        temperature: float = 0.05,
    ) -> None:
        """
        Initialize the LegalMemoryOrchestrator.

        Args:
            encoder: Concrete instance of `BaseEncoderPort`. Defaults to `HuggingFaceEncoderAdapter`.
            value_dim: Target decision vector dimension.
            temperature: Softmax attention scaling temperature.
        """
        self.encoder = encoder or HuggingFaceEncoderAdapter()
        self.value_dim = value_dim
        self.hope_module = HopeModule(
            embed_dim=self.encoder.embedding_dim,
            value_dim=value_dim,
            temperature=temperature,
        )

    def update_memory(self, rule_text: str, action_vector: list[float]) -> None:
        """
        Update the model's active memory with a new legal directive without backpropagating through base model weights.

        Args:
            rule_text: Human-readable text string of the legal rule or statute.
            action_vector: List of floating-point values representing target decision outputs.

        Raises:
            InvalidMemoryVectorError: If input parameters violate dimension or text bounds.
            EncoderInferenceError: If semantic key embedding generation fails.
        """
        if not isinstance(rule_text, str) or len(rule_text.strip()) == 0:
            raise InvalidMemoryVectorError("Rule text must be a non-empty string.")

        if not isinstance(action_vector, list) or len(action_vector) != self.value_dim:
            raise InvalidMemoryVectorError(
                f"Action vector must be a list of floats with length equal to {self.value_dim}.",
                payload={"action_vector": action_vector, "expected_length": self.value_dim},
            )

        try:
            key_embed = self.encoder.get_embedding([rule_text])
        except Exception as e:
            if isinstance(e, EncoderInferenceError):
                raise
            raise EncoderInferenceError(f"Failed to generate embedding for rule text: {e}") from e

        val_tensor = torch.tensor([action_vector], dtype=torch.float32)
        self.hope_module.memory.add_memory(key_embed, val_tensor, rule_text)

    def predict(self, query_text: str) -> PredictionResult:
        """
        Predict decision vectors and retrieve explainable attention weights for a query string.

        Args:
            query_text: Legal query or case scenario text string.

        Returns:
            A populated `PredictionResult` domain object.

        Raises:
            InvalidMemoryVectorError: If query text is empty.
            EncoderInferenceError: If query embedding generation fails.
        """
        if not isinstance(query_text, str) or len(query_text.strip()) == 0:
            raise InvalidMemoryVectorError("Query text must be a non-empty string.")

        try:
            query_embed = self.encoder.get_embedding([query_text])
        except Exception as e:
            if isinstance(e, EncoderInferenceError):
                raise
            raise EncoderInferenceError(f"Failed to generate embedding for query text: {e}") from e

        predicted_action, attention_weights, fast_slow_gate = self.hope_module(query_embed)

        pred_action_list = predicted_action.squeeze().tolist()
        if isinstance(pred_action_list, float):
            pred_action_list = [pred_action_list]

        result = PredictionResult(
            query=query_text,
            predicted_action_vector=pred_action_list,
            fast_slow_gate=fast_slow_gate,
        )

        if attention_weights is not None:
            weights = attention_weights.squeeze().tolist()
            if isinstance(weights, float):
                weights = [weights]

            result.attention_weights = weights
            top_idx = int(torch.argmax(attention_weights, dim=-1).item())
            result.most_relevant_rule = self.hope_module.memory.texts[top_idx]
            result.confidence = weights[top_idx]

        return result
