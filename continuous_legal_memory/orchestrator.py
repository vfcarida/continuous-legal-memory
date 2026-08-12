"""
Hexagonal Application Service (Legal Memory Orchestrator).

Serves as the central entry point for the Multi-Tier Continuous Legal Memory engine, coordinating
Working Memory, Episodic Ledger, Semantic Knowledge Graph, and neural continuum adaptation networks.
"""

from datetime import datetime, timezone

import torch

from continuous_legal_memory.adapters.encoders import HuggingFaceEncoderAdapter
from continuous_legal_memory.adapters.ollama import OllamaGemmaAdapter
from continuous_legal_memory.core.hope_module import HopeModule
from continuous_legal_memory.core.memory_tiers.episodic_memory import EpisodicMemory
from continuous_legal_memory.core.memory_tiers.semantic_memory import SemanticKnowledgeGraph
from continuous_legal_memory.core.memory_tiers.working_memory import WorkingMemory
from continuous_legal_memory.domain.exceptions import (
    EncoderInferenceError,
    InvalidMemoryVectorError,
)
from continuous_legal_memory.domain.interfaces import BaseEncoderPort
from continuous_legal_memory.domain.models import MemoryTier, PredictionResult


class LegalMemoryOrchestrator:
    """
    Main Hexagonal Application Orchestrator for Continuous Legal Memory operations.

    Rationale:
        Synthesizes multi-tiered cognitive memory (Working, Episodic, Semantic) with neural associative
        learning (`HopeModule` / `ContinuousMemory`). Supports strict privacy mode for local edge execution.
    """

    def __init__(
        self,
        encoder: BaseEncoderPort | None = None,
        value_dim: int = 2,
        temperature: float = 0.05,
        strict_privacy_mode: bool = False,
    ) -> None:
        """
        Initialize the LegalMemoryOrchestrator.

        Args:
            encoder: Instance of `BaseEncoderPort`. If None, defaults to `HuggingFaceEncoderAdapter`
                     or `OllamaGemmaAdapter` when `strict_privacy_mode` is True.
            value_dim: Target decision vector dimension.
            temperature: Softmax attention scaling temperature.
            strict_privacy_mode: Enforces local offline execution.
        """
        self.strict_privacy_mode = strict_privacy_mode

        if encoder is not None:
            self.encoder = encoder
        elif self.strict_privacy_mode:
            self.encoder = OllamaGemmaAdapter(strict_privacy_mode=True)
        else:
            self.encoder = HuggingFaceEncoderAdapter()

        self.value_dim = value_dim
        self.hope_module = HopeModule(
            embed_dim=self.encoder.embedding_dim,
            value_dim=value_dim,
            temperature=temperature,
        )

        # Multi-Tier Cognitive Memory System
        self.working_memory = WorkingMemory(capacity=10)
        self.episodic_memory = EpisodicMemory()
        self.semantic_graph = SemanticKnowledgeGraph()

    def update_memory(
        self,
        rule_text: str,
        action_vector: list[float],
        valid_from: datetime | None = None,
        valid_to: datetime | None = None,
        metadata: dict | None = None,
    ) -> None:
        """
        Ingest a new legal directive into active neural memory and multi-tier ledgers.

        Args:
            rule_text: Human-readable text string of the legal rule or statute.
            action_vector: Target decision outputs list.
            valid_from: Start timestamp of legal validity.
            valid_to: Expiry or temporal invalidation timestamp.
            metadata: Custom audit metadata dictionary.

        Raises:
            InvalidMemoryVectorError: If input parameters violate dimension or text bounds.
            EncoderInferenceError: If key embedding generation fails.
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

        # 1. Adapt neural associative continuum memory
        self.hope_module.memory.add_memory(key_embed, val_tensor, rule_text)

        # 2. Record in Episodic Memory ledger
        self.episodic_memory.append(
            text=rule_text,
            key_vector=key_embed,
            value_vector=val_tensor,
            valid_from=valid_from,
            valid_to=valid_to,
            metadata=metadata,
        )

        # 3. Add to Working Memory
        self.working_memory.add(
            text=rule_text,
            key_vector=key_embed,
            value_vector=val_tensor,
            metadata=metadata,
        )

    def predict(self, query_text: str, at_time: datetime | None = None) -> PredictionResult:
        """
        Predict decision vectors across multi-tier memory networks for a query string.

        Args:
            query_text: Legal query or case scenario text string.
            at_time: Datetime timestamp to evaluate temporal validity.

        Returns:
            A populated `PredictionResult` domain object.

        Raises:
            InvalidMemoryVectorError: If query text is empty.
            EncoderInferenceError: If query embedding generation fails.
        """
        if not isinstance(query_text, str) or len(query_text.strip()) == 0:
            raise InvalidMemoryVectorError("Query text must be a non-empty string.")

        eval_time = at_time or datetime.now(timezone.utc)
        _ = self.episodic_memory.get_valid_records(eval_time)

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
            source_tier=MemoryTier.SEMANTIC if self.semantic_graph.nodes else MemoryTier.EPISODIC,
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
