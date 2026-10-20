"""
LangChain Memory Adapter for Continuous Legal Memory.

Provides drop-in integration with LangChain chains, agents, and conversational runtimes,
enabling LangChain agents to query deterministic statutory precedence, enforce multi-tenant
boundaries, inspect defeasible proof traces and Sparse Merkle state roots, respect conformal
abstention guardrails, and retain verified legal context states.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import torch

from continuous_legal_memory.domain.models import PredictionResult
from continuous_legal_memory.orchestrator import LegalMemoryOrchestrator

# Optional import of LangChain BaseMemory
try:
    from langchain_core.memory import BaseMemory as _LangChainBaseMemory
except ImportError:
    try:
        from langchain.schema import BaseMemory as _LangChainBaseMemory
    except ImportError:
        _LangChainBaseMemory = object


class ContinuousLegalMemoryLangChain(_LangChainBaseMemory):
    """
    LangChain-compatible memory component powered by Continuous Legal Memory.

    Attributes:
        orchestrator: Configured LegalMemoryOrchestrator instance.
        memory_key: Output dictionary key containing retrieved legal context.
        input_key: Input dictionary key containing user prompt or legal query.
        output_key: Output dictionary key containing agent legal response.
        tenant_id: Active tenant identifier for multi-tenant boundary isolation.
        return_messages: Whether to return LangChain BaseMessage objects or raw strings.
        include_proof_trace: Whether to append the winning defeasible canon to context.
        include_merkle_root: Whether to append the 256-bit Sparse Merkle state root to context.
        jurisdiction: Optional default jurisdiction filter.
        connecting_factors: Optional Rome I / Rome II / GDPR connecting factors dictionary.
    """

    def __init__(
        self,
        orchestrator: LegalMemoryOrchestrator,
        memory_key: str = "legal_context",
        input_key: str = "input",
        output_key: str = "output",
        tenant_id: str = "default",
        return_messages: bool = False,
        include_proof_trace: bool = False,
        include_merkle_root: bool = False,
        jurisdiction: str | list[str] | None = None,
        connecting_factors: dict[str, Any] | None = None,
    ) -> None:
        self.orchestrator = orchestrator
        self.memory_key = memory_key
        self.input_key = input_key
        self.output_key = output_key
        self.tenant_id = tenant_id
        self.return_messages = return_messages
        self.include_proof_trace = include_proof_trace
        self.include_merkle_root = include_merkle_root
        self.jurisdiction = jurisdiction
        self.connecting_factors = connecting_factors
        self._last_prediction: PredictionResult | None = None

    @property
    def memory_variables(self) -> list[str]:
        """Return the string keys injected by this memory module into prompt inputs."""
        return [self.memory_key]

    def get_last_prediction(self) -> PredictionResult | None:
        """Return the full `PredictionResult` from the most recent `load_memory_variables` call."""
        return self._last_prediction

    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """
        Retrieve governing legal rules and precedent for the incoming prompt.

        Args:
            inputs: Dictionary containing conversation/chain inputs and optional
                    `at_time`, `as_of_transaction_time`, `jurisdiction`, or `connecting_factors`.

        Returns:
            Dictionary mapping `self.memory_key` to retrieved legal context string.
        """
        query_str = ""
        if self.input_key in inputs and isinstance(inputs[self.input_key], str):
            query_str = inputs[self.input_key]
        else:
            for val in inputs.values():
                if isinstance(val, str):
                    query_str = val
                    break

        if not query_str.strip():
            return {self.memory_key: "" if not self.return_messages else []}

        at_time: datetime | None = inputs.get("at_time")
        as_of_tx: datetime | None = inputs.get("as_of_transaction_time")
        juris = inputs.get("jurisdiction", self.jurisdiction)
        factors = inputs.get("connecting_factors", self.connecting_factors)

        with self.orchestrator.tenant(self.tenant_id):
            result = self.orchestrator.predict(
                query_str,
                at_time=at_time,
                as_of_transaction_time=as_of_tx,
                jurisdiction=juris,
                connecting_factors=factors,
            )
        self._last_prediction = result

        if result.abstained:
            abstention_text = (
                f"[EPISTEMIC ABSTENTION] Reason: {result.abstention_reason or 'uncertainty_threshold'}"
            )
            if self.include_merkle_root and result.merkle_root:
                abstention_text += f"\nMerkle State Root: {result.merkle_root}"
            return {self.memory_key: abstention_text if not self.return_messages else [abstention_text]}

        if not result.most_relevant_rule:
            return {self.memory_key: "" if not self.return_messages else []}

        context_lines = [f"Governing Legal Norm: {result.most_relevant_rule}"]
        if result.retrieved_snippets:
            for snip in result.retrieved_snippets:
                context_lines.append(f"Statutory Snippet: {snip}")
        if result.predicted_action_vector:
            context_lines.append(f"Recommended Decision Vector: {result.predicted_action_vector}")
        if self.include_proof_trace and result.proof_trace:
            canon = result.proof_trace.get("winning_canon", "N/A")
            context_lines.append(f"Defeasible Winning Canon: {canon}")
        if self.include_merkle_root and result.merkle_root:
            context_lines.append(f"Merkle State Root: {result.merkle_root}")

        context_text = "\n".join(context_lines)

        if self.return_messages:
            try:
                from langchain_core.messages import SystemMessage

                return {self.memory_key: [SystemMessage(content=context_text)]}
            except ImportError:
                pass

        return {self.memory_key: context_text}

    def save_context(self, inputs: dict[str, Any], outputs: dict[str, Any]) -> None:
        """
        Record interaction into the orchestrator's tenant-scoped working memory.

        Args:
            inputs: Input dictionary passed to the chain.
            outputs: Output dictionary generated by the chain/agent.
        """
        user_input = inputs.get(self.input_key, "")
        agent_output = outputs.get(self.output_key, "")

        if user_input:
            key_embed = self.orchestrator.encoder.get_embedding([user_input])
            val_tensor = torch.zeros(1, self.orchestrator.value_dim)
            self.orchestrator.working_memory.add(
                f"User: {user_input}",
                key_vector=key_embed,
                value_vector=val_tensor,
                tenant_id=self.tenant_id,
            )
        if agent_output:
            key_embed = self.orchestrator.encoder.get_embedding([agent_output])
            val_tensor = torch.zeros(1, self.orchestrator.value_dim)
            self.orchestrator.working_memory.add(
                f"Agent: {agent_output}",
                key_vector=key_embed,
                value_vector=val_tensor,
                tenant_id=self.tenant_id,
            )

    def clear(self) -> None:
        """Clear active working memory context for the current tenant."""
        self.orchestrator.working_memory.clear(tenant_id=self.tenant_id)
        self._last_prediction = None
