"""
LlamaIndex Retriever Adapter for Continuous Legal Memory.

Provides drop-in integration with LlamaIndex query engines, RAG pipelines, and agent tools,
enabling LlamaIndex applications to retrieve governing statutory norms with character-level
snippets, defeasible proof traces, Sparse Merkle state roots, conformal prediction sets,
and bi-temporal / choice-of-law filtering.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from continuous_legal_memory.orchestrator import LegalMemoryOrchestrator

# Optional import of LlamaIndex BaseRetriever
try:
    from llama_index.core.base.base_retriever import BaseRetriever as _LlamaIndexBaseRetriever
    from llama_index.core.schema import NodeWithScore, TextNode

    _HAS_LLAMA_INDEX = True
except ImportError:
    _LlamaIndexBaseRetriever = object
    NodeWithScore = None
    TextNode = None
    _HAS_LLAMA_INDEX = False


class LegalNodeResult:
    """Lightweight result node representing a retrieved legal rule and its precedence score."""

    def __init__(
        self,
        text: str,
        score: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.text = text
        self.score = score
        self.metadata = metadata or {}

    def __repr__(self) -> str:
        return f"LegalNodeResult(text={self.text[:30]!r}..., score={self.score:.4f})"


class ContinuousLegalMemoryLlamaRetriever(_LlamaIndexBaseRetriever):
    """
    LlamaIndex-compatible retriever backed by Continuous Legal Memory.

    Attributes:
        orchestrator: Configured LegalMemoryOrchestrator instance.
        tenant_id: Active tenant identifier for multi-tenant boundary isolation.
        top_k: Maximum number of retrieved candidate legal provisions.
        include_conformal_set: If True and conformal calibration is active, emits additional
                               nodes for rules in the conformal prediction set.
    """

    def __init__(
        self,
        orchestrator: LegalMemoryOrchestrator,
        tenant_id: str = "default",
        top_k: int = 5,
        include_conformal_set: bool = False,
        jurisdiction: str | list[str] | None = None,
        connecting_factors: dict[str, Any] | None = None,
    ) -> None:
        self.orchestrator = orchestrator
        self.tenant_id = tenant_id
        self.top_k = top_k
        self.include_conformal_set = include_conformal_set
        self.jurisdiction = jurisdiction
        self.connecting_factors = connecting_factors

    def _retrieve(self, query_bundle: Any) -> list[Any]:
        """
        Internal retrieval logic compatible with LlamaIndex BaseRetriever.

        Args:
            query_bundle: LlamaIndex QueryBundle or raw string.

        Returns:
            List of NodeWithScore (or LegalNodeResult) objects.
        """
        query_str = getattr(query_bundle, "query_str", str(query_bundle))
        return self.retrieve(query_str)

    def retrieve(
        self,
        str_or_query_bundle: Any,
        at_time: datetime | None = None,
        as_of_transaction_time: datetime | None = None,
        jurisdiction: str | list[str] | None = None,
        connecting_factors: dict[str, Any] | None = None,
    ) -> list[Any]:
        """
        Execute statutory retrieval over Continuous Legal Memory.

        Args:
            str_or_query_bundle: Natural language legal query string or QueryBundle.
            at_time: Optional valid-time evaluation timestamp.
            as_of_transaction_time: Optional transaction-time cutoff for bi-temporal replay.
            jurisdiction: Optional jurisdiction override.
            connecting_factors: Optional Rome I / Rome II / GDPR connecting factors.

        Returns:
            List of retrieved nodes sorted by precedence score.
        """
        query_str = getattr(str_or_query_bundle, "query_str", str(str_or_query_bundle))
        eff_juris = jurisdiction if jurisdiction is not None else self.jurisdiction
        eff_factors = connecting_factors if connecting_factors is not None else self.connecting_factors

        with self.orchestrator.tenant(self.tenant_id):
            result = self.orchestrator.predict(
                query_str,
                at_time=at_time,
                as_of_transaction_time=as_of_transaction_time,
                jurisdiction=eff_juris,
                connecting_factors=eff_factors,
            )

        results: list[Any] = []
        if result.most_relevant_rule and not result.abstained:
            meta: dict[str, Any] = {
                "tenant_id": self.tenant_id,
                "predicted_action_vector": result.predicted_action_vector,
                "fast_slow_gate": result.fast_slow_gate,
                "abstained": result.abstained,
                "abstention_reason": result.abstention_reason,
                "merkle_root": result.merkle_root,
                "proof_trace": result.proof_trace,
                "conformal_set": result.conformal_set,
                "choice_of_law": result.choice_of_law,
            }
            if result.retrieved_snippets:
                meta["retrieved_snippets"] = result.retrieved_snippets

            score = float(result.confidence) if result.confidence is not None else 1.0
            results.append(self._make_node(result.most_relevant_rule, score=score, metadata=meta))

            if self.include_conformal_set and result.conformal_set:
                for alt_rule in result.conformal_set:
                    alt_text = str(alt_rule)
                    if alt_text == result.most_relevant_rule or len(results) >= self.top_k:
                        continue
                    alt_meta = dict(meta)
                    alt_meta["conformal_alternative"] = True
                    results.append(
                        self._make_node(alt_text, score=max(score * 0.9, 0.01), metadata=alt_meta)
                    )

        return results

    @staticmethod
    def _make_node(text: str, score: float, metadata: dict[str, Any]) -> Any:
        if _HAS_LLAMA_INDEX and NodeWithScore is not None and TextNode is not None:
            node = TextNode(text=text, metadata=metadata)
            return NodeWithScore(node=node, score=score)
        return LegalNodeResult(text=text, score=score, metadata=metadata)
