"""
LlamaIndex Retriever Adapter for Continuous Legal Memory.

Provides drop-in integration with LlamaIndex query engines, RAG pipelines, and agent tools,
enabling LlamaIndex applications to retrieve governing statutory norms with character-level
snippets and legal precedence scoring.
"""

from __future__ import annotations

from typing import Any

from continuous_legal_memory.orchestrator import LegalMemoryOrchestrator

# Optional import of LlamaIndex BaseRetriever
try:
    from llama_index.core.base.base_retriever import BaseRetriever as _LlamaIndexBaseRetriever
    from llama_index.core.schema import NodeWithScore, TextNode
    _HAS_LLAMA_INDEX = True
except ImportError:
    _LlamaIndexBaseRetriever = object  # type: ignore[misc,assignment]
    NodeWithScore = None  # type: ignore[assignment,misc]
    TextNode = None  # type: ignore[assignment,misc]
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
    """

    def __init__(
        self,
        orchestrator: LegalMemoryOrchestrator,
        tenant_id: str = "default",
        top_k: int = 5,
    ) -> None:
        self.orchestrator = orchestrator
        self.tenant_id = tenant_id
        self.top_k = top_k

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

    def retrieve(self, str_or_query_bundle: Any) -> list[Any]:
        """
        Execute statutory retrieval over Continuous Legal Memory.

        Args:
            str_or_query_bundle: Natural language legal query string or QueryBundle.

        Returns:
            List of retrieved nodes sorted by precedence score.
        """
        query_str = getattr(str_or_query_bundle, "query_str", str(str_or_query_bundle))

        with self.orchestrator.tenant(self.tenant_id):
            result = self.orchestrator.predict(query_str)

        results: list[Any] = []
        if result.most_relevant_rule:
            meta = {
                "tenant_id": self.tenant_id,
                "predicted_action_vector": result.predicted_action_vector,
                "fast_slow_gate": result.fast_slow_gate,
            }
            if result.retrieved_snippets:
                meta["retrieved_snippets"] = result.retrieved_snippets

            score = 1.0  # Top governing rule

            if _HAS_LLAMA_INDEX and NodeWithScore is not None and TextNode is not None:
                node = TextNode(text=result.most_relevant_rule, metadata=meta)
                results.append(NodeWithScore(node=node, score=score))
            else:
                results.append(LegalNodeResult(text=result.most_relevant_rule, score=score, metadata=meta))

        return results
