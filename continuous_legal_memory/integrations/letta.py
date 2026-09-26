"""
Letta (formerly MemGPT) Integration Adapter for Continuous Legal Memory.

Provides a Letta-compatible memory block and tool interfaces, allowing autonomous agents
to query legal precedence, execute multi-hop graph associations, and ingest compliance rules.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from continuous_legal_memory.orchestrator import LegalMemoryOrchestrator


class ContinuousLegalMemoryBlock:
    """
    Letta/MemGPT-compatible Memory Block interface for legal precedence memory.

    Rationale:
        Letta structures long-term agent memory into distinct blocks within the agent's context window.
        `ContinuousLegalMemoryBlock` bridges Continuous Legal Memory into Letta's block interface,
        exposing an auto-updating textual summary of active governing statutes while backing queries
        with deterministic precedence resolution and multi-tenant isolation.
    """

    def __init__(
        self,
        orchestrator: LegalMemoryOrchestrator,
        name: str = "legal_statutes",
        label: str = "legal_compliance",
        limit: int = 2000,
        tenant_id: str | None = None,
    ) -> None:
        """
        Initialize the ContinuousLegalMemoryBlock.

        Args:
            orchestrator: Configured LegalMemoryOrchestrator instance.
            name: Block identifier name.
            label: Block category label.
            limit: Maximum character budget for the compiled block text.
            tenant_id: Optional tenant identifier for multi-tenant isolation.
        """
        self.orchestrator = orchestrator
        self.name = name
        self.label = label
        self.limit = limit
        self.tenant_id = tenant_id

    @property
    def value(self) -> str:
        """Render the current memory block content for context window compilation."""
        return self.compile()

    def compile(self) -> str:
        """
        Compile active working memory and governing rules into a concise context block.

        Returns:
            Formatted string summarizing active statutory directives.
        """
        active_records = self.orchestrator.working_memory.get_active_context(tenant_id=self.tenant_id)
        if not active_records:
            return "[Legal Precedence Memory: No active statutes loaded]"

        lines = ["[Legal Precedence Memory - Active Directives]"]
        char_count = len(lines[0])

        for rec in active_records:
            line = f"- (Rank {rec.authority_rank}) {rec.text}"
            if char_count + len(line) + 1 > self.limit:
                lines.append("... [Truncated due to character limit]")
                break
            lines.append(line)
            char_count += len(line) + 1

        return "\n".join(lines)

    def insert_rule(
        self,
        rule_text: str,
        action_vector: list[float],
        authority_rank: int = 1,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Ingest a new legal directive into memory.

        Args:
            rule_text: Text snippet of legal rule or policy.
            action_vector: Decision vector list.
            authority_rank: Hierarchy rank integer.
            metadata: Custom metadata dictionary.

        Returns:
            Confirmation string with record ID.
        """
        rec = self.orchestrator.update_memory(
            rule_text=rule_text,
            action_vector=action_vector,
            authority_rank=authority_rank,
            metadata=metadata,
            tenant_id=self.tenant_id or "default",
        )
        return f"Successfully ingested rule '{rec.record_id}' (Rank {authority_rank})."

    def search(
        self,
        query: str,
        at_time: datetime | None = None,
    ) -> dict[str, Any]:
        """
        Query legal memory for the governing directive and decision vector.

        Args:
            query: Query string describing the legal factual scenario.
            at_time: Optional datetime for temporal validity evaluation.

        Returns:
            Dictionary with governing rule, predicted action vector, and snippets.
        """
        result = self.orchestrator.predict(
            query_text=query,
            at_time=at_time,
            tenant_id=self.tenant_id,
        )
        return {
            "query": query,
            "most_relevant_rule": result.most_relevant_rule,
            "predicted_action_vector": result.predicted_action_vector,
            "retrieved_snippets": result.retrieved_snippets,
            "confidence": result.confidence,
        }

    def associate(
        self,
        seed_clause_ids: list[str],
        max_results: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Execute HippoRAG Personalized PageRank to discover topologically linked legal prerequisites.

        Args:
            seed_clause_ids: List of seed node identifiers.
            max_results: Maximum associated nodes to return.

        Returns:
            List of dictionaries with node_id, label, description, and PPR score.
        """
        results = self.orchestrator.associate_statutes(
            seed_nodes=seed_clause_ids,
            max_results=max_results,
            tenant_id=self.tenant_id,
        )
        return [
            {
                "node_id": node.node_id,
                "label": node.label,
                "description": node.description,
                "entity_type": node.entity_type.value,
                "score": score,
            }
            for node, score in results
        ]


def create_letta_tools(
    orchestrator: LegalMemoryOrchestrator,
    tenant_id: str | None = None,
) -> dict[str, Any]:
    """
    Create callable tools for Letta/MemGPT agent tool registries.

    Args:
        orchestrator: Initialized LegalMemoryOrchestrator instance.
        tenant_id: Optional tenant identifier for scoping.

    Returns:
        Dictionary mapping tool names to callable execution functions.
    """
    block = ContinuousLegalMemoryBlock(orchestrator=orchestrator, tenant_id=tenant_id)

    def legal_memory_search(query: str) -> str:
        """Search legal memory for governing directives and recommended decisions."""
        res = block.search(query)
        if not res["most_relevant_rule"]:
            return "No governing legal rule found in active memory."
        return (
            f"Governing Rule: {res['most_relevant_rule']}\n"
            f"Action Vector: {res['predicted_action_vector']}\n"
            f"Snippets: {res.get('retrieved_snippets', [])}"
        )

    def legal_memory_insert(rule_text: str, action_vector: list[float], authority_rank: int = 1) -> str:
        """Insert a new legal statute or compliance policy into continuous memory."""
        return block.insert_rule(rule_text, action_vector, authority_rank=authority_rank)

    def legal_memory_associate(seed_clause_id: str) -> str:
        """Discover multi-hop legal dependencies via HippoRAG Personalized PageRank."""
        associations = block.associate([seed_clause_id], max_results=5)
        if not associations:
            return f"No topological associations found for '{seed_clause_id}'."
        lines = [f"Associations for {seed_clause_id}:"]
        for a in associations:
            lines.append(f"- [{a['entity_type']}] {a['node_id']}: {a['label']} (PPR: {a['score']:.4f})")
        return "\n".join(lines)

    return {
        "legal_memory_search": legal_memory_search,
        "legal_memory_insert": legal_memory_insert,
        "legal_memory_associate": legal_memory_associate,
    }
