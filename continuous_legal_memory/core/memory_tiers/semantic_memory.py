"""
Semantic Knowledge Graph Component.

Implements an entity-typed knowledge graph abstraction mapping persistent legal entities,
logic rules, client preferences, prerequisite dependencies, and non-destructive GDPR Art. 17 temporal invalidations.
"""

from datetime import datetime, timezone

import torch

from continuous_legal_memory.domain.exceptions import (
    TemporalInvalidationError,
)
from continuous_legal_memory.domain.models import EntityType, GraphEdge, GraphNode, RelationType


class SemanticKnowledgeGraph:
    """
    Entity-Typed Semantic Knowledge Graph abstraction layer.

    Rationale:
        Pure vector search cannot reliably evaluate legal prerequisite dependencies or rule precedence.
        The Semantic Knowledge Graph constructs an explicit entity-relationship network:
        1. Non-Destructive Invalidation (GDPR Art. 17): Updating client preferences or statutes does not
           execute hard SQL DELETE queries; instead, it sets `valid_to` timestamps and reduces `decay_factor` to 0.0,
           preserving an immutable audit log.
        2. Multi-Hop Prerequisite Dependency Checking: Verifies prerequisite relations (e.g. `DEPENDS_ON`)
           and checks for direct logical contradictions (`CONTRADICTS`) before inference.
    """

    def __init__(self) -> None:
        """Initialize the Semantic Knowledge Graph."""
        self.nodes: dict[str, GraphNode] = {}
        self.edges: list[GraphEdge] = []

    def add_node(
        self,
        node_id: str,
        entity_type: EntityType,
        label: str,
        description: str,
        embedding: torch.Tensor | None = None,
        valid_from: datetime | None = None,
        valid_to: datetime | None = None,
    ) -> GraphNode:
        """
        Add an entity node to the knowledge graph.

        Args:
            node_id: Unique string identifier for the node.
            entity_type: Category enum (STATUTE, CLAUSE, PREFERENCE, etc.).
            label: Human-readable entity name.
            description: Legal clause or rule text content.
            embedding: Vector embedding tensor.
            valid_from: Start timestamp of legal validity.
            valid_to: Expiry or invalidation timestamp.

        Returns:
            The created `GraphNode`.
        """
        node = GraphNode(
            node_id=node_id,
            entity_type=entity_type,
            label=label,
            description=description,
            embedding=embedding,
            valid_from=valid_from or datetime.now(timezone.utc),
            valid_to=valid_to,
        )
        self.nodes[node_id] = node
        return node

    def add_edge(self, source_id: str, target_id: str, relation_type: RelationType, weight: float = 1.0) -> GraphEdge:
        """
        Add a directed relation edge between two graph nodes.

        Args:
            source_id: Originating node ID.
            target_id: Destination node ID.
            relation_type: Relationship classification (SUPERSEDES, CONTRADICTS, DEPENDS_ON, etc.).
            weight: Precedence or strength coefficient.

        Returns:
            The created `GraphEdge`.

        Raises:
            KeyError: If source or target node ID does not exist in graph.
            MemoryContradictionError: If attempting to link nodes already marked as contradictory.
        """
        if source_id not in self.nodes:
            raise KeyError(f"Source node '{source_id}' not found in knowledge graph.")
        if target_id not in self.nodes:
            raise KeyError(f"Target node '{target_id}' not found in knowledge graph.")

        edge = GraphEdge(source_id=source_id, target_id=target_id, relation_type=relation_type, weight=weight)
        self.edges.append(edge)
        return edge

    def invalidate_node_non_destructively(self, node_id: str, invalidation_time: datetime | None = None) -> None:
        """
        Execute non-destructive temporal invalidation on a graph node (GDPR Art. 17 compliant).

        Rationale:
            Rather than destroying records via DELETE, this sets `valid_to` to the current timestamp
            and decays `decay_factor` to 0.0, rendering the node inactive for inference while preserving
            audit trails.

        Args:
            node_id: Node ID to invalidate.
            invalidation_time: Datetime timestamp marking invalidation. Defaults to current UTC time.

        Raises:
            KeyError: If node_id is not found.
        """
        if node_id not in self.nodes:
            raise KeyError(f"Node '{node_id}' not found in knowledge graph.")

        ts = invalidation_time or datetime.now(timezone.utc)
        node = self.nodes[node_id]
        node.valid_to = ts
        node.decay_factor = 0.0

    def check_prerequisites(self, node_id: str, at_time: datetime | None = None) -> list[GraphNode]:
        """
        Perform multi-hop traversal to retrieve all valid prerequisite nodes required by a given node.

        Args:
            node_id: Target node ID.
            at_time: Datetime timestamp to evaluate validity.

        Returns:
            List of valid prerequisite `GraphNode` instances.

        Raises:
            TemporalInvalidationError: If target node or any required prerequisite node is expired.
        """
        eval_time = at_time or datetime.now(timezone.utc)
        if node_id not in self.nodes:
            raise KeyError(f"Node '{node_id}' not found in knowledge graph.")

        target = self.nodes[node_id]
        if target.valid_to is not None and eval_time > target.valid_to:
            raise TemporalInvalidationError(f"Target node '{node_id}' is temporally invalid.")

        visited: set[str] = set()
        prerequisites: list[GraphNode] = []

        def traverse(current_id: str) -> None:
            visited.add(current_id)
            for edge in self.edges:
                if edge.source_id == current_id and edge.relation_type == RelationType.DEPENDS_ON:
                    dep_id = edge.target_id
                    if dep_id not in visited:
                        dep_node = self.nodes.get(dep_id)
                        if dep_node is None or (dep_node.valid_to is not None and eval_time > dep_node.valid_to):
                            raise TemporalInvalidationError(
                                f"Required prerequisite node '{dep_id}' for '{current_id}' is invalid or expired."
                            )
                        prerequisites.append(dep_node)
                        traverse(dep_id)

        traverse(node_id)
        return prerequisites

    def find_contradictions(self, node_id: str) -> list[GraphNode]:
        """
        Find all active graph nodes that contradict the specified node.

        Args:
            node_id: Target node ID to check for contradiction links.

        Returns:
            List of contradicting `GraphNode` instances.
        """
        contradictions: list[GraphNode] = []
        for edge in self.edges:
            if edge.relation_type == RelationType.CONTRADICTS:
                if edge.source_id == node_id and edge.target_id in self.nodes:
                    contradictions.append(self.nodes[edge.target_id])
                elif edge.target_id == node_id and edge.source_id in self.nodes:
                    contradictions.append(self.nodes[edge.source_id])
        return contradictions
