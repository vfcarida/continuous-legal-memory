"""
Semantic Knowledge Graph Component.

Implements an entity-typed knowledge graph abstraction mapping persistent legal entities,
logic rules, client preferences, prerequisite dependencies, and non-destructive GDPR Art. 17 temporal invalidations,
enforcing strict multi-tenant isolation.
"""

from __future__ import annotations

from datetime import datetime, timezone

import torch

from continuous_legal_memory.domain.exceptions import (
    MemoryContradictionError,
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
        3. Strict Multi-Tenant Isolation: Enforces tenant namespace partitioning; blocks cross-tenant edges
           and partitions graph traversals.
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
        tenant_id: str = "default",
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
            tenant_id: Tenant namespace identifier.

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
            tenant_id=tenant_id,
        )
        self.nodes[node_id] = node
        return node

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relation_type: RelationType,
        weight: float = 1.0,
        tenant_id: str | None = None,
    ) -> GraphEdge:
        """
        Add a directed relation edge between two graph nodes.

        Args:
            source_id: Originating node ID.
            target_id: Destination node ID.
            relation_type: Relationship classification (SUPERSEDES, CONTRADICTS, DEPENDS_ON, etc.).
            weight: Precedence or strength coefficient.
            tenant_id: Optional tenant identifier. If omitted, inherits source node's tenant.

        Returns:
            The created `GraphEdge`.

        Raises:
            KeyError: If source or target node ID does not exist in graph.
            MemoryContradictionError: If attempting to link nodes across different tenants or nodes already contradictory.
        """
        if source_id not in self.nodes:
            raise KeyError(f"Source node '{source_id}' not found in knowledge graph.")
        if target_id not in self.nodes:
            raise KeyError(f"Target node '{target_id}' not found in knowledge graph.")

        source_node = self.nodes[source_id]
        target_node = self.nodes[target_id]

        # Enforce multi-tenant isolation: forbid linking nodes from different tenants
        if source_node.tenant_id != target_node.tenant_id:
            raise MemoryContradictionError(
                f"Cross-tenant graph edge prohibited: source node '{source_id}' belongs to tenant "
                f"'{source_node.tenant_id}' while target node '{target_id}' belongs to tenant '{target_node.tenant_id}'.",
                payload={
                    "source_id": source_id,
                    "target_id": target_id,
                    "source_tenant": source_node.tenant_id,
                    "target_tenant": target_node.tenant_id,
                },
            )

        # Check contradiction constraints: cannot link nodes already marked as contradictory
        if relation_type != RelationType.CONTRADICTS:
            contradicting_ids = {c.node_id for c in self.find_contradictions(source_id, tenant_id=source_node.tenant_id)}
            if target_id in contradicting_ids:
                raise MemoryContradictionError(
                    f"Cannot link '{source_id}' to '{target_id}' with relation '{relation_type}': "
                    "nodes are already marked as contradictory.",
                    payload={"source_id": source_id, "target_id": target_id, "relation_type": relation_type},
                )

        edge_tenant = tenant_id or source_node.tenant_id
        edge = GraphEdge(
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            weight=weight,
            tenant_id=edge_tenant,
        )
        self.edges.append(edge)
        return edge

    def invalidate_node_non_destructively(
        self,
        node_id: str,
        invalidation_time: datetime | None = None,
        tenant_id: str | None = None,
    ) -> None:
        """
        Execute non-destructive temporal invalidation on a graph node (GDPR Art. 17 compliant).

        Rationale:
            Rather than destroying records via DELETE, this sets `valid_to` to the current timestamp
            and decays `decay_factor` to 0.0, rendering the node inactive for inference while preserving
            audit trails.

        Args:
            node_id: Node ID to invalidate.
            invalidation_time: Datetime timestamp marking invalidation. Defaults to current UTC time.
            tenant_id: Optional tenant identifier to enforce authorization.

        Raises:
            KeyError: If node_id is not found or tenant mismatch occurs.
        """
        if node_id not in self.nodes:
            raise KeyError(f"Node '{node_id}' not found in knowledge graph.")

        if tenant_id is not None and self.nodes[node_id].tenant_id != tenant_id:
            raise KeyError(f"Node '{node_id}' does not belong to tenant '{tenant_id}'.")

        ts = invalidation_time or datetime.now(timezone.utc)
        node = self.nodes[node_id]
        node.valid_to = ts
        node.decay_factor = 0.0

    def check_prerequisites(
        self,
        node_id: str,
        at_time: datetime | None = None,
        tenant_id: str | None = None,
    ) -> list[GraphNode]:
        """
        Perform multi-hop traversal to retrieve all valid prerequisite nodes required by a given node,
        strictly scoped to tenant boundary.

        Args:
            node_id: Target node ID.
            at_time: Datetime timestamp to evaluate validity.
            tenant_id: Optional tenant identifier to enforce tenant isolation.

        Returns:
            List of valid prerequisite `GraphNode` instances.

        Raises:
            KeyError: If target node does not exist or tenant mismatch occurs.
            TemporalInvalidationError: If target node or any required prerequisite node is expired.
        """
        eval_time = at_time or datetime.now(timezone.utc)
        if node_id not in self.nodes:
            raise KeyError(f"Node '{node_id}' not found in knowledge graph.")

        target = self.nodes[node_id]
        effective_tenant = tenant_id or target.tenant_id

        if tenant_id is not None and target.tenant_id != tenant_id:
            raise KeyError(f"Node '{node_id}' does not belong to tenant '{tenant_id}'.")

        if target.valid_to is not None and eval_time > target.valid_to:
            raise TemporalInvalidationError(f"Target node '{node_id}' is temporally invalid.")

        visited: set[str] = set()
        prerequisites: list[GraphNode] = []

        def traverse(current_id: str) -> None:
            visited.add(current_id)
            for edge in self.edges:
                if (
                    edge.source_id == current_id
                    and edge.relation_type == RelationType.DEPENDS_ON
                    and edge.tenant_id == effective_tenant
                ):
                    dep_id = edge.target_id
                    if dep_id not in visited:
                        dep_node = self.nodes.get(dep_id)
                        if (
                            dep_node is None
                            or dep_node.tenant_id != effective_tenant
                            or (dep_node.valid_to is not None and eval_time > dep_node.valid_to)
                        ):
                            raise TemporalInvalidationError(
                                f"Required prerequisite node '{dep_id}' for '{current_id}' is invalid or expired."
                            )
                        prerequisites.append(dep_node)
                        traverse(dep_id)

        traverse(node_id)
        return prerequisites

    def find_contradictions(
        self,
        node_id: str,
        tenant_id: str | None = None,
    ) -> list[GraphNode]:
        """
        Find all active graph nodes that contradict the specified node within tenant boundary.

        Args:
            node_id: Target node ID to check for contradiction links.
            tenant_id: Optional tenant identifier.

        Returns:
            List of contradicting `GraphNode` instances.
        """
        if node_id not in self.nodes:
            return []

        target_node = self.nodes[node_id]
        effective_tenant = tenant_id or target_node.tenant_id

        if tenant_id is not None and target_node.tenant_id != tenant_id:
            return []

        contradictions: list[GraphNode] = []
        for edge in self.edges:
            if edge.relation_type == RelationType.CONTRADICTS and edge.tenant_id == effective_tenant:
                if edge.source_id == node_id and edge.target_id in self.nodes:
                    other = self.nodes[edge.target_id]
                    if other.tenant_id == effective_tenant:
                        contradictions.append(other)
                elif edge.target_id == node_id and edge.source_id in self.nodes:
                    other = self.nodes[edge.source_id]
                    if other.tenant_id == effective_tenant:
                        contradictions.append(other)
        return contradictions

    def remove_node(self, node_id: str, tenant_id: str | None = None) -> None:
        """
        Permanently remove an entity node and all incident edges from the knowledge graph.

        Args:
            node_id: Identifier of node to delete.
            tenant_id: Optional tenant identifier to enforce authorization.

        Raises:
            KeyError: If node_id does not belong to specified tenant.
        """
        if node_id in self.nodes:
            if tenant_id is not None and self.nodes[node_id].tenant_id != tenant_id:
                raise KeyError(f"Node '{node_id}' does not belong to tenant '{tenant_id}'.")
            del self.nodes[node_id]
        self.edges = [
            e for e in self.edges
            if not ((e.source_id == node_id or e.target_id == node_id) and (tenant_id is None or e.tenant_id == tenant_id))
        ]

    def is_superseded(
        self,
        node_id: str,
        active_node_ids: set[str] | None = None,
        tenant_id: str | None = None,
    ) -> bool:
        """
        Check if an entity node is superseded by an active superseding relation within tenant boundary.

        Args:
            node_id: Node ID to check.
            active_node_ids: Optional set of active node IDs to restrict supersession to.
            tenant_id: Optional tenant identifier.

        Returns:
            True if an active node supersedes node_id.
        """
        if node_id in self.nodes and tenant_id is not None and self.nodes[node_id].tenant_id != tenant_id:
            return False

        effective_tenant = tenant_id or (self.nodes[node_id].tenant_id if node_id in self.nodes else None)

        for edge in self.edges:
            if (
                edge.target_id == node_id
                and edge.relation_type == RelationType.SUPERSEDES
                and (effective_tenant is None or edge.tenant_id == effective_tenant)
                and (active_node_ids is None or edge.source_id in active_node_ids)
            ):
                return True
        return False

    def get_nodes(self, tenant_id: str | None = None) -> list[GraphNode]:
        """Return all nodes, optionally filtered by tenant."""
        if tenant_id is None:
            return list(self.nodes.values())
        return [n for n in self.nodes.values() if n.tenant_id == tenant_id]

    def get_edges(self, tenant_id: str | None = None) -> list[GraphEdge]:
        """Return all edges, optionally filtered by tenant."""
        if tenant_id is None:
            return list(self.edges)
        return [e for e in self.edges if e.tenant_id == tenant_id]


