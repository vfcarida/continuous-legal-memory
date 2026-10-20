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
        transaction_time: datetime | None = None,
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
            transaction_time: Optional ingestion transaction timestamp for bi-temporal auditing.

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
            transaction_time=transaction_time or datetime.now(timezone.utc),
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

        # Enforce acyclic invariant for normative hierarchy and dependency edges
        if relation_type in (
            RelationType.SUPERSEDES,
            RelationType.DEPENDS_ON,
            RelationType.DERIVES_FROM,
        ):
            if source_id == target_id:
                raise MemoryContradictionError(
                    f"Self-referential '{relation_type.value}' cycle prohibited on node '{source_id}'.",
                    payload={"source_id": source_id, "target_id": target_id, "relation_type": relation_type},
                )
            # Check if source_id is reachable from target_id via existing edges of the same relation_type
            visited_cycle: set[str] = set()
            frontier = [target_id]
            while frontier:
                curr = frontier.pop()
                if curr == source_id:
                    raise MemoryContradictionError(
                        f"Adding edge '{source_id}' -> '{target_id}' with relation '{relation_type.value}' "
                        "would create a normative cycle.",
                        payload={"source_id": source_id, "target_id": target_id, "relation_type": relation_type},
                    )
                if curr not in visited_cycle:
                    visited_cycle.add(curr)
                    for existing_edge in self.edges:
                        if (
                            existing_edge.source_id == curr
                            and existing_edge.relation_type == relation_type
                            and existing_edge.tenant_id == edge_tenant
                        ):
                            frontier.append(existing_edge.target_id)

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
        as_of_transaction_time: datetime | None = None,
    ) -> list[GraphNode]:
        """
        Perform multi-hop traversal to retrieve all valid prerequisite nodes required by a given node,
        strictly scoped to tenant boundary.

        Args:
            node_id: Target node ID.
            at_time: Datetime timestamp to evaluate validity.
            tenant_id: Optional tenant identifier to enforce tenant isolation.
            as_of_transaction_time: Optional transaction-time cutoff for bi-temporal auditing.

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

        if (target.valid_to is not None and eval_time > target.valid_to) or not target.is_known_as_of(
            as_of_transaction_time
        ):
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
                            or not dep_node.is_known_as_of(as_of_transaction_time)
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

    def personalized_pagerank(
        self,
        seed_weights: dict[str, float] | list[str],
        damping: float = 0.85,
        max_iterations: int = 50,
        convergence_tol: float = 1e-6,
        tenant_id: str | None = None,
        at_time: datetime | None = None,
        as_of_transaction_time: datetime | None = None,
        deontic_polarity: bool = True,
    ) -> dict[str, float]:
        """
        Compute HippoRAG-style Personalized PageRank (PPR) over the legal entity graph
        with Deontic Polarity-Aware Graph Diffusion.

        Rationale:
            Standard semantic search retrieves only nodes that directly match the query text.
            In legal architectures, relevant statutory provisions often form multi-hop chains
            (e.g., General Rule -> Exception Clause -> Enforcement Directive).
            PPR diffuses activation mass from query seed nodes across semantic edges, discovering
            topologically and legally associated statutes while respecting tenant isolation.

            When `deontic_polarity=True` (default):
            1. `CONTRADICTS` edges are excluded from positive transition walks so activation mass
               never leaks into logically contradictory norms.
            2. `SUPERSEDES` edges (`new_norm -> old_norm`) reverse their diffusion flow so that
               querying an outdated/superseded norm (`old_norm`) redirects teleport and walk mass
               toward the active superseding norm (`new_norm`), while preventing active norms from
               leaking mass backward into superseded norms.

        Args:
            seed_weights: Either a dict mapping node_id -> float relevance score, or a list of node_ids.
            damping: Teleportation probability balance alpha (typically 0.85).
            max_iterations: Maximum power iteration steps.
            convergence_tol: L1 norm convergence threshold.
            tenant_id: Optional tenant identifier to enforce multi-tenant isolation.
            at_time: Datetime timestamp to filter out expired nodes. Defaults to current UTC time.
            as_of_transaction_time: Optional transaction-time cutoff for bi-temporal replay.
            deontic_polarity: Whether to enforce deontic edge polarity (`SUPERSEDES` redirection
                              and `CONTRADICTS` blocking).

        Returns:
            Dictionary mapping node_id to its stationary PPR probability, sorted descending.
        """
        eval_time = at_time or datetime.now(timezone.utc)
        valid_nodes = [
            n for n in self.nodes.values()
            if (tenant_id is None or n.tenant_id == tenant_id)
            and (n.valid_to is None or eval_time <= n.valid_to)
            and n.is_known_as_of(as_of_transaction_time)
        ]
        if not valid_nodes:
            return {}

        node_ids = [n.node_id for n in valid_nodes]
        node_set = set(node_ids)
        n_nodes = len(node_ids)
        idx_to_id = dict(enumerate(node_ids))
        id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

        # Build personalization distribution vector v
        if isinstance(seed_weights, list):
            seed_dict = {s: 1.0 for s in seed_weights if s in node_set}
        else:
            seed_dict = {s: float(w) for s, w in seed_weights.items() if s in node_set and w > 0}

        # Deontic polarity seed redirection: if a seeded node is superseded by an active node in node_set,
        # transfer `damping` fraction of its initial teleport mass to its superseding active node(s).
        if deontic_polarity and seed_dict:
            superseding_map: dict[str, list[tuple[str, float]]] = {}
            for edge in self.edges:
                if tenant_id is not None and edge.tenant_id != tenant_id:
                    continue
                if (
                    edge.relation_type == RelationType.SUPERSEDES
                    and edge.source_id in node_set
                    and edge.target_id in node_set
                ):
                    superseding_map.setdefault(edge.target_id, []).append(
                        (edge.source_id, max(0.01, float(edge.weight)))
                    )
            if superseding_map:
                adjusted_seeds: dict[str, float] = {}
                for nid, w in seed_dict.items():
                    superseders = superseding_map.get(nid)
                    if not superseders:
                        adjusted_seeds[nid] = adjusted_seeds.get(nid, 0.0) + w
                    else:
                        retained = (1.0 - damping) * w
                        if retained > 0:
                            adjusted_seeds[nid] = adjusted_seeds.get(nid, 0.0) + retained
                        total_sup_w = sum(sw for _, sw in superseders)
                        transferred = damping * w
                        for sup_id, sw in superseders:
                            adjusted_seeds[sup_id] = (
                                adjusted_seeds.get(sup_id, 0.0) + transferred * (sw / total_sup_w)
                            )
                seed_dict = adjusted_seeds

        total_seed_weight = sum(seed_dict.values())
        if total_seed_weight > 0:
            p_teleport = [seed_dict.get(nid, 0.0) / total_seed_weight for nid in node_ids]
        else:
            if len(seed_weights) > 0:
                return {}
            p_teleport = [1.0 / n_nodes] * n_nodes

        # Build adjacency matrix and out-degree sums
        adj: dict[int, list[tuple[int, float]]] = {i: [] for i in range(n_nodes)}
        out_weights: list[float] = [0.0] * n_nodes

        for edge in self.edges:
            if tenant_id is not None and edge.tenant_id != tenant_id:
                continue
            if edge.source_id in id_to_idx and edge.target_id in id_to_idx:
                if deontic_polarity and edge.relation_type == RelationType.CONTRADICTS:
                    # Block positive activation flow across contradictory norms
                    continue
                w = max(0.01, float(edge.weight))
                if deontic_polarity and edge.relation_type == RelationType.SUPERSEDES:
                    # Redirect random walk from superseded target_id -> superseding source_id
                    src_idx = id_to_idx[edge.target_id]
                    tgt_idx = id_to_idx[edge.source_id]
                else:
                    src_idx = id_to_idx[edge.source_id]
                    tgt_idx = id_to_idx[edge.target_id]
                adj[src_idx].append((tgt_idx, w))
                out_weights[src_idx] += w

        # Power iteration
        p = list(p_teleport)

        for _ in range(max_iterations):
            p_next = [(1.0 - damping) * t for t in p_teleport]

            # Handle dangling nodes (out_weight == 0)
            dangling_mass = sum(p[i] for i in range(n_nodes) if out_weights[i] == 0.0)
            if dangling_mass > 0:
                for i in range(n_nodes):
                    p_next[i] += damping * dangling_mass * p_teleport[i]

            for src_idx in range(n_nodes):
                if out_weights[src_idx] > 0:
                    src_prob = p[src_idx]
                    total_out = out_weights[src_idx]
                    for tgt_idx, weight in adj[src_idx]:
                        p_next[tgt_idx] += damping * src_prob * (weight / total_out)

            # Check L1 convergence
            diff = sum(abs(p_next[i] - p[i]) for i in range(n_nodes))
            p = p_next
            if diff < convergence_tol:
                break

        return dict(
            sorted(
                {idx_to_id[i]: p[i] for i in range(n_nodes)}.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        )



