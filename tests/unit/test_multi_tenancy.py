"""
Unit tests for multi-tenant isolation across all memory tiers and storage layers.
"""

from pathlib import Path

import pytest
import torch

from continuous_legal_memory.core.memory_tiers.semantic_memory import SemanticKnowledgeGraph
from continuous_legal_memory.core.memory_tiers.working_memory import WorkingMemory
from continuous_legal_memory.domain.exceptions import MemoryContradictionError
from continuous_legal_memory.domain.interfaces import BaseEncoderPort
from continuous_legal_memory.domain.models import EntityType, RelationType
from continuous_legal_memory.orchestrator import LegalMemoryOrchestrator


def test_working_memory_tenant_isolation() -> None:
    """Validate that WorkingMemory strictly partitions active context by tenant_id."""
    wm = WorkingMemory(capacity=10)
    v_key = torch.zeros(1, 16)
    v_val = torch.tensor([[1.0, 0.0]])

    wm.add("Alpha prompt 1", v_key, v_val, tenant_id="tenant_alpha")
    wm.add("Alpha prompt 2", v_key, v_val, tenant_id="tenant_alpha")
    wm.add("Beta prompt 1", v_key, v_val, tenant_id="tenant_beta")
    wm.add("Shared text prompt", v_key, v_val, tenant_id="tenant_alpha")
    wm.add("Shared text prompt", v_key, v_val, tenant_id="tenant_beta")

    assert len(wm) == 5

    alpha_ctx = wm.get_active_context(tenant_id="tenant_alpha")
    beta_ctx = wm.get_active_context(tenant_id="tenant_beta")

    assert len(alpha_ctx) == 3
    assert all(r.tenant_id == "tenant_alpha" for r in alpha_ctx)

    assert len(beta_ctx) == 2
    assert all(r.tenant_id == "tenant_beta" for r in beta_ctx)

    # Remove by text should only affect specified tenant
    wm.remove_by_text("Shared text prompt", tenant_id="tenant_alpha")
    assert len(wm.get_active_context(tenant_id="tenant_alpha")) == 2
    assert len(wm.get_active_context(tenant_id="tenant_beta")) == 2

    # Clear tenant_alpha only
    wm.clear(tenant_id="tenant_alpha")
    assert len(wm.get_active_context(tenant_id="tenant_alpha")) == 0
    assert len(wm.get_active_context(tenant_id="tenant_beta")) == 2


def test_semantic_knowledge_graph_tenant_isolation() -> None:
    """Validate that SemanticKnowledgeGraph enforces tenant boundaries and forbids cross-tenant edges."""
    graph = SemanticKnowledgeGraph()

    graph.add_node("alpha_statute_1", EntityType.STATUTE, "Alpha Statute", "Content A1", tenant_id="tenant_alpha")
    graph.add_node("alpha_clause_1", EntityType.CLAUSE, "Alpha Clause", "Content A2", tenant_id="tenant_alpha")
    graph.add_node("beta_statute_1", EntityType.STATUTE, "Beta Statute", "Content B1", tenant_id="tenant_beta")

    # Add intra-tenant edge
    graph.add_edge("alpha_clause_1", "alpha_statute_1", RelationType.DEPENDS_ON)
    assert len(graph.get_edges(tenant_id="tenant_alpha")) == 1
    assert len(graph.get_edges(tenant_id="tenant_beta")) == 0

    # Cross-tenant edge attempt must raise MemoryContradictionError
    with pytest.raises(MemoryContradictionError):
        graph.add_edge("beta_statute_1", "alpha_statute_1", RelationType.SUPERSEDES)

    # Prerequisite traversal should not cross tenant boundary
    prereqs = graph.check_prerequisites("alpha_clause_1", tenant_id="tenant_alpha")
    assert len(prereqs) == 1
    assert prereqs[0].node_id == "alpha_statute_1"

    with pytest.raises(KeyError):
        # Attempting to check prerequisites with mismatched tenant must fail
        graph.check_prerequisites("alpha_clause_1", tenant_id="tenant_beta")

    # Node deletion with mismatched tenant must fail
    with pytest.raises(KeyError):
        graph.remove_node("alpha_statute_1", tenant_id="tenant_beta")

    assert "alpha_statute_1" in graph.nodes


def test_orchestrator_multi_tenant_isolation(offline_encoder: BaseEncoderPort) -> None:
    """Validate end-to-end multi-tenant isolation in LegalMemoryOrchestrator."""
    orch = LegalMemoryOrchestrator(value_dim=2, encoder=offline_encoder)

    rec_a = orch.update_memory(
        "Tenant Alpha Rule: Customers can request deletion.",
        [1.0, 0.0],
        tenant_id="tenant_alpha",
    )
    rec_b = orch.update_memory(
        "Tenant Beta Rule: Customers must retain all records.",
        [0.0, 1.0],
        tenant_id="tenant_beta",
    )
    assert rec_b.tenant_id == "tenant_beta"


    # Check Working Memory scoping
    wm_alpha = orch.get_working_memory_context(tenant_id="tenant_alpha")
    wm_beta = orch.get_working_memory_context(tenant_id="tenant_beta")
    assert len(wm_alpha) == 1
    assert wm_alpha[0].text == "Tenant Alpha Rule: Customers can request deletion."
    assert len(wm_beta) == 1
    assert wm_beta[0].text == "Tenant Beta Rule: Customers must retain all records."

    # Check Semantic Graph scoping
    sg_alpha = orch.get_semantic_graph_nodes(tenant_id="tenant_alpha")
    sg_beta = orch.get_semantic_graph_nodes(tenant_id="tenant_beta")
    assert len(sg_alpha) == 1
    assert sg_alpha[0].tenant_id == "tenant_alpha"
    assert len(sg_beta) == 1
    assert sg_beta[0].tenant_id == "tenant_beta"

    # Cross-tenant deletion attempt must fail
    with pytest.raises(KeyError):
        orch.delete_rule(rec_a.record_id, tenant_id="tenant_beta")

    # Legitimate deletion succeeds and only affects tenant_alpha
    audit = orch.delete_rule(rec_a.record_id, tenant_id="tenant_alpha")
    assert audit["status"] == "ERASED"
    assert len(orch.get_working_memory_context(tenant_id="tenant_alpha")) == 0
    assert len(orch.get_working_memory_context(tenant_id="tenant_beta")) == 1
    assert len(orch.get_semantic_graph_nodes(tenant_id="tenant_alpha")) == 0
    assert len(orch.get_semantic_graph_nodes(tenant_id="tenant_beta")) == 1


def test_sqlite_multi_tenant_roundtrip(tmp_path: Path, offline_encoder: BaseEncoderPort) -> None:
    """Validate that SqliteMemoryStore preserves tenant_id on semantic graph nodes and edges across serialization."""
    db_file = tmp_path / "tenant_test.db"
    orch = LegalMemoryOrchestrator(value_dim=2, encoder=offline_encoder)

    orch.update_memory(
        "Alpha Rule 101",
        [1.0, 0.0],
        tenant_id="alpha_corp",
        metadata={"node_id": "alpha_n1"},
    )
    orch.update_memory(
        "Beta Rule 202",
        [0.0, 1.0],
        tenant_id="beta_corp",
        metadata={"node_id": "beta_n1"},
    )

    orch.save_to_disk(db_file)

    # Restore in new orchestrator
    restored = LegalMemoryOrchestrator(value_dim=2, encoder=offline_encoder)
    restored.load_from_disk(db_file)

    alpha_nodes = restored.get_semantic_graph_nodes(tenant_id="alpha_corp")
    beta_nodes = restored.get_semantic_graph_nodes(tenant_id="beta_corp")

    assert len(alpha_nodes) == 1
    assert alpha_nodes[0].tenant_id == "alpha_corp"
    assert alpha_nodes[0].node_id == "alpha_n1"

    assert len(beta_nodes) == 1
    assert beta_nodes[0].tenant_id == "beta_corp"
    assert beta_nodes[0].node_id == "beta_n1"
