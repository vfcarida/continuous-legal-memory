"""
Unit tests for HippoRAG-style Personalized PageRank over SemanticKnowledgeGraph.
"""

from datetime import datetime, timedelta, timezone

from continuous_legal_memory.core.memory_tiers.semantic_memory import SemanticKnowledgeGraph
from continuous_legal_memory.domain.interfaces import BaseEncoderPort
from continuous_legal_memory.domain.models import EntityType, RelationType
from continuous_legal_memory.orchestrator import LegalMemoryOrchestrator


def test_personalized_pagerank_basic_diffusion() -> None:
    """Verify PageRank activation spreads from seed node across legal dependency edges."""
    graph = SemanticKnowledgeGraph()

    # Create legal entity nodes: General Statute -> Exception Clause -> Regulatory Guidance
    graph.add_node("statute_10", EntityType.STATUTE, "Tax Exemption", "General corporate tax exemption")
    graph.add_node("clause_10a", EntityType.CLAUSE, "Export Exception", "Export activity exception clause")
    graph.add_node("guidance_01", EntityType.OBLIGATION, "Filing Procedure", "Procedural guidance on export declarations")
    graph.add_node("unrelated_statute", EntityType.STATUTE, "Maritime Law", "Unrelated vessel registration rule")

    # Connect with directed dependency edges
    graph.add_edge("statute_10", "clause_10a", RelationType.DEPENDS_ON, weight=1.0)
    graph.add_edge("clause_10a", "guidance_01", RelationType.DEPENDS_ON, weight=1.0)

    # Seed on statute_10
    ppr_scores = graph.personalized_pagerank(seed_weights=["statute_10"], damping=0.85)

    assert "statute_10" in ppr_scores
    assert "clause_10a" in ppr_scores
    assert "guidance_01" in ppr_scores
    assert "unrelated_statute" in ppr_scores

    # Associated nodes should have higher stationary rank than disconnected unrelated statute
    assert ppr_scores["statute_10"] > ppr_scores["unrelated_statute"]
    assert ppr_scores["clause_10a"] > ppr_scores["unrelated_statute"]
    # Probabilities should sum to approximately 1.0
    assert abs(sum(ppr_scores.values()) - 1.0) < 1e-4


def test_personalized_pagerank_tenant_isolation() -> None:
    """Verify PageRank strictly confines activation to the requesting tenant's partition."""
    graph = SemanticKnowledgeGraph()

    # Tenant Alpha nodes
    graph.add_node("alpha_doc1", EntityType.STATUTE, "Alpha 1", "Alpha clause 1", tenant_id="tenant_alpha")
    graph.add_node("alpha_doc2", EntityType.CLAUSE, "Alpha 2", "Alpha clause 2", tenant_id="tenant_alpha")
    graph.add_edge("alpha_doc1", "alpha_doc2", RelationType.DEPENDS_ON, tenant_id="tenant_alpha")

    # Tenant Beta nodes
    graph.add_node("beta_doc1", EntityType.STATUTE, "Beta 1", "Beta clause 1", tenant_id="tenant_beta")
    graph.add_node("beta_doc2", EntityType.CLAUSE, "Beta 2", "Beta clause 2", tenant_id="tenant_beta")
    graph.add_edge("beta_doc1", "beta_doc2", RelationType.DEPENDS_ON, tenant_id="tenant_beta")

    # Query scoped to Tenant Alpha
    alpha_scores = graph.personalized_pagerank(seed_weights=["alpha_doc1"], tenant_id="tenant_alpha")
    assert "alpha_doc1" in alpha_scores
    assert "alpha_doc2" in alpha_scores
    assert "beta_doc1" not in alpha_scores
    assert "beta_doc2" not in alpha_scores

    # Query scoped to Tenant Beta
    beta_scores = graph.personalized_pagerank(seed_weights=["beta_doc1"], tenant_id="tenant_beta")
    assert "beta_doc1" in beta_scores
    assert "beta_doc2" in beta_scores
    assert "alpha_doc1" not in beta_scores
    assert "alpha_doc2" not in beta_scores


def test_personalized_pagerank_temporal_filtering() -> None:
    """Verify expired or temporally invalidated nodes are excluded from PageRank graph."""
    graph = SemanticKnowledgeGraph()
    now = datetime.now(timezone.utc)

    graph.add_node("active_rule", EntityType.STATUTE, "Active Rule", "Currently valid directive")
    graph.add_node(
        "expired_clause",
        EntityType.CLAUSE,
        "Expired Clause",
        "Repealed historical clause",
        valid_to=now - timedelta(days=1),
    )
    graph.add_edge("active_rule", "expired_clause", RelationType.DEPENDS_ON)

    scores = graph.personalized_pagerank(seed_weights=["active_rule"], at_time=now)
    assert "active_rule" in scores
    assert "expired_clause" not in scores


def test_orchestrator_associate_statutes(offline_encoder: BaseEncoderPort) -> None:
    """Verify LegalMemoryOrchestrator.associate_statutes top-level API integration."""
    orch = LegalMemoryOrchestrator(encoder=offline_encoder, value_dim=2, engine_mode="structured")

    with orch.tenant("corporate_law"):
        orch.semantic_graph.add_node(
            "gdpr_art15",
            EntityType.STATUTE,
            "GDPR Art 15",
            "Right of access",
            tenant_id="corporate_law",
        )
        orch.semantic_graph.add_node(
            "gdpr_art12",
            EntityType.STATUTE,
            "GDPR Art 12",
            "Transparent communication modalities",
            tenant_id="corporate_law",
        )
        orch.semantic_graph.add_edge(
            "gdpr_art15",
            "gdpr_art12",
            RelationType.DEPENDS_ON,
            tenant_id="corporate_law",
        )

        results = orch.associate_statutes(seed_nodes=["gdpr_art15"], max_results=5)
        assert len(results) == 2
        node_ids = [r[0].node_id for r in results]
        assert "gdpr_art15" in node_ids
        assert "gdpr_art12" in node_ids
        assert results[0][1] >= results[1][1]
