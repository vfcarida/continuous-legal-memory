"""
Unit tests for Ecosystem Integrations (LangChain & LlamaIndex).
"""

from continuous_legal_memory.domain.interfaces import BaseEncoderPort
from continuous_legal_memory.integrations.langchain import ContinuousLegalMemoryLangChain
from continuous_legal_memory.integrations.llamaindex import (
    ContinuousLegalMemoryLlamaRetriever,
    LegalNodeResult,
)
from continuous_legal_memory.orchestrator import LegalMemoryOrchestrator


def test_langchain_memory_adapter_lifecycle(offline_encoder: BaseEncoderPort) -> None:
    """Verify LangChain memory variables, context loading, saving, and tenant scoping."""
    orch = LegalMemoryOrchestrator(encoder=offline_encoder, value_dim=2, engine_mode="structured")

    # Ingest legal rule
    orch.update_memory(
        "Article 15 GDPR: Right of access by the data subject.",
        [1.0, 0.0],
        authority_rank=7,
        tenant_id="client_legal",
    )

    lc_mem = ContinuousLegalMemoryLangChain(
        orchestrator=orch,
        memory_key="statutory_context",
        input_key="prompt",
        output_key="response",
        tenant_id="client_legal",
    )

    assert lc_mem.memory_variables == ["statutory_context"]

    # Load context for query matching the rule
    vars_dict = lc_mem.load_memory_variables({"prompt": "Can a user request access to their personal data?"})
    assert "statutory_context" in vars_dict
    context_str = vars_dict["statutory_context"]
    assert "Article 15 GDPR: Right of access by the data subject." in context_str
    assert "Recommended Decision Vector" in context_str

    # Test saving context into working memory
    lc_mem.save_context(
        {"prompt": "Data subject submitted an access form."},
        {"response": "Request acknowledged under Art. 15."},
    )
    active_wm = orch.working_memory.get_active_context(tenant_id="client_legal")
    assert len(active_wm) == 3
    assert "User: Data subject submitted an access form." in active_wm[1].text
    assert "Agent: Request acknowledged under Art. 15." in active_wm[2].text

    # Test clearing memory
    lc_mem.clear()
    assert len(orch.working_memory.get_active_context(tenant_id="client_legal")) == 0


def test_langchain_tenant_isolation(offline_encoder: BaseEncoderPort) -> None:
    """Verify LangChain memory isolates queries between tenants."""
    orch = LegalMemoryOrchestrator(encoder=offline_encoder, value_dim=2, engine_mode="structured")

    orch.update_memory(
        "Tenant Alpha Proprietary Policy: Remote work mandatory.",
        [1.0, 0.0],
        authority_rank=3,
        tenant_id="tenant_alpha",
    )

    lc_mem_beta = ContinuousLegalMemoryLangChain(
        orchestrator=orch,
        tenant_id="tenant_beta",
    )

    # Tenant Beta should receive empty memory for tenant Alpha's rule
    vars_beta = lc_mem_beta.load_memory_variables({"input": "What is the policy on remote work?"})
    assert vars_beta["legal_context"] == ""


def test_llamaindex_retriever_adapter(offline_encoder: BaseEncoderPort) -> None:
    """Verify LlamaIndex retriever interface, node generation, and metadata."""
    orch = LegalMemoryOrchestrator(encoder=offline_encoder, value_dim=2, engine_mode="structured")

    orch.update_memory(
        "Commercial Code 88: Bills of lading must be signed by the carrier.",
        [0.0, 1.0],
        authority_rank=5,
        tenant_id="logistics_dept",
    )

    retriever = ContinuousLegalMemoryLlamaRetriever(
        orchestrator=orch,
        tenant_id="logistics_dept",
    )

    nodes = retriever.retrieve("Requirements for signature on bill of lading")
    assert len(nodes) == 1
    node = nodes[0]

    assert isinstance(node, LegalNodeResult) or hasattr(node, "node")
    text = node.text if isinstance(node, LegalNodeResult) else node.node.text
    assert "Commercial Code 88: Bills of lading must be signed by the carrier." in text

    # Verify internal _retrieve hook
    nodes_internal = retriever._retrieve("bill of lading")
    assert len(nodes_internal) == 1


def test_letta_memory_block_lifecycle(offline_encoder: BaseEncoderPort) -> None:
    """Verify Letta memory block compilation, budget truncation, search, and graph association."""
    from continuous_legal_memory.domain.models import EntityType, RelationType
    from continuous_legal_memory.integrations.letta import ContinuousLegalMemoryBlock

    orch = LegalMemoryOrchestrator(encoder=offline_encoder, value_dim=2, engine_mode="structured")
    block = ContinuousLegalMemoryBlock(orchestrator=orch, tenant_id="tenant_letta", limit=2000)

    # Empty block
    assert block.compile() == "[Legal Precedence Memory: No active statutes loaded]"
    assert block.value == "[Legal Precedence Memory: No active statutes loaded]"

    # Insert rules
    msg1 = block.insert_rule("Article 6 GDPR: Lawfulness of processing.", [1.0, 0.0], authority_rank=8)
    assert "Successfully ingested rule" in msg1
    assert "(Rank 8)" in msg1

    msg2 = block.insert_rule("Internal Data Policy: Retain logs for 7 years.", [0.0, 1.0], authority_rank=3)
    assert "(Rank 3)" in msg2

    # Compiled block contains both directives
    compiled = block.compile()
    assert "[Legal Precedence Memory - Active Directives]" in compiled
    assert "(Rank 8) Article 6 GDPR: Lawfulness of processing." in compiled
    assert "(Rank 3) Internal Data Policy: Retain logs for 7 years." in compiled

    # Test character budget truncation
    tiny_block = ContinuousLegalMemoryBlock(orchestrator=orch, tenant_id="tenant_letta", limit=70)
    truncated = tiny_block.compile()
    assert "... [Truncated due to character limit]" in truncated

    # Test search
    res = block.search("Lawfulness of personal data processing")
    assert res["query"] == "Lawfulness of personal data processing"
    assert "Article 6 GDPR" in (res["most_relevant_rule"] or "")
    assert isinstance(res["predicted_action_vector"], list)

    # Test HippoRAG graph association via block
    orch.semantic_graph.add_node("gdpr_art6", EntityType.STATUTE, "GDPR Art 6", "Lawfulness", tenant_id="tenant_letta")
    orch.semantic_graph.add_node("dpa_guidance", EntityType.OBLIGATION, "DPA Guidance", "Enforcement", tenant_id="tenant_letta")
    orch.semantic_graph.add_edge("gdpr_art6", "dpa_guidance", RelationType.DEPENDS_ON, weight=1.0, tenant_id="tenant_letta")

    assoc = block.associate(["gdpr_art6"], max_results=2)
    assert len(assoc) > 0
    node_ids = [a["node_id"] for a in assoc]
    assert "gdpr_art6" in node_ids

    # Tenant isolation: another tenant should see empty block
    other_block = ContinuousLegalMemoryBlock(orchestrator=orch, tenant_id="isolated_tenant")
    assert other_block.compile() == "[Legal Precedence Memory: No active statutes loaded]"


def test_letta_agent_tools_execution(offline_encoder: BaseEncoderPort) -> None:
    """Verify Letta agent tool creation and string outputs for autonomous agents."""
    from continuous_legal_memory.domain.models import EntityType, RelationType
    from continuous_legal_memory.integrations.letta import create_letta_tools

    orch = LegalMemoryOrchestrator(encoder=offline_encoder, value_dim=2, engine_mode="structured")
    tools = create_letta_tools(orchestrator=orch, tenant_id="compliance_agent")

    assert "legal_memory_search" in tools
    assert "legal_memory_insert" in tools
    assert "legal_memory_associate" in tools

    # 1. Ingestion tool
    insert_res = tools["legal_memory_insert"](
        "FCPA: Prohibition of bribery of foreign public officials.",
        [1.0, 0.0],
        authority_rank=10,
    )
    assert "Successfully ingested rule" in insert_res
    assert "Rank 10" in insert_res

    # 2. Search tool
    search_res = tools["legal_memory_search"]("FCPA foreign bribery")
    assert "Governing Rule: FCPA: Prohibition of bribery of foreign public officials." in search_res
    assert "Action Vector: [1.0, 0.0]" in search_res

    # 3. Associate tool with empty graph
    empty_assoc = tools["legal_memory_associate"]("unknown_node")
    assert "No topological associations found" in empty_assoc

    # 4. Associate tool with graph nodes
    orch.semantic_graph.add_node("fcpa_base", EntityType.STATUTE, "FCPA", "Anti-Bribery", tenant_id="compliance_agent")
    orch.semantic_graph.add_node("sec_guideline", EntityType.OBLIGATION, "SEC Guidance", "Disclose payments", tenant_id="compliance_agent")
    orch.semantic_graph.add_edge("fcpa_base", "sec_guideline", RelationType.DEPENDS_ON, weight=1.0, tenant_id="compliance_agent")

    assoc_res = tools["legal_memory_associate"]("fcpa_base")
    assert "Associations for fcpa_base:" in assoc_res
    assert "fcpa_base" in assoc_res
    assert "PPR:" in assoc_res

