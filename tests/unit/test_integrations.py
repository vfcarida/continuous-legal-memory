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
