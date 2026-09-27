"""
Framework Integrations Guide: LangChain, LlamaIndex, and Letta (MemGPT).

This script demonstrates:
1. Connecting Continuous Legal Memory to LangChain via `ContinuousLegalMemoryLangChain`.
2. Querying legal precedence in LlamaIndex via `ContinuousLegalMemoryLlamaRetriever`.
3. Sinking active governing statutes into Letta (MemGPT) context via `ContinuousLegalMemoryBlock`.
"""

from __future__ import annotations

from continuous_legal_memory.adapters.mock_encoder import SemanticMockEncoder
from continuous_legal_memory.integrations.langchain import ContinuousLegalMemoryLangChain
from continuous_legal_memory.integrations.letta import ContinuousLegalMemoryBlock
from continuous_legal_memory.integrations.llamaindex import ContinuousLegalMemoryLlamaRetriever
from continuous_legal_memory.orchestrator import LegalMemoryOrchestrator


def run_integrations_demo() -> None:
    print("=" * 70)
    print("1. Initializing Core Legal Memory Orchestrator")
    print("=" * 70)

    encoder = SemanticMockEncoder(embedding_dim=64)
    orch = LegalMemoryOrchestrator(encoder=encoder, value_dim=2)

    # Ingest baseline rule
    orch.ingest_rule(
        rule_text="Master Services Agreement Section 12: Aggregate liability is capped at 12 months fees.",
        action_vector=[0.0, 1.0],
        authority_rank=5,
        tenant_id="legal_dept",
    )

    print("=" * 70)
    print("2. LangChain Memory Component (ContinuousLegalMemoryLangChain)")
    print("=" * 70)

    lc_mem = ContinuousLegalMemoryLangChain(
        orchestrator=orch,
        memory_key="legal_context",
        tenant_id="legal_dept",
    )
    lc_mem.save_context(
        inputs={"input": "What is the liability cap under the vendor agreement?"},
        outputs={"output": "Under MSA Section 12, liability is capped at 12 months fees."},
    )

    loaded_vars = lc_mem.load_memory_variables({"input": "liability cap"})
    print(f"Retrieved LangChain Legal Context:\n{loaded_vars.get('legal_context')}")
    assert "MSA" in str(loaded_vars) or "liability" in str(loaded_vars).lower()

    print("=" * 70)
    print("3. LlamaIndex Legal Retriever (ContinuousLegalMemoryLlamaRetriever)")
    print("=" * 70)

    llama_retriever = ContinuousLegalMemoryLlamaRetriever(
        orchestrator=orch,
        tenant_id="legal_dept",
        top_k=3,
    )
    nodes = llama_retriever.retrieve("vendor liability limitations")
    print(f"LlamaIndex Retrieved Nodes: {len(nodes)}")
    for node in nodes:
        print(f"  Node Text: {getattr(node, 'text', str(node))[:60]}... Score: {getattr(node, 'score', 1.0)}")
    assert len(nodes) >= 1

    print("=" * 70)
    print("4. Letta / MemGPT Memory Block (ContinuousLegalMemoryBlock)")
    print("=" * 70)

    letta_block = ContinuousLegalMemoryBlock(
        orchestrator=orch,
        name="legal_compliance_block",
        tenant_id="legal_dept",
    )
    compiled_text = letta_block.compile()
    print(f"Compiled Letta Memory Block Text ({len(compiled_text)} chars):\n{compiled_text}")
    assert "Active Legal Provisions" in compiled_text or "Section 12" in compiled_text

    print("=" * 70)
    print("Framework Integrations Demo completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    run_integrations_demo()
