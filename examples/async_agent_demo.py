"""
Asynchronous Legal Agent Reasoning Loop Example.

This script demonstrates:
1. Using non-blocking async methods (`async_ingest_rule`, `async_predict`).
2. High-throughput concurrent legal checks via `asyncio.gather`.
3. Seamless integration into modern asynchronous agentic architectures.
"""

from __future__ import annotations

import asyncio

from continuous_legal_memory.adapters.mock_encoder import SemanticMockEncoder
from continuous_legal_memory.orchestrator import LegalMemoryOrchestrator


async def run_async_agent_demo() -> None:
    print("=" * 70)
    print("1. Initializing Asynchronous Legal Memory Engine")
    print("=" * 70)

    encoder = SemanticMockEncoder(embedding_dim=64)
    orch = LegalMemoryOrchestrator(
        encoder=encoder,
        value_dim=2,
    )

    print("=" * 70)
    print("2. Ingesting Regulatory Directives Asynchronously")
    print("=" * 70)

    rules = [
        ("EU AI Act Article 5: Prohibited artificial intelligence practices involving biometric categorisation.", [1.0, 0.0], 10),
        ("EU AI Act Article 9: Risk management system requirements for high-risk AI systems.", [0.0, 1.0], 9),
        ("Internal Guidance: Preliminary research exploratory testing of biometric facial clustering.", [0.5, 0.5], 2),
    ]

    for text, action, rank in rules:
        rec = await orch.async_ingest_rule(
            rule_text=text,
            action_vector=action,
            authority_rank=rank,
            jurisdiction="EU",
        )
        print(f"Ingested async: {rec.text[:60]}... (Rank {rec.authority_rank})")

    print("=" * 70)
    print("3. Executing Concurrent Compliance Audits with asyncio.gather")
    print("=" * 70)

    cases = [
        "Deploying real-time public biometric facial identification in railway stations.",
        "Establishing continuous post-market monitoring and risk management for medical diagnostic AI.",
        "Evaluating customer feedback clustering algorithm without biometric inputs.",
    ]

    tasks = [orch.async_predict(case) for case in cases]
    results = await asyncio.gather(*tasks)

    for case, res in zip(cases, results):
        print(f"\nScenario: '{case}'")
        print(f"Governing Directive: {res.most_relevant_rule}")
        print(f"Decision Vector:     {res.predicted_action_vector}")

    print("=" * 70)
    print("Asynchronous Legal Agent Demo completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(run_async_agent_demo())
