"""
Quickstart Guide for Continuous Legal Memory (CLM).

This script demonstrates:
1. Initializing the LegalMemoryOrchestrator in structured mode.
2. Ingesting legal directives with bitemporal timestamps and authority ranks.
3. Evaluating legal precedence (lex superior derogat legi inferiori).
4. Running HippoRAG Personalized PageRank over the semantic knowledge graph.
"""

from __future__ import annotations

from datetime import datetime, timezone

from continuous_legal_memory.adapters.mock_encoder import SemanticMockEncoder
from continuous_legal_memory.domain.models import EntityType, RelationType
from continuous_legal_memory.orchestrator import LegalMemoryOrchestrator


def run_quickstart_demo() -> None:
    print("=" * 70)
    print("1. Initializing Continuous Legal Memory Engine")
    print("=" * 70)

    # Use SemanticMockEncoder for instant, offline, deterministic semantic embeddings
    encoder = SemanticMockEncoder(embedding_dim=64)
    orch = LegalMemoryOrchestrator(
        encoder=encoder,
        value_dim=2,  # Binary decision space: [Deny/Reject, Grant/Allow]
        engine_mode="structured",  # Default sub-10ms deterministic engine
    )

    print("=" * 70)
    print("2. Ingesting Hierarchical Legal Directives")
    print("=" * 70)

    # Ingest Federal Statute (Authority Rank 10)
    # Action [0.0, 1.0] corresponds to "Grant Access / Retain Record"
    statute = orch.ingest_rule(
        rule_text="Federal Banking Act: Banks must retain transaction audit logs for 5 years.",
        action_vector=[0.0, 1.0],
        authority_rank=10,
        jurisdiction="US-FED",
        metadata={
            "node_id": "statute_banking_act",
            "entity_type": EntityType.STATUTE,
            "label": "Banking Act 5y Retention",
        },
    )
    print(f"Ingested Statute: ID={statute.record_id}, Rank={statute.authority_rank}")

    # Ingest Lower-Ranked Corporate Policy (Authority Rank 2)
    # Action [1.0, 0.0] corresponds to "Deny Access / Purge Record"
    policy = orch.ingest_rule(
        rule_text="Internal Corporate Policy: Erase all customer logs after 6 months to minimize storage costs.",
        action_vector=[1.0, 0.0],
        authority_rank=2,
        jurisdiction="CORP-INTERNAL",
        metadata={
            "node_id": "policy_storage_cleanup",
            "entity_type": EntityType.CLAUSE,
            "label": "Internal 6m Cleanup",
            "relations": [
                {
                    "target_id": "statute_banking_act",
                    "relation_type": RelationType.CONTRADICTS,
                    "weight": 1.0,
                }
            ],
        },
    )
    print(f"Ingested Policy:  ID={policy.record_id}, Rank={policy.authority_rank}")

    print("=" * 70)
    print("3. Querying with Conflict Resolution (Lex Superior)")
    print("=" * 70)

    query = "Audit department requests customer banking transaction records from 2 years ago."
    prediction = orch.predict(query, at_time=datetime.now(timezone.utc))

    print(f"Query: '{query}'")
    print(f"Most Relevant Statute: {prediction.most_relevant_rule}")
    print(f"Decision Vector:       {prediction.predicted_action_vector}")

    # Index 1 (Retain / Allow) must win because Rank 10 overrides Rank 2
    if prediction.predicted_action_vector[1] > prediction.predicted_action_vector[0]:
        print("Outcome: RETAIN / GRANT (Federal Statute strictly overrides Corporate Policy)")
    else:
        print("Outcome: PURGE / DENY")

    print("=" * 70)
    print("4. HippoRAG Knowledge Graph Associative Reasoning")
    print("=" * 70)

    associated = orch.associate_statutes(
        seed_nodes=["policy_storage_cleanup"],
        damping=0.85,
        max_results=5,
    )
    for node, score in associated:
        print(f"Node: {node.node_id:<25} Label: {node.label:<25} PPR Score: {score:.4f}")

    print("=" * 70)
    print("Quickstart Demo completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    run_quickstart_demo()
