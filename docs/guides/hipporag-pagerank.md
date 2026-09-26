# HippoRAG-Style Personalized PageRank Guide

Continuous Legal Memory features an associative memory layer over its `SemanticKnowledgeGraph` inspired by **HippoRAG** (Gutiérrez et al., NeurIPS 2024), diffusing query activation mass across legal entities via Personalized PageRank (PPR).

## Why Graph Diffusion for Legal Memory?

In legal reasoning, statutes and regulations are rarely self-contained. A governing legal clause frequently depends on:
1. Definitions defined in introductory sections.
2. Cross-referenced administrative exceptions.
3. Subordinate procedural obligations.

Standard vector retrieval matches only the text snippet directly similar to the prompt keywords, missing prerequisite obligations.

## Algorithmic Formulation

Given a query, initial seed nodes $\mathbf{v}$ are identified. The stationary activation vector $\mathbf{p}$ is solved via power iteration:

$$\mathbf{p}^{(t+1)} = (1 - \alpha) \mathbf{v} + \alpha \mathbf{W} \mathbf{p}^{(t)}$$

where $\alpha$ is the damping factor (default $0.85$) and $\mathbf{W}$ is the edge weight transition matrix.

## Usage Example

```python
from continuous_legal_memory.orchestrator import LegalMemoryOrchestrator
from continuous_legal_memory.domain.models import EntityType, RelationType

orchestrator = LegalMemoryOrchestrator(value_dim=2, engine_mode="structured")

with orchestrator.tenant("firm_tax"):
    # Add nodes
    orchestrator.semantic_graph.add_node("art_10", EntityType.STATUTE, "Tax Relief", "General corporate relief")
    orchestrator.semantic_graph.add_node("art_10_sec2", EntityType.CLAUSE, "Export Exception", "Must report export turnover")
    orchestrator.semantic_graph.add_edge("art_10", "art_10_sec2", RelationType.DEPENDS_ON)

    # Perform HippoRAG associative retrieval from seed
    associated = orchestrator.associate_statutes(seed_nodes=["art_10"], max_results=5)
    for node, score in associated:
        print(f"Node: {node.node_id} ({node.label}) -> PPR Score: {score:.4f}")
```
