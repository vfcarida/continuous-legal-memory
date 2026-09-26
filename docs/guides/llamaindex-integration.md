# LlamaIndex Integration Guide

Continuous Legal Memory provides native retrieval integration with LlamaIndex through [`ContinuousLegalMemoryLlamaRetriever`](../reference/api-orchestrator.md), implementing the standard `BaseRetriever` interface.

## Overview

`ContinuousLegalMemoryLlamaRetriever` wraps the Continuous Legal Memory engine into an index retriever. Query evaluation executes legal precedence arbitration (*lex superior*, *lex posterior*, and explicit supersession), returning ranked nodes packaged with metadata attributes (e.g. `authority_rank`, `predicted_action_vector`, `retrieved_snippets`).

## Installation

```bash
pip install continuous-legal-memory llama-index-core
```

## Basic Usage

```python
from continuous_legal_memory.orchestrator import LegalMemoryOrchestrator
from continuous_legal_memory.integrations.llamaindex import ContinuousLegalMemoryLlamaRetriever

# 1. Initialize orchestrator
orchestrator = LegalMemoryOrchestrator(value_dim=2, engine_mode="structured")

# 2. Ingest legal directives
orchestrator.update_memory(
    rule_text="Commercial Code Sec. 88: Bills of lading must be authenticated by the carrier.",
    action_vector=[0.0, 1.0],
    authority_rank=5,
    tenant_id="logistics_dept",
)

# 3. Create LlamaIndex retriever
retriever = ContinuousLegalMemoryLlamaRetriever(
    orchestrator=orchestrator,
    tenant_id="logistics_dept",
)

# 4. Retrieve nodes
nodes = retriever.retrieve("Requirements for bill of lading carrier signatures")
for n in nodes:
    print(n.text, n.score, n.metadata)
```
