# LangChain Integration Guide

Continuous Legal Memory provides native integration with LangChain through [`ContinuousLegalMemoryLangChain`](../reference/api-orchestrator.md), implementing the standard `BaseMemory` interface.

## Overview

Unlike standard conversational buffers that treat dialogue history as a flat queue of strings, `ContinuousLegalMemoryLangChain` intercepts incoming prompts, queries the underlying multi-tier legal engine, and injects governing statutory rules, authority rankings, and recommended decision vectors directly into the prompt context.

## Installation

Ensure LangChain is available in your environment:

```bash
pip install continuous-legal-memory langchain-core
```

## Basic Usage

```python
from continuous_legal_memory.orchestrator import LegalMemoryOrchestrator
from continuous_legal_memory.integrations.langchain import ContinuousLegalMemoryLangChain

# 1. Initialize the orchestrator
orchestrator = LegalMemoryOrchestrator(value_dim=2, engine_mode="structured")

# 2. Ingest legal directives into memory
orchestrator.update_memory(
    rule_text="Article 15 GDPR: The data subject shall have the right to obtain confirmation as to whether or not personal data concerning him or her are being processed.",
    action_vector=[1.0, 0.0],
    authority_rank=7,
    tenant_id="enterprise_client",
)

# 3. Create LangChain memory adapter
memory = ContinuousLegalMemoryLangChain(
    orchestrator=orchestrator,
    memory_key="statutory_context",
    input_key="input",
    output_key="output",
    tenant_id="enterprise_client",
)

# 4. Load memory variables for a user prompt
variables = memory.load_memory_variables({"input": "Can the user request their stored activity logs?"})
print(variables["statutory_context"])
```

## Multi-Tenant Context Scoping

Every memory adapter can be scoped to a specific tenant to prevent cross-client leakage:

```python
tenant_a_memory = ContinuousLegalMemoryLangChain(orchestrator=orchestrator, tenant_id="tenant_a")
tenant_b_memory = ContinuousLegalMemoryLangChain(orchestrator=orchestrator, tenant_id="tenant_b")
```
