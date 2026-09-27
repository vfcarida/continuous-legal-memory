# Asynchronous Agent Workflows

Continuous Legal Memory provides first-class asynchronous methods designed specifically for event-driven legal agents, FastAPI / Starlette microservices, and asynchronous agent frameworks (such as AutoGen, LangGraph, and CrewAI).

---

## 1. Asynchronous API Overview

All core operations in `LegalMemoryOrchestrator` offer non-blocking coroutine counterparts that delegate synchronous computation and SQLite writes to worker threads via `asyncio.to_thread`:

| Synchronous Method | Asynchronous Method | Description |
| :--- | :--- | :--- |
| `update_memory(...)` / `ingest_rule(...)` | `async_ingest_rule(...)` / `async_update_memory(...)` | Non-blocking rule ingestion and cognitive graph update |
| `predict(...)` | `async_predict(...)` | Non-blocking decision prediction and precedence scoring |
| `delete_rule(...)` | `async_delete_rule(...)` | Non-blocking Right-to-Erasure multi-tier purge |
| `associate_statutes(...)` | `async_associate_statutes(...)` | Non-blocking HippoRAG Personalized PageRank diffusion |

---

## 2. Ingesting and Querying Asynchronously

```python
import asyncio
from continuous_legal_memory.orchestrator import LegalMemoryOrchestrator
from continuous_legal_memory.adapters.mock_encoder import SemanticMockEncoder

async def main() -> None:
    encoder = SemanticMockEncoder(embedding_dim=64)
    orch = LegalMemoryOrchestrator(encoder=encoder, value_dim=2)

    # Ingest without blocking the event loop
    record = await orch.async_ingest_rule(
        rule_text="GDPR Article 17: Data subjects have the right to request erasure without undue delay.",
        action_vector=[1.0, 0.0],
        authority_rank=10,
        jurisdiction="EU",
    )
    print(f"Ingested record: {record.record_id}")

    # Predict decision asynchronously
    result = await orch.async_predict(
        query_text="Customer exercises right to be forgotten for account profile."
    )
    print(f"Governing rule: {result.most_relevant_rule}")
    print(f"Action vector:  {result.predicted_action_vector}")

asyncio.run(main())
```

---

## 3. High-Throughput Concurrent Audits with `asyncio.gather`

Because `LegalMemoryOrchestrator` uses reentrant locking (`threading.RLock`) and thread-safe memory buffers, multiple asynchronous queries can be processed concurrently across worker threads:

```python
import asyncio

async def audit_cases(orch: LegalMemoryOrchestrator, case_descriptions: list[str]) -> list:
    tasks = [orch.async_predict(case) for case in case_descriptions]
    # Dispatches all queries concurrently
    results = await asyncio.gather(*tasks)
    return results
```

---

## 4. Asynchronous Right-to-Erasure (GDPR Art. 17)

When processing automated data subject access requests (DSARs), asynchronous deletion purges the target record across the Episodic ledger, Working Memory, Semantic Knowledge Graph, and HopeModule retrieval buffers while generating an immutable tombstone digest in the cryptographic audit trail:

```python
audit_report = await orch.async_delete_rule(
    rule_id="target_record_id_123",
    tenant_id="client_tenant_alpha",
)
print(f"Erasure timestamp: {audit_report.get('timestamp')}")
print(f"Hash chain sequence: {audit_report.get('sequence')}")
```

---

## 5. Runnable Example

A complete end-to-end asynchronous agent script is provided in the repository at [examples/async_agent_demo.py](https://github.com/vfcarida/continuous-legal-memory/blob/main/examples/async_agent_demo.py). Execute it locally:

```bash
python examples/async_agent_demo.py
```
