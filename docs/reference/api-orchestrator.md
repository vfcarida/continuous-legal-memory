# API Reference: LegalMemoryOrchestrator

::: continuous_legal_memory.orchestrator.LegalMemoryOrchestrator

## Overview

The `LegalMemoryOrchestrator` is the unified facade providing the public interface for memory ingestion, rule deletion, tenant-scoped predictions, and cryptographic state attestation.

### Constructor Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `encoder` | `BaseEncoderPort \| None` | `None` | Embedding adapter. Defaults to Portuguese BERT or mock encoder in tests. |
| `value_dim` | `int` | `2` | Dimensionality of the legal decision / action vector. |
| `engine_mode` | `str` | `"structured"` | Execution mode: `"structured"` (deterministic precedence) or `"hybrid"` (neural research). |
| `memory_store` | `BaseMemoryStorePort \| None` | `None` | Persistent storage adapter (e.g. `SqliteMemoryStore`). |
| `db_path` | `str \| None` | `None` | SQLite database filepath if memory store is omitted. |
| `temperature` | `float` | `0.05` | Attention temperature for softmax scaling. |
| `seed` | `int \| None` | `42` | Random seed for deterministic reproducibility. |

### Key Methods

- `update_memory(text, action_vector, ...)`: Ingests a new legal rule or policy.
- `predict(query, ...)`: Retrieves governing rules and computes predicted decision vector.
- `tenant(tenant_id)`: Context manager for tenant-scoped operations.
- `delete_rule(record_id)`: Executes GDPR Art. 17 right to erasure.
