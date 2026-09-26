# Multi-Tenancy & Partitioning Guide

Continuous Legal Memory enforces strict enterprise-grade multi-tenant partitioning across all symbolic, subsymbolic, and persistent data tiers.

## Tenant Isolation Guarantees

In multi-tenant legal applications (e.g. multi-client law practice management, distinct corporate departments, multi-jurisdiction compliance), data from Tenant Alpha must never influence decisions or be accessible to Tenant Beta.

Continuous Legal Memory guarantees:
1. **Working Memory**: Sliding window interaction turns are strictly segregated by `tenant_id`.
2. **Episodic Ledger**: Valid time index lookups, tombstone tracking, and hash-chain audits evaluate only matching tenant records.
3. **Semantic Knowledge Graph**: Cross-tenant edge insertion is strictly forbidden, raising `MemoryContradictionError`. Multi-hop dependency checks and Personalized PageRank traversals are isolated to the tenant partition.
4. **SQLite Persistence**: Schema-level indexing and queries enforce `WHERE tenant_id = ?` boundaries.

## Usage: Tenant Context Manager

The recommended pattern is using the `orchestrator.tenant(tenant_id)` context manager:

```python
with orchestrator.tenant("client_acme"):
    orchestrator.update_memory("Directive: Trade secrets policy.", [1.0, 0.0])
    result = orchestrator.predict("Can trade secret data be shared?")
```

All operations executed inside the `with` block implicitly inherit the specified tenant scope.
