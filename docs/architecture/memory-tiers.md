# Multi-Tier Cognitive Memory Hierarchy

Continuous Legal Memory organizes legal knowledge into three distinct, specialized tiers:

---

## 1. Working Memory (Short-Term Active Context)

- **Class**: `WorkingMemory`
- **Scope**: Current conversational session / active legal matter.
- **Eviction**: Sliding window capacity (`max_items`). When capacity is exceeded, oldest non-pinned items are evicted.
- **Tenant Isolation**: Separate context buffers per tenant.

---

## 2. Episodic Memory (Bi-Temporal Ledger)

- **Class**: `EpisodicMemory`
- **Scope**: Historical legal events, past contract interpretations, and ingested regulatory rulings.
- **Integrity**: Append-only hash chain linking each ingested event to its predecessor via SHA-256 digests.
- **Temporal Validity**: Validates `valid_from` and `valid_to` timestamps at inference time to prevent stale recall of expired rules.

---

## 3. Semantic Knowledge Graph (Structural Relationships)

- **Class**: `SemanticKnowledgeGraph`
- **Scope**: Structural statutory hierarchy, definitions, cross-references, and explicit `SUPERSEDES` relationships.
- **Conflict Prevention**: Validates new edges against existing nodes to prevent contradictory links.
